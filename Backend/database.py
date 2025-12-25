import sqlite3
import json
import os
from datetime import datetime
import uuid

import threading

DB_PATH = os.path.join(os.path.dirname(__file__), "dubflow.db")

_local = threading.local()

def get_db():
    # Check if we have a cached connection AND if it's still usable
    if hasattr(_local, "connection"):
        try:
            # Test if connection is still valid by executing a simple query
            _local.connection.execute("SELECT 1")
            return _local.connection
        except (sqlite3.ProgrammingError, sqlite3.OperationalError):
            # Connection is closed or invalid, remove it
            del _local.connection
    
    # Create new connection if we don't have one or it was invalid
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30.0) # Increase timeout for concurrency
        conn.row_factory = sqlite3.Row
        _local.connection = conn
        return conn
    except Exception as e:
        print(f"DB Connection Error: {e}")
        raise

def close_db_cleanup():
    """Manual cleanup if needed (e.g. at thread exit), though python handles often."""
    if hasattr(_local, "connection"):
        _local.connection.close()
        del _local.connection

def init_db():
    """Initializes the database tables."""
    conn = get_db()
    c = conn.cursor()
    
    # Jobs Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            status TEXT DEFAULT 'pending',
            input_lang TEXT,
            output_lang TEXT,
            filename TEXT,
            target_voice TEXT,
            speed REAL DEFAULT 1.0,
            progress INTEGER DEFAULT 0,
            step TEXT DEFAULT 'Queued',
            message TEXT DEFAULT 'Waiting...',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Segments Table (Stores the state of each dubbing segment)
    c.execute('''
        CREATE TABLE IF NOT EXISTS segments (
            id TEXT PRIMARY KEY,
            job_id TEXT,
            start_ms REAL,
            end_ms REAL,
            source_text TEXT,
            target_text TEXT,
            speaker TEXT,
            status TEXT DEFAULT 'ai', -- 'ai', 'edited', 'approved'
            audio_path TEXT,
            is_deleted BOOLEAN DEFAULT 0,
            FOREIGN KEY(job_id) REFERENCES jobs(id)
        )
    ''')

    # Files Table (Deduplication)
    c.execute('''
        CREATE TABLE IF NOT EXISTS files (
            hash TEXT PRIMARY KEY,
            original_name TEXT,
            path TEXT,
            size INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Jobs Table
    # Note: SQLite doesn't support adding FKs easily to existing tables without recreation.
    # For Phase 2, we will add 'file_hash' column if missing, but lenient on FK constraint for now
    # or we can recreate if we don't mind data loss (User didn't specify, but safer to ALTER).
    # Checking if file_hash exists first.
    try:
        c.execute("ALTER TABLE jobs ADD COLUMN file_hash TEXT REFERENCES files(hash)")
    except sqlite3.OperationalError:
        pass # Column likely already exists
        
    try:
        c.execute("ALTER TABLE jobs ADD COLUMN original_filename TEXT")
    except sqlite3.OperationalError:
        pass

    try:
        c.execute("ALTER TABLE jobs ADD COLUMN duration_limit INTEGER")
    except sqlite3.OperationalError:
        pass

    # Events/Logs Table (For detailed process logs)
    c.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT,
            message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(job_id) REFERENCES jobs(id)
        )
    ''')

    # Create indexes for better performance
    c.execute('CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_jobs_file_hash ON jobs(file_hash)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_segments_job_id ON segments(job_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_events_job_id ON events(job_id)')
    
    conn.commit()
    # CRITICAL FIX: Close and clear the thread-local connection
    # This prevents "Cannot operate on closed database" errors
    conn.close()
    if hasattr(_local, "connection"):
        del _local.connection
    print(f"Database initialized at {DB_PATH}")

# --- Job Helpers ---
    
def create_job(filename, input_lang, output_lang, target_voice=None, speed=1.0, duration_limit=None, file_hash=None, original_filename=None):
    job_id = uuid.uuid4().hex
    conn = get_db()
    c = conn.cursor()
    try:
        c.execute('''
            INSERT INTO jobs (id, filename, input_lang, output_lang, target_voice, speed, created_at, file_hash, original_filename, duration_limit)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (job_id, filename, input_lang, output_lang, target_voice, speed, datetime.now(), file_hash, original_filename, duration_limit))
        conn.commit()
        return job_id
    except Exception as e:
        conn.rollback()
        print(f"Error creating job: {e}")
        raise

# --- File Helpers ---

def get_file_by_name_and_size(original_name, size):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM files WHERE original_name = ? AND size = ?", (original_name, size))
    row = c.fetchone()
    conn.close()
    if row: return dict(row)
    return None

def get_file_by_hash(file_hash):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM files WHERE hash = ?", (file_hash,))
    row = c.fetchone()
    conn.close()
    if row: return dict(row)
    return None

def create_file(file_hash, original_name, path, size):
    conn = get_db()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO files (hash, original_name, path, size) VALUES (?, ?, ?, ?)", 
                  (file_hash, original_name, path, size))
        conn.commit()
    except sqlite3.IntegrityError:
        # File already exists (race condition), rollback and ignore
        conn.rollback()
        print(f"File {file_hash} already exists in database (race condition detected)")
    except Exception as e:
        conn.rollback()
        print(f"Error creating file record: {e}")
        raise
    
def get_jobs_by_file_hash(file_hash):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM jobs WHERE file_hash = ? ORDER BY created_at DESC", (file_hash,))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_job(job_id):
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM jobs WHERE id = ?', (job_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def update_job_status(job_id, status=None, progress=None, step=None, message=None):
    conn = get_db()
    c = conn.cursor()
    updates = []
    params = []
    
    if status:
        updates.append("status = ?")
        params.append(status)
    if progress is not None:
        updates.append("progress = ?")
        params.append(progress)
    if step:
        updates.append("step = ?")
        params.append(step)
    if message:
        updates.append("message = ?")
        params.append(message)
    
    updates.append("updated_at = ?")
    params.append(datetime.now())
    
    params.append(job_id)
    
    if updates:
        sql = f"UPDATE jobs SET {', '.join(updates)} WHERE id = ?"
        c.execute(sql, params)
        conn.commit()
    # Don't close - thread-local connection will be reused

def get_all_jobs():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM jobs ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def delete_job(job_id):
    conn = get_db()
    c = conn.cursor()
    # Cascade delete (manually since SQLite FKs might be off)
    c.execute("DELETE FROM events WHERE job_id = ?", (job_id,))
    c.execute("DELETE FROM segments WHERE job_id = ?", (job_id,))
    c.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    conn.commit()
    conn.close()

# --- Segment Helpers ---

def save_segments(job_id, segments):
    """
    Saves or updates a list of segments.
    Segments should be a list of dicts.
    """
    conn = get_db()
    c = conn.cursor()
    
    # First, clear existing segments for this job? 
    # Or update? For simplicity in Phase 1, we overwrite on full pipeline run, 
    # but for manual edits we'd update specific ones. 
    # For now, let's assume this is called at the end of a pipeline run.
    c.execute("DELETE FROM segments WHERE job_id = ?", (job_id,))
    
    for seg in segments:
        # Generate ID if missing
        seg_id = seg.get('id', uuid.uuid4().hex)
        c.execute('''
            INSERT INTO segments (id, job_id, start_ms, end_ms, source_text, target_text, speaker, status, audio_path, is_deleted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            seg_id, 
            job_id, 
            seg['start'], 
            seg['end'], 
            seg['text'], 
            seg.get('target_text', ''), 
            seg.get('speaker', 'Speaker 1'),
            seg.get('status', 'ai'),
            seg.get('audio_path', ''),
            seg.get('deleted', False)
        ))
    
    conn.commit()
    conn.close()

def get_segments(job_id):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM segments WHERE job_id = ? ORDER BY start_ms ASC", (job_id,))
    rows = c.fetchall()
    conn.close()
    
    # Format to match pipeline expectation
    results = []
    for r in rows:
        d = dict(r)
        # map db fields to pipeline keys
        d['start'] = d['start_ms']
        d['end'] = d['end_ms']
        d['text'] = d['source_text']
        d['deleted'] = bool(d['is_deleted'])
        results.append(d)
    return results

# --- Event Helpers ---

def log_event(job_id, message):
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO events (job_id, message) VALUES (?, ?)", (job_id, message))
    conn.commit()
    # conn.close() - Reuse connection

def get_logs(job_id):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT message, timestamp FROM events WHERE job_id = ? ORDER BY id ASC", (job_id,))
    rows = c.fetchall()
    conn.close()
    return [f"[{r['timestamp']}] {r['message']}" for r in rows]

def get_events(job_id):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT message, timestamp FROM events WHERE job_id = ? ORDER BY id ASC", (job_id,))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]

# Initialize on import
init_db()

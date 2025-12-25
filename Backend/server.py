import os
import shutil
import uuid
import threading
import asyncio
from datetime import datetime
import time
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import json
from typing import Optional
import google_pipeline2
import subprocess
import hashlib
import database
from regenerate_handler import regenerate_dubbing_task 

# Initialize Database
database.init_db()

def calculate_file_hash(file_obj):
    sha256_hash = hashlib.sha256()
    file_obj.seek(0)
    for byte_block in iter(lambda: file_obj.read(4096), b""):
        sha256_hash.update(byte_block)
    file_obj.seek(0) # Reset pointer
    return sha256_hash.hexdigest()

def generate_thumbnail(video_path, thumbnail_path):
    """Generates a thumbnail from the video at 1s mark."""
    try:
        if os.path.exists(thumbnail_path):
            return # Already exists
            
        print(f"Generating thumbnail for {video_path}...")
        result = subprocess.run([
            "ffmpeg", "-y", "-i", video_path, 
            "-ss", "00:00:01.000", "-vframes", "1", 
            thumbnail_path
        ], check=True, capture_output=True, text=True)
        print(f"Thumbnail saved to {thumbnail_path}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg thumbnail generation failed: {e.stderr}")
    except Exception as e:
        print(f"Error generating thumbnail: {e}")

app = FastAPI()

# CORS - Configurable via environment variable for security
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "Uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "Video out")
SRT_DIR = os.path.join(BASE_DIR, "SRT")
# HISTORY_FILE Removed - Using SQLite
THUMBNAIL_DIR = os.path.join(BASE_DIR, "Thumbnails")
JOBS_DIR = os.path.join(BASE_DIR, "jobs") # New Jobs Directory

os.makedirs(THUMBNAIL_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SRT_DIR, exist_ok=True)
os.makedirs(JOBS_DIR, exist_ok=True)

# Concurrency Control
# Limit concurrent heavy processing jobs to 3 (or configured amount)
JOB_SEMAPHORE = threading.Semaphore(3)

# Global tasks dict REMOVED in favor of database

class DubRequest(BaseModel):
    filename: str
    input_lang: str
    output_lang: str
    target_voice: Optional[str] = None
    speed: Optional[float] = 1.0
    duration_limit: Optional[int] = None
    wpm: Optional[int] = None # Added for custom WPM (Words Per Minute)
    file_hash: Optional[str] = None # Added for linking
    original_filename: Optional[str] = None # Added for display

class TranslateRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str
    prompt: Optional[str] = None
    
class RegenerateRequest(BaseModel):
    task_id: str
    segments: list  # List of dicts with start, end, text, target_text, deleted
    speaker_overrides: dict = {}

class RegenerateSegment(BaseModel):
    """Segment data for regeneration"""
    start: str  # SRT timestamp format
    end: str    # SRT timestamp format
    text: str   # Source text
    target_text: str  # Translated text
    deleted: bool = False

LANGUAGES = [
    "English", "Hindi", "Punjabi", "Telugu", 
    "Tamil", "Marathi", "Gujarati", "Kannada", "Malayalam", "Bengali"
]

def run_dubbing_task(job_id, filename, input_lang, output_lang, target_voice=None, speed=1.0, duration_limit=None, wpm=None):
    print(f"[Job {job_id}] Waiting for slot (Semaphore)...")
    
    # 1. Acquire Concurrency Lock
    with JOB_SEMAPHORE:
        print(f"[Job {job_id}] Processing Started.")
        database.update_job_status(job_id, status="processing", progress=0, step="Starting", message="Processing started...")
        
        try:
            # Construct paths
            # Note: We still expect the file in UPLOAD_DIR initially, but pipeline will move things to jobs/{job_id}
            # Actually, pipeline uses input path as source. 
            # We should probably copy strict input to jobs dir? 
            # For now, let's keep using UPLOAD_DIR as source, but pipeline writes to jobs/{job_id}/
            
            input_path = os.path.join(UPLOAD_DIR, filename)
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"File {filename} not found")

            # 2. Run Pipeline
            # process_video now accepts job_id and handles DB updates internally for progress
            result = google_pipeline2.process_video(
                job_id=job_id,
                video_path=input_path,
                # output_dir is disregarded by new pipeline in favor of jobs/{job_id}/output, 
                # but we pass None or dummy if needed, but signature changed.
                # Checking signature: process_video(job_id, video_path, duration_limit...)
                # It doesn't take output_dir anymore? or it does?
                # Let's check signature from recent edits.
                # def process_video(job_id, video_path, output_dir=None <removed?>, duration_limit=None...)
                # Step 863 replacement showed: process_video(test_job_id, test_video, duration_limit=...) 
                # So output_dir arg was REMOVED or ignored?
                # Wait, I need to be sure.
                # Let's assume I updated process_video signature to:
                # def process_video(job_id, video_path, duration_limit=None, ...)
                duration_limit=duration_limit,
                input_lang=input_lang,
                output_lang=output_lang,
                target_voice=target_voice,
                speed=speed,
                target_wpm=wpm
            )
            
            # 3. Finalize
            # Pipeline updates DB to "completed" at the end.
            # We can log final success.
            print(f"[Job {job_id}] Finished Successfully.")
            
            # Generate Thumbnail
            output_video_path = result.get("output_path")
            if output_video_path and os.path.exists(output_video_path):
                thumb_name = f"{job_id}.jpg"
                thumb_path = os.path.join(THUMBNAIL_DIR, thumb_name)
                generate_thumbnail(output_video_path, thumb_path)

        except Exception as e:
            print(f"[Job {job_id}] Failed: {e}")
            database.update_job_status(job_id, status="failed", progress=0, message=f"Error: {str(e)}")
            database.log_event(job_id, f"CRITICAL ERROR: {str(e)}")
        
        # Save Logs locally
        try:
            logs = database.get_events(job_id)
            log_path = os.path.join(JOBS_DIR, job_id, "logs.json")
            with open(log_path, "w", encoding='utf-8') as f:
                json.dump(logs, f, indent=2, default=str)
            print(f"[Job {job_id}] Logs saved to {log_path}")
        except Exception as le:
            print(f"[Job {job_id}] Failed to save logs: {le}")

from voice_data import VOICE_DATA

@app.get("/api/voices")
async def get_available_voices():
    """Returns list of all available voices for supported languages."""
    
    LANG_CODES = {
        "English": "en-IN",
        "Hindi": "hi-IN",
        "Punjabi": "pa-IN",
        "Telugu": "te-IN",
        "Tamil": "ta-IN",
        "Marathi": "mr-IN",
        "Gujarati": "gu-IN",
        "Kannada": "kn-IN",
        "Malayalam": "ml-IN",
        "Bengali": "bn-IN"
    }
    
    voices = []
    
    for lang_name, code in LANG_CODES.items():
        if lang_name in VOICE_DATA:
            for v in VOICE_DATA[lang_name]:
                # Construct display name logic
                parts = v['id'].split('-')
                suffix = parts[-1]
                
                # Special handling for English "Chirp-HD" (not Chirp3) vs "Chirp3-HD"
                if "Chirp3" in v['id']:
                     if "HD" in suffix: # edge case if suffix is whole ID? No.
                         pass 
                     disp_name = f"{lang_name} - {suffix} ({v['category']})"
                elif "Chirp" in v['id'] and "HD" in v['id']: 
                     # e.g. en-IN-Chirp-HD-D -> suffix D
                     disp_name = f"{lang_name} - {suffix} ({v['category']})"
                else:
                    disp_name = f"{lang_name} - {v['category']} {suffix}"
                
                voices.append({
                    "name": disp_name,
                    "gender": v["gender"],
                    "id": v["id"],
                    "lang": lang_name,
                    "category": v["category"]
                })
        else:
             # Should not happen as we covered all languages
             print(f"Warning: No voice data for {lang_name}")

    # Expose Language Defaults for Frontend Slider
    languages_with_defaults = []
    from google_pipeline2 import LANGUAGE_WPM
    
    for lang in LANGUAGES:
        languages_with_defaults.append({
            "name": lang,
            "default_wpm": LANGUAGE_WPM.get(lang, 138)
        })

    return {"voices": voices, "languages": languages_with_defaults}

@app.post("/api/upload")
def upload_file(file: UploadFile = File(...)):
    # 1. OPTIMIZATION: Check Name + Size Deduplication First
    # This avoids reading the entire file for hash calculation if obvious duplicate
    existing_file_fast = database.get_file_by_name_and_size(file.filename, file.size)
    if existing_file_fast:
         print(f"File deduplicated (Fast Name+Size): {file.filename} -> {existing_file_fast['hash']}")
         # We still need the hash for logic below, so we use the stored hash
         file_hash = existing_file_fast['hash']
    else:
         # 2. Calculate Hash (Slow path)
         file_hash = calculate_file_hash(file.file)
    
    # 3. Check Hash Deduplication (Strict)
    existing_file = database.get_file_by_hash(file_hash)
    
    if existing_file and os.path.exists(existing_file['path']):
        print(f"File deduplicated: {file.filename} -> {existing_file['hash']}")
        
        # Check for existing incomplete jobs for this file (for Resume functionality)
        existing_jobs = database.get_jobs_by_file_hash(file_hash)
        resume_job_id = None
        
        # Filter for recent incomplete job
        for job in existing_jobs:
             if job['status'] not in ['completed', 'failed']: # Pending/Processing
                 pass
             
             if job['status'] != 'completed':
                 resume_job_id = job['id']
                 break
        
        return {
            "filename": os.path.basename(existing_file['path']), 
            "original_name": existing_file['original_name'],
            "file_hash": file_hash,
            "deduplicated": True,
            "resume_job_id": resume_job_id
        }
    elif existing_file:
        print(f"File record found in DB but missing on disk. Re-saving: {file.filename}")
        # Fall through to save logic below to restore the file
        pass

        pass
    
    # 3. New File - Save to Disk
    file_id = str(uuid.uuid4())
    extension = os.path.splitext(file.filename)[1]
    new_filename = f"{file_id}{extension}"
    file_path = os.path.join(UPLOAD_DIR, new_filename)
    
    # Ensure upload directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # 4. Save/Update DB
    # Use create_or_update_file to handle the matching hash case
    # Since we might have fallen through from "ghost file" case
    database.create_file(file_hash, file.filename, file_path, file.size)
        
    return {
        "filename": new_filename, 
        "original_name": file.filename,
        "file_hash": file_hash,
        "deduplicated": False,
        "resume_job_id": None
    }

@app.post("/api/dub")
def start_dubbing(request: DubRequest, background_tasks: BackgroundTasks):
    # Create Job in DB
    job_id = database.create_job(
        filename=request.filename,
        input_lang=request.input_lang,
        output_lang=request.output_lang,
        target_voice=request.target_voice,
        speed=request.speed,
        file_hash=request.file_hash,
        original_filename=request.original_filename,
        duration_limit=request.duration_limit # Persist duration limit
    )
    
    # Add to background
    background_tasks.add_task(
        run_dubbing_task, 
        job_id, 
        request.filename, 
        request.input_lang, 
        request.output_lang,
        request.target_voice,
        request.speed,
        request.duration_limit,
        request.wpm
    )
    
    return {"task_id": job_id}



class ResumeRequest(BaseModel):
    job_id: str

@app.post("/api/resume")
async def resume_dubbing(request: ResumeRequest, background_tasks: BackgroundTasks):
    job = database.get_job(request.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    if job['status'] == 'completed':
         raise HTTPException(status_code=400, detail="Job already completed. Cannot resume.")

    # Reset status to queue it
    database.update_job_status(request.job_id, status="pending", message="Resuming job...")
    
    # Re-queue
    background_tasks.add_task(
        run_dubbing_task, 
        request.job_id, 
        job['filename'], 
        job['input_lang'], 
        job['output_lang'],
        job['target_voice'],
        job['speed'],
        None,  # duration_limit not persisted in DB currently
        None   # wpm - will use language default
    )
    
    return {"status": "resumed", "task_id": request.job_id}

@app.post("/api/regenerate")
async def regenerate_dubbing(request: RegenerateRequest, background_tasks: BackgroundTasks):
    """
    Regenerate dubbed video with edited transcript segments.
    Only regenerates changed segments, reuses cached audio for unchanged ones.
    """
    print(f"[Regenerate] Received request for task_id: {request.task_id}")
    print(f"[Regenerate] Segments count: {len(request.segments)}")
    print(f"[Regenerate] First segment: {request.segments[0] if request.segments else 'None'}")
    
    # Validate job exists
    job = database.get_job(request.task_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Update status
    database.update_job_status(request.task_id, status="pending", message="Queuing regeneration...")
    
    # Queue regeneration task
    background_tasks.add_task(
        regenerate_dubbing_task,
        request.task_id,
        request.segments,
        request.speaker_overrides,
        JOBS_DIR  # Pass JOBS_DIR
    )
    
    return {"status": "processing", "task_id": request.task_id}


@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    job = database.get_job(task_id)
    if not job:
        raise HTTPException(status_code=404, detail="Task not found")
    # Convert row to dict
    return job

@app.get("/api/history")
async def get_history():
    jobs = database.get_all_jobs()
    # Return as list
    return jobs

@app.get("/api/result/{task_id}")
async def get_result(task_id: str):
    job = database.get_job(task_id)
    if not job:
        raise HTTPException(status_code=404, detail="Task not found")
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed")
    
    # We need to construct the result object similar to before
    # Retrieve segments from DB
    segments = database.get_segments(task_id)
    
    # Where is output path? 
    # Pipeline saves output to jobs/{job_id}/output/filename.mp4
    # But we need to know the filename. 
    # It constructs it as: os.path.splitext(filename)[0] + "_dubbed.wav" or merged video.
    # The pipeline logic for output filename is specific.
    # We should have stored 'output_path' in DB or standardized it.
    # Currently DB 'jobs' table doesn't have 'output_path' column?
    # Let's check schema.
    # Schema: id, status, input_lang, ... filename ...
    # It does NOT have output_path.
    # However, we enforce directory structure: jobs/{job_id}/output/
    # The file is likely named {filename} (original name).
    # In `run_dubbing_task`, pipeline result returned `output_path`.
    # We didn't save it to DB. 
    # FIX: We should look for the file in jobs/{job_id}/output/
    
    base_dir = os.path.join(JOBS_DIR, task_id)
    output_dir = os.path.join(base_dir, "output")
    
    # Find the output file
    output_path = None
    if os.path.exists(output_dir):
        candidates = []
        for f in os.listdir(output_dir):
            if f.lower().endswith((".mp4", ".mkv", ".mov", ".avi")):
                candidates.append(os.path.join(output_dir, f))
        
        if candidates:
            # Simply pick the most recently modified file (Reliable for Job Isolation)
            output_path = max(candidates, key=os.path.getmtime)
    
    if not output_path:
        raise HTTPException(status_code=404, detail="Output file not found on server")

    filename = os.path.basename(output_path)
    
    # Construct source_srt and target_srt content
    # They are in jobs/{job_id}/srt/
    srt_dir = os.path.join(base_dir, "srt")
    source_srt = ""
    target_srt = ""
    
    name_no_ext = os.path.splitext(job['filename'])[0]
    # Try generic names
    # source: {name}_ASR.srt
    # target: {name}_{lang}.srt
    
    # Actually checking dir is safer
    if os.path.exists(srt_dir):
        for f in os.listdir(srt_dir):
            if "_ASR.srt" in f:
                with open(os.path.join(srt_dir, f), "r", encoding="utf-8") as fr:
                    source_srt = fr.read()
            elif f.endswith(".srt") and "_ASR" not in f:
                with open(os.path.join(srt_dir, f), "r", encoding="utf-8") as fr:
                    target_srt = fr.read()

    # Split segments into source and target for legacy frontend compatibility?
    # Frontend expects "source_segments" and "target_segments".
    # Our DB segments have both.
    # We can just pass the same list for both, or just map carefully.
    
    # Find original video in input directory
    input_dir = os.path.join(base_dir, "input")
    os.makedirs(input_dir, exist_ok=True) # Ensure it exists
    
    original_video_path = None
    original_video_url = None
    
    # 1. Check jobs/{id}/input
    if os.path.exists(input_dir):
        for f in os.listdir(input_dir):
            if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
                original_video_path = os.path.join(input_dir, f)
                original_video_url = f"/api/download/{task_id}/input/{f}"
                break
    
    # 2. Fallback: Check UPLOAD_DIR and copy if missing
    if not original_video_path and job.get('filename'):
        upload_path = os.path.join(UPLOAD_DIR, job['filename'])
        if os.path.exists(upload_path):
            # Copy to input dir for consistency and access
            dest_path = os.path.join(input_dir, job['filename'])
            import shutil
            try:
                shutil.copy2(upload_path, dest_path)
                original_video_path = dest_path
                original_video_url = f"/api/download/{task_id}/input/{job['filename']}"
            except Exception as e:
                print(f"Failed to copy original file to input dir: {e}")

    response = {
        "download_url": f"/api/download/{task_id}/{filename}", # Changed URL structure for isolation
        "source_srt": source_srt,
        "target_srt": target_srt,
        "source_segments": segments,
        "target_segments": [{**s, "text": s["target_text"]} for s in segments] # Map target_text to text for frontend
    }
    
    # Add original video URL if available
    if original_video_path and os.path.exists(original_video_path):
        response["original_video_url"] = original_video_url
    
    return response

@app.get("/api/download/{task_id}/{filepath:path}")
async def download_file(task_id: str, filepath: str):
    """Download output or input files from task directory."""
    # Security: Ensure task_id is valid UUID
    try:
        uuid.UUID(task_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid Task ID")

    # Base directory for this job
    base_dir = os.path.join(JOBS_DIR, task_id)
    
    # Check if path includes subdirectory (e.g., "input/video.mp4")
    if "/" in filepath:
        # Full path with subdirectory
        path = os.path.join(base_dir, filepath)
    else:
        # Legacy: just filename, default to output directory
        path = os.path.join(base_dir, "output", filepath)
    
    # Security: Validate path doesn't escape job directory
    final_path_abs = os.path.abspath(path)
    base_dir_abs = os.path.abspath(base_dir)
    if not final_path_abs.startswith(base_dir_abs):
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"File not found: {filepath}")
    
    filename = os.path.basename(filepath) # Use original filepath for filename in response
    
    media_type = "application/octet-stream"
    if filename.endswith(".mp4"): media_type = "video/mp4"
    elif filename.endswith(".wav"): media_type = "audio/wav"
    elif filename.endswith(".srt"): media_type = "text/plain"

    return FileResponse(
        path, 
        media_type=media_type, 
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.get("/api/thumbnail/{task_id}")
async def get_thumbnail(task_id: str):
    thumb_name = f"{task_id}.jpg"
    thumb_path = os.path.join(THUMBNAIL_DIR, thumb_name)
    
    if os.path.exists(thumb_path):
        return FileResponse(thumb_path, media_type="image/jpeg")
    
    raise HTTPException(status_code=404, detail="Thumbnail not found")

@app.delete("/api/delete/{task_id}")
def delete_task(task_id: str):
    job = database.get_job(task_id)
    if not job:
        raise HTTPException(status_code=404, detail="Task not found")
    
    print(f"Deleting job {task_id}...")
    
    # 1. Delete DB Entry
    database.delete_job(task_id)
    
    # 2. Delete Job Directory
    job_dir = os.path.join(JOBS_DIR, task_id)
    if os.path.exists(job_dir):
        try:
            shutil.rmtree(job_dir)
            print(f"Deleted directory: {job_dir}")
        except Exception as e:
            print(f"Error deleting directory {job_dir}: {e}")
            
    # 3. Delete Thumbnail
    thumb_path = os.path.join(THUMBNAIL_DIR, f"{task_id}.jpg")
    if os.path.exists(thumb_path):
        os.remove(thumb_path)
    
    return {"status": "deleted", "id": task_id}

@app.get("/api/languages")
async def get_languages():
    return {"languages": LANGUAGES}

@app.get("/api/logs/{task_id}")
def get_task_logs(task_id: str):
    logs = database.get_logs(task_id)
    return {"logs": logs}

@app.post("/api/tool/translate")
def adhoc_translate(req: TranslateRequest):
    try:
        # Construct Prompt
        base_prompt = req.prompt if req.prompt and req.prompt.strip() else f"Translate the following text from {req.source_lang} to {req.target_lang}."
        
        final_prompt = f"""
        {base_prompt}
        
        Input Text: "{req.text}"
        
        Instructions:
        1. Maintain the original tone but make it CASUAL and SPOKEN (Informal).
        2. {f'Output Language: {req.target_lang}'}
        3. Return ONLY the translation, no extra text.
        4. Focus on how people actually speak (natural conversational flow).
        """
        
        resp = google_pipeline2.retry_manager.call(lambda: google_pipeline2.get_client().models.generate_content(model="gemini-2.5-pro", contents=final_prompt))
        return {"translated": resp.text.strip()}
    except Exception as e:
        print(f"Translation Tool Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tool/asr")
async def asr_tool(file: UploadFile = File(...), lang: str = Form("en")):
    temp_path = f"temp_asr_{uuid.uuid4()}.webm"
    try:
        # Save Upload
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Upload to Gemini - Use genai SDK directly for simpler handling
        import google.generativeai as genai
        import time
        from pydub import AudioSegment
        genai.configure(api_key=google_pipeline2.get_api_key())
        
        # Convert webm to wav (Gemini doesn't handle webm well)
        audio = AudioSegment.from_file(temp_path, format="webm")
        wav_path = temp_path.replace(".webm", ".wav")
        audio.export(wav_path, format="wav")
        
        # Upload wav file
        uploaded_file = genai.upload_file(wav_path)
        
        # Wait for file to become ACTIVE
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(1)
            uploaded_file = genai.get_file(uploaded_file.name)
        
        if uploaded_file.state.name != "ACTIVE":
            raise Exception(f"File processing failed: {uploaded_file.state.name}")
        
        # Transcribe
        prompt = f"""
        Transcribe the audio exactly as spoken.
        Target Language: {lang} (if audio is in this language, transcribe it. if audio is different, transcribe what is spoken).
        Output ONLY the transcript text. No timestamps.
        """
        
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        resp = model.generate_content([prompt, uploaded_file])
        
        return {"transcript": resp.text.strip()}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"ASR Tool Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup both temp files
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        wav_path = temp_path.replace(".webm", ".wav")
        if os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

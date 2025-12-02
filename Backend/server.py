import os
import shutil
import uuid
import threading
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import json
from datetime import datetime
from typing import Optional

# Import the pipeline
import google_pipeline2

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "Videos", "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "Video out")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# In-memory task store
# Persistence
HISTORY_FILE = "history.json"
SRT_DIR = os.path.join(BASE_DIR, "SRT")

# In-memory task store
tasks = {}

def parse_srt(srt_content):
    segments = []
    blocks = srt_content.strip().split('\n\n')
    for block in blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            try:
                times = lines[1]
                text = " ".join(lines[2:])
                if ' --> ' in times:
                    start_str, end_str = times.split(' --> ')
                    segments.append({
                        "start": start_str.strip(),
                        "end": end_str.strip(),
                        "text": text.strip()
                    })
            except Exception:
                continue
    return segments

def scan_existing_files():
    print("Scanning for existing files...")
    if not os.path.exists(OUTPUT_DIR):
        return

    count = 0
    for filename in os.listdir(OUTPUT_DIR):
        if filename.endswith(".mp4"):
            task_id = os.path.splitext(filename)[0]
            
            # Skip if already in tasks (loaded from history)
            found = False
            for t in tasks.values():
                if t.get("result", {}).get("output_path", "").endswith(filename):
                    found = True
                    break
            if found:
                continue

            # Reconstruct task
            print(f"Recovering task for {filename}...")
            
            # Try to find SRTs
            source_srt_path = os.path.join(SRT_DIR, f"{task_id}_ASR.srt")
            target_srt_path = None
            output_lang = "Unknown"
            
            # Find target SRT to infer language
            if os.path.exists(SRT_DIR):
                for f in os.listdir(SRT_DIR):
                    if f.startswith(task_id) and not f.endswith("_ASR.srt") and f.endswith(".srt"):
                        target_srt_path = os.path.join(SRT_DIR, f)
                        # Extract language from filename: {uuid}_{Lang}.srt
                        output_lang = f.replace(f"{task_id}_", "").replace(".srt", "")
                        break
            
            source_srt = ""
            target_srt = ""
            source_segments = []
            target_segments = []

            if source_srt_path and os.path.exists(source_srt_path):
                try:
                    with open(source_srt_path, "r", encoding="utf-8") as f:
                        source_srt = f.read()
                        source_segments = parse_srt(source_srt)
                except Exception as e:
                    print(f"Error reading source SRT: {e}")

            if target_srt_path and os.path.exists(target_srt_path):
                try:
                    with open(target_srt_path, "r", encoding="utf-8") as f:
                        target_srt = f.read()
                        target_segments = parse_srt(target_srt)
                except Exception as e:
                    print(f"Error reading target SRT: {e}")

            # Create task entry
            tasks[task_id] = {
                "id": task_id,
                "filename": filename, # Use UUID filename as display name if original unknown
                "input_lang": "Auto",
                "output_lang": output_lang,
                "status": "completed",
                "progress": 100,
                "step": "Restored",
                "message": "Restored from storage",
                "timestamp": str(datetime.fromtimestamp(os.path.getmtime(os.path.join(OUTPUT_DIR, filename)))),
                "result": {
                    "output_path": os.path.join(OUTPUT_DIR, filename),
                    "source_srt": source_srt,
                    "target_srt": target_srt,
                    "source_segments": source_segments,
                    "target_segments": target_segments
                }
            }
            count += 1
            
    # Cleanup: Remove tasks where the file no longer exists
    tasks_to_remove = []
    for t_id, task in tasks.items():
        if task.get("status") == "completed":
            out_path = task.get("result", {}).get("output_path")
            if out_path and not os.path.exists(out_path):
                print(f"File missing for task {t_id}: {out_path}. Removing from history.")
                tasks_to_remove.append(t_id)
    
    for t_id in tasks_to_remove:
        del tasks[t_id]
        count += 1 # Count removals as changes to trigger save

    if count > 0:
        print(f"Updated {count} tasks (restored or removed).")
        save_history()

def load_history():
    global tasks
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                tasks = json.load(f)
            print(f"Loaded {len(tasks)} tasks from history.")
        except Exception as e:
            print(f"Error loading history: {e}")
    
    # Scan for files not in history
    scan_existing_files()

def save_history():
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2)
    except Exception as e:
        print(f"Error saving history: {e}")

# Load history on startup
load_history()

class DubRequest(BaseModel):
    filename: str
    input_lang: str
    output_lang: str
    target_voice: Optional[str] = None
    speed: Optional[float] = 1.0

LANGUAGES = [
    "English", "Hindi", "Assamese", "Punjabi", "Telugu", 
    "Tamil", "Marathi", "Gujarati", "Kannada", "Malayalam", "Odia"
]

def run_dubbing_task(task_id, filename, input_lang, output_lang):
    tasks[task_id]["status"] = "processing"
    save_history() # Save initial processing state
    
    def progress_callback(step, progress, message):
        tasks[task_id]["progress"] = progress
        tasks[task_id]["step"] = step
        tasks[task_id]["message"] = message
        print(f"Task {task_id}: [{step}] {progress}% - {message}")

    try:
        input_path = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"File {filename} not found")

        # Run pipeline
        result = google_pipeline2.process_video(
            video_path=input_path,
            output_dir=OUTPUT_DIR,
            input_lang=input_lang,
            output_lang=output_lang,
            progress_callback=progress_callback
        )
        
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = 100
        tasks[task_id]["message"] = "Done"
        tasks[task_id]["result"] = result
        tasks[task_id]["timestamp"] = str(datetime.now())
        
    except Exception as e:
        print(f"Task {task_id} failed: {e}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
    
    save_history() # Save final state

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    extension = os.path.splitext(file.filename)[1]
    new_filename = f"{file_id}{extension}"
    file_path = os.path.join(UPLOAD_DIR, new_filename)
    
    # Ensure upload directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {"filename": new_filename, "original_name": file.filename}

@app.post("/api/dub")
async def start_dubbing(request: DubRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "id": task_id,
        "filename": request.filename, # Store filename for display
        "input_lang": request.input_lang,
        "output_lang": request.output_lang,
        "status": "pending",
        "progress": 0,
        "step": "Queued",
        "message": "Waiting to start...",
        "result": None,
        "timestamp": str(datetime.now())
    }
    save_history() # Save pending state
    
    background_tasks.add_task(
        run_dubbing_task, 
        task_id, 
        request.filename, 
        request.input_lang, 
        request.output_lang
    )
    
    return {"task_id": task_id}

@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]

@app.get("/api/history")
async def get_history():
    # Return list of completed tasks, sorted by timestamp desc
    completed_tasks = [
        t for t in tasks.values() 
        if t.get("status") == "completed"
    ]
    # Sort by timestamp if available, else random
    completed_tasks.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return completed_tasks

@app.get("/api/result/{task_id}")
async def get_result(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    if tasks[task_id]["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed")
    
    # Construct download URL for the output file
    output_path = tasks[task_id]["result"]["output_path"]
    filename = os.path.basename(output_path)
    
    return {
        "download_url": f"/api/download/{filename}",
        "source_srt": tasks[task_id]["result"]["source_srt"],
        "target_srt": tasks[task_id]["result"]["target_srt"],
        "source_segments": tasks[task_id]["result"]["source_segments"],
        "target_segments": tasks[task_id]["result"]["target_segments"]
    }

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    # Check in Output dir
    path = os.path.join(OUTPUT_DIR, filename)
    print(f"DEBUG: Checking download path: {path}")
    if not os.path.exists(path):
        print(f"DEBUG: Not found in output. Checking upload dir...")
        # Check in Upload dir (for testing or playback of source)
        path = os.path.join(UPLOAD_DIR, filename)
        print(f"DEBUG: Checking upload path: {path}")
    
    if not os.path.exists(path):
        print(f"DEBUG: File not found: {filename}")
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    media_type = "application/octet-stream"
    if filename.endswith(".mp4"):
        media_type = "video/mp4"
    elif filename.endswith(".wav"):
        media_type = "audio/wav"
    elif filename.endswith(".srt"):
        media_type = "text/plain"

    return FileResponse(
        path, 
        media_type=media_type, 
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename={filename}"} if "download" in filename else None
    )

@app.get("/api/languages")
async def get_languages():
    return {"languages": LANGUAGES}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

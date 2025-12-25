#!/usr/bin/env python
# coding: utf-8

# # Google Pipeline: ASR, MT, TTS & Dubbing (Multi-Language)
# 
# This script implements a complete video dubbing pipeline with an optimized TTS-MT feedback loop:
# 
# 1.  **ASR (Gemini 2.5 Pro):**
#     - Transcribes video in the specified `input_lang`.
#     - Tags significant noises (e.g., `[NOISE: traffic]`) and silences (>1s).
#     - Removes filler words and ensures sentence-level segmentation.
#     - Handles code-switching (e.g., English terms in Hindi) naturally.
# 
# 2.  **Adaptive MT (Gemini 2.5 Pro):**
#     - Translates to `output_lang` (via English pivot if needed).
#     - **Pre-emptive Length Estimation:** Estimates target word count based on duration to minimize retries.
#     - **Duration-Aware:** Elaborates or summarizes text to fit the time slot.
# 
# 3.  **TTS (Google Cloud):**
#     - Generates audio in `output_lang` (default: Indian English).
#     - **Strict Speed:** No artificial speed-up/slow-down. Natural speaking rate only.
# 
# 4.  **Feedback Loop (Optimization):**
#     - Checks if generated audio fits the target duration.
#     - **If Mismatch:** Retries (Max 1 attempt) with explicit "Elaborate" or "Summarize" instructions.
#     - **Strict Sync (Atempo):** If audio is still too long, uses ffmpeg `atempo` filter to compress it to exact duration.
# 
# 5.  **Dubbing:** Merges generated audio with the original video.
# 6.  **Trimming:** Optionally trim input video to save credits/time (Handles pre-trimmed files).

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import time
import json
import re
import io
import base64
import requests
import subprocess
import shutil
import hashlib
import pickle
import threading
import logging
import wave
import contextlib
import math
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional, Callable
from dotenv import load_dotenv
from pydub import AudioSegment
from google import genai
from google.genai import types
import database
import config
import srt_utils

# Load environment variables
# Global client variable, initialized lazily
client = None
GEMINI_API_KEY = None
JOBS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobs")

def get_client():
    global client, GEMINI_API_KEY
    if client:
        return client
    
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not found in .env file.")
        
    client = genai.Client(api_key=GEMINI_API_KEY)
    print("API Key loaded and Client initialized.")
    return client

def get_api_key():
    get_client() # Ensure initialized
    return GEMINI_API_KEY

# Add ffmpeg to PATH (Assuming system install)


# --- Infrastructure Classes ---

class MetricsLogger:
    def __init__(self):
        self.metrics = {
            "api_latency": [],
            "api_costs": 0.0, # Placeholder
            "retries": 0,
            "tts_duration_mismatch": 0,
            "segments_processed": 0
        }
        self.lock = threading.Lock()

    def log_latency(self, endpoint, duration_ms):
        with self.lock:
            self.metrics["api_latency"].append({"endpoint": endpoint, "duration": duration_ms})

    def log_retry(self):
        with self.lock:
            self.metrics["retries"] += 1

    def log_mismatch(self):
        with self.lock:
            self.metrics["tts_duration_mismatch"] += 1
            
    def save_report(self, path="metrics.json"):
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=2)

# Metrics removed as global. Will be instanced per job if needed, or logged to DB.

class CacheManager:
    def __init__(self, cache_dir=".cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_path(self, key):
        hashed = hashlib.md5(key.encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir, f"{hashed}.pkl")

    def get(self, key):
        path = self._get_path(key)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Cache retrieval error for key {key}: {e}")
                return None
        return None

    def set(self, key, value):
        path = self._get_path(key)
        with open(path, "wb") as f:
            pickle.dump(value, f)
    
    def generate_key(self, text, voice_id, output_lang):
        """
        Generate cache key from text, voice, and language.
        
        Args:
            text: Text to be synthesized
            voice_id: Voice identifier
            output_lang: Output language
            
        Returns:
            Cache key string
        """
        return f"{text}|{voice_id}|{output_lang}"

# Cache removed as global. Instantiated per job.

class RetryManager:
    def __init__(self, max_retries=3, backoff_factor=1.5, circuit_breaker_threshold=5):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.failures = 0
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.last_failure_time = 0
        self.lock = threading.Lock()

    def call(self, func, *args, **kwargs):
        with self.lock:
            if self.failures >= self.circuit_breaker_threshold:
                if time.time() - self.last_failure_time < 60: # 1 min cool-off
                    print("Circuit breaker open. Skipping call.")
                    raise Exception("Circuit breaker open. Rate limit protection active.")
                else:
                    self.failures = 0 # Reset

        for attempt in range(self.max_retries + 1):
            try:
                start = time.time()
                result = func(*args, **kwargs)
                # metrics.log_latency(func.__name__, (time.time() - start) * 1000)
                
                with self.lock:
                    self.failures = 0 # Reset on success
                return result
            except Exception as e:
                print(f"Error in {func.__name__} (Attempt {attempt+1}/{self.max_retries+1}): {e}")
                # metrics.log_retry()
                if attempt < self.max_retries:
                    sleep_time = self.backoff_factor ** attempt
                    time.sleep(sleep_time)
                else:
                    with self.lock:
                        self.failures += 1
                        self.last_failure_time = time.time()
                    raise e

retry_manager = RetryManager()


def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini."""
    def _upload():
        file = get_client().files.upload(file=path, config={'mime_type': mime_type})
        print(f"Uploaded file '{file.name}' as: {file.uri}")
        return file
    return retry_manager.call(_upload)

def wait_for_files_active(files):
    """Waits for the given files to be active."""
    print("Waiting for file processing...")
    for name in (file.name for file in files):
        file = get_client().files.get(name=name)
        while file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(2)
            file = get_client().files.get(name=name)
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process")
    print("...all files ready")


# ## 1. ASR (Gemini 2.5 Pro)


from pydub.silence import detect_silence

# Calibrated Words Per Minute (WPM) Table
# Updated to match actual Google TTS speeds (significantly faster than natural speech)
# Values calibrated from real test observations at 1.0x TTS speed
LANGUAGE_WPM = {
    "English": 180,     # Reduced from 220 to allow more breathing room
    "Hindi": 210,       # Actual TTS speed observed ~195-210 WPM
    "Tamil": 200,       # Increased to match TTS reality
    "Malayalam": 195,   # Increased to match TTS reality
    "Telugu": 205,      # Increased to match TTS reality
    "Kannada": 200,     # Increased to match TTS reality
    "Marathi": 210,     # Increased to match TTS reality
    "Bengali": 210,     # Increased to match TTS reality
    "Punjabi": 210,     # Increased to match TTS reality
    "Gujarati": 205     # Increased to match TTS reality
}

def detect_silences_ffmpeg(video_path, noise_db=-30, duration_sec=0.5, log_callback=None):
    """
    Detects silence intervals using ffmpeg silencedetect filter.
    Returns a list of (start, end) tuples in seconds.
    """
    def log(msg):
        if log_callback: log_callback(msg)
    
    log(f"Detecting silence (Noise: {noise_db}dB, Min Dur: {duration_sec}s)...")
    
    cmd = [
        "ffmpeg", "-i", video_path, 
        "-af", f"silencedetect=noise={noise_db}dB:d={duration_sec}", 
        "-f", "null", "-"
    ]
    
    try:
        # ffmpeg outputs silence info to stderr
        result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True, check=True)
        output = result.stderr
    except subprocess.CalledProcessError as e:
        log(f"Error detecting silence: {e}")
        return []

    silences = []
    current_start = None
    
    # Parse output
    # [silencedetect @ ...] silence_start: 12.345
    # [silencedetect @ ...] silence_end: 14.567 | silence_duration: 2.222
    for line in output.splitlines():
        if "silence_start:" in line:
            try:
                current_start = float(line.split("silence_start:")[1].strip())
            except ValueError:
                pass
        elif "silence_end:" in line and current_start is not None:
            try:
                end = float(line.split("silence_end:")[1].split("|")[0].strip())
                silences.append((current_start, end))
                current_start = None
            except ValueError:
                pass
                
    log(f"Found {len(silences)} silence intervals.")
    return silences

def split_audio_chunks_optimized(video_path, job_id, chunk_length_ms=300000, log_callback=None):
    """
    Splits audio into chunks using ffmpeg segment muxer with silence-aware splitting.
    """
    def log(msg):
        print(f"[Job {job_id}] {msg}")
        if log_callback: log_callback(msg)

    chunk_length_sec = chunk_length_ms / 1000.0
    log(f"Splitting audio (Optimized) (Target: {chunk_length_sec}s/chunk)...")
    
    # Use jobs/{job_id}/temp/chunks
    temp_dir = os.path.join(JOBS_DIR, job_id, "temp", "chunks")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    # 1. Detect Silences
    silences = detect_silences_ffmpeg(video_path, log_callback=log)
    
    # 2. Calculate Split Points
    # We want splits roughly every chunk_length_sec
    total_duration_sec = get_media_duration_ms(video_path) / 1000.0
    split_points = []
    current_time = 0
    
    # Sort silences by start time (detect_silences_ffmpeg returns sorted, but ensure)
    silences.sort(key=lambda x: x[0])
    import bisect
    
    # Pre-compute start points for binary search
    silence_starts = [s[0] for s in silences]

    while current_time + chunk_length_sec < total_duration_sec:
        target_time = current_time + chunk_length_sec
        best_split = target_time
        
        # Look for silence within +/- 30 seconds of target (or half chunk length if smaller)
        window = min(30, chunk_length_sec / 4) 
        
        # Binary search for silences near target time
        # We want silences that start before (target + window) and end after (target - window)
        # Using bisect to find the first silence that starts >= (target - window)
        idx = bisect.bisect_left(silence_starts, target_time - window)
        
        candidates = []
        # Iterate forward from the found index until we go out of window range
        # This reduces search from O(N) to O(k) where k is small number of silences in window
        for k in range(idx, len(silences)):
            s = silences[k]
            if s[0] > (target_time + window):
                break # Passed the window
            
            # Check overlap logic: start < (target+window) AND end > (target-window)
            if s[1] > (target_time - window):
                candidates.append(s)

        found_silence = False
        if candidates:
            # Pick the silence closest to target_time
            # Ideally split in the middle of silence
            best_candidate = min(candidates, key=lambda s: abs((s[0] + s[1])/2 - target_time))
            
            # Check if this silence is reasonably close
            mid_silence = (best_candidate[0] + best_candidate[1]) / 2
            
            # Additional check: we prefer not to deviate too much unless it's a really good silence
            if abs(mid_silence - target_time) < window:
                best_split = mid_silence
                found_silence = True
                log(f"  Found silence at {best_split:.2f}s (Target: {target_time:.2f}s)")
        
        if not found_silence:
             log(f"  No silence found near {target_time:.2f}s. Using hard cut.")
             
        split_points.append(f"{best_split:.3f}")
        current_time = best_split

    segment_times_str = ",".join(split_points)
    segment_pattern = os.path.join(temp_dir, "chunk_%03d.wav")

    cmd = [
        "ffmpeg", "-y", "-i", video_path, 
        "-f", "segment", 
        "-c", "pcm_s16le", "-ar", "16000", "-ac", "1", 
        "-reset_timestamps", "1"
    ]
    
    if segment_times_str:
        cmd.extend(["-segment_times", segment_times_str])
    else:
        # Fallback to fixed if logic yielded nothing (e.g. short file or failed)
        cmd.extend(["-segment_time", str(chunk_length_sec)])

    cmd.append(segment_pattern)

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"Error splitting audio: {e}")
        raise

    chunks = []
    # Collect generated files
    for f in sorted(os.listdir(temp_dir)):
        if f.endswith(".wav"):
            chunks.append(os.path.join(temp_dir, f))
            
    log(f"Exported {len(chunks)} chunks using FFmpeg.")
    return chunks, temp_dir

class VoiceManager:
    def __init__(self):
        # Pool of distinct voices (Indian English)
        self.voice_pool = [
            "en-IN-Neural2-B", # Male
            "en-IN-Neural2-C", # Male
            "en-IN-Neural2-A", # Female
            "en-IN-Neural2-D", # Female
            "en-IN-Standard-B", # Male
            "en-IN-Standard-C", # Male
            "en-IN-Standard-A", # Female
            "en-IN-Standard-D", # Female
        ]
        
        # Chirp 3 Personalities
        self.chirp_3_personalities = {
            "Aoede": "Female",
            "Kore": "Female",
            "Leda": "Female",
            "Zephyr": "Female",
            "Erinome": "Female",
            "Puck": "Male",
            "Charon": "Male",
            "Fenrir": "Male",
            "Orus": "Male",
            "Achird": "Male",
            "Alnilam": "Male",
            "Enceladus": "Male",
            "Achernar": "Female",
            "Algenib": "Male",
            "Algieba": "Male",
            "Autonoe": "Female",
            "Callirrhoe": "Female",
            "Despina": "Female",
            "Gacrux": "Female",
            "Iapetus": "Male",
            "Laomedeia": "Female",
            "Pulcherrima": "Female",
            "Rasalgethi": "Male",
            "Sadachbia": "Male",
            "Sadaltager": "Male",
            "Schedar": "Male",
            "Sulafat": "Female",
            "Umbriel": "Male",
            "Vindemiatrix": "Female",
            "Zubenelgenubi": "Male"
        }
        
        self.speaker_map = {}
        self.lock = threading.Lock()

    def get_voice(self, speaker_label, voice_preference=None, output_lang="English"):
            # Check if preference is provided
            if voice_preference:
                # Case 1: Preference is already a Full ID (e.g. en-IN-Chirp3-HD-Aoede)
                # We need to extract the "Personality" (Aoede) and re-apply it to the Target Language.
                
                parts = voice_preference.split('-')
                personality = None
                
                if "Chirp3" in voice_preference or "Chirp" in voice_preference:
                    # Try to find known personality
                    for p in self.chirp_3_personalities:
                        # Check strict suffix match or presence
                        if voice_preference.endswith(f"-{p}") or f"-{p}-" in voice_preference:
                            personality = p
                            break
                    
                    # Fallback for generic Chirp (e.g. Chirp-HD-D)
                    if not personality and len(parts) > 0:
                         # Last part usually is ID/Name
                         personality = parts[-1]

                if personality:
                     # Map output language to BCP-47 code
                    LANGUAGE_CODE_MAP = {
                        "English": "en-IN",
                        "Hindi": "hi-IN",
                        "Tamil": "ta-IN",
                        "Telugu": "te-IN",
                        "Kannada": "kn-IN",
                        "Malayalam": "ml-IN",
                        "Marathi": "mr-IN",
                        "Gujarati": "gu-IN",
                        "Bengali": "bn-IN",
                        "Punjabi": "pa-IN"
                    }
                    lang_code = LANGUAGE_CODE_MAP.get(output_lang, "en-IN")
                    
                    # Reconstruct ID: {lang_code}-Chirp3-HD-{personality}
                    # Note: older Chirp IDs might differ, but assuming Chirp3 HD standard for this app
                    return f"{lang_code}-Chirp3-HD-{personality}"
                
                # Case 2: Preference didn't match Chirp logic but is provided?
                # Maybe Neural2? Just return it as-is if we can't adapt it safely?
                # Or assume it's just a raw override.
                # Case 2: Standard/Wavenet/Polyglot/Neural2 or other valid IDs
                # If it looks like a valid ID (contains hyphens and language code), use it.
                # Use simple heuristic: if it has at least 2 hyphens (lang-region-name) or known types.
                if any(t in voice_preference for t in ["Neural2", "Standard", "Wavenet", "Polyglot", "Studio"]):
                     return voice_preference
                
                # Fallback: if it looks like a full ID, return it
                if voice_preference.count('-') >= 2:
                    return voice_preference

            if not speaker_label:
                return "en-IN-Neural2-B" # Default
            
            # Normalize label
            label = speaker_label.strip().upper()
            
            if label not in self.speaker_map:
                # Assign next available voice
                voice_idx = len(self.speaker_map) % len(self.voice_pool)
                self.speaker_map[label] = self.voice_pool[voice_idx]
                print(f"Assigned voice {self.speaker_map[label]} to {label}")
                
            return self.speaker_map[label]

# Global voice_manager removed. Instantiated per job.

def parse_gemini_json(response_text, offset_ms=0, chunk_duration_ms=None, log_callback=None):
    """Parses Gemini JSON response and applies time offset. Validates against chunk duration."""
    # Extract JSON from code blocks if present
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0]
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0]
        
    try:
        segments = json.loads(response_text)
    except json.JSONDecodeError:
        msg = f"Error decoding JSON: {response_text}"
        if log_callback: log_callback(msg)
        else: print(msg)
        processed_segments = []
        return processed_segments

    processed_segments = []
    
    # Validation loop
    for seg in segments:
        if seg.get('type') == 'silence':
            continue
            
        try:
            local_start = srt_utils.parse_srt_time(seg['start'])
            local_end = srt_utils.parse_srt_time(seg['end'])
            
            # Validate no negative timestamps
            if local_start < 0 or local_end < 0:
                raise ValueError(f"Negative timestamp detected: start={local_start}ms, end={local_end}ms")
            
            # Basic Validity Check
            if local_end <= local_start:
                # STRICT FIX: User requested RETRY, not skip/correct.
                raise ValueError(f"Invalid segment duration detected (End {local_end} <= Start {local_start}) at source.")

            # Sanity Check: If local timestamp exceeds chunk duration significantly, it's a hallucination.
            if chunk_duration_ms:
                if local_start > (chunk_duration_ms + config.HALLUCINATION_BUFFER_MS):
                     raise ValueError(f"Hallucinated segment detected: Start {local_start}ms > Chunk Duration {chunk_duration_ms}ms")
                if local_end > (chunk_duration_ms + config.HALLUCINATION_BUFFER_MS):
                     raise ValueError(f"Hallucinated segment detected: End {local_end}ms > Chunk Duration {chunk_duration_ms}ms")

            start_ms = local_start + offset_ms
            end_ms = local_end + offset_ms
            
            # Extract speaker first
            speaker = seg.get('speaker', 'Speaker 1')
            
            # Clean inline speaker tags from text (e.g., "[Speaker 1] [Speaker 2] text" -> "text")
            # This fixes duplicate tags where Gemini adds tags to both the speaker field AND inline in text
            clean_text = seg['text']
            clean_text = re.sub(r'\[Speaker \d+\]\s*', '', clean_text).strip()
            
            processed_segments.append({
                "start": start_ms,
                "end": end_ms,
                "text": clean_text,
                "speaker": speaker,
                "start_str": srt_utils.format_srt_time(start_ms),
                "end_str": srt_utils.format_srt_time(end_ms)
            })
        except ValueError as e:
            # CRITICAL: Always re-raise ValueError (Hallucinations/Invalid segments) to trigger Retry Loop
            raise e
        except Exception as e:
            msg = f"Skipping malformed segment: {seg} - {e}"
            if log_callback: log_callback(msg)
            else: print(msg)
            
    return processed_segments

def segments_to_srt(segments):
    lines = []
    for i, seg in enumerate(segments):
        speaker = seg.get('speaker', '')
        text = seg['text']
        
        # Only add speaker tag if it's not already in the text
        if speaker and not text.strip().startswith(f"[{speaker}]"):
            text = f"[{speaker}] {text}"
        
        lines.append(f"{i+1}")
        lines.append(f"{seg['start_str']} --> {seg['end_str']}")
        lines.append(f"{text}\n")
    return "\n".join(lines)

def fix_srt_timestamps(srt_content):
    """
    Fixes SRT timestamps by parsing them and re-formatting to strict HH:MM:SS,mmm.
    Delegates to srt_utils for consistent strict parsing.
    """
    lines = srt_content.splitlines()
    new_lines = []
    for line in lines:
        if "-->" in line:
            parts = line.split("-->")
            if len(parts) == 2:
                # Use srt_utils to consistently repair/fmt logic
                start = srt_utils.normalize_timestamp(parts[0])
                end = srt_utils.normalize_timestamp(parts[1])
                new_lines.append(f"{start} --> {end}")
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
    return "\n".join(new_lines)

def merge_segments_to_sentences(srt_content_or_segments):
    """Merges SRT segments into full sentences based on punctuation."""
    if isinstance(srt_content_or_segments, list):
         segments = srt_content_or_segments
    else:
         segments = srt_utils.parse_srt(srt_content_or_segments)
         
    merged_segments = []
    current_text = ""
    current_start = None
    current_speaker = None
    
    terminals = ('.', '?', '!', '|', '।')
    
    for i, seg in enumerate(segments):
        text = seg['text'].strip()
        start = seg['start']
        end = seg['end']
        speaker = seg.get('speaker', 'Speaker 1')
        
        if current_start is None:
            current_start = start
            current_speaker = speaker
            
        # If speaker changes, force split
        if speaker != current_speaker and current_text:
            # Use the previous segment's end time as the cut-off
            prev_end = segments[i-1]['end'] if i > 0 else start
            
            merged_segments.append({
                "start": current_start,
                "end": prev_end,
                "text": current_text.strip(),
                "speaker": current_speaker
            })
            current_text = ""
            current_start = start
            current_speaker = speaker

        if current_text:
            current_text += " " + text
        else:
            current_text = text
            
        if current_text.strip().endswith(terminals):
            merged_segments.append({
                "start": current_start,
                "end": end,
                "text": current_text.strip(),
                "speaker": current_speaker
            })
            current_text = ""
            current_start = None
            current_speaker = None
            
    if current_text and current_start is not None:
        merged_segments.append({
            "start": current_start,
            "end": segments[-1]['end'],
            "text": current_text.strip(),
            "speaker": current_speaker
        })
        
    lines = []
    for i, seg in enumerate(merged_segments):
        # Handle both numeric (ms) and string timestamps
        start_ms = seg['start'] if isinstance(seg['start'], (int, float)) else srt_utils.parse_srt_time(seg['start'])
        end_ms = seg['end'] if isinstance(seg['end'], (int, float)) else srt_utils.parse_srt_time(seg['end'])
        
        s = datetime.utcfromtimestamp(start_ms/1000).strftime('%H:%M:%S,%f')[:-3]
        e = datetime.utcfromtimestamp(end_ms/1000).strftime('%H:%M:%S,%f')[:-3]
        
        # Only add speaker tag if not already in text
        text = seg['text']
        if seg.get('speaker') and not text.strip().startswith(f"[{seg['speaker']}]"):
            text = f"[{seg['speaker']}] {text}"
        
        lines.append(f"{i+1}\n{s} --> {e}\n{text}\n")
    return "\n".join(lines)

def generate_srt_gemini(video_path: str, input_lang: str = "Hindi", model: str = "gemini-2.5-pro", duration_limit: Optional[int] = None, job_id: Optional[str] = None, log_callback: Optional[Callable[[str], None]] = None) -> str:
    """
    Generates SRT using Google Gemini (default 2.5 Pro) with optimized chunking and strict JSON mode.
    """
    def log(msg):
        print(f"[Job {job_id}] {msg}")
        if job_id: database.log_event(job_id, msg)
        if log_callback: log_callback(msg)

    # 1. Use Optimized Splitting (Low RAM)
    chunk_paths, temp_dir = split_audio_chunks_optimized(video_path, job_id, chunk_length_ms=config.ASR_CHUNK_LENGTH_MS, log_callback=log)
    
    all_segments = []
    
    # 2. Pre-calculate exact durations and offsets
    log("Calculating chunk durations...")
    chunk_metadata = []
    current_offset = 0
    
    for path in chunk_paths:
        try:
            # Use ffprobe for duration (ms) - RAM efficient
            cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path]
            dur_sec = float(subprocess.check_output(cmd).strip())
            dur_ms = int(dur_sec * 1000)
            
            chunk_metadata.append({
                "path": path,
                "duration_ms": dur_ms,
                "offset_ms": current_offset,
                "file_ref": None # To be filled after upload
            })
            current_offset += dur_ms
        except Exception as e:
            log(f"Error calculating duration for {path}: {e}")
            raise

    try:
        # 3. Sequential Uploads (Prevent Rate Limits)
        log(f"Uploading {len(chunk_metadata)} chunks to Gemini sequentially...")
        for i, meta in enumerate(chunk_metadata):
            meta['file_ref'] = upload_to_gemini(meta['path'], mime_type="audio/wav")
            log(f"  Uploaded Chunk {i+1}/{len(chunk_metadata)}")
        
        # Wait for all files to be active
        wait_for_files_active([m['file_ref'] for m in chunk_metadata])


        # 4. Define Processing Function
        def process_chunk_task(index, meta):
            attempt = 0
            max_retries = 5
            
            chunk_duration_sec = meta['duration_ms'] / 1000
            
            # JSON Mode Configuration
            generation_config = {
                "temperature": 0.0,
                "response_mime_type": "application/json"
            }

            def format_time_local(ms):
                s = int(ms / 1000)
                ms = int(ms % 1000)
                m = int(s / 60)
                s = s % 60
                h = int(m / 60)
                m = m % 60
                return f"{h:02}:{m:02}:{s:02},{ms:03}"

            prompt = f"""
        Role: You are a high-precision ASR system.

        INPUT: Audio Chunk ({index+1}/{len(chunk_metadata)})
        DURATION: {chunk_duration_sec:.2f} seconds.
        TASK: Transcribe audio.

        STRICT REQUIREMENTS:
        1.  **Time Alignment:** Timestamps MUST be relative to start (00:00:00) and MUST NOT exceed {chunk_duration_sec:.2f}s.
        2.  **Diarization:** Identify speakers.
        3.  **Granularity:** One sentence per segment.
        4.  **Silence:** Tag silences > 1.0s.
        5.  **Clean-up:** Remove filler words (uh, um, mm).
        6.  **Language:** Transcribe {input_lang} natively.

        OUTPUT FORMAT (JSON):
        [
          {{
            "start": "HH:MM:SS,mmm",
            "end": "HH:MM:SS,mmm",
            "speaker": "Speaker 1",
            "text": "Text"
          }}
        ]
        """
            def _generate_and_parse():
                response = get_client().models.generate_content(
                    model=model,
                    contents=[meta['file_ref'], prompt],
                    config=generation_config
                )
                
                # Validation happens inside logic robust parser
                # Any ValueError -> RetryManager -> Backoff -> Retry
                segments = parse_gemini_json(
                    response.text, 
                    offset_ms=meta['offset_ms'], 
                    chunk_duration_ms=meta['duration_ms'],
                    log_callback=log
                )
                
                if segments:
                     # Double check logic - Parser does most, but safeguard here
                     last_end_ms = segments[-1]['end']
                     chunk_end_limit = meta['offset_ms'] + meta['duration_ms'] + config.HALLUCINATION_BUFFER_MS
                     if last_end_ms > chunk_end_limit:
                          raise ValueError(f"Hallucination detected (Last End {last_end_ms} > Limit {chunk_end_limit})")
                          
                return segments

            # Use Global Retry Manager (Handles Backoff + Circuit Breaker)
            try:
                segments = retry_manager.call(_generate_and_parse)
                return segments  # Explicit return
            except Exception as e:
                log(f"Chunk {index+1} FAILED after retries: {e}")
                raise ValueError(f"Failed to process chunk {index+1}: {e}")

        # 5. Parallel Processing
        log("Transcribing chunks in parallel...")
        # SERVERLESS OPTIMIZATION: Use Env Var to control concurrency. 
        # Default to 10 to prevent one job from starving the container or hitting API limits.
        try:
             max_workers = int(os.getenv("MAX_CONCURRENT_CHUNKS", "10"))
        except: 
             max_workers = 10
             
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(process_chunk_task, i, meta): i 
                for i, meta in enumerate(chunk_metadata)
            }
            
            results = [None] * len(chunk_metadata)
            
            for future in as_completed(future_to_index):
                i = future_to_index[future]
                try:
                    res = future.result()
                    results[i] = res
                    log(f"  Chunk {i+1} processed ({len(res)} segments).")
                except Exception as e:
                    log(f"  Chunk {i+1} FAILED: {e}")
                    raise

            # Collect results
            for res in results:
                if res:
                    all_segments.extend(res)

        # 6. Sort Segments (Correction for threading order or any partial drift)
        all_segments.sort(key=lambda x: x['start'])

    finally:
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
            
    return segments_to_srt(all_segments)

def translate_adaptive(text, target_duration_sec, input_lang, output_lang, feedback=None, time_diff=0.0, words_to_add=0, gender=None, cache=None, target_wpm=None, tts_speed=1.0):
    """Adaptive translation with word budget based on WPM and duration."""
    
    # Use fixed WPM from language table
    # If target_wpm override is passed, use it, otherwise use LANGUAGE_WPM table
    base_target_wpm = target_wpm if target_wpm else LANGUAGE_WPM.get(output_lang, 138)
    
    # Language-pair-specific buffer (addresses source→target characteristics)
    language_pair = (input_lang, output_lang)
    buffer = config.LANGUAGE_PAIR_BUFFERS.get(language_pair, config.DEFAULT_BUFFER)
    
    # Word Budget Calculation
    # Scale by TTS Speed: Slower TTS (0.85x) means we can fit FEWER words.
    target_words = int(base_target_wpm * (target_duration_sec / 60.0) * buffer * tts_speed)
    
    # Min floor
    if target_words < 2: target_words = 2
    
    # Log the calculation for debugging
    buffer_note = f" (pair-specific {buffer:.2f}x)" if language_pair in config.LANGUAGE_PAIR_BUFFERS else ""
    print(f"  [WPM Logic] {input_lang}→{output_lang}: {int(base_target_wpm)}wpm × {target_duration_sec:.2f}s × {tts_speed:.2f}x{buffer_note} = {target_words} words")

    extra_instruction = ""
    if feedback == "panic":
        # Panic Mode: Drastic reduction for stubborn segments
        extra_instruction = f"""
        CRITICAL: Previous summary was STILL too long by {time_diff:.2f}s.
        ACTION: TELEGRAPHIC STYLE. Remove all fillers, adjectives, and grammar glue words. 
        Keep ONLY essential nouns/verbs. 
        Example: "We are going to the settings" -> "Go to settings".
        """
    elif feedback == "elaborate":
        extra_instruction = f"""IMPORTANT: The previous translation was TOO SHORT (Gap: {time_diff:.2f}s). 
        You MUST ELABORATE by adding approximately {words_to_add} words to the output.
        Add descriptive adjectives, natural fillers, or repeat key concepts naturally to fill the time."""
    elif feedback == "summarize":
        # Use words_to_add as words_to_remove for summarization
        words_to_remove = words_to_add if words_to_add > 0 else 2  # Default to 2 if not specified
        extra_instruction = f"""IMPORTANT: The previous translation was TOO LONG (Excess: {time_diff:.2f}s). 
        You MUST SUMMARIZE by removing approximately {words_to_remove} words.
        Use shorter words and concise phrasing. Remove adjectives, fillers, and redundant words.
        Target: Reduce from current length to approximately {target_words - words_to_remove} words maximum."""

    gender_instruction = ""
    if gender:
        gender_instruction = f"""
    SPEAKER GENDER: {gender}
    
    GRAMMAR INSTRUCTIONS:
    1. **Self-Reference:** For sentences where the speaker refers to themselves ("I am going", "I felt"), use {gender} verb and adjective forms.
    2. **Object Integrity:** Do NOT force speaker gender onto inanimate objects. If the target language has rules where verbs agree with objects (e.g., Hindi ergative 'ne' constructions), priority goes to the object's grammatical gender, NOT the speaker.
    """
    
    SCRIPT_MAPPING = {
        "Hindi": "Devanagari (e.g., नमस्ते)",
        "Tamil": "Tamil Script (e.g., வணக்கம்)",
        "Telugu": "Telugu Script (e.g., నమస్కారం)",
        "Kannada": "Kannada Script (e.g., ನಮಸ್ಕಾರ)",
        "Malayalam": "Malayalam Script (e.g., നമസ്കാരം)",
        "Marathi": "Devanagari (e.g., नमस्कार)",
        "Gujarati": "Gujarati Script (e.g., નમસ્તે)",
        "Punjabi": "Gurmukhi (e.g., ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ)",
        "Bengali": "Bengali Script (e.g., নমস্কার)"
    }
    target_script = SCRIPT_MAPPING.get(output_lang, "Native Script")

    prompt = f"""
Translate this subtitle from {input_lang} to {output_lang}:

"{text}"

REQUIREMENTS:

1. WORD COUNT: Aim for {target_words} words (between {int(target_words * 0.85)}-{target_words} words is ideal)
   - This ensures the audio matches the video timing
   - Use more words if the original is too short, fewer if too long

2. SCRIPT: Write in {target_script}, NOT romanized English
   - Example: "नमस्ते" (correct), NOT "Namaste" (wrong)
   - Exception: Keep technical terms in English (e.g., "smartphone", "WiFi")

3. STYLE: Casual and conversational
   - Use words people actually speak in everyday life
   - Avoid formal or bookish language

OUTPUT: Return ONLY the translated text
"""
    
    cache_key = f"mt_{text}_{target_duration_sec}_{feedback}_{output_lang}"
    cached = cache.get(cache_key) if cache else None
    if cached: return cached
    
    try:
        def _generate():
            return get_client().models.generate_content(model="gemini-2.5-pro", contents=prompt)
        resp = retry_manager.call(_generate)
        result = resp.text.strip()
        if cache: cache.set(cache_key, result)
        return result
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def generate_google_tts(text, api_key, voice_name="en-IN-Chirp3-HD-Achird", speaking_rate=1, cache=None):
    """Generates audio using Google Cloud TTS."""
    if not text or not text.strip(): return None
    
    cache_key = f"tts_{text}_{voice_name}_{speaking_rate}"
    cached = cache.get(cache_key) if cache else None
    if cached: return cached

    url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={api_key}"
    payload = {
        "input": {"text": text},
        "voice": {"languageCode": "en-IN", "name": voice_name},
        "audioConfig": {"audioEncoding": "LINEAR16", "speakingRate": speaking_rate}
    }
    
    # Chirp requires specific config
    if "Chirp" in voice_name:
        # Extract language code from voice name (e.g., "ta-IN" from "ta-IN-Chirp3-HD-Aoede")
        parts = voice_name.split("-")
        if len(parts) >= 2:
            payload["voice"]["languageCode"] = f"{parts[0]}-{parts[1]}"
        else:
            payload["voice"]["languageCode"] = "en-IN" # Fallback

    
    def _call_tts():
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return base64.b64decode(response.json().get('audioContent'))
        else:
            # Fallback for Chirp failures (e.g. language not supported for specific personality)
            if "Chirp" in voice_name:
                print(f"  -> Chirp voice {voice_name} failed (Status {response.status_code}). Falling back to Neural2...")
                # Dynamic fallback based on original voice language
                parts = voice_name.split("-")
                lang_code = f"{parts[0]}-{parts[1]}"
                fallback_voice = f"{lang_code}-Standard-A" 
                payload["voice"]["name"] = fallback_voice
                payload["voice"]["languageCode"] = lang_code
                response_retry = requests.post(url, json=payload)
                if response_retry.status_code == 200:
                    return base64.b64decode(response_retry.json().get('audioContent'))
            
            print(f"TTS Error: {response.text}")
            return None
        
    args = [] # Removed retry manager from here to simplify or keep it global? It relies on global retry_manager which is fine.
    audio_bytes = retry_manager.call(_call_tts)
    if audio_bytes and cache: cache.set(cache_key, audio_bytes)
    return audio_bytes

def get_media_duration_ms(path):
    """Returns duration of media file in milliseconds using ffprobe."""
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path]
        dur_sec = float(subprocess.check_output(cmd).strip())
        return int(dur_sec * 1000)
    except Exception as e:
        print(f"Error getting duration for {path}: {e}")
        return 0

def process_segment_task(seg, i, total, input_lang, output_lang, target_voice=None, speed=1.0, forced_target_text=None, status_reporter=None, status=None, log_callback=None, voice_manager=None, cache=None, temp_dir=None, job_id=None, target_wpm=None):
    """Processes a single segment. If forced_target_text is provided, skips MT and text adaptation."""
    def log(msg):
        # Prefer database logging if job_id available
        if job_id: 
            # We don't flood DB with per-segment logs unless error/critical, but console is fine
            print(f"[Job {job_id}] {msg}") 
            database.log_event(job_id, msg) # Explicitly log to DB for UI
        else:
            print(msg)
        
        if log_callback: log_callback(msg)

    source_text = seg['text']
    
    # -------------------------------------------------------------------------
    # FIX: Clean up speaker tags from text to prevent TTS from reading them
    # Handles: "[Speaker 1] Text", "Speaker 1: Text", etc.
    # -------------------------------------------------------------------------
    import re # Ensure re is available locally if not global
    
    # -------------------------------------------------------------------------
    # FIX V2: Aggressive Speaker Tag Cleaning (Loop)
    # Handles multiple tags e.g. "[Speaker 1] [Speaker 1] Text"
    # -------------------------------------------------------------------------
    # Regex to match "[Speaker X]" or "Speaker X:" or "Speaker X" at start
    speaker_pattern = r"^(\[Speaker \d+\]|Speaker \d+:|Speaker \d+)\s*"
    
    while re.match(speaker_pattern, source_text, flags=re.IGNORECASE):
        source_text = re.sub(speaker_pattern, "", source_text, count=1, flags=re.IGNORECASE)
        source_text = source_text.strip()
    
    # Parse timestamps if they're strings (from srt_utils.parse_srt)
    start_time = seg['start'] if isinstance(seg['start'], (int, float)) else srt_utils.parse_srt_time(seg['start'])
    end_time = seg['end'] if isinstance(seg['end'], (int, float)) else srt_utils.parse_srt_time(seg['end'])
    
    speaker = seg.get('speaker', 'Speaker 1')
    target_duration_ms = end_time - start_time
    target_duration_sec = target_duration_ms / 1000.0
    
    # Helper to clean text for TTS (Target Side Safeguard)
    def clean_text_for_tts(text):
        if not text: return text
        # Remove speaker tags from TARGET text (MT hallucinations)
        cleaned = text
        while re.match(speaker_pattern, cleaned, flags=re.IGNORECASE):
            cleaned = re.sub(speaker_pattern, "", cleaned, count=1, flags=re.IGNORECASE)
            cleaned = cleaned.strip()
        return cleaned
    
    # log(f"[{i+1}/{total}] [{speaker}] {target_duration_sec:.2f}s | {source_text[:30]}...")
    if status_reporter: status_reporter(f"Seg {i+1}: Analyzing...")
    
    if "[SILENCE]" in source_text.upper() or "[NOISE" in source_text.upper():
        return i, AudioSegment.silent(duration=target_duration_ms), source_text, start_time, end_time

    # Use passed voice_manager if available
    vm = voice_manager if voice_manager else VoiceManager() 
    voice_name = vm.get_voice(speaker, voice_preference=target_voice, output_lang=output_lang)
    
    # Determine Target Text
    if forced_target_text:
        current_translated_text = forced_target_text
        # If forced, we skip the adaptive MT step
    else:
        # Pass tts_speed (which is 'speed' argument) to adaptive logic
        current_translated_text = translate_adaptive(source_text, target_duration_sec, input_lang, output_lang, gender=target_voice, cache=cache, target_wpm=target_wpm, tts_speed=speed)
    
    best_audio = None
    best_text = current_translated_text
    
    # SAFEGUARD: Clean target text before TTS
    tts_input_text = clean_text_for_tts(current_translated_text)
    
    # 1. Try normal generation
    audio_bytes = generate_google_tts(tts_input_text, get_api_key(), voice_name=voice_name, speaking_rate=speed, cache=cache)
    if audio_bytes:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
        
        # Calculate Diff
        diff_ms = len(audio) - target_duration_ms
        log(f"Seg {i+1}: Target: {target_duration_sec:.2f}s | Generated: {len(audio)/1000:.2f}s | Diff: {diff_ms/1000:+.2f}s")

        # 2. Progressive Feedback Loop (NEW SYSTEM - Only if NOT forced text)
        if not forced_target_text:
            abs_diff_ms = abs(diff_ms)
            diff_sec = diff_ms / 1000.0
            
            # ═══════════════════════════════════════════════════════════════
            # TIER 1: SMALL DEVIATION - Just use atempo
            # ═══════════════════════════════════════════════════════════════
            if abs_diff_ms < config.ATEMPO_ONLY_THRESHOLD_MS:
                log(f"Seg {i+1}: Small deviation ({diff_sec:+.2f}s), will use atempo only")
                # Skip feedback, go straight to atempo adjustment below
            
            # ═══════════════════════════════════════════════════════════════
            # TIER 2: EXTREME DEVIATION - Immediate panic (skip word feedback)
            # ═══════════════════════════════════════════════════════════════
            elif abs_diff_ms > config.PANIC_IMMEDIATE_THRESHOLD_MS:
                log(f"Seg {i+1}: EXTREME deviation ({diff_sec:+.2f}s), immediate PANIC MODE")
                if status_reporter: status_reporter(f"Seg {i+1}: PANIC Mode (Extreme deviation)...")
                
                # Aggressive feedback
                panic_text = translate_adaptive(
                    source_text, target_duration_sec, input_lang, output_lang,
                    feedback="panic", time_diff=abs(diff_sec),
                    gender=target_voice, cache=cache, target_wpm=target_wpm,
                    tts_speed=speed
                )
                # SAFEGUARD
                audio_bytes_panic = generate_google_tts(clean_text_for_tts(panic_text), get_api_key(), voice_name=voice_name, speaking_rate=speed, cache=cache)
                
                if audio_bytes_panic:
                    audio_panic = AudioSegment.from_file(io.BytesIO(audio_bytes_panic), format="wav")
                    # Accept if it's closer to target
                    if abs(len(audio_panic) - target_duration_ms) < abs_diff_ms:
                        audio = audio_panic
                        best_text = panic_text
                        diff_ms = len(audio) - target_duration_ms
                        log(f"Seg {i+1}: PANIC Result: {len(audio)/1000:.2f}s | Diff: {diff_ms/1000:+.2f}s")
            
            # ═══════════════════════════════════════════════════════════════
            # TIER 3: MODERATE DEVIATION - Try word-based feedback first
            # ═══════════════════════════════════════════════════════════════
            else:  # abs_diff_ms >= WORD_FEEDBACK_THRESHOLD_MS and <= PANIC_IMMEDIATE_THRESHOLD_MS
                
                # Calculate word adjustment
                local_wpm = target_wpm if target_wpm else LANGUAGE_WPM.get(output_lang, 138)
                wps = local_wpm / 60.0
                words_to_adjust = max(1, int(abs(diff_sec) * wps))
                
                # Too LONG - Summarize
                if diff_ms > 0:
                    # Calculate how many words to reduce (same WPM logic as elaboration)
                    local_wpm = target_wpm if target_wpm else LANGUAGE_WPM.get(output_lang, 138)
                    wps = local_wpm / 60.0
                    words_to_remove = max(1, int(diff_sec * wps))
                    
                    log(f"Seg {i+1}: Too long by {diff_sec:.2f}s. Summarizing by ~{words_to_remove} words...")
                    if status_reporter: status_reporter(f"Seg {i+1}: Summarizing...")
                    
                    summarized_text = translate_adaptive(
                        source_text, target_duration_sec, input_lang, output_lang,
                        feedback="summarize", time_diff=diff_sec, words_to_add=words_to_remove,  # Reusing words_to_add param
                        gender=target_voice, cache=cache, target_wpm=target_wpm,
                        tts_speed=speed
                    )
                    # SAFEGUARD
                    audio_bytes_v2 = generate_google_tts(clean_text_for_tts(summarized_text), get_api_key(), voice_name=voice_name, speaking_rate=speed, cache=cache)
                    
                    if audio_bytes_v2:
                        audio_v2 = AudioSegment.from_file(io.BytesIO(audio_bytes_v2), format="wav")
                        if len(audio_v2) < len(audio):  # Accept if shorter
                            audio = audio_v2
                            best_text = summarized_text
                            diff_ms = len(audio) - target_duration_ms
                            log(f"Seg {i+1}: After summarization: {len(audio)/1000:.2f}s | Diff: {diff_ms/1000:+.2f}s")
                
                # Too SHORT - Elaborate
                else:  # diff_ms < 0
                    log(f"Seg {i+1}: Too short by {abs(diff_sec):.2f}s. Elaborating by ~{words_to_adjust} words...")
                    if status_reporter: status_reporter(f"Seg {i+1}: Elaborating...")
                    
                    elaborated_text = translate_adaptive(
                        source_text, target_duration_sec, input_lang, output_lang,
                        feedback="elaborate", time_diff=abs(diff_sec), words_to_add=words_to_adjust,
                        gender=target_voice, cache=cache, target_wpm=target_wpm,
                        tts_speed=speed
                    )
                    # SAFEGUARD
                    audio_bytes_v2 = generate_google_tts(clean_text_for_tts(elaborated_text), get_api_key(), voice_name=voice_name, speaking_rate=speed, cache=cache)
                    
                    if audio_bytes_v2:
                        audio_v2 = AudioSegment.from_file(io.BytesIO(audio_bytes_v2), format="wav")
                        if len(audio_v2) > len(audio):  # Accept if longer
                            audio = audio_v2
                            best_text = elaborated_text
                            diff_ms = len(audio) - target_duration_ms
                            log(f"Seg {i+1}: After elaboration: {len(audio)/1000:.2f}s | Diff: {diff_ms/1000:+.2f}s")
                
                # ═══════════════════════════════════════════════════════════
                # POST-FEEDBACK CHECK: Still >1s off? → PANIC MODE
                # ═══════════════════════════════════════════════════════════
                if abs(diff_ms) > config.PANIC_POST_FEEDBACK_THRESHOLD_MS:
                    gap_v2 = diff_ms / 1000.0
                    log(f"Seg {i+1}: Still {abs(gap_v2):.2f}s off after feedback. Entering PANIC MODE...")
                    if status_reporter: status_reporter(f"Seg {i+1}: PANIC Mode (post-feedback)...")
                    
                    panic_text = translate_adaptive(
                        source_text, target_duration_sec, input_lang, output_lang,
                        feedback="panic", time_diff=abs(gap_v2),
                        gender=target_voice, cache=cache, target_wpm=target_wpm,
                        tts_speed=speed
                    )
                    # SAFEGUARD
                    audio_bytes_v3 = generate_google_tts(clean_text_for_tts(panic_text), get_api_key(), voice_name=voice_name, speaking_rate=speed, cache=cache)
                    
                    if audio_bytes_v3:
                        audio_v3 = AudioSegment.from_file(io.BytesIO(audio_bytes_v3), format="wav")
                        # Accept if it's closer to target
                        if abs(len(audio_v3) - target_duration_ms) < abs(diff_ms):
                            audio = audio_v3
                            best_text = panic_text
                            diff_ms = len(audio) - target_duration_ms
                            log(f"Seg {i+1}: PANIC Result: {len(audio)/1000:.2f}s | Diff: {diff_ms/1000:+.2f}s")

        # 3. Final Speed Adjustment (atempo) if still too long
        # We apply this even for forced text to ensure sync
        if len(audio) > target_duration_ms:
            speed_factor = len(audio) / target_duration_ms
            if speed_factor > 1.02: # Only if > 2% diff
                eff_speed = min(speed_factor, config.MAX_ATEMPO_SPEEDUP) # Cap at config limit
                log(f"Seg {i+1}: Adjusting speed by {eff_speed:.2f}x (Required: {speed_factor:.2f}x)")
                if status_reporter: status_reporter(f"Seg {i+1}: Syncing speed ({eff_speed:.2f}x)...")
                
                if speed_factor > config.MAX_ATEMPO_SPEEDUP:
                    log(f"Seg {i+1}: WARNING: Audio too long! Desync of {(len(audio)/config.MAX_ATEMPO_SPEEDUP - target_duration_ms)/1000:.2f}s remaining.")

                temp_in = f"temp_seg_{i}_{int(time.time())}.wav"
                temp_out = f"temp_seg_{i}_{int(time.time())}_fast.wav"
                audio.export(temp_in, format="wav")
                
                cmd = ["ffmpeg", "-y", "-i", temp_in, "-filter:a", f"atempo={eff_speed}", "-vn", temp_out]
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    audio = AudioSegment.from_file(temp_out, format="wav")
                except Exception as e:
                    print(f"  -> FFmpeg atempo failed: {e}")
                finally:
                    if os.path.exists(temp_in): os.remove(temp_in)
                    if os.path.exists(temp_out): os.remove(temp_out)

        best_audio = audio

    return i, best_audio, best_text, start_time, end_time

def process_video(job_id: str, video_path: str, duration_limit: Optional[int] = None, input_lang: str = "Malayalam", output_lang: str = "English", target_voice: Optional[str] = None, speed: float = 1.0, target_wpm: Optional[float] = None) -> None:
    
    # Setup Isolation Paths
    base_dir = os.path.join(JOBS_DIR, job_id)
    output_dir = os.path.join(base_dir, "output")
    srt_dir = os.path.join(base_dir, "srt")
    temp_dir = os.path.join(base_dir, "temp")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(srt_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Init Local State
    # Use GLOBAL cache for TTS reuse across jobs
    # We still use local cache wrapper but point it to a shared dir or mix?
    # User requested: "for MT and tts it depends if the language was the same, then use..."
    # A shared cache dir allows this.
    GLOBAL_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
    local_cache = CacheManager(GLOBAL_CACHE_DIR)
    local_vm = VoiceManager()
    
    def log(msg):
        print(f"[Job {job_id}] {msg}")
        database.log_event(job_id, msg)
        database.update_job_status(job_id, message=msg)

    filename = os.path.basename(video_path)
    database.update_job_status(job_id, status="processing", progress=0, step="Preprocessing")
    log(f"Starting processing for {filename}...")

    # Calculate File Hash for Reuse Lookup
    file_hash = None
    try:
        # Hash the video file for deduplication (chunked for memory efficiency)
        sha256_hash = hashlib.sha256()
        with open(video_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        file_hash = sha256_hash.hexdigest()
    except Exception as e:
        log(f"Warning: Failed to calculate hash: {e}")
        file_hash = None

    # --- INPUT PREPARATION & TRIMMING ---
    # Strategy: Always work on a COPY in job_dir/input
    # This prevents modifying the source (Uploads/file) and handles trimming cleanly.
    
    job_input_dir = os.path.join(base_dir, "input")
    os.makedirs(job_input_dir, exist_ok=True)
    
    working_video_path = os.path.join(job_input_dir, filename)
    
    # 1. Copy Source -> Working Path
    if not os.path.exists(working_video_path):
        import shutil
        shutil.copy2(video_path, working_video_path)
        log(f"Copied input to workspace: {working_video_path}")
    
    # Update video_path to point to our working copy
    video_path = working_video_path
    
    # 2. Trim (if requested)
    if duration_limit:
        log(f"Applying duration limit: {duration_limit}s")
        # Trim IN PLACE (since it's already a copy)
        temp_trim_path = os.path.join(job_input_dir, f"temp_trim_{uuid.uuid4().hex}.mp4")
        try:
            cmd = ["ffmpeg", "-y", "-i", video_path, "-t", str(duration_limit), "-c", "copy", temp_trim_path]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.5)
            os.replace(temp_trim_path, video_path)
            log(f"Trimmed working copy to {duration_limit}s")
        except Exception as e:
            log(f"Trimming failed: {e}")
            if os.path.exists(temp_trim_path): os.remove(temp_trim_path)
            raise
    
    name_no_ext = os.path.splitext(filename)[0]

    # Load metadata Checkpoint if exists (Speaker Map)
    metadata_path = os.path.join(base_dir, "metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                meta = json.load(f)
                if 'speaker_map' in meta:
                    local_vm.speaker_map = meta['speaker_map']
            log("Resuming: Loaded existing speaker map.")
        except Exception as e:
            log(f"Warning: Failed to load metadata: {e}")

    # Check if input is audio-only
    is_audio_only = filename.lower().endswith(('.mp3', '.wav', '.aac', '.flac', '.m4a'))
    
    # Trimming logic moved up ^
    # Cleanup old logic references
    pass
    
    log(f"--- Step 1: Generating SRT for {filename} ---")
    database.update_job_status(job_id, progress=20, step="Transcribing")
    
    # Checkpoint: SRT
    srt_file_path = os.path.join(srt_dir, f"{name_no_ext}_ASR.srt")
    # reuse_asr_from_job logic
    # Check if we can reuse ASR from a previous job for this file hash
    reused_asr = False
    
    if file_hash:
        previous_jobs = database.get_jobs_by_file_hash(file_hash)
        for prev_job in previous_jobs:
            if prev_job['id'] == job_id: continue # Skip self
            
            # Logic: Input Lang Match + Duration Limit Match
            # If duration limit is different, ASR timing would be wrong
            # Note: stored 'duration_limit' in DB might be None if column new
            
            prev_limit = prev_job.get('duration_limit')
            # Normalize to compare (None vs None, or Int vs Int)
            
            if prev_job['input_lang'] == input_lang and str(prev_limit) == str(duration_limit):
                # Found Candidate!
                # Check for SRT file
                prev_srt_path = os.path.join(JOBS_DIR, prev_job['id'], "srt", f"{os.path.splitext(prev_job['filename'])[0]}_ASR.srt")
                
                # Check name mismatch (if filename changed but hash same)
                if not os.path.exists(prev_srt_path):
                     # Try listing dir
                     prev_srt_dir = os.path.join(JOBS_DIR, prev_job['id'], "srt")
                     if os.path.exists(prev_srt_dir):
                         for f in os.listdir(prev_srt_dir):
                             if f.endswith("_ASR.srt"):
                                 prev_srt_path = os.path.join(prev_srt_dir, f)
                                 break
                
                if os.path.exists(prev_srt_path):
                    try:
                        log(f"Smart Reuse: Found existing ASR from Job {prev_job['id']}")
                        with open(prev_srt_path, 'r', encoding='utf-8') as src:
                            srt_content = src.read()
                        
                        # Copy to current job
                        with open(srt_file_path, 'w', encoding='utf-8') as dst:
                            dst.write(srt_content)
                            
                        reused_asr = True
                        break
                    except Exception as e:
                         log(f"Warning: Failed to reuse ASR: {e}")

    if reused_asr:
         pass # srt_content loaded
    elif os.path.exists(srt_file_path):
        log("Resuming: Found existing ASR SRT. Skipping transcription.")
        with open(srt_file_path, "r", encoding="utf-8") as f:
            srt_content = f.read()
    else:
        # Generate SRT (Pass job_id for internal logging if needed)
        srt_content = generate_srt_gemini(video_path, input_lang=input_lang, model="gemini-2.5-pro", job_id=job_id)
        srt_content = fix_srt_timestamps(srt_content)
        srt_content = merge_segments_to_sentences(srt_content)
        
        with open(srt_file_path, "w", encoding="utf-8") as f:
            f.write(srt_content)
            
    # Save Metadata (Speaker Map might be updated during ASR/Dub?)
    # Actually Speaker Map updates during Dubbing.
    # We should save it at end.
    
    log(f"--- Step 2: Translating & Generating Audio (Parallel) ---")
    database.update_job_status(job_id, progress=40, step="Translating & Dubbing")
    
    segments = srt_utils.parse_srt(srt_content)
    

    processed_results = [None] * len(segments)
    
    total_segments = len(segments)
    completed_segments = 0

    # Reuse the same max_workers logic
    try:
         max_workers_gen = int(os.getenv("MAX_CONCURRENT_CHUNKS", "10"))
    except: 
         max_workers_gen = 10

    with ThreadPoolExecutor(max_workers=max_workers_gen) as executor:
        futures = []
        for i, seg in enumerate(segments):
            # Pass local managers and temp dir
            futures.append(executor.submit(
                process_segment_task, 
                seg, 
                i, 
                len(segments), 
                input_lang, 
                output_lang, 
                target_voice, 
                speed, 
                forced_target_text=None, 
                voice_manager=local_vm, # Pass local VM
                cache=local_cache,      # Pass local Cache
                temp_dir=temp_dir,      # Pass local Temp
                job_id=job_id,          # For logging
                target_wpm=target_wpm,   # User WPM preference

            ))
        
        for future in as_completed(futures):
            try:
                i, audio_seg, text, start, end = future.result()
                processed_results[i] = (audio_seg, text, start, end)
                
                completed_segments += 1
                prog = 40 + int((completed_segments / total_segments) * 40) # 40% to 80%
                database.update_job_status(job_id, progress=prog, message=f"Segment {completed_segments}/{total_segments}")
     
            except Exception as e:
                log(f"Error in segment task: {e}")

    # Hardening: Check if we have ANY success
    success_count = sum(1 for r in processed_results if r is not None)
    if success_count == 0 and len(segments) > 0:
        log("CRITICAL: All segments failed processing. Aborting pipeline.")
        raise RuntimeError(f"Pipeline failed: 0/{len(segments)} segments processed successfully. Check logs for details.")

    # Save Metadata Checkpoint (Speaker Map)
    try:
        metadata_path = os.path.join(base_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump({
                "speaker_map": local_vm.speaker_map,
                "updated_at": str(datetime.now())
            }, f, indent=2)
        log("Checkpoint: Saved speaker map.")
    except Exception as e:
        log(f"Warning: Failed to save metadata: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # OVERLAP PROTECTION: Validate and fix segment timing overlaps
    # ═══════════════════════════════════════════════════════════════════════
    log("Validating segment boundaries for overlaps...")
    overlap_count = 0
    severe_overlap_count = 0
    
    for i in range(len(processed_results) - 1):
        if not processed_results[i] or not processed_results[i+1]:
            continue
            
        current = processed_results[i]
        next_seg = processed_results[i+1]
        
        current_audio, current_text, current_start, current_end = current
        next_audio, next_text, next_start, next_end = next_seg
        
        overlap_ms = current_end - next_start
        
        if overlap_ms > 0:  # Overlap detected!
            overlap_count += 1
            
            if overlap_ms <= config.MAX_ACCEPTABLE_OVERLAP_MS:
                # Small overlap (<500ms) - trim current segment
                log(f"Seg {i} overlaps Seg {i+1} by {overlap_ms}ms. Trimming...")
                
                # Adjust end time to prevent overlap
                new_end = next_start - config.SEGMENT_BUFFER_MS
                trim_amount_ms = current_end - new_end
                
                # Trim audio segment
                if current_audio and trim_amount_ms > 0:
                    trimmed_audio = current_audio[:-trim_amount_ms]
                    processed_results[i] = (trimmed_audio, current_text, current_start, new_end)
                    log(f"  → Trimmed {trim_amount_ms}ms from Seg {i} (new duration: {len(trimmed_audio)}ms)")
                    
            else:
                # Large overlap (>500ms) - trust the pipeline, just log
                severe_overlap_count += 1
                log(f"⚠️  Seg {i} overlaps Seg {i+1} by {overlap_ms}ms (>500ms threshold)")
                log(f"  → Trusting pipeline timing. Video length may change slightly.")
    
    if overlap_count > 0:
        log(f"Overlap validation complete: {overlap_count} overlaps detected, {overlap_count - severe_overlap_count} fixed")
        if severe_overlap_count > 0:
            log(f"⚠️  {severe_overlap_count} severe overlaps (>500ms) left as-is per pipeline trust policy")
    else:
        log("✅ No segment overlaps detected")

    log("--- Assembling Audio (Legacy Sequential Mode) ---")
    database.update_job_status(job_id, progress=85, step="Finalizing", message="Stitching audio...")
    
    # Legacy Assembly Logic: Strict Sequential Appending (No Overlaps)
    # We ignore exact absolute timestamp if it causes overlap.
    # Logic: 
    #   Next Start = max(Original Start, Previous End)
    #   Gap = Next Start - Previous End. If > 0, insert silence. 
    #   If < 0, it means we are late, so we just append (effectively shifting start).
    
    combined_audio = AudioSegment.empty()
    translated_segments = []
    
    for i, res in enumerate(processed_results):
        if not res: continue
        audio_seg, text, start_time, end_time = res
        
        current_duration_ms = len(combined_audio)
        silence_gap = start_time - current_duration_ms
        
        # Only add silence if we are early (positive gap).
        # If silence_gap is negative, it means the previous segment ran long, 
         # so we just append immediately (pushing this segment's start time forward).
        if silence_gap > 0:
            combined_audio += AudioSegment.silent(duration=silence_gap)
            
        if audio_seg:
            combined_audio += audio_seg
        else:
            # If segment failed, we should probably fill the gap to maintain sync?
            # Or just skip? Legacy usually skipped or added silence.
            # Let's add silence for the intended duration of this segment to minimize drift.
            log(f"Warning: Segment at {start_time}ms failed. Filling with silence.")
            target_dur = end_time - start_time
            combined_audio += AudioSegment.silent(duration=target_dur)

        translated_segments.append({
            "start": start_time, 
            "end": end_time, 
            "text": segments[i]['text'],     # Source from original list
            "target_text": text,             # Translated from result
            "status": "ai"
        })
            
    # Save Segments to DB
    database.save_segments(job_id, translated_segments)

    mt_srt_path = os.path.join(srt_dir, f"{name_no_ext}_{output_lang}.srt")
    with open(mt_srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(translated_segments):
            s = datetime.utcfromtimestamp(seg['start']/1000).strftime('%H:%M:%S,%f')[:-3]
            e = datetime.utcfromtimestamp(seg['end']/1000).strftime('%H:%M:%S,%f')[:-3]
            f.write(f"{i+1}\n{s} --> {e}\n{seg['target_text']}\n\n")

    audio_path = os.path.join(output_dir, f"{name_no_ext}_dubbed.wav")
    combined_audio.export(audio_path, format="wav")
    
    final_output_path = ""
    
    if is_audio_only:
        log(f"--- Audio Only: Saved to {audio_path} ---")
        final_output_path = audio_path
        database.update_job_status(job_id, progress=100, step="Completed", message="Audio ready.")
    else:
        output_video_path = os.path.join(output_dir, f"{name_no_ext}_dubbed{os.path.splitext(filename)[1]}")
        log(f"--- Step 3: Merging into {output_video_path} ---")
        database.update_job_status(job_id, progress=90, step="Muxing")
        
        cmd = ["ffmpeg", "-y", "-i", video_path, "-i", audio_path, "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest", "-movflags", "+faststart", output_video_path]
        subprocess.run(cmd, check=True)
        log("Done! Video saved.")
        final_output_path = output_video_path
        database.update_job_status(job_id, progress=100, step="Completed", status="completed", message="Video ready.")

    # Return structured data (Legacy support, but DB is primary)
    with open(mt_srt_path, "r", encoding="utf-8") as f:
        target_srt_content = f.read()
    
    return {
        "output_path": final_output_path,
        "source_srt": srt_content,
        "target_srt": target_srt_content,
        "source_segments": segments, 
        "target_segments": translated_segments
    }

def regenerate_video(job_id, video_path, new_segments, old_segments, input_lang="English", output_lang="Hindi", target_voice=None, speed=1.0, speaker_overrides={}):
    """
    [DEPRECATED] DO NOT USE. 
    Use regenerate_handler.py:regenerate_dubbing_task instead.
    
    Regenerates the dubbing based on updated segments (Job Isolated).
    """
    base_dir = os.path.join(JOBS_DIR, job_id)
    output_dir = os.path.join(base_dir, "output")
    srt_dir = os.path.join(base_dir, "srt")
    temp_dir = os.path.join(base_dir, "temp")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(srt_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    local_cache = CacheManager(os.path.join(base_dir, ".cache"))
    local_vm = VoiceManager()
    
    def log(msg):
        print(f"[Job {job_id}] {msg}")
        database.log_event(job_id, msg)
        database.update_job_status(job_id, message=msg)

    filename = os.path.basename(video_path)
    name_no_ext = os.path.splitext(filename)[0]
    
    log(f"--- Regenerating Dub for {filename} ---")
    database.update_job_status(job_id, status="processing", progress=0, step="Regenerating")

    processed_results = [None] * len(new_segments)
    total_segments = len(new_segments)
    completed_segments = 0

    # Create a map of fast lookup for old segments to handle reordering/deletions robustly
    def get_seg_id(seg):
        # ID based on Start Time + Content. 
        # Using hash of text helps differentiate segments at same time (rare) 
        # and ensures simple text changes break the ID (forcing re-translation).
        return f"{int(seg['start'])}_{hash(seg['text'])}"

    old_seg_map = {get_seg_id(s): s for s in old_segments}

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i, new_seg in enumerate(new_segments):
            # Robust Lookup: Match by ID instead of Index
            seg_id = get_seg_id(new_seg)
            old_seg = old_seg_map.get(seg_id)
            
            # If old_seg is found, it means Source Text & Start Time are IDENTICAL.
            # So 'source_changed' is effectively False.
            
            source_changed = (old_seg is None) 
            target_changed = old_seg and (new_seg.get('target_text') != old_seg.get('target_text'))
            
            forced_target = None
            
            if new_seg.get('deleted'):
                print(f"Segment {i} marked as DELETED. Silencing.")
                # Create silent segment
                target_dur_ms = new_seg['end'] - new_seg['start']
                silent_audio = AudioSegment.silent(duration=target_dur_ms)
                processed_results[i] = (silent_audio, "", new_seg['start'], new_seg['end'])
                continue # Skip processing task

            if source_changed:
                print(f"Segment {i}: New or Modified Source. Translating...")
                # forced_target remains None, so MT runs
            elif target_changed:
                target_val = new_seg.get('target_text')
                # FIX: If user CLEARED the target text (empty string), treat it as a request to re-translate from source.
                # Only use forced_target if it has actual content.
                if target_val is None or target_val.strip() == "":
                    print(f"Segment {i}: Target cleared. Re-translating...")
                    forced_target = None 
                else:
                    print(f"Segment {i}: Target Modified manually. Using forced text.")
                    forced_target = target_val
            else:
                # No change? Check if target is mysteriously missing in object but expected?
                # Usually if old_seg matched, it has target text. 
                # But just in case
                target_val = new_seg.get('target_text')
                if target_val is None or target_val.strip() == "":
                     # Should not happen if old_seg matched and had text, unless old_seg didn't have text?
                     if old_seg.get('target_text'):
                         # Weird mismatch. Trust old seg? Or re-translate?
                         # Assume re-translate if missing.
                         print(f"Segment {i}: Target missing despite ID match. Re-translating.")
                         forced_target = None
                     else:
                         print(f"Segment {i}: No Target in old segment. Re-translating.")
                         forced_target = None
                else:
                    # Use Cache (TTS) by passing the existing target text to skipping MT
                    forced_target = target_val
            
            # Determine voice for this segment
            # Use override if present for this speaker, else fallback to global target_voice
            seg_speaker = new_seg.get('speaker', 'Speaker 1')
            effective_voice = speaker_overrides.get(seg_speaker, target_voice)

            # We must construct a 'seg' object that matches what process_segment_task expects
            # It expects 'text' to be the SOURCE text.
            task_seg = {
                "text": new_seg['text'], # Source Text
                "start": new_seg['start'],
                "end": new_seg['end'],
                "speaker": new_seg.get('speaker', 'Speaker 1')
            }
            
            futures.append(executor.submit(
                process_segment_task, 
                task_seg, i, total_segments, 
                input_lang, output_lang, effective_voice, speed, 
                forced_target_text=forced_target,
                status_reporter=lambda msg, idx=i, tot=total_segments: database.update_job_status(job_id, progress=int((idx/tot)*90), message=msg),
                voice_manager=local_vm,
                cache=local_cache,
                temp_dir=temp_dir,
                job_id=job_id
            ))
        
        for future in as_completed(futures):
            try:
                i, audio_seg, text, start, end = future.result()
                processed_results[i] = (audio_seg, text, start, end)
                
                completed_segments += 1
                prog = int((completed_segments / total_segments) * 80)
                database.update_job_status(job_id, progress=prog, step="Regenerating", message=f"Segment {completed_segments}/{total_segments}")
                    
            except Exception as e:
                print(f"Error in regeneration task {i}: {e}")

    print("--- Assembling Audio (Legacy Sequential Mode) ---")
    database.update_job_status(job_id, progress=90, step="Finalizing", message="Stitching audio...")
    
    # Legacy Assembly Logic: Strict Sequential Appending
    combined_audio = AudioSegment.empty()
    translated_segments = []
    
    for i, res in enumerate(processed_results):
        if not res: continue
        audio_seg, text, start_time, end_time = res
        
        current_duration_ms = len(combined_audio)
        silence_gap = start_time - current_duration_ms
        
        if silence_gap > 0:
            combined_audio += AudioSegment.silent(duration=silence_gap)
            
        if audio_seg:
            combined_audio += audio_seg
        else:
            # Fill failed segment duration with silence
            target_dur = end_time - start_time
            combined_audio += AudioSegment.silent(duration=target_dur)

        translated_segments.append({
            "start": start_time, 
            "end": end_time, 
            "text": new_segments[i]['text'],  # Source
            "target_text": text               # Target
        })

    # Save new Target SRT
    mt_srt_path = os.path.join(srt_dir, f"{name_no_ext}_{output_lang}.srt")
    with open(mt_srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(translated_segments):
            s = datetime.utcfromtimestamp(seg['start']/1000).strftime('%H:%M:%S,%f')[:-3]
            e = datetime.utcfromtimestamp(seg['end']/1000).strftime('%H:%M:%S,%f')[:-3]
            f.write(f"{i+1}\n{s} --> {e}\n{seg['target_text']}\n\n")

    audio_path = os.path.join(output_dir, f"{name_no_ext}_dubbed.wav")
    combined_audio.export(audio_path, format="wav")
    
    # Use standard naming for output video
    output_video_path = os.path.join(output_dir, f"{name_no_ext}_dubbed{os.path.splitext(filename)[1]}")
    
    # Check if audio only
    is_audio_only = filename.lower().endswith(('.mp3', '.wav', '.aac', '.flac', '.m4a'))
    final_output_path = ""

    if is_audio_only:
        final_output_path = audio_path
    else:
        print(f"--- Merging into {output_video_path} ---")
        cmd = ["ffmpeg", "-y", "-i", video_path, "-i", audio_path, "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest", "-movflags", "+faststart", output_video_path]
        subprocess.run(cmd, check=True)
        final_output_path = output_video_path

    database.update_job_status(job_id, status="completed", progress=100, step="Completed", message="Regeneration successful.")
    
    # CRITICAL FIX: Save updated segments to DB so frontend sees changes
    database.save_segments(job_id, translated_segments)

    source_srt_path = os.path.join(srt_dir, f"{name_no_ext}_ASR.srt")
    target_srt_path = mt_srt_path
    
    source_srt_content = ""
    if os.path.exists(source_srt_path):
        with open(source_srt_path, "r", encoding="utf-8") as f:
            source_srt_content = f.read()
        
    target_srt_content = ""
    if os.path.exists(target_srt_path):
        with open(target_srt_path, "r", encoding="utf-8") as f:
            target_srt_content = f.read()

    return {
        "output_path": final_output_path,
        "source_srt": source_srt_content,
        "target_srt": target_srt_content,
        "source_segments": new_segments, 
        "target_segments": translated_segments
    }

# === RUN ===
# === RUN ===
if __name__ == "__main__":
    # USER REQUESTED TEST CONFIGURATION
    test_video = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Videos", "dubflow test.mp4")
    output_dir = "Video out"
    
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(test_video):
        print(f"--- Running Test on {test_video} ---")
        test_job_id = uuid.uuid4().hex
        print(f"Test Job ID: {test_job_id}")
        process_video(
            test_job_id,
            test_video, 
            duration_limit=None, 
            input_lang="English", 
            output_lang="Tamil", 
            target_voice="Male (Chirp 3 - Achird)", 
            speed=1
        )
    else:
        print(f"Test video not found: {test_video}")
    
    # metrics.save_report()
    # print("Metrics saved to metrics.json")

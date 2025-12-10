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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dotenv import load_dotenv
from pydub import AudioSegment
from google import genai
from google.genai import types

# Load environment variables
# Global client variable, initialized lazily
client = None
GEMINI_API_KEY = None

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

# Add ffmpeg to PATH

# Add ffmpeg to PATH (Assuming system install or local bin)
# os.environ["PATH"] += os.pathsep + r"C:\Users\anand\miniconda3\envs\tf_env\Library\bin"


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

metrics = MetricsLogger()

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
            except:
                return None
        return None

    def set(self, key, value):
        path = self._get_path(key)
        with open(path, "wb") as f:
            pickle.dump(value, f)

cache = CacheManager()

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
                    return None
                else:
                    self.failures = 0 # Reset

        for attempt in range(self.max_retries + 1):
            try:
                start = time.time()
                result = func(*args, **kwargs)
                metrics.log_latency(func.__name__, (time.time() - start) * 1000)
                
                with self.lock:
                    self.failures = 0 # Reset on success
                return result
            except Exception as e:
                print(f"Error in {func.__name__} (Attempt {attempt+1}/{self.max_retries+1}): {e}")
                metrics.log_retry()
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

def split_audio_chunks(video_path, chunk_length_ms=300000):
    """
    Splits audio from video into chunks using ffmpeg (faster than pydub).
    Returns list of chunk file paths and the temp directory.
    """
    chunk_length_sec = chunk_length_ms / 1000
    print(f"Splitting audio into {chunk_length_sec}s chunks using ffmpeg...")
    
    temp_dir = "temp_chunks"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    # Output pattern for ffmpeg
    output_pattern = os.path.join(temp_dir, "chunk_%03d.wav")
    
    cmd = [
        "ffmpeg", "-i", video_path,
        "-f", "segment",
        "-segment_time", str(chunk_length_sec),
        "-c:a", "pcm_s16le",
        "-vn", # No video
        output_pattern
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"Error splitting audio: {e}")
        raise
        
    chunks = sorted([os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".wav")])
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
            # Check if specific Chirp voice is requested
            if voice_preference and "Chirp 3" in voice_preference:
                # Extract personality from preference string e.g. "Female (Chirp 3 - Aoede)"
                personality = None
                for p in self.chirp_3_personalities:
                    if p in voice_preference:
                        personality = p
                        break
                
                if personality:
                    # Map output language to BCP-47 code
                    lang_code = "en-IN" # Default
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
                    
                    # Construct Dynamic ID: {lang_code}-Chirp3-HD-{personality}
                    return f"{lang_code}-Chirp3-HD-{personality}"

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

voice_manager = VoiceManager()

def parse_gemini_json(response_text, offset_ms=0):
    """Parses Gemini JSON response and applies time offset."""
    # Extract JSON from code blocks if present
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0]
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0]
        
    try:
        segments = json.loads(response_text)
    except json.JSONDecodeError:
        print(f"Error decoding JSON: {response_text}")
        return []

    def parse_time(t_str):
        t_str = t_str.strip().replace('.', ',')
        parts = t_str.split(':')
        
        if len(parts) == 3: # HH:MM:SS,mmm
            h, m, s_ms = parts
        elif len(parts) == 2: # MM:SS,mmm
            h = "0"
            m, s_ms = parts
        else:
            raise ValueError(f"Invalid time format: {t_str}")
            
        if ',' in s_ms:
            s, ms = s_ms.split(',')
        else:
            s = s_ms
            ms = "000"
            
        return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)

    def format_time(ms):
        s = int(ms / 1000)
        ms = int(ms % 1000)
        m = int(s / 60)
        s = s % 60
        h = int(m / 60)
        m = m % 60
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    processed_segments = []
    for seg in segments:
        if seg.get('type') == 'silence':
            continue
            
        try:
            start_ms = parse_time(seg['start']) + offset_ms
            end_ms = parse_time(seg['end']) + offset_ms
            
            processed_segments.append({
                "start": start_ms,
                "end": end_ms,
                "text": seg['text'],
                "speaker": seg.get('speaker', 'Speaker 1'),
                "start_str": format_time(start_ms),
                "end_str": format_time(end_ms)
            })
        except Exception as e:
            print(f"Skipping malformed segment: {seg} - {e}")
        
    return processed_segments

def segments_to_srt(segments):
    lines = []
    for i, seg in enumerate(segments):
        speaker = seg.get('speaker', '')
        text = f"[{speaker}] {seg['text']}" if speaker else seg['text']
        lines.append(f"{i+1}")
        lines.append(f"{seg['start_str']} --> {seg['end_str']}")
        lines.append(f"{text}\n")
    return "\n".join(lines)

def fix_srt_timestamps(srt_content):
    """
    Fixes SRT timestamps by parsing them and re-formatting to strict HH:MM:SS,mmm.
    """
    def normalize_timestamp(ts_str):
        ts_str = ts_str.strip().replace('.', ',')
        if ',' in ts_str:
            main, ms = ts_str.split(',')
        else:
            main = ts_str
            ms = "000"
        parts = main.split(':')
        total_seconds = 0
        try:
            if len(parts) == 3:
                h, m, s = map(int, parts)
                total_seconds = h * 3600 + m * 60 + s
            elif len(parts) == 2:
                m, s = map(int, parts)
                total_seconds = m * 60 + s
            else:
                return ts_str
        except ValueError:
            return ts_str
        h = total_seconds // 3600
        m = (total_seconds % 3600) // 60
        s = total_seconds % 60
        return f"{h:02}:{m:02}:{s:02},{ms.ljust(3, '0')[:3]}"

    lines = srt_content.splitlines()
    new_lines = []
    for line in lines:
        if "-->" in line:
            parts = line.split("-->")
            if len(parts) == 2:
                start = normalize_timestamp(parts[0])
                end = normalize_timestamp(parts[1])
                new_lines.append(f"{start} --> {end}")
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
    return "\n".join(new_lines)

def merge_segments_to_sentences(srt_content):
    """Merges SRT segments into full sentences based on punctuation."""
    segments = parse_srt(srt_content)
    merged_segments = []
    current_text = ""
    current_start = None
    current_speaker = None
    
    terminals = ('.', '?', '!', '|', '।')
    
    for seg in segments:
        text = seg['text'].strip()
        start = seg['start']
        end = seg['end']
        speaker = seg.get('speaker', 'Speaker 1')
        
        if current_start is None:
            current_start = start
            current_speaker = speaker
            
        # If speaker changes, force split
        if speaker != current_speaker and current_text:
            merged_segments.append({
                "start": current_start,
                "end": segments[segments.index(seg)-1]['end'],
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
        s = datetime.utcfromtimestamp(seg['start']/1000).strftime('%H:%M:%S,%f')[:-3]
        e = datetime.utcfromtimestamp(seg['end']/1000).strftime('%H:%M:%S,%f')[:-3]
        speaker_tag = f"[{seg['speaker']}] " if seg.get('speaker') else ""
        lines.append(f"{i+1}\n{s} --> {e}\n{speaker_tag}{seg['text']}\n")
    return "\n".join(lines)

def generate_srt_gemini(video_path, input_lang="English", model="gemini-2.5-pro", duration_limit=None):
    """Generates SRT using Gemini with updated prompt for silence/noise/diarization."""
    chunk_size_ms = 300000
    chunk_paths, temp_dir = split_audio_chunks(video_path, chunk_length_ms=chunk_size_ms)
    all_segments = []
    
    def process_chunk(i, chunk_path):
        print(f"Processing chunk {i+1}/{len(chunk_paths)}: {chunk_path}")
        cache_key = f"asr_{model}_{input_lang}_{os.path.basename(chunk_path)}_{os.path.getsize(chunk_path)}"
        cached_result = cache.get(cache_key)
        if cached_result:
            print(f"Chunk {i+1} found in cache.")
            return cached_result

        chunk_file = upload_to_gemini(chunk_path, mime_type="audio/wav")
        wait_for_files_active([chunk_file])
        
        prompt = f"""
        Role: You are a high-precision ASR system specialized in time-aligned transcription and speaker diarization.

        INPUT: Audio Chunk ({i+1}/{len(chunk_paths)})
        TASK: Transcribe the spoken audio into a structured segment list.

        STRICT REQUIREMENTS:
        1.  **Time Alignment:** Align start/end times strictly to the audio waveform.
        2.  **Diarization:** Identify speakers (e.g., "Speaker 1", "Speaker 2").
        3.  **Granularity:** One sentence per segment. Do not break mid-sentence unless there is a significant pause.
        4.  **Silence:** 
            - Tag silences > 1.0s as "type": "silence".
        5.  **Clean-up:** Remove filler words (uh, um, mm).
        6.  **Language:** Transcribe {input_lang} natively. Keep only English words spoken in English(Latin Script).

        OUTPUT FORMAT (JSON):
        [
          {{
            "start": "HH:MM:SS,mmm",
            "end": "HH:MM:SS,mmm",
            "speaker": "Speaker 1",
            "text": "Actual spoken text."
          }},
          {{
            "start": "HH:MM:SS,mmm",
            "end": "HH:MM:SS,mmm",
            "type": "silence" 
          }}
        ]
        """
        
        generation_config = {"temperature": 0.0}
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                def _generate():
                    return get_client().models.generate_content(
                        model=model,
                        contents=[chunk_file, prompt],
                        config=generation_config
                    )
                # retry_manager handles API errors (500s, etc)
                response = retry_manager.call(_generate)
                offset_ms = i * chunk_size_ms
                segments = parse_gemini_json(response.text, offset_ms=offset_ms)
                
                # Validate timestamps
                if segments:
                    last_end = segments[-1]['end']
                    chunk_end_limit = offset_ms + chunk_size_ms + 5000 # 5s tolerance
                    if last_end > chunk_end_limit:
                        print(f"WARNING: Chunk {i+1} hallucinated timestamps (Attempt {attempt+1}/{max_retries})! End: {last_end} > Limit: {chunk_end_limit}. Retrying...")
                        continue # Retry loop
                else:
                    print(f"WARNING: Chunk {i+1} returned empty segments (Attempt {attempt+1}/{max_retries}). Retrying...")
                    continue # Retry loop
                
                cache.set(cache_key, segments)
                return segments
            except Exception as e:
                print(f"Error processing chunk {i+1} (Attempt {attempt+1}/{max_retries}): {e}")
                # If it's the last attempt, we let it fall through to raise
        
        # If we get here, all retries failed
        raise ValueError(f"Failed to transcribe chunk {i+1} after {max_retries} attempts due to hallucinations or errors.")

    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_index = {executor.submit(process_chunk, i, path): i for i, path in enumerate(chunk_paths)}
            results = [None] * len(chunk_paths)
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    print(f"Chunk {index+1} FAILED: {e}")
                    raise e # Abort the whole process
            
            for res in results:
                if res is None:
                    raise ValueError("One or more chunks failed to process.")
                all_segments.extend(res)
    finally:
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
            
    return segments_to_srt(all_segments)

def translate_adaptive(text, target_duration_sec, input_lang="English", output_lang="Hindi", feedback=None, current_text=None, word_budget=None, time_diff=None):
    """Adaptive translation with user's specific feedback prompts."""
    target_chars = int(target_duration_sec * 15)
    
    extra_instruction = ""
    if feedback == "elaborate":
        extra_instruction = f"""IMPORTANT: The previous translation was TOO SHORT (Gap: {time_diff:.2f}s). 
        You MUST ELABORATE MORE. Add descriptive words, polite fillers, or repeat key concepts naturally to fill the time. 
        Make it significantly longer."""
    elif feedback == "summarize":
        extra_instruction = f"""IMPORTANT: The previous translation was TOO LONG (Excess: {time_diff:.2f}s). 
        You MUST SUMMARIZE. Use shorter words and concise phrasing."""

    SCRIPT_MAPPING = {
        "Hindi": "Devanagari (e.g., नमस्ते)",
        "Tamil": "Tamil Script (e.g., வணக்கம்)",
        "Telugu": "Telugu Script (e.g., నమస్కారం)",
        "Kannada": "Kannada Script (e.g., ನಮಸ್ಕಾರ)",
        "Malayalam": "Malayalam Script (e.g., നമസ്കാരം)",
        "Marathi": "Devanagari (e.g., नमस्कार)",
        "Gujarati": "Gujarati Script (e.g., નમસ્તે)",
        "Punjabi": "Gurmukhi (e.g., ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ)",
        "Bengali": "Bengali Script (e.g., নমস্কার)",
        "Assamese": "Assamese Script (e.g., নমস্কাৰ)",
        "Odia": "Odia Script (e.g., ନମସ୍କାର)"
    }
    target_script = SCRIPT_MAPPING.get(output_lang, "Native Script")

    prompt = f"""
    Translate the following subtitle from {input_lang} to {output_lang}.

    Input Text: "{text}"
    Target Duration: {target_duration_sec:.2f} seconds
    Target Length: Approximately {target_chars} characters

    {extra_instruction}

    CRITICAL INSTRUCTION:
    You MUST write the output in {target_script}.Keep Technical terms in English(Latin Script) 
    Example: If translating to Hindi, write "नमस्ते", NOT "Namaste".

    CONSTRAINTS:
    1. SCRIPT: Use {target_script}.
    2. ADAPTATION:
       - If text is short but duration is long, ELABORATE naturally.
       - If text is long but duration is short, SUMMARIZE.
    3. TRANSLATION: Translation must be in casual, spoken {output_lang}. It should NOT be in formal and textbook language.  
    4. TONE: Casual and calm.
    5. OUTPUT: Return ONLY the translated text in {target_script}.
    6. Technical Terms: Do not translate technical terms. keep them in English.`
    """
    
    cache_key = f"mt_{text}_{target_duration_sec}_{feedback}_{output_lang}"
    cached = cache.get(cache_key)
    if cached: return cached
    
    try:
        def _generate():
            return get_client().models.generate_content(model="gemini-2.5-pro", contents=prompt)
        resp = retry_manager.call(_generate)
        result = resp.text.strip()
        cache.set(cache_key, result)
        return result
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def generate_google_tts(text, api_key, voice_name="en-IN-Chirp3-HD-Achird", speaking_rate=0.9):
    """Generates audio using Google Cloud TTS."""
    if not text or not text.strip(): return None
    
    cache_key = f"tts_{text}_{voice_name}_{speaking_rate}"
    cached = cache.get(cache_key)
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
                fallback_voice = "en-IN-Neural2-B" # Generic fallback
                payload["voice"]["name"] = fallback_voice
                payload["voice"]["languageCode"] = "en-IN"
                response_retry = requests.post(url, json=payload)
                if response_retry.status_code == 200:
                    return base64.b64decode(response_retry.json().get('audioContent'))
            
            print(f"TTS Error: {response.text}")
            return None
        
    audio_bytes = retry_manager.call(_call_tts)
    if audio_bytes: cache.set(cache_key, audio_bytes)
    return audio_bytes

def parse_srt_time(time_str):
    """Parses SRT timestamp to milliseconds."""
    time_str = time_str.strip().replace(',', '.')
    if time_str.count(':') == 3:
        parts = time_str.rsplit(':', 1)
        time_str = '.'.join(parts)
    try:
        if '.' in time_str:
            main, frac = time_str.split('.')
            frac = frac.ljust(6, '0')[:6]
            time_str = f"{main}.{frac}"
        t = datetime.strptime(time_str, "%H:%M:%S.%f")
        return (t.hour * 3600 + t.minute * 60 + t.second) * 1000 + t.microsecond / 1000
    except:
        return 0

def parse_srt(srt_content):
    """Parses SRT content string into a list of segments."""
    segments = []
    lines = srt_content.splitlines()
    n = len(lines)
    current_seg = {}
    i = 0
    while i < n:
        line = lines[i].strip()
        if line.isdigit() and i+1 < n and '-->' in lines[i+1]:
            if 'text' in current_seg: segments.append(current_seg)
            current_seg = {}
            time_parts = lines[i+1].strip().split(' --> ')
            if len(time_parts) == 2:
                current_seg['start'] = parse_srt_time(time_parts[0])
                current_seg['end'] = parse_srt_time(time_parts[1])
            i += 2
            continue
        if 'start' in current_seg and line:
            current_seg['text'] = (current_seg.get('text', '') + ' ' + line).strip()
        i += 1
    if 'text' in current_seg: segments.append(current_seg)
    
    for seg in segments:
        text = seg.get('text', '')
        match = re.match(r'^\[(.*?)\]\s*(.*)', text)
        if match:
            seg['speaker'] = match.group(1)
            seg['text'] = match.group(2)
        else:
            seg['speaker'] = 'Speaker 1'
    return segments

def process_segment_task(seg, i, total, input_lang, output_lang, target_voice=None, speed=1.0, forced_target_text=None):
    """Processes a single segment. If forced_target_text is provided, skips MT and text adaptation."""
    source_text = seg['text']
    start_time = seg['start']
    end_time = seg['end']
    speaker = seg.get('speaker', 'Speaker 1')
    target_duration_ms = end_time - start_time
    target_duration_sec = target_duration_ms / 1000.0
    
    print(f"[{i+1}/{total}] [{speaker}] {target_duration_sec:.2f}s | {source_text[:30]}...")
    
    if "[SILENCE]" in source_text.upper() or "[NOISE" in source_text.upper():
        return i, AudioSegment.silent(duration=target_duration_ms), source_text, start_time, end_time

    voice_name = voice_manager.get_voice(speaker, voice_preference=target_voice, output_lang=output_lang)
    
    # Determine Target Text
    if forced_target_text:
        current_translated_text = forced_target_text
        # If forced, we skip the adaptive MT step
    else:
        current_translated_text = translate_adaptive(source_text, target_duration_sec, input_lang, output_lang)
    
    best_audio = None
    best_text = current_translated_text
    
    # 1. Try normal generation
    audio_bytes = generate_google_tts(current_translated_text, get_api_key(), voice_name=voice_name, speaking_rate=speed)
    if audio_bytes:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
        
        # Calculate Diff
        diff_ms = len(audio) - target_duration_ms
        print(f"  -> Target: {target_duration_sec:.2f}s | Generated: {len(audio)/1000:.2f}s | Diff: {diff_ms/1000:+.2f}s")

        # 2. Feedback Loop (ONLY if NOT forced text)
        if not forced_target_text:
            # Check if Too Long (Strict trigger for summarization)
            if diff_ms > 500: # If > 0.5s longer
                gap = diff_ms / 1000.0
                print(f"  -> Too long by {gap:.2f}s. Attempting to SUMMARIZE...")
                current_translated_text = translate_adaptive(source_text, target_duration_sec, input_lang, output_lang, feedback="summarize", time_diff=gap)
                audio_bytes_v2 = generate_google_tts(current_translated_text, get_api_key(), voice_name=voice_name, speaking_rate=speed)
                if audio_bytes_v2:
                    audio_v2 = AudioSegment.from_file(io.BytesIO(audio_bytes_v2), format="wav")
                    # If shorter than original, take it.
                    if len(audio_v2) < len(audio):
                        audio = audio_v2
                        best_text = current_translated_text
                        diff_ms = len(audio) - target_duration_ms # Recalculate diff
                        print(f"  -> New Duration: {len(audio)/1000:.2f}s | Diff: {diff_ms/1000:+.2f}s")

            # Check if Too Short (Elaborate)
            elif diff_ms < -500: # If > 0.5s shorter
                gap = abs(diff_ms) / 1000.0
                print(f"  -> Too short by {gap:.2f}s. Attempting to ELABORATE...")
                current_translated_text = translate_adaptive(source_text, target_duration_sec, input_lang, output_lang, feedback="elaborate", time_diff=gap)
                audio_bytes_v2 = generate_google_tts(current_translated_text, get_api_key(), voice_name=voice_name, speaking_rate=speed)
                if audio_bytes_v2:
                    audio_v2 = AudioSegment.from_file(io.BytesIO(audio_bytes_v2), format="wav")
                    if len(audio_v2) > len(audio): # Accepted if longer
                        audio = audio_v2
                        best_text = current_translated_text
                        diff_ms = len(audio) - target_duration_ms # Recalculate diff
                        print(f"  -> New Duration: {len(audio)/1000:.2f}s | Diff: {diff_ms/1000:+.2f}s")

        # 3. Final Speed Adjustment (atempo) if still too long
        # We apply this even for forced text to ensure sync
        if len(audio) > target_duration_ms:
            speed_factor = len(audio) / target_duration_ms
            if speed_factor > 1.02: # Only if > 2% diff
                eff_speed = min(speed_factor, 1.15) # Cap at 1.15x
                print(f"  -> Adjusting speed by {eff_speed:.2f}x (Required: {speed_factor:.2f}x)")
                
                if speed_factor > 1.15:
                    print(f"  -> WARNING: Audio too long! Desync of {(len(audio)/1.15 - target_duration_ms)/1000:.2f}s remaining.")

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

def process_video(video_path, output_dir, duration_limit=None, input_lang="Malayalam", output_lang="English", progress_callback=None, target_voice=None, speed=1.0):
    filename = os.path.basename(video_path)
    
    if progress_callback: progress_callback("Preprocessing", 0, "Checking file...")

    # Check if input is audio-only
    is_audio_only = filename.lower().endswith(('.mp3', '.wav', '.aac', '.flac', '.m4a'))
    
    if duration_limit and "_trimmed_" not in filename:
        if progress_callback: progress_callback("Preprocessing", 10, "Trimming media...")
        print(f"Trimming media to {duration_limit} seconds...")
        trimmed_path = os.path.join(os.path.dirname(video_path), f"{os.path.splitext(filename)[0]}_trimmed_{duration_limit}s{os.path.splitext(filename)[1]}")
        # ffmpeg command differs slightly if we want to keep it generic, but -t works for both
        cmd = ["ffmpeg", "-y", "-i", video_path, "-t", str(duration_limit), "-c", "copy", trimmed_path]
        subprocess.run(cmd, check=True)
        video_path = trimmed_path
        filename = os.path.basename(video_path)

    name_no_ext = os.path.splitext(filename)[0]
    srt_dir = "SRT"
    os.makedirs(srt_dir, exist_ok=True)
    
    print(f"--- Step 1: Generating SRT for {filename} ---")
    if progress_callback: progress_callback("Transcribing", 20, "Generating SRT with Gemini...")
    
    # Pass callback to generate_srt_gemini if possible, or just update here
    srt_content = generate_srt_gemini(video_path, input_lang=input_lang, duration_limit=duration_limit)
    srt_content = fix_srt_timestamps(srt_content)
    srt_content = merge_segments_to_sentences(srt_content)
    
    with open(os.path.join(srt_dir, f"{name_no_ext}_ASR.srt"), "w", encoding="utf-8") as f:
        f.write(srt_content)
    
    print(f"--- Step 2: Translating & Generating Audio (Parallel) ---")
    if progress_callback: progress_callback("Translating & Dubbing", 40, "Processing segments...")
    
    segments = parse_srt(srt_content)
    processed_results = [None] * len(segments)
    
    total_segments = len(segments)
    completed_segments = 0

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i, seg in enumerate(segments):
            futures.append(executor.submit(process_segment_task, seg, i, len(segments), input_lang, output_lang, target_voice, speed))
        
        for future in as_completed(futures):
            try:
                i, audio_seg, text, start, end = future.result()
                processed_results[i] = (audio_seg, text, start, end)
                metrics.metrics["segments_processed"] += 1
                
                completed_segments += 1
                if progress_callback:
                    prog = 40 + int((completed_segments / total_segments) * 40) # 40% to 80%
                    progress_callback("Translating & Dubbing", prog, f"Segment {completed_segments}/{total_segments}")
                    
            except Exception as e:
                print(f"Error in segment task: {e}")

    print("--- Assembling Audio ---")
    if progress_callback: progress_callback("Finalizing", 85, "Stitching audio...")
    
    combined_audio = AudioSegment.empty()
    current_time = 0
    translated_segments = []
    
    for res in processed_results:
        if not res: continue
        audio_seg, text, start_time, end_time = res
        
        # Calculate gap based on ACTUAL audio length vs Next Segment Start
        actual_current_time = len(combined_audio)
        silence_gap = start_time - actual_current_time
        
        if silence_gap > 0:
            combined_audio += AudioSegment.silent(duration=silence_gap)
            
        if audio_seg:
            combined_audio += audio_seg
        else:
            print(f"Warning: Segment {i} failed. Filling with silence.")
            target_dur = end_time - start_time
            combined_audio += AudioSegment.silent(duration=target_dur)

        translated_segments.append({"start": start_time, "end": end_time, "text": text})
            
    mt_srt_path = os.path.join(srt_dir, f"{name_no_ext}_{output_lang}.srt")
    with open(mt_srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(translated_segments):
            s = datetime.utcfromtimestamp(seg['start']/1000).strftime('%H:%M:%S,%f')[:-3]
            e = datetime.utcfromtimestamp(seg['end']/1000).strftime('%H:%M:%S,%f')[:-3]
            f.write(f"{i+1}\n{s} --> {e}\n{seg['text']}\n\n")

    audio_path = os.path.join(output_dir, f"{name_no_ext}_dubbed.wav")
    combined_audio.export(audio_path, format="wav")
    
    final_output_path = ""
    
    if is_audio_only:
        print(f"--- Audio Only: Saved to {audio_path} ---")
        final_output_path = audio_path
        if progress_callback: progress_callback("Completed", 100, "Audio ready.")
    else:
        output_video_path = os.path.join(output_dir, filename)
        print(f"--- Step 3: Merging into {output_video_path} ---")
        if progress_callback: progress_callback("Muxing", 90, "Merging audio and video...")
        
        # Use -shortest to avoid extending video if audio is slightly longer, or map length
        cmd = ["ffmpeg", "-y", "-i", video_path, "-i", audio_path, "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest", "-movflags", "+faststart", output_video_path]
        subprocess.run(cmd, check=True)
        print("Done! Video saved.")
        final_output_path = output_video_path
        if progress_callback: progress_callback("Completed", 100, "Video ready.")

    # Return structured data
    return {
        "output_path": final_output_path,
        "source_srt": srt_content,
        "target_srt": open(mt_srt_path, "r", encoding="utf-8").read(),
        "source_segments": segments, # List of dicts
        "target_segments": translated_segments # List of dicts
    }

def regenerate_video(video_path, output_dir, new_segments, old_segments, input_lang="English", output_lang="Hindi", target_voice=None, speed=1.0, progress_callback=None):
    """
    Regenerates the dubbing based on updated segments.
    - If Source Text changed -> Re-run MT + TTS
    - If Target Text changed (but Source same) -> Re-run TTS (Skip MT)
    - If Neither changed -> Use Cache (TTS)
    """
    filename = os.path.basename(video_path)
    name_no_ext = os.path.splitext(filename)[0]
    srt_dir = "SRT"
    
    print(f"--- Regenerating Dub for {filename} ---")
    if progress_callback: progress_callback("Regenerating", 0, "Analyzing changes...")

    processed_results = [None] * len(new_segments)
    total_segments = len(new_segments)
    completed_segments = 0

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i, new_seg in enumerate(new_segments):
            # Find corresponding old segment (assuming index alignment for now)
            # TODO: Better alignment if segments were added/removed? 
            # For now, we assume strict 1:1 mapping based on index.
            old_seg = old_segments[i] if i < len(old_segments) else None
            
            source_changed = old_seg and (new_seg['text'] != old_seg['text'])
            target_changed = old_seg and (new_seg.get('target_text') != old_seg.get('target_text'))
            
            forced_target = None
            
            if source_changed:
                print(f"Segment {i}: Source changed. Re-translating...")
                # forced_target remains None, so MT runs
            elif target_changed:
                print(f"Segment {i}: Target changed manually. Skipping MT.")
                forced_target = new_seg.get('target_text')
            else:
                # No change, but we pass forced_target to skip MT and hit TTS cache
                forced_target = new_seg.get('target_text')
                
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
                input_lang, output_lang, target_voice, speed, 
                forced_target_text=forced_target
            ))
        
        for future in as_completed(futures):
            try:
                i, audio_seg, text, start, end = future.result()
                processed_results[i] = (audio_seg, text, start, end)
                
                completed_segments += 1
                if progress_callback:
                    prog = int((completed_segments / total_segments) * 80)
                    progress_callback("Regenerating", prog, f"Segment {completed_segments}/{total_segments}")
                    
            except Exception as e:
                print(f"Error in regeneration task {i}: {e}")

    print("--- Assembling Audio ---")
    if progress_callback: progress_callback("Finalizing", 90, "Stitching audio...")
    
    combined_audio = AudioSegment.empty()
    translated_segments = []
    
    for i, res in enumerate(processed_results):
        if not res: continue
        audio_seg, text, start_time, end_time = res
        
        actual_current_time = len(combined_audio)
        silence_gap = start_time - actual_current_time
        
        if silence_gap > 0:
            combined_audio += AudioSegment.silent(duration=silence_gap)
            
        if audio_seg:
            combined_audio += audio_seg
        else:
            target_dur = end_time - start_time
            combined_audio += AudioSegment.silent(duration=target_dur)

        translated_segments.append({"start": start_time, "end": end_time, "text": text})

    # Save new Target SRT
    mt_srt_path = os.path.join(srt_dir, f"{name_no_ext}_{output_lang}.srt")
    with open(mt_srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(translated_segments):
            s = datetime.utcfromtimestamp(seg['start']/1000).strftime('%H:%M:%S,%f')[:-3]
            e = datetime.utcfromtimestamp(seg['end']/1000).strftime('%H:%M:%S,%f')[:-3]
            f.write(f"{i+1}\n{s} --> {e}\n{seg['text']}\n\n")

    audio_path = os.path.join(output_dir, f"{name_no_ext}_dubbed.wav")
    combined_audio.export(audio_path, format="wav")
    
    output_video_path = os.path.join(output_dir, filename)
    
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

    if progress_callback: progress_callback("Completed", 100, "Regeneration done.")

    return {
        "output_path": final_output_path,
        "source_srt": open(os.path.join(srt_dir, f"{name_no_ext}_ASR.srt"), "r", encoding="utf-8").read(), # Original Source SRT (or should we update it if source changed? Let's keep original file but return updated segments)
        "target_srt": open(mt_srt_path, "r", encoding="utf-8").read(),
        "source_segments": new_segments, # Return the NEW source segments as the current state
        "target_segments": translated_segments
    }

# === RUN ===
if __name__ == "__main__":
    input_dir = "Videos/test"
    output_dir = "Video out"
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    video_files = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.mov', '.avi'))]
    
    if not video_files:
        print(f"No video files found in {input_dir}")
    else:
        for filename in video_files:
            full_path = os.path.join(input_dir, filename)
            process_video(full_path, output_dir, duration_limit=120, input_lang="English", output_lang="Hindi", target_voice="Male (Chirp 3 - Alnilam)", speed=0.85)
    
    metrics.save_report()
    print("Metrics saved to metrics.json")

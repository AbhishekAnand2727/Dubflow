"""
Regenerate endpoint implementation for handling edited transcript segments.
PROPERLY IMPLEMENTED - Uses actual pipeline functions
"""

import os
import subprocess
from pathlib import Path
import io
import json
from pydub import AudioSegment
import math
import traceback
import uuid

import database
from srt_utils import parse_srt, parse_srt_time, format_srt_time, write_srt
from google_pipeline2 import (
    CacheManager,
    VoiceManager,
    process_segment_task
)
from config import AUDIO_SAMPLE_RATE, AUDIO_CHANNELS, AUDIO_BITRATE

def get_media_duration_ms(file_path):
    """Get duration of media file in milliseconds using ffprobe."""
    try:
        cmd = [
            "ffprobe", 
            "-v", "error", 
            "-show_entries", "format=duration", 
            "-of", "default=noprint_wrappers=1:nokey=1", 
            file_path
        ]
        output = subprocess.check_output(cmd).strip()
        if not output:
            return 0
        return int(float(output) * 1000)
    except Exception as e:
        print(f"Error getting duration for {file_path}: {e}")
        return 0

def extract_audio_segment(video_path, start_ms, end_ms):
    """
    Extract a specific audio segment from a video file.
    Returns AudioSegment or None if extraction fails.
    """
    try:
        # Use ffmpeg to extract the specific audio segment
        temp_audio = f"temp_extract_{uuid.uuid4().hex}.wav"
        start_sec = start_ms / 1000.0
        duration_sec = (end_ms - start_ms) / 1000.0
        
        subprocess.run([
            "ffmpeg", "-y",
            "-i", video_path,
            "-ss", str(start_sec),
            "-t", str(duration_sec),
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # Raw audio
            "-ar", "44100",  # Sample rate
            "-ac", "2",  # Stereo
            temp_audio
        ], check=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        
        audio_segment = AudioSegment.from_file(temp_audio, format="wav")
        os.remove(temp_audio)
        return audio_segment
    except Exception as e:
        print(f"Failed to extract audio segment: {e}")
        return None

def regenerate_dubbing_task(task_id: str, edited_segments: list, speaker_overrides: dict = {}, jobs_dir: str = None):
    """
    Background task to regenerate dubbed video with edited segments.
    
    Args:
        task_id: Job ID
        edited_segments: List of segment dicts with start, end, text, target_text, deleted
        speaker_overrides: Dict mapping segment index to voice ID
        jobs_dir: Path to jobs directory
    """
    try:
        # Validate inputs
        if not task_id:
            raise ValueError("task_id is required")
        
        if not edited_segments:
            raise ValueError("edited_segments cannot be empty")
        
        if not jobs_dir:
            raise ValueError("jobs_dir is required")
        
        # Validate task_id format (prevent path traversal)
        if not task_id.replace('-', '').replace('_', '').isalnum():
            raise ValueError(f"Invalid task_id format: {task_id}")
        
        # Update job status
        database.update_job_status(task_id, status="processing", message="Starting regeneration...")
        
        # Log received speaker overrides (will save to DB after comparison)
        print(f"[Regenerate] Received speaker_overrides: {speaker_overrides}")
        database.log_event(task_id, f"Speaker overrides received: {json.dumps(speaker_overrides)}")
        
        job = database.get_job(task_id)
        if not job:
            raise Exception("Job not found")
        
        job_dir = os.path.join(jobs_dir, task_id)
        output_dir = os.path.join(job_dir, "output")
        srt_dir = os.path.join(job_dir, "srt")
        cache_dir = os.path.join(job_dir, ".cache")
        temp_dir = os.path.join(job_dir, "temp")
        
        # Find existing dubbed video (ROBUST SELECTION)
        existing_dubbed = None
        
        candidates = []
        for file in os.listdir(output_dir):
            if file.lower().endswith(('.mp4', '.mkv', '.mov', '.avi')):
                path = os.path.join(output_dir, file)
                # Filter out temp/regenerated files if possible to find "source" dub
                if "_regenerated" in file:
                    continue
                # Also ignore original source video if it happens to be in output?
                # Usually source is in root job dir or not in output at all.
                candidates.append(path)
        
        if not candidates:
            # Fallback retry including regenerated if nothing else
            candidates = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.lower().endswith(('.mp4', '.mkv'))]

        if not candidates:
             raise Exception("No video file found in output directory")
        
        # Simply pick the most recently modified file (Reliable for Job Isolation)
        existing_dubbed = max(candidates, key=os.path.getmtime)
        
        print(f"[Regenerate] Using base video: {existing_dubbed}")
        
        # Get original SRT paths
        original_srt_path = None
        target_srt_path = None
        
        candidates = os.listdir(srt_dir)
        
        # 1. Find Original Source SRT (Usually ends with _ASR.srt)
        for file in candidates:
            if file.endswith("_ASR.srt"):
                original_srt_path = os.path.join(srt_dir, file)
                break
        
        # Fallback: check input language match
        if not original_srt_path:
            for file in candidates:
                if file.endswith(f"_{job['input_lang']}.srt"):
                    original_srt_path = os.path.join(srt_dir, file)
                    break

        # 2. Find Target SRT (Ends with _{output_lang}.srt)
        # Prefer file that matches video base name if possible
        video_basename = os.path.splitext(os.path.basename(existing_dubbed))[0]
        # Clean suffixes to find "root" name
        clean_base = video_basename.replace("_dubbed", "").replace("_regenerated", "")
        
        best_target = None
        for file in candidates:
            if file.endswith(f"_{job['output_lang']}.srt"):
                path = os.path.join(srt_dir, file)
                if clean_base in file:
                    best_target = path
                    break # Exact match on base
                if not best_target:
                    best_target = path # First match fallback
        
        target_srt_path = best_target
        
        if not original_srt_path or not target_srt_path:
            # List available files for debugging error
            raise Exception(f"Original SRT files not found in {srt_dir}. Candidates: {candidates}")
        
        # CRITICAL: Load original segments BEFORE overwriting
        database.update_job_status(task_id, status="processing", message="Loading original segments...")
        
        original_source_segments = parse_srt(original_srt_path)
        original_target_segments = parse_srt(target_srt_path)
        
        # Validate edited segments
        database.update_job_status(task_id, status="processing", message="Validating segments...")
        from segment_validation import validate_segments_list
        
        is_valid, error_msg = validate_segments_list(edited_segments)
        if not is_valid:
            raise ValueError(f"Segment validation failed: {error_msg}")
        

        
        print(f"[Regenerate] Validated {len(edited_segments)} segments successfully")
        
        # DEBUG LOGGING
        if edited_segments:
            database.log_event(task_id, f"DEBUG: Receive Seg 0 Target: '{edited_segments[0].get('target_text', 'N/A')}'")
        
        # Debug: Show what we loaded from original files
        print(f"[Regenerate] Loaded {len(original_source_segments)} source segments and {len(original_target_segments)} target segments from SRT files")
        if original_source_segments:
            print(f"[Regenerate] Original Source Seg 0: '{original_source_segments[0].get('text', 'N/A')}'")
        if original_target_segments:
            print(f"[Regenerate] Original Target Seg 0: '{original_target_segments[0].get('text', 'N/A')}'")
        
        # Identify changed segments by comparing with original
        database.update_job_status(task_id, status="processing", message="Comparing segments...")
        
        # Get previous voice overrides from database for comparison
        old_speaker_overrides = {}
        if job.get('speaker_overrides'):
            try:
                old_speaker_overrides = json.loads(job['speaker_overrides'])
            except:
                old_speaker_overrides = {}
        
        print(f"[Regenerate] Old speaker overrides from DB: {old_speaker_overrides}")
        print(f"[Regenerate] New speaker overrides from request: {speaker_overrides}")
        
        changed_indices = []
        for i, edited_seg in enumerate(edited_segments):
            if edited_seg.get('deleted'):
                changed_indices.append(i)
                continue
            
            # Check if this segment exists in original and if it changed
            if i >= len(original_target_segments) or i >= len(original_source_segments):
                # New segment or mismatch
                changed_indices.append(i)
            else:
                orig_target = original_target_segments[i]
                orig_source = original_source_segments[i]
                
                try:
                    edited_start_ms = parse_srt_time(edited_seg['start'])
                    edited_end_ms = parse_srt_time(edited_seg['end'])
                except Exception as e:
                    print(f"CRASH PARSING TIMESTAMP (Edited): '{edited_seg.get('start')}' or '{edited_seg.get('end')}'. Error: {e}")
                    raise

                try:
                    orig_start_ms = parse_srt_time(orig_target.get('start', '00:00:00,000'))
                    orig_end_ms = parse_srt_time(orig_target.get('end', '00:00:00,000'))
                except Exception as e:
                    print(f"CRASH PARSING TIMESTAMP (Original): '{orig_target.get('start')}'. Error: {e}")
                    raise
                
                # Normalize text for comparison (strip speaker tags)
                import re
                def normalize_text(text):
                    """Remove speaker tags and extra whitespace for comparison"""
                    if not text:
                        return ""
                    # Remove [Speaker X] tags
                    text = re.sub(r'\[Speaker \d+\]\s*', '', text)
                    # Remove "Speaker X:" format
                    text = re.sub(r'Speaker \d+:\s*', '', text)
                    return text.strip()
                
                edited_source_normalized = normalize_text(edited_seg['text'])
                orig_source_normalized = normalize_text(orig_source.get('text', ''))
                edited_target_normalized = normalize_text(edited_seg['target_text'])
                orig_target_normalized = normalize_text(orig_target.get('text', ''))
                
                source_changed = edited_source_normalized != orig_source_normalized
                target_changed = edited_target_normalized != orig_target_normalized
                timing_changed = (edited_start_ms != orig_start_ms or edited_end_ms != orig_end_ms)
                
                # Check if voice assignment changed for this speaker
                speaker = edited_seg.get('speaker', 'Speaker 1')
                old_voice = old_speaker_overrides.get(speaker, job.get('target_voice'))
                new_voice = speaker_overrides.get(speaker, job.get('target_voice'))
                voice_changed = (old_voice != new_voice)
                
                # Debug logging for first 3 segments
                if i < 3:
                    print(f"[Regenerate] Seg {i} comparison:")
                    print(f"  Source: '{edited_source_normalized}' vs '{orig_source_normalized}' = {source_changed}")
                    print(f"  Target: '{edited_target_normalized}' vs '{orig_target_normalized}' = {target_changed}")
                    print(f"  Timing: {edited_start_ms},{edited_end_ms} vs {orig_start_ms},{orig_end_ms} = {timing_changed}")
                    print(f"  Voice: {new_voice} vs {old_voice} = {voice_changed}")
                
                # Debug voice comparison for all segments with voice overrides
                if speaker in speaker_overrides or speaker in old_speaker_overrides:
                    print(f"[Regenerate] Seg {i} ({speaker}): old_voice={old_voice}, new_voice={new_voice}, changed={voice_changed}")
                
                if source_changed or target_changed or timing_changed or voice_changed:
                    changed_indices.append(i)
                    print(f"[Regenerate] Seg {i}: CHANGED - source={source_changed}, target={target_changed}, timing={timing_changed}, voice={voice_changed}")
        
        database.update_job_status(task_id, status="processing", 
                                  message=f"Found {len(changed_indices)} changed segments out of {len(edited_segments)}")
        
        print(f"[Regenerate] Changed segments: {changed_indices}")
        
        # NOW save speaker overrides to database (after comparison)
        database.update_speaker_overrides(task_id, speaker_overrides)
        print(f"[Regenerate] Saved new speaker overrides to database")
        
        # Now save edited segments to SRT files
        # Save edited target SRT
        target_segments_for_srt = [
            {
                'start': seg['start'],
                'end': seg['end'],
                'text': seg['target_text'],
                'deleted': seg.get('deleted', False)
            }
            for seg in edited_segments
        ]
        write_srt(target_segments_for_srt, target_srt_path)
        
        # Save edited source SRT
        source_segments_for_srt = [
            {
                'start': seg['start'],
                'end': seg['end'],
                'text': seg['text'],
                'deleted': seg.get('deleted', False)
            }
            for seg in edited_segments
        ]
        write_srt(source_segments_for_srt, original_srt_path)
        
        # Generate audio for segments (changed=new TTS, unchanged=use cache)
        # We perform full assembly to ensure sync (overlay approach)
        cache = CacheManager(cache_dir)
        voice_manager = VoiceManager()
        
        database.update_job_status(task_id, status="processing", 
                                  message=f"Processing {len(edited_segments)} segments...")
        
        # Get video duration for the canvas
        video_duration_ms = get_media_duration_ms(existing_dubbed)
        if video_duration_ms == 0:
             # Fallback: estimate from last segment end
             last_end = parse_srt_time(edited_segments[-1]['end'])
             video_duration_ms = last_end + 1000
             print(f"[Regenerate] Estimated duration: {video_duration_ms}ms")

        # Create silent canvas
        combined_audio = AudioSegment.silent(duration=video_duration_ms)
        
        # List to collect final segments for DB update
        final_db_segments = []
        
        for i, seg in enumerate(edited_segments):
            # Track deleted state and persist to DB
            is_deleted = seg.get('deleted', False)
            if is_deleted:
                # Add deleted segment marker to DB
                final_db_segments.append({
                    'start': parse_srt_time(seg['start']),
                    'end': parse_srt_time(seg['end']),
                    'text': seg['text'],
                    'target_text': seg['target_text'],
                    'deleted': True,
                    'speaker': seg.get('speaker', 'Speaker 1'),
                     'status': 'edited'
                })
                continue
            
            # Convert SRT timestamps to milliseconds
            start_ms = parse_srt_time(seg['start'])
            end_ms = parse_srt_time(seg['end'])
            
            # Extract speaker from segment to lookup voice override
            speaker = seg.get('speaker', 'Speaker 1')
            
            # Look up voice override by speaker name (e.g., "Speaker 2")
            # Falls back to job's default target_voice if no override found
            voice_id = speaker_overrides.get(speaker, job.get('target_voice'))
            
            print(f"[Regenerate] Seg {i}: speaker={speaker}, voice_id={voice_id}, overrides={speaker_overrides}")
            database.log_event(task_id, f"Seg {i}: Using speaker={speaker}, voice={voice_id}")
            
            # Create segment in format expected by process_segment_task
            segment_for_processing = {
                'text': seg['text'], # Pass SOURCE text for internal logic
                'start': start_ms,
                'end': end_ms,
                'speaker': seg.get('speaker', 'Speaker 1')
            }
            
            audio_segment = None
            
            if i in changed_indices:
                # Segment has changed - regenerate TTS
                print(f"[Regenerate] Seg {i}: Generating TTS for changed segment")
                database.update_job_status(task_id, status="processing", 
                                          message=f"Generating audio for segment {i+1}/{len(edited_segments)}...")
                
                try:
                    # Use process_segment_task which handles TTS generation with cache
                    # Pass forced_target_text since we already have the translation
                    _, audio_segment, _, _, _ = process_segment_task(
                        seg=segment_for_processing,
                        i=i,
                        total=len(edited_segments),
                        input_lang=job['input_lang'],
                        output_lang=job['output_lang'],
                        target_voice=voice_id,
                        speed=job.get('speed', 1.0),
                        forced_target_text=seg['target_text'],  # Use edited text directly
                        voice_manager=voice_manager,
                        cache=cache,
                        temp_dir=temp_dir,
                        job_id=task_id,
                        target_wpm=job.get('wpm')
                    )
                except Exception as e:
                    print(f"[Regenerate] Error generating TTS for segment {i}: {e}")
                    raise
            else:
                # Segment unchanged - extract from existing video (much faster!)
                print(f"[Regenerate] Seg {i}: Extracting audio from existing video (unchanged)")
                database.update_job_status(task_id, status="processing", 
                                          message=f"Extracting audio for segment {i+1}/{len(edited_segments)}...")
                
                try:
                    audio_segment = extract_audio_segment(existing_dubbed, start_ms, end_ms)
                    if not audio_segment:
                        # Fallback to regeneration if extraction fails
                        print(f"[Regenerate] Seg {i}: Extraction failed, falling back to TTS regeneration")
                        _, audio_segment, _, _, _ = process_segment_task(
                            seg=segment_for_processing,
                            i=i,
                            total=len(edited_segments),
                            input_lang=job['input_lang'],
                            output_lang=job['output_lang'],
                            target_voice=voice_id,
                            speed=job.get('speed', 1.0),
                            forced_target_text=seg['target_text'],
                            voice_manager=voice_manager,
                            cache=cache,
                            temp_dir=temp_dir,
                            job_id=task_id,
                            target_wpm=job.get('wpm')
                        )
                except Exception as e:
                    print(f"[Regenerate] Error extracting audio for segment {i}: {e}")
                    raise
            
            # Overlay audio on canvas (outside if/else - applies to all segments)
            if audio_segment:
                combined_audio = combined_audio.overlay(audio_segment, position=start_ms)
            else:
                print(f"[Regenerate] Warning: No audio generated for segment {i}")
                database.log_event(task_id, f"Warning: No audio for segment {i}")
            
            final_db_segments.append({
                'start': start_ms,
                'end': end_ms,
                'text': seg['text'],
                'target_text': seg['target_text'],
                'deleted': False,
                'speaker': seg.get('speaker', 'Speaker 1'),
                'status': 'edited' # Mark as edited
            })
        
        # Export to file
        merged_audio_path = os.path.join(output_dir, "merged_audio_regen.wav")
        combined_audio.export(merged_audio_path, format="wav")
        
        # Replace audio in existing dubbed video
        database.update_job_status(task_id, status="processing", message="Replacing audio in video...")
        
        regenerated_video_path = os.path.join(output_dir, f"{Path(existing_dubbed).stem}_regenerated.mp4")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", existing_dubbed,
            "-i", merged_audio_path,
            "-c:v", "copy",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            regenerated_video_path
        ], check=True, capture_output=True)
        
        # Replace the original dubbed file
        # CRITICAL FIX: Save updated segments to DB so Frontend sees them!
        # MUST BE DONE BEFORE MARKING JOB AS COMPLETED TO AVOID RACE CONDITION
        database.save_segments(task_id, final_db_segments)
        
        # Replace the original dubbed file safely
        # It's already in 'regenerated_video_path', but we want to overwrite 'existing_dubbed'
        # to ensure the filename remains consistent (or updated to specific pattern)
        # Actually, let's stick to replacing existing_dubbed for now to keep filename stable
        
        try:
             # Retry loop for Windows file lock issues
             import time
             for _ in range(3):
                 try:
                     os.replace(regenerated_video_path, existing_dubbed)
                     break
                 except PermissionError:
                     time.sleep(1)
             else:
                 # If replace fails after retries (e.g. open in player), force a new name?
                 # Or just log error?
                 # Let's try shutil.move which might be more robust
                 import shutil
                 shutil.move(regenerated_video_path, existing_dubbed)
                 
        except Exception as e:
             print(f"[Regenerate] Warning: Could not overwrite original file: {e}")
             # Return the new path instead
             existing_dubbed = regenerated_video_path # Update reference for return value
        
        database.update_job_status(task_id, status="completed", message="Regeneration complete!")
        unchanged_count = len(edited_segments) - len(changed_indices)
        print(f"[Regenerate] Complete! Regenerated {len(changed_indices)} segments, reused {unchanged_count} from existing video")
        database.log_event(task_id, f"Optimization: {len(changed_indices)} regenerated, {unchanged_count} extracted from cache")
        
        # Save logs on success
        try:
            logs = database.get_events(task_id)
            log_path = os.path.join(jobs_dir, task_id, "logs.json")
            with open(log_path, "w", encoding='utf-8') as f:
                json.dump(logs, f, indent=2, default=str)
        except Exception as le:
            print(f"[Regenerate] Failed to save logs: {le}")

        # Read SRT content cleanly
        with open(original_srt_path, 'r', encoding='utf-8') as f:
            src_content = f.read()
        with open(target_srt_path, 'r', encoding='utf-8') as f:
            tgt_content = f.read()

        return {
            "output_path": existing_dubbed,
            "source_srt": src_content,
            "target_srt": tgt_content,
            "target_segments": final_db_segments
        }
        
    except ValueError as e:
        # Validation errors
        error_msg = f"Validation error: {str(e)}"
        print(f"[Regenerate] {error_msg}")
        database.update_job_status(task_id, status="failed", message=error_msg)
    except FileNotFoundError as e:
        # Missing files
        error_msg = f"File not found: {str(e)}"
        print(f"[Regenerate] {error_msg}")
        database.update_job_status(task_id, status="failed", message=error_msg)
    except subprocess.CalledProcessError as e:
        # FFmpeg errors
        error_msg = f"Audio processing failed: {e.stderr.decode() if e.stderr else str(e)}"
        print(f"[Regenerate] {error_msg}")
        database.update_job_status(task_id, status="failed", message=error_msg)
    except Exception as e:
        # Generic errors
        error_msg = f"Regeneration failed: {str(e)}"
        print(f"[Regenerate] {error_msg}")
        traceback.print_exc()
        database.update_job_status(task_id, status="failed", message=error_msg)
        
        # Log deep trace to DB
        database.log_event(task_id, f"CRASH TRACE: {traceback.format_exc()}")
    
    # Save logs on failure
    try:
        logs = database.get_events(task_id)
        log_path = os.path.join(jobs_dir, task_id, "logs.json")
        with open(log_path, "w", encoding='utf-8') as f:
            json.dump(logs, f, indent=2, default=str)
    except Exception as le:
        print(f"[Regenerate] Failed to save logs: {le}")

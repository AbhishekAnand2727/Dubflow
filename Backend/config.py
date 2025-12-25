"""
Configuration constants for the dubbing application
"""

# Audio Processing
AUDIO_SAMPLE_RATE = 44100  # Hz
AUDIO_CHANNELS = 2  # Stereo
AUDIO_BITRATE = "192k"

# Segment Constraints
MIN_SEGMENT_DURATION_MS = 100  # Minimum 0.1 seconds

# Polling
FRONTEND_POLL_INTERVAL_MS = 2000  # frontend updates every 2 seconds

# FFmpeg
FFMPEG_CONCAT_FORMAT = "concat"
FFMPEG_VIDEO_CODEC = "copy"  # Don't re-encode video

# Paths
JOBS_DIR_NAME = "jobs"
OUTPUT_DIR_NAME = "output"
SRT_DIR_NAME = "srt"
CACHE_DIR_NAME = "cache"
THUMBNAILS_DIR_NAME = "Thumbnails"
UPLOADS_DIR_NAME = "Uploads"
# Logic / Heuristics
HALLUCINATION_BUFFER_MS = 5000     # Word Budget Buffer - Simplified to 1.0 (neutral)
# After extensive testing, discovered translation behavior is content-dependent
# (technical vs conversational), making universal optimization impossible.
# Using neutral 1.0 buffer - panic modes handle edge cases.
LANGUAGE_PAIR_BUFFERS = {}
DEFAULT_BUFFER = 1.0

MAX_ATEMPO_SPEEDUP = 1.15          # Max speedup factor before splitting
MAX_ATEMPO_SLOWDOWN = 0.85         # Max slowdown factor

# ASR Configuration
ASR_CHUNK_LENGTH_MS = 300000       # 5 minutes - Audio chunk size for ASR processing

# Progressive Feedback Thresholds (New System - Replaces old panic logic)
ATEMPO_ONLY_THRESHOLD_MS = 500           # < 0.5s: Just use atempo speedup/slowdown
WORD_FEEDBACK_THRESHOLD_MS = 500         # >= 0.5s: Trigger word-based feedback (elaborate/summarize)
PANIC_IMMEDIATE_THRESHOLD_MS = 3000      # > 3s: Immediate panic mode (skip word feedback)
PANIC_POST_FEEDBACK_THRESHOLD_MS = 1000  # After feedback, only panic if still > 1s off

# Segment Overlap Protection
MAX_ACCEPTABLE_OVERLAP_MS = 500    # Merge/trim overlaps < 0.5s; >0.5s trusted as-is
SEGMENT_BUFFER_MS = 10             # Minimum gap between segments (10ms safety buffer)

# Legacy Panic Mode Constants (Deprecated - kept for backward compatibility)
PANIC_ELABORATION_THRESHOLD_MS = 1000  # Old: 1 second - Trigger aggressive elaboration
PANIC_ELABORATION_MULTIPLIER = 1.5     # Old: Multiplier for panic mode word addition

# Concurrency (Default if Env Var missing)
DEFAULT_MAX_WORKERS = 10

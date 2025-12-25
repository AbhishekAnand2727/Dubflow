"""
SRT Utilities - Parsing, formatting, and time conversion for SRT files
Extracted from google_pipeline2.py for reusability
"""

import re

def parse_srt_time(time_str):
    """
    Parses complex SRT/time timestamps to milliseconds.
    Handles:
    - HH:MM:SS,mmm (Standard)
    - MM:SS,mmm (Short)
    - HH:MM:SS:mmm (Colon separator)
    - Raw milliseconds (int/float/str)
    """
    if isinstance(time_str, (int, float)):
        return int(time_str)

    t_str = str(time_str).strip().replace('.', ',') # Normalise decimal
    
    # 1. Check Standard SRT (00:00:00,000)
    # 2. Check Alt (00:00:00:000)
    # 3. Check Short (00:00,000)
    
    parts = t_str.replace(':', ',').split(',')
    # Logic:
    # 3 parts -> MM:SS,mmm (or HH:MM:SS?) -> Ambiguity. Usually HH:MM:SS.
    # 4 parts -> HH:MM:SS,mmm
    
    # Let's try explicit regex for safer parsing
    # HH:MM:SS,mmm or HH:MM:SS.mmm
    match_full = re.match(r'(\d+):(\d+):(\d+)[:,](\d+)', t_str)
    if match_full:
        h, m, s, ms = map(int, match_full.groups())
        return (h * 3600 + m * 60 + s) * 1000 + ms

    # MM:SS,mmm
    match_short = re.match(r'(\d+):(\d+)[:,](\d+)', t_str)
    if match_short:
        m, s, ms = map(int, match_short.groups())
        return (m * 60 + s) * 1000 + ms
        
    # HH:MM:SS (No ms)
    match_hms = re.match(r'(\d+):(\d+):(\d+)', t_str)
    if match_hms:
        h, m, s = map(int, match_hms.groups())
        return (h * 3600 + m * 60 + s) * 1000

    # Just number
    try:
        return int(float(str(time_str).replace(',', '.')))
    except:
        pass

    raise ValueError(f"Invalid timestamp format: {time_str}")

def format_srt_time(ms):
    """
    Converts milliseconds to SRT timestamp format (HH:MM:SS,mmm).
    
    Args:
        ms: Milliseconds as integer
        
    Returns:
        SRT timestamp string
    """
    ms = int(ms)
    hours = ms // 3600000
    ms %= 3600000
    minutes = ms // 60000
    ms %= 60000
    seconds = ms // 1000
    milliseconds = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def parse_srt(srt_content):
    """
    Parses SRT content string into a list of segments.
    
    Args:
        srt_content: SRT file content as string or file path
        
    Returns:
        List of dicts with keys: index, start, end, text
    """
    # If it's a file path, read it
    if isinstance(srt_content, str) and '\n' not in srt_content and len(srt_content) < 500:
        try:
            with open(srt_content, 'r', encoding='utf-8') as f:
                srt_content = f.read()
        except FileNotFoundError:
            # Maybe it's actual SRT content, not a path
            pass
    
    segments = []
    blocks = srt_content.strip().split('\n\n')
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
            
        try:
            index = int(lines[0])
            time_line = lines[1]
            text = '\n'.join(lines[2:])
            
            # Parse timestamps
            if '-->' not in time_line:
                continue
                
            start_str, end_str = time_line.split('-->')
            start_str = start_str.strip()
            end_str = end_str.strip()
            
            segments.append({
                'index': index,
                'start': start_str,
                'end': end_str,
                'text': text
            })
        except (ValueError, IndexError) as e:
            # Skip malformed blocks
            continue
    
    return segments

def write_srt(segments, filepath):
    """
    Writes segments to SRT file.
    
    Args:
        segments: List of segment dicts with start, end, text keys
        filepath: Output SRT file path
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        segment_num = 1
        for seg in segments:
            if seg.get('deleted'):
                continue
            
            f.write(f"{segment_num}\n")
            f.write(f"{seg['start']} --> {seg['end']}\n")
            f.write(f"{seg.get('text', '')}\n\n")
            segment_num += 1

def normalize_timestamp(ts_str):
    """
    Normalizes SRT timestamp to strict HH:MM:SS,mmm format.
    
    Args:
        ts_str: Timestamp string (various formats)
        
    Returns:
        Normalized timestamp string
    """
    # Convert to milliseconds and back to ensure format
    try:
        ms = parse_srt_time(ts_str)
        return format_srt_time(ms)
    except ValueError:
        # If parsing fails, return as-is
        return ts_str

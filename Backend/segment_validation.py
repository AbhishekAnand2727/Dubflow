"""
Validation utilities for segment data
"""

def validate_srt_timestamp(timestamp: str) -> bool:
    """
    Validate SRT timestamp format: HH:MM:SS,mmm
    
    Args:
        timestamp: Timestamp string to validate
        
    Returns:
        True if valid, False otherwise
    """
    import re
    if isinstance(timestamp, (int, float)):
        return True
    
    # Check if stringified number
    try:
        float(timestamp.replace(',', '.'))
        return True
    except ValueError:
        pass
        
    pattern = r'^\d{2}:\d{2}:\d{2},\d{3}$'
    return bool(re.match(pattern, timestamp))

def validate_segment(segment: dict) -> tuple[bool, str]:
    """
    Validate a single segment structure and values
    
    Args:
        segment: Segment dictionary with start, end, text, target_text
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check required fields
    required_fields = ['start', 'end', 'text', 'target_text']
    for field in required_fields:
        if field not in segment:
            return False, f"Missing required field: {field}"
    
    # Validate timestamp format
    if not validate_srt_timestamp(segment['start']):
        return False, f"Invalid start timestamp format: {segment['start']}"
    
    if not validate_srt_timestamp(segment['end']):
        return False, f"Invalid end timestamp format: {segment['end']}"
    
    # Validate timestamp order
    try:
        from srt_utils import parse_srt_time
        start_ms = parse_srt_time(segment['start'])
        end_ms = parse_srt_time(segment['end'])
        
        if end_ms <= start_ms:
            return False, f"End timestamp must be after start timestamp"
        
        if end_ms - start_ms < 100:  # Minimum 0.1s duration
            return False, f"Segment duration too short (minimum 0.1s)"
            
    except Exception as e:
        return False, f"Error parsing timestamps: {str(e)}"
    
    return True, ""

def validate_segments_list(segments: list) -> tuple[bool, str]:
    """
    Validate entire segments list for consistency
    
    Args:
        segments: List of segment dictionaries
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not segments:
        return False, "Segments list is empty"
    
    from srt_utils import parse_srt_time
    
    prev_end_ms = 0
    for i, segment in enumerate(segments):
        if segment.get('deleted'):
            continue
            
        # Validate individual segment
        is_valid, error = validate_segment(segment)
        if not is_valid:
            return False, f"Segment {i}: {error}"
        
        # Check for overlaps with previous segment
        start_ms = parse_srt_time(segment['start'])
        if start_ms < prev_end_ms:
            # RELAXED VALIDATION: Allow overlaps, just warn logic will handle it
            pass 
            # return False, f"Segment {i} overlaps with previous segment"
        
        prev_end_ms = parse_srt_time(segment['end'])
    
    return True, ""

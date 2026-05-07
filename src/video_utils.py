import yt_dlp
import logging
from typing import List, Set, Tuple
from .models import VideoInfo

logger = logging.getLogger(__name__)

def get_video_info(url: str) -> VideoInfo:
    """
    Uses yt-dlp to extract the video duration and the direct stream URL.
    This avoids downloading the actual video.
    """
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]/best',
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            if not info:
                raise ValueError("Could not extract video info.")
                
            duration = info.get('duration', 0)
            if not duration:
                raise ValueError("Could not extract video duration.")
                
            video_id = info.get('id', '')
            direct_url = info.get('url', '')
            
            return VideoInfo(
                url=url,
                video_id=video_id,
                duration=duration,
                direct_url=direct_url
            )
    except Exception as e:
        logger.error(f"Error fetching info for {url}: {str(e)}")
        raise

def generate_timestamps(duration: float, num_splits: int, checked_timestamps: Set[float], tolerance: float = 2.0) -> List[float]:
    """
    Generates new timestamps for a video given a split count.
    
    Args:
        duration: The video duration in seconds.
        num_splits: The number of intervals to split the video into.
        checked_timestamps: Set of previously checked timestamps.
        tolerance: If a generated timestamp is within `tolerance` seconds of an already
                   checked timestamp, we skip it to avoid redundant checks.
                   
    Returns:
        A list of new timestamps to check.
    """
    if num_splits <= 1:
        return []
        
    interval = duration / num_splits
    new_timestamps = []
    
    for i in range(1, num_splits):
        ts = i * interval
        
        # Check if this timestamp is too close to any already checked timestamp
        is_duplicate = False
        for checked_ts in checked_timestamps:
            if abs(ts - checked_ts) <= tolerance:
                is_duplicate = True
                break
                
        if not is_duplicate:
            new_timestamps.append(ts)
            
    return new_timestamps

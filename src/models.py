from dataclasses import dataclass
from typing import Optional

@dataclass
class VideoResult:
    video_url: str
    detected: str # "yes", "no", "error"
    transcript: Optional[str] = None
    detection_timestamp: Optional[float] = None
    frames_checked: int = 0
    processing_time: float = 0.0
    transcript_language: Optional[str] = None
    detection_confidence: Optional[float] = None
    status: str = "pending" # "completed", "failed", "pending"
    error_message: Optional[str] = None

@dataclass
class VideoInfo:
    url: str
    video_id: str
    duration: float
    direct_url: str # the direct m3u8/mp4 url for ffmpeg

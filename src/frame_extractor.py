import subprocess
import logging
import cv2
import numpy as np
from typing import Optional, List
import concurrent.futures

logger = logging.getLogger(__name__)

class FrameExtractor:
    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def extract_frame(self, video_url: str, timestamp: float) -> Optional[np.ndarray]:
        """
        Extracts a single frame from the given video URL at the specified timestamp.
        Uses ffmpeg and pipes the output directly to memory to avoid disk I/O.
        """
        command = [
            'ffmpeg',
            '-ss', str(timestamp),    # fast seek before input
            '-i', video_url,          # input URL (should be direct m3u8 or mp4)
            '-vframes', '1',          # extract only 1 frame
            '-f', 'image2pipe',       # pipe output
            '-vcodec', 'mjpeg',       # use mjpeg codec
            '-hide_banner',           # reduce log spam
            '-loglevel', 'error',     # only show errors
            '-'                       # output to stdout
        ]

        try:
            process = subprocess.Popen(
                command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            out, err = process.communicate(timeout=self.timeout)

            if process.returncode != 0:
                logger.error(f"FFmpeg error for timestamp {timestamp}: {err.decode('utf-8')}")
                return None

            if not out:
                logger.warning(f"No frame extracted for timestamp {timestamp}")
                return None

            # Convert bytes to numpy array then to OpenCV image
            nparr = np.frombuffer(out, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            return frame

        except subprocess.TimeoutExpired:
            process.kill()
            logger.error(f"FFmpeg timeout after {self.timeout}s for timestamp {timestamp}")
            return None
        except Exception as e:
            logger.error(f"Failed to extract frame at {timestamp}: {str(e)}")
            return None

    def extract_frame_sequence(self, video_url: str, base_timestamp: float) -> List[np.ndarray]:
        """
        Extracts a sequence of frames at [t, t+0.5, t+1.0] to capture motion.
        """
        timestamps = [base_timestamp, base_timestamp + 0.5, base_timestamp + 1.0]
        frames = []
        
        # We can extract them sequentially or in parallel. Sequential is safer for bandwidth.
        for ts in timestamps:
            frame = self.extract_frame(video_url, ts)
            if frame is not None:
                frames.append(frame)
                
        return frames

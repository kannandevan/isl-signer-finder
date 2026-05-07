import logging
import time
import os
import cv2
from typing import Set

from .config import Config, get_config
from .models import VideoResult, VideoInfo
from .video_utils import get_video_info, generate_timestamps
from .frame_extractor import FrameExtractor
from .detector import SignLanguageDetector
from .transcript_fetcher import fetch_transcript
from .state_manager import StateManager

logger = logging.getLogger(__name__)

def process_video(video_url: str, config_path: str = "config.yaml") -> VideoResult:
    """
    Processes a single video URL. This function is designed to be run in a worker process.
    """
    config = get_config(config_path)
    state_manager = StateManager(config.processing.db_path)
    extractor = FrameExtractor(timeout=config.processing.ffmpeg_timeout)
    detector = SignLanguageDetector(
        crop_left_ratio=config.detection.crop_left_ratio,
        confidence_threshold=config.detection.confidence_threshold
    )
    
    start_time = time.time()
    frames_checked = 0
    checked_timestamps: Set[float] = set()
    
    result = VideoResult(video_url=video_url, detected="no", status="pending")
    
    # 1. Check if already processed
    if state_manager.is_processed(video_url):
        logger.info(f"Skipping {video_url}, already processed.")
        result.status = "completed" # Already completed previously
        return result
        
    try:
        # 2. Get video info
        logger.info(f"Fetching info for {video_url}")
        info = get_video_info(video_url)
        
        # 3. Progressive checking loop
        # Start at 5 splits, up to max_splits (usually 10)
        found_interpreter = False
        
        for num_splits in range(5, config.processing.max_splits + 1):
            if found_interpreter:
                break
                
            new_timestamps = generate_timestamps(info.duration, num_splits, checked_timestamps)
            
            for ts in new_timestamps:
                logger.debug(f"[{video_url}] Extracting frame at {ts:.2f}s (split level {num_splits})")
                frame = extractor.extract_frame(info.direct_url, ts)
                checked_timestamps.add(ts)
                frames_checked += 1
                
                if frame is not None:
                    # 4. Analyze Frame
                    detected, confidence = detector.analyze_frame(frame)
                    
                    if detected:
                        logger.info(f"[{video_url}] Interpreter detected at {ts:.2f}s with confidence {confidence:.2f}")
                        found_interpreter = True
                        result.detected = "yes"
                        result.detection_timestamp = ts
                        result.detection_confidence = confidence
                        
                        # Optionally save frame
                        if config.output.save_frames:
                            frame_path = os.path.join(config.output.frames_dir, f"{info.video_id}_{ts:.2f}.jpg")
                            cv2.imwrite(frame_path, frame)
                            
                        break # Stop checking timestamps
                        
        # 5. Fetch Transcript if detected
        if found_interpreter:
            logger.info(f"[{video_url}] Fetching transcript...")
            transcript_text, lang = fetch_transcript(info.video_id)
            if transcript_text:
                result.transcript = transcript_text
                result.transcript_language = lang
            else:
                result.transcript = "no" # Could not fetch
        else:
            result.transcript = "no"
            
        result.status = "completed"
        
    except Exception as e:
        logger.error(f"Error processing {video_url}: {str(e)}")
        result.status = "failed"
        result.error_message = str(e)
        result.detected = "error"
        
    finally:
        result.processing_time = time.time() - start_time
        result.frames_checked = frames_checked
        
        # Save to state manager
        if result.status != "pending":
            state_manager.save_result(result)
            
    return result

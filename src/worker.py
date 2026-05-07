import logging
import time
import os
import cv2
from typing import Set

from .config import Config, get_config
from .models import VideoResult, VideoInfo
from .video_utils import get_video_info, generate_timestamps, generate_early_timestamps
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
    
    if state_manager.is_processed(video_url):
        logger.info(f"Skipping {video_url}, already processed.")
        result.status = "completed"
        return result
        
    try:
        logger.info(f"Fetching info for {video_url}")
        info = get_video_info(video_url)
        
        found_interpreter = False
        
        # --- PHASE 1: EARLY DENSE SCAN ---
        logger.info(f"[{video_url}] Phase 1: Early Dense Scan")
        early_timestamps = generate_early_timestamps(
            config.processing.early_scan_timestamps, 
            info.duration, 
            checked_timestamps
        )
        
        for ts in early_timestamps:
            if found_interpreter:
                break
                
            logger.debug(f"[{video_url}] Extracting frame sequence at {ts:.2f}s")
            frames = extractor.extract_frame_sequence(info.direct_url, ts)
            checked_timestamps.add(ts)
            frames_checked += len(frames)
            
            if frames:
                detected, confidence, debug_frame = detector.analyze_frame_sequence(frames)
                
                if detected:
                    logger.info(f"[{video_url}] Interpreter detected at {ts:.2f}s (Conf: {confidence:.2f})")
                    found_interpreter = True
                    result.detected = "yes"
                    result.detection_timestamp = ts
                    result.detection_confidence = confidence
                    
                    if config.output.save_frames and debug_frame is not None:
                        frame_path = os.path.join(config.output.frames_dir, f"{info.video_id}_{ts:.2f}_pos.jpg")
                        cv2.imwrite(frame_path, debug_frame)
                elif config.output.save_failed_frames and debug_frame is not None:
                    # Save a sample of the false negative check for debugging
                    frame_path = os.path.join(config.output.frames_dir, f"{info.video_id}_{ts:.2f}_neg.jpg")
                    cv2.imwrite(frame_path, debug_frame)

        # --- PHASE 2: ADAPTIVE INTERVAL SCANNING ---
        if not found_interpreter:
            logger.info(f"[{video_url}] Phase 2: Adaptive Interval Scan")
            for num_splits in range(5, config.processing.max_splits + 1):
                if found_interpreter:
                    break
                    
                new_timestamps = generate_timestamps(info.duration, num_splits, checked_timestamps)
                
                for ts in new_timestamps:
                    logger.debug(f"[{video_url}] Extracting frame sequence at {ts:.2f}s (split level {num_splits})")
                    frames = extractor.extract_frame_sequence(info.direct_url, ts)
                    checked_timestamps.add(ts)
                    frames_checked += len(frames)
                    
                    if frames:
                        detected, confidence, debug_frame = detector.analyze_frame_sequence(frames)
                        
                        if detected:
                            logger.info(f"[{video_url}] Interpreter detected at {ts:.2f}s (Conf: {confidence:.2f})")
                            found_interpreter = True
                            result.detected = "yes"
                            result.detection_timestamp = ts
                            result.detection_confidence = confidence
                            
                            if config.output.save_frames and debug_frame is not None:
                                frame_path = os.path.join(config.output.frames_dir, f"{info.video_id}_{ts:.2f}_pos.jpg")
                                cv2.imwrite(frame_path, debug_frame)
                            break
                        elif config.output.save_failed_frames and debug_frame is not None:
                            frame_path = os.path.join(config.output.frames_dir, f"{info.video_id}_{ts:.2f}_neg.jpg")
                            cv2.imwrite(frame_path, debug_frame)

        # --- TRANSCRIPT FETCHING ---
        if found_interpreter:
            logger.info(f"[{video_url}] Fetching transcript...")
            transcript_text, lang = fetch_transcript(info.video_id)
            if transcript_text:
                result.transcript = transcript_text
                result.transcript_language = lang
            else:
                result.transcript = "no" 
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
        
        if result.status != "pending":
            state_manager.save_result(result)
            
        detector.close()
            
    return result

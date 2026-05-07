import cv2
import logging
import numpy as np
from typing import Tuple, List, Optional
import mediapipe as mp

logger = logging.getLogger(__name__)

class SignLanguageDetector:
    def __init__(self, crop_left_ratio: float = 0.50, confidence_threshold: float = 0.6):
        self.crop_left_ratio = crop_left_ratio
        self.confidence_threshold = confidence_threshold
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=True, 
            min_detection_confidence=0.5
        )
        self.hands_detector = self.mp_hands.Hands(
            static_image_mode=True, 
            max_num_hands=2, 
            min_detection_confidence=0.3
        )

    def _crop_left(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        crop_width = int(width * self.crop_left_ratio)
        return frame[:, :crop_width]

    def analyze_frame_sequence(self, frames: List[np.ndarray]) -> Tuple[bool, float, Optional[np.ndarray]]:
        """
        Analyzes a sequence of frames for the presence of a sign language interpreter.
        
        Args:
            frames: A list of numpy arrays representing BGR images.
            
        Returns:
            A tuple of (detected: bool, confidence: float, debug_frame: Optional[np.ndarray])
        """
        if not frames:
            return False, 0.0, None

        pose_detections = 0
        hand_detections = 0
        
        # We will use the first frame as the base debug frame
        base_crop = self._crop_left(frames[0])
        debug_frame = base_crop.copy()

        hand_positions = []

        for frame in frames:
            if frame is None or frame.size == 0:
                continue
                
            crop = self._crop_left(frame)
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # Detect Pose
            pose_results = self.pose_detector.process(rgb_crop)
            if pose_results.pose_landmarks:
                pose_detections += 1
                
            # Detect Hands
            hands_results = self.hands_detector.process(rgb_crop)
            if hands_results.multi_hand_landmarks:
                hand_detections += 1
                
                # Track wrist positions to calculate movement variance later
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                    hand_positions.append((wrist.x, wrist.y))

        num_frames = len(frames)
        confidence = 0.0
        
        # Base confidence from pose consistency
        if pose_detections > 0:
            confidence += 0.4 * (pose_detections / num_frames)
            
        # Confidence from hand presence
        if hand_detections > 0:
            confidence += 0.3 * (hand_detections / num_frames)
            
        # Confidence from movement variance (if hands moved between frames)
        if len(hand_positions) > 1:
            xs = [p[0] for p in hand_positions]
            ys = [p[1] for p in hand_positions]
            var_x = np.var(xs)
            var_y = np.var(ys)
            
            if var_x > 0.0001 or var_y > 0.0001:
                confidence += 0.2  # Bonus for detected motion
                
        # Draw some debug info on the debug frame if it's reasonably confident
        if confidence >= self.confidence_threshold:
            cv2.putText(debug_frame, f"Conf: {confidence:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        return confidence >= self.confidence_threshold, confidence, debug_frame

    def close(self):
        self.pose_detector.close()
        self.hands_detector.close()

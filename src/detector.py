import cv2
import os
import logging
import numpy as np
from typing import Tuple

logger = logging.getLogger(__name__)

class SignLanguageDetector:
    def __init__(self, crop_left_ratio: float = 0.40, confidence_threshold: float = 0.6):
        self.crop_left_ratio = crop_left_ratio
        self.confidence_threshold = confidence_threshold
        
        # Initialize Haar cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Initialize HOG descriptor for person detection (fallback)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def analyze_frame(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Analyzes a single frame for the presence of a sign language interpreter.
        
        Args:
            frame: A numpy array representing the BGR image from OpenCV.
            
        Returns:
            A tuple of (detected: bool, confidence: float)
        """
        if frame is None or frame.size == 0:
            return False, 0.0
            
        height, width = frame.shape[:2]
        
        # Crop to the left side where the interpreter box is usually located
        crop_width = int(width * self.crop_left_ratio)
        left_crop = frame[:, :crop_width]
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(left_crop, cv2.COLOR_BGR2GRAY)
        
        # First, try to detect a face using Haar Cascade (fastest)
        # We tune parameters to avoid false positives. 
        # minNeighbors=5, scaleFactor=1.1
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            # We found a face on the left side. 
            # In a news video, a prominent face on the left 40% in a box is very likely the interpreter.
            # We assign a high confidence.
            return True, 0.85
            
        # Fallback to HOG person detection if face is missed but a person is visible
        boxes, weights = self.hog.detectMultiScale(left_crop, winStride=(8,8))
        
        if len(boxes) > 0:
            # HOG returns weights for each detection
            max_weight = float(np.max(weights)) if len(weights) > 0 else 0.0
            
            # Normalize confidence somewhat heuristically
            confidence = min(max_weight / 2.0, 0.95)
            
            if confidence >= self.confidence_threshold:
                return True, confidence
                
        return False, 0.0

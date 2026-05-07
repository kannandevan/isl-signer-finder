import sqlite3
import logging
from typing import List, Optional
from .models import VideoResult
from .config import get_config

logger = logging.getLogger(__name__)

class StateManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS video_results (
                    video_url TEXT PRIMARY KEY,
                    detected TEXT,
                    transcript TEXT,
                    detection_timestamp REAL,
                    frames_checked INTEGER,
                    processing_time REAL,
                    transcript_language TEXT,
                    detection_confidence REAL,
                    status TEXT,
                    error_message TEXT
                )
            ''')
            conn.commit()

    def save_result(self, result: VideoResult):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO video_results (
                        video_url, detected, transcript, detection_timestamp, 
                        frames_checked, processing_time, transcript_language, 
                        detection_confidence, status, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result.video_url, result.detected, result.transcript, 
                    result.detection_timestamp, result.frames_checked, 
                    result.processing_time, result.transcript_language, 
                    result.detection_confidence, result.status, result.error_message
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save result for {result.video_url}: {str(e)}")

    def is_processed(self, video_url: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT status FROM video_results WHERE video_url = ?', (video_url,))
            row = cursor.fetchone()
            if row and row[0] == "completed":
                return True
            return False

    def get_all_results(self) -> List[VideoResult]:
        results = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM video_results')
            rows = cursor.fetchall()
            for row in rows:
                results.append(VideoResult(
                    video_url=row[0],
                    detected=row[1],
                    transcript=row[2],
                    detection_timestamp=row[3],
                    frames_checked=row[4],
                    processing_time=row[5],
                    transcript_language=row[6],
                    detection_confidence=row[7],
                    status=row[8],
                    error_message=row[9]
                ))
        return results

def get_state_manager() -> StateManager:
    config = get_config()
    return StateManager(config.processing.db_path)

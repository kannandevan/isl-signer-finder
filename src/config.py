import yaml
import os
import logging
from dataclasses import dataclass
from typing import Optional

@dataclass
class ProcessingConfig:
    num_workers: int
    max_splits: int
    db_path: str
    log_level: str
    ffmpeg_timeout: int

@dataclass
class DetectionConfig:
    crop_left_ratio: float
    confidence_threshold: float
    use_haarcascade: bool

@dataclass
class OutputConfig:
    csv_path: str
    xlsx_path: str
    save_frames: bool
    frames_dir: str

class Config:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self._load_config()
        self._setup_logging()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file {self.config_path} not found.")

        with open(self.config_path, "r") as f:
            data = yaml.safe_load(f)

        self.processing = ProcessingConfig(**data.get("processing", {}))
        self.detection = DetectionConfig(**data.get("detection", {}))
        self.output = OutputConfig(**data.get("output", {}))
        
        # Ensure output directories exist
        os.makedirs(os.path.dirname(self.output.csv_path) or ".", exist_ok=True)
        if self.output.save_frames:
            os.makedirs(self.output.frames_dir, exist_ok=True)

    def _setup_logging(self):
        level = getattr(logging, self.processing.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("processing.log"),
                logging.StreamHandler()
            ]
        )

# Global config instance can be initialized here or lazily
_config: Optional[Config] = None

def get_config(config_path: str = "config.yaml") -> Config:
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config

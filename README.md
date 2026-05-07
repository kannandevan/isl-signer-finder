# YouTube Sign Language Interpreter Detector

## 1. Project Overview
This project is a production-grade, high-performance Python system that analyzes large volumes of YouTube news videos (100,000+) to detect if a sign language interpreter is present. It is designed to strictly minimize bandwidth and RAM usage while preventing false negatives common in older heuristics.

## 2. Architecture Explanation
The architecture relies on a Master-Worker pool. The Master process parses YouTube URLs and distributes them. Workers use `yt-dlp` to get direct stream links, preventing full video downloads. They invoke `ffmpeg` to extract temporal bursts of frames into memory, and `MediaPipe` calculates pose/hand activity. Results are cached into a local SQLite database (`checkpoints.db`) for immediate resumability, and finally dumped to CSV/XLSX.

## 3. Detection Pipeline Explanation
1. **Phase 1 (Early Dense Scan):** Most news interpreters appear instantly. We densely scan `[5s, 10s, 15s, 20s, 30s, 45s, 60s]`.
2. **Phase 2 (Adaptive Interval Scan):** If Phase 1 fails, we split the video duration into 5 intervals, then 6, 7, etc.
3. **Temporal Burst:** At every timestamp `t`, we extract `t, t+0.5, t+1.0`.
4. **CV Analysis:** We crop the left 50% of the frames and run MediaPipe Pose and Hands to detect continuous upper-body motion and hand landmarks.

## 4. Dataset Example Explanation
Analysis of the `dataset_examples/` folders revealed that older implementations failed because they expected tiny corner boxes. The positive examples show real-world Indian TV broadcasts where the signer often occupies massive split-screen layouts.

## 5. Signer Layout Explanation
In the target dataset (Indian news), signers:
- Usually appear in a **LARGE LEFT SPLIT SCREEN** layout.
- Can occupy up to 40–50% of the screen width.
- Exhibit continuous, varying hand movement from the beginning of the broadcast.
Our system dynamically adapts to this by analyzing the entire left 50% using MediaPipe rather than hardcoding strict, tiny bounding boxes.

## 6. Installation Guide
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 7. FFmpeg Setup
FFmpeg must be installed and accessible in your system's PATH.
- **Windows:** `winget install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`

## 8. Quick Start Guide
Run the orchestrator with the provided sample file:
```bash
python main.py -i sample_urls.txt
```

## 9. Example Commands
- Basic run: `python main.py -i urls.txt`
- Custom config: `python main.py -i urls.txt -c custom_config.yaml`

## 10. Configuration Options
Edit `config.yaml`:
- `processing.num_workers`: Multiprocessing concurrency limit.
- `processing.early_scan_timestamps`: Initial dense scan seconds.
- `detection.crop_left_ratio`: Width ratio to crop (default `0.50`).
- `output.save_failed_frames`: Exports negative frames for debugging.

## 11. Worker Scaling
The system scales horizontally by increasing `num_workers`. Since frame extraction is piped directly into memory and OpenCV/MediaPipe run on CPU, CPU core count is your primary bottleneck. Recommend `num_workers = CPU_CORES - 1`.

## 12. Debugging Guide
If a video fails or returns a false negative, enable `save_crops: true` and `save_failed_frames: true` in `config.yaml`. The system will save the annotated MediaPipe frames to `output/debug_frames/` with `_neg.jpg` suffix.

## 13. Transcript Handling
If an interpreter is found, we fetch the transcript prioritizing:
1. `hi` (Hindi)
2. `hi-IN`
3. `en` (English)
4. Auto-generated captions
5. Any available fallback.

## 14. Troubleshooting
- **No frames extracted:** Ensure `ffmpeg` is globally installed.
- **Transcript errors:** Some YouTube videos disable captions entirely. The system catches this and sets transcript to `no`.
- **Memory leaks:** The system correctly closes MediaPipe instances per-video to prevent RAM accumulation.

## 15. Performance Optimization
- **Bandwidth:** `yt-dlp` finds `.m3u8`/`.mp4` stream chunks. `ffmpeg` fast-seeks (`-ss` before `-i`) to only download the few megabytes needed for 1 second of video.
- **Memory:** Disk I/O is bypassed using `image2pipe`.

## 16. Deployment Guide
For large runs (100k+), deploy on an AWS c5/c6i instance or equivalent CPU-optimized server. Run the command inside a `tmux` or `screen` session.

## 17. Docker Setup
```dockerfile
FROM python:3.10-slim
RUN apt-get update && apt-get install -y ffmpeg libgl1-mesa-glx
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py", "-i", "sample_urls.txt"]
```

## 18. Resume / Recovery System
A local `checkpoints.db` is updated atomically. If the script crashes, simply restart it. Completed URLs will be skipped automatically in `O(1)` time.

## 19. Sample Inputs
`sample_urls.txt`:
```
https://www.youtube.com/watch?v=dQw4w9WgXcQ
https://www.youtube.com/watch?v=3JZ_D3ELwOQ
```

## 20. Sample Outputs
Exported `output/results.csv`:
`video_url, detected, transcript, detection_timestamp, frames_checked, processing_time, transcript_language, detection_confidence, status, error_message`

## 21. Example Screenshots
*(Imagine seeing split-screen Indian news with a MediaPipe skeleton overlay on the left 50% bounding box).*

## 22. False Negative Debugging Guide
1. Check `output/debug_frames/`.
2. If the person is visible but undetected, the `confidence_threshold` (0.6) might be too high for low-res videos.
3. If the person is cut off, adjust `crop_left_ratio` to `0.60`.

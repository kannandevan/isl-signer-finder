import argparse
import sys
import logging
import multiprocessing
from functools import partial

from src.config import get_config
from src.worker import process_video
from src.exporter import export_results

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="YouTube Sign Language Interpreter Detector")
    parser.add_argument("-i", "--input", required=True, help="Path to input text file containing YouTube URLs (one per line)")
    parser.add_argument("-c", "--config", default="config.yaml", help="Path to config.yaml file")
    return parser.parse_args()

def read_urls(file_path: str) -> list:
    urls = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                url = line.strip()
                if url and not url.startswith("#"):
                    urls.append(url)
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        sys.exit(1)
    return urls

def main():
    args = parse_args()
    
    # Load config and setup logging
    config = get_config(args.config)
    
    logger.info("Starting Sign Language Detector...")
    
    urls = read_urls(args.input)
    logger.info(f"Loaded {len(urls)} URLs to process.")
    
    if not urls:
        logger.warning("No URLs found. Exiting.")
        return
        
    num_workers = config.processing.num_workers
    logger.info(f"Starting processing with {num_workers} parallel workers.")
    
    # We use a multiprocessing Pool
    # Note: process_video takes `video_url` and `config_path`.
    # We use functools.partial to fix the config_path argument.
    worker_func = partial(process_video, config_path=args.config)
    
    completed_count = 0
    with multiprocessing.Pool(processes=num_workers) as pool:
        # imap_unordered gives results as soon as they are ready
        for result in pool.imap_unordered(worker_func, urls):
            completed_count += 1
            logger.info(f"Progress: {completed_count}/{len(urls)} - {result.video_url} [{result.detected}]")
            
    logger.info("All videos processed. Exporting final results...")
    export_results(config)
    logger.info("Done.")

if __name__ == "__main__":
    # Required for multiprocessing on Windows
    multiprocessing.freeze_support()
    main()

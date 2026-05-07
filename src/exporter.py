import pandas as pd
import logging
from dataclasses import asdict
from .state_manager import StateManager
from .config import Config

logger = logging.getLogger(__name__)

def export_results(config: Config):
    """
    Reads all processed results from the local SQLite database
    and exports them to CSV and XLSX.
    """
    logger.info("Exporting results...")
    state_manager = StateManager(config.processing.db_path)
    results = state_manager.get_all_results()
    
    if not results:
        logger.warning("No results to export.")
        return
        
    # Convert list of dataclass objects to a list of dicts
    data = [asdict(r) for r in results]
    
    # Create pandas DataFrame
    df = pd.DataFrame(data)
    
    # Reorder columns as requested
    columns_order = [
        "video_url", "detected", "transcript", "detection_timestamp", 
        "frames_checked", "processing_time", "transcript_language", 
        "detection_confidence", "status", "error_message"
    ]
    df = df[columns_order]
    
    # Export to CSV
    csv_path = config.output.csv_path
    try:
        df.to_csv(csv_path, index=False)
        logger.info(f"Results exported successfully to {csv_path}")
    except Exception as e:
        logger.error(f"Failed to export CSV: {e}")
        
    # Export to XLSX
    xlsx_path = config.output.xlsx_path
    try:
        # Use openpyxl engine
        df.to_excel(xlsx_path, index=False, engine='openpyxl')
        logger.info(f"Results exported successfully to {xlsx_path}")
    except Exception as e:
        logger.error(f"Failed to export XLSX: {e}")

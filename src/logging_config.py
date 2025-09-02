import logging
import os
from datetime import datetime
from pathlib import Path

def setup_logging(logs_dir: str = "./logs", log_level: str = "INFO") -> None:
    """
    Configure logging with console and file handlers.
    
    Args:
        logs_dir (str): Directory to store log files.
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(logs_dir, f"rag_system_{timestamp}.log")

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    logger.handlers = []

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(file_handler)

    logger.info(f"Logging configured. Logs will be saved to {log_file}")
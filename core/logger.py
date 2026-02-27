# =============================================================================
# core/logger.py — Centralized Logging Utility
# =============================================================================

import logging
import os
from datetime import datetime
from config import LOGS_DIR

def get_logger(name: str) -> logging.Logger:
    """
    Returns a named logger that writes to both console and a dated log file.
    Usage:  from core.logger import get_logger
            log = get_logger(__name__)
            log.info("Message")
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Avoid adding duplicate handlers

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    log_filename = datetime.now().strftime("dms_%Y%m%d.log")
    log_path = os.path.join(LOGS_DIR, log_filename)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
"""
logger.py - Structured logging configuration.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from app.core.config import settings


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for *name*."""
    logger = logging.getLogger(name)

    if logger.handlers:           # avoid duplicate handlers on re-import
        return logger

    logger.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Rotating file
    log_file = settings.LOGS_DIR / "forecasting_system.log"
    fh = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger

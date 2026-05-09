"""Logging configuration using loguru.

Industry best practice: Structured logging with rotation and proper formatting.
"""

import sys
from pathlib import Path

from loguru import logger

from src.utils.config_loader import get_config


def setup_logger() -> None:
    """Configure logger with file and console outputs.
    
    Best practice: 
    - Separate log levels for console and file
    - Log rotation to prevent large files
    - Structured format with timestamp, level, location, and message
    """
    config = get_config()
    
    # Remove default handler
    logger.remove()
    
    # Console handler (INFO and above)
    log_level = config.get("logging.level", "INFO")
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
        colorize=True,
    )
    
    # File handler (DEBUG and above) with rotation
    log_file = Path(config.get("logging.log_file", "./logs/rag_benchmark.log"))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_file,
        level="DEBUG",
        format=config.get("logging.format"),
        rotation=config.get("logging.rotation", "100 MB"),
        retention="10 days",
        compression="zip",
    )
    
    logger.info(f"Logger initialized with level: {log_level}")


def get_logger(name: str):
    """Get logger instance for a module.
    
    Args:
        name: Module name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logger.bind(name=name)

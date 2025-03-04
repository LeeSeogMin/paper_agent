# -*- coding: utf-8 -*-

"""
Logging utility module
Provides functionality for logging activities in the paper writing process.
"""

import os
import logging
from datetime import datetime

# Logger configuration
logger = logging.getLogger("paper_agent")

def configure_logging(log_level="INFO", log_file=None):
    """
    Configure the logging system
    
    Args:
        log_level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file (str, optional): Log file path
    """
    # Set log level
    level = getattr(logging, log_level.upper())
    
    # Configure logger
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # Add file handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    logger.info(f"Logging configuration completed (level: {log_level})")
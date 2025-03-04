"""
Application settings and configuration

This module contains global configuration settings for the paper writing agent.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# OpenAI API settings
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", "0.7"))
TEMPERATURE = OPENAI_TEMPERATURE  # Add alias for backward compatibility

# Paper settings
DEFAULT_TEMPLATE = "academic"
DEFAULT_STYLE_GUIDE = "Standard Academic"
DEFAULT_CITATION_STYLE = "APA"
DEFAULT_FORMAT = "markdown"
MAX_SECTION_TOKENS = 2000  # 섹션당 최대 토큰 수 추가

# Output settings
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Agent settings
MAX_RETRIES = 3
TIMEOUT = 60  # seconds

# Research settings
MAX_SOURCES = 10
MAX_SEARCH_DEPTH = 3

# Vector DB settings
VECTOR_DB_PATH = os.path.join(BASE_DIR, "data", "vector_db")
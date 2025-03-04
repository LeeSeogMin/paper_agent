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

# XAI API settings
# XAI_MODEL = os.environ.get("XAI_MODEL", "grok-2-1212")

# Paper settings
DEFAULT_TEMPLATE = "academic"
DEFAULT_STYLE_GUIDE = "Standard Academic"
DEFAULT_CITATION_STYLE = "APA"
DEFAULT_FORMAT = "markdown"
MAX_SECTION_TOKENS = 2000  # 섹션당 최대 토큰 수 추가

# Output settings
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Ensure all required directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Data directories
DATA_DIR = os.path.join(BASE_DIR, "data")
PAPERS_DIR = os.path.join(DATA_DIR, "papers")
VECTOR_DB_DIR = os.path.join(DATA_DIR, "vector_db")
PDF_STORAGE_DIR = os.path.join(DATA_DIR, "pdf")

# Create all data directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PAPERS_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)
os.makedirs(PDF_STORAGE_DIR, exist_ok=True)

# Agent settings
MAX_RETRIES = 3
TIMEOUT = 60  # seconds

# Research settings
MAX_SOURCES = 10
MAX_SEARCH_DEPTH = 3

# Vector DB settings
VECTOR_DB_PATH = os.path.join(BASE_DIR, "data", "vector_db")
VECTOR_DB_DIR = VECTOR_DB_PATH  # 호환성을 위한 별칭 추가
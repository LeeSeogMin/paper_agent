"""
Application settings and configuration

This module contains global configuration settings for the paper writing agent.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# API Selection
USE_OPENAI_API = os.environ.get("USE_OPENAI_API", "false").lower() == "true"
USE_XAI_API = os.environ.get("USE_XAI_API", "false").lower() == "true"
USE_ANTHROPIC_API = os.environ.get("USE_ANTHROPIC_API", "true").lower() == "true"

# OpenAI API settings
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", "0.7"))
TEMPERATURE = OPENAI_TEMPERATURE  # Add alias for backward compatibility

# XAI API settings
XAI_MODEL = os.environ.get("XAI_MODEL", "grok-2")
XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
XAI_TEMPERATURE = float(os.environ.get("XAI_TEMPERATURE", "0.7"))

# Anthropic API settings
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-3-7-sonnet-20240229")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_TEMPERATURE = float(os.environ.get("ANTHROPIC_TEMPERATURE", "0.7"))

# Default model settings (will be set based on selected API)
if USE_OPENAI_API:
    DEFAULT_MODEL = OPENAI_MODEL
    DEFAULT_TEMPERATURE = OPENAI_TEMPERATURE
elif USE_XAI_API:
    DEFAULT_MODEL = XAI_MODEL
    DEFAULT_TEMPERATURE = XAI_TEMPERATURE
elif USE_ANTHROPIC_API:
    DEFAULT_MODEL = ANTHROPIC_MODEL
    DEFAULT_TEMPERATURE = ANTHROPIC_TEMPERATURE
else:
    # Fallback to Anthropic if no API is selected
    DEFAULT_MODEL = ANTHROPIC_MODEL
    DEFAULT_TEMPERATURE = ANTHROPIC_TEMPERATURE

TEMPERATURE = DEFAULT_TEMPERATURE  # Add alias for backward compatibility

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
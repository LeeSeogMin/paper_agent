"""
API keys configuration module

This module loads and provides API keys for various services used by the paper writing agent.
"""

import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# OpenAI API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    logger.warning("OpenAI API key not found. Some functionality may be limited.")

# XAI API key
XAI_API_KEY = os.environ.get("XAI_API_KEY", "")
if not XAI_API_KEY:
    logger.warning("XAI API key not found. Using XAI API will not be possible.")

# Semantic Scholar API key
# If API key is not provided, we'll use the API as an unauthorized user
# This will have lower rate limits but still function
SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
# Automatically use unauthorized access if no API key is provided
SEMANTIC_SCHOLAR_USE_AUTH = bool(SEMANTIC_SCHOLAR_API_KEY)
if not SEMANTIC_SCHOLAR_API_KEY:
    logger.warning("Semantic Scholar API key not found. Using unauthorized access with lower rate limits.")

# Google Scholar API key
# Google Scholar API는 GOOGLE_API_KEY와 GOOGLE_CSE_ID를 사용할 수 있음
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID", "")
GOOGLE_SCHOLAR_API_KEY = GOOGLE_API_KEY  # GOOGLE_API_KEY를 GOOGLE_SCHOLAR_API_KEY로 사용

if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
    logger.warning("Google API key or CSE ID not found. Google Scholar search may not be available.")

# Function to check if required API keys are available
def check_required_api_keys():
    """
    Check if all required API keys are available.
    Returns a tuple of (success, missing_keys)
    """
    required_keys = {
        "OPENAI_API_KEY": OPENAI_API_KEY,  # XAI_API_KEY 대신 OPENAI_API_KEY만 확인
    }
    
    missing_keys = [key for key, value in required_keys.items() if not value]
    
    if missing_keys:
        logger.error(f"Missing required API keys: {', '.join(missing_keys)}")
        return False, missing_keys
    
    return True, []
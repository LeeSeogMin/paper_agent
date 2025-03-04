"""
API keys configuration module

This module loads and provides API keys for various services used by the paper writing agent.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Semantic Scholar API key
# If API key is not provided, we'll use the API as an unauthorized user
# This will have lower rate limits but still function
SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
# Automatically use unauthorized access if no API key is provided
SEMANTIC_SCHOLAR_USE_AUTH = bool(SEMANTIC_SCHOLAR_API_KEY)

# Google Scholar API key
GOOGLE_SCHOLAR_API_KEY = os.environ.get("GOOGLE_SCHOLAR_API_KEY", "")
"""
유틸리티 패키지
논문 작성 AI 에이전트 시스템에서 사용되는 다양한 유틸리티 함수를 제공합니다.
"""

from utils.logger import logger, configure_logging
from utils.api_clients import search_academic_papers, download_pdf
from utils.pdf_processor import extract_text_from_pdf, extract_sections_from_pdf, PDFProcessor
from utils.vector_db import create_vector_db, search_vector_db, list_vector_dbs

import os
from pathlib import Path
from typing import List

from config.settings import (
    DATA_DIR, PAPERS_DIR, VECTOR_DB_DIR, PDF_STORAGE_DIR,
    OUTPUT_DIR, LOGS_DIR
)

def ensure_directories_exist():
    """
    Ensure all required directories exist.
    """
    directories = [
        DATA_DIR,
        PAPERS_DIR,
        VECTOR_DB_DIR,
        PDF_STORAGE_DIR,
        OUTPUT_DIR,
        LOGS_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")

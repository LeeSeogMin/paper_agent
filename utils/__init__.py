"""
유틸리티 패키지
논문 작성 AI 에이전트 시스템에서 사용되는 다양한 유틸리티 함수를 제공합니다.
"""

from utils.logger import logger, configure_logging
# 모듈만 노출하고 개별 함수는 노출하지 않음
import utils.api_clients
import utils.pdf_processor
import utils.vector_db

# 학술 검색 통합 인터페이스 - 새로 추가됨
import utils.academic_search
import utils.rag_integration
import utils.serpapi_scholar
import utils.openalex_api

# 원래 이 파일에 정의되어 있던 함수 사용 (외부 파일로 옮기는 코드 제거)
# from utils._directory_utils import ensure_directories_exist

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

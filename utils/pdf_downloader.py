"""
PDF 다운로드 및 관리 유틸리티
"""

import os
import requests
import hashlib
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from utils.logger import logger

# PDF 저장 기본 경로
PDF_STORAGE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "pdfs")

def ensure_pdf_directory():
    """PDF 저장 디렉토리 확인 및 생성"""
    os.makedirs(PDF_STORAGE_PATH, exist_ok=True)

def download_pdf(url: str, output_path: Optional[str] = None) -> Optional[str]:
    """
    PDF URL에서 파일을 다운로드하여 로컬에 저장
    
    Args:
        url: PDF 파일의 URL
        output_path: 저장할 경로 (없으면 자동 생성)
        
    Returns:
        저장된 파일의 경로 또는 실패 시 None
    """
    logger.info(f"PDF 다운로드 시작: {url}")
    
    try:
        # URL 유효성 확인
        if not url or not url.startswith(('http://', 'https://')):
            logger.warning(f"유효하지 않은 URL: {url}")
            return None
        
        # 출력 경로가 지정되지 않은 경우 자동 생성
        if not output_path:
            ensure_pdf_directory()
            
            # URL에서 파일명 추출 시도
            url_path = urlparse(url).path
            filename = os.path.basename(url_path)
            
            # 파일명이 없거나 .pdf로 끝나지 않는 경우 해시 기반 이름 생성
            if not filename or not filename.lower().endswith('.pdf'):
                url_hash = hashlib.md5(url.encode()).hexdigest()
                filename = f"{url_hash}.pdf"
            
            output_path = os.path.join(PDF_STORAGE_PATH, filename)
        
        # 이미 파일이 존재하는지 확인
        if os.path.exists(output_path):
            logger.info(f"파일이 이미 존재함: {output_path}")
            return output_path
        
        # PDF 다운로드
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()  # HTTP 오류 발생 시 예외 발생
        
        # 콘텐츠 타입 확인 (PDF인지)
        content_type = response.headers.get('Content-Type', '')
        if 'application/pdf' not in content_type and not url.lower().endswith('.pdf'):
            logger.warning(f"PDF가 아닌 콘텐츠: {content_type}, URL: {url}")
        
        # 파일 저장
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"PDF 다운로드 완료: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"PDF 다운로드 실패: {str(e)}", exc_info=True)
        return None

def get_local_pdf_path(paper_id: str, url: str = None) -> Optional[str]:
    """
    로컬에 저장된 PDF 파일 경로 조회 또는 다운로드
    
    Args:
        paper_id: 논문 ID (파일명 생성에 사용)
        url: PDF 다운로드 URL (없으면 로컬 파일만 검색)
        
    Returns:
        로컬 PDF 파일 경로 또는 None
    """
    ensure_pdf_directory()
    
    # 가능한 파일명 생성
    filename = f"{paper_id}.pdf"
    local_path = os.path.join(PDF_STORAGE_PATH, filename)
    
    # 로컬 파일 존재 여부 확인
    if os.path.exists(local_path):
        return local_path
    
    # URL이 제공된 경우 다운로드 시도
    if url:
        return download_pdf(url, local_path)
    
    return None 
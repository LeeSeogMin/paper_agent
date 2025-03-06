"""
API client utilities for accessing external research services.

This module provides functions to interact with various academic research APIs
such as Google Scholar, CrossRef, and arXiv.
"""

import os
import json
import time
import random
import requests
import re
from typing import List, Dict, Any, Optional, Union
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from utils.logger import logger
from config.api_keys import (
    GOOGLE_SCHOLAR_API_KEY,
    GOOGLE_CSE_ID
)
from config.settings import MAX_RETRIES


class APIRateLimitError(Exception):
    """API 속도 제한 오류"""
    pass


class APIAuthenticationError(Exception):
    """API 인증 오류"""
    pass


class APIResponseError(Exception):
    """API 응답 오류"""
    pass


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type((requests.exceptions.RequestException, APIRateLimitError)),
    reraise=True
)
def search_semantic_scholar(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Semantic Scholar 검색 (가짜 결과 반환)
    
    Args:
        query: 검색 쿼리
        limit: 반환할 최대 결과 수
        
    Returns:
        List[Dict[str, Any]]: 검색 결과 목록
    """
    logger.info(f"Semantic Scholar 검색 (가짜 결과): '{query}', 최대 {limit}개 결과")
    return _generate_mock_results(query, limit)


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type((requests.exceptions.RequestException, APIRateLimitError)),
    reraise=True
)
def search_google_scholar(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Google Scholar API를 사용하여 학술 논문 검색
    
    Args:
        query: 검색 쿼리
        limit: 반환할 최대 결과 수
        
    Returns:
        List[Dict[str, Any]]: 검색 결과 목록
    """
    logger.info(f"Google Scholar 검색: '{query}', 최대 {limit}개 결과")
    
    if not GOOGLE_SCHOLAR_API_KEY or not GOOGLE_CSE_ID:
        logger.warning("Google Scholar API 키 또는 CSE ID가 없습니다.")
        return []
    
    base_url = "https://www.googleapis.com/customsearch/v1"
    
    params = {
        "key": GOOGLE_SCHOLAR_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "num": min(limit, 10),  # Google API는 한 번에 최대 10개 결과
        "start": 1
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        if "items" in data:
            for item in data["items"]:
                result = {
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "authors": [],  # Google Scholar API에서는 저자 정보를 직접 제공하지 않음
                    "year": "",  # 연도 정보도 직접 제공하지 않음
                    "source": "google_scholar"
                }
                
                # 제목에서 연도 추출 시도
                year_match = re.search(r'\b(19|20)\d{2}\b', item.get("title", ""))
                if year_match:
                    result["year"] = year_match.group(0)
                
                results.append(result)
        
        logger.info(f"Google Scholar 검색 결과: {len(results)}개 논문 찾음")
        return results
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Google Scholar API 요청 오류: {str(e)}")
        raise


def search_academic_papers(query: str, limit: int = 10, sources: List[str] = None) -> List[Dict[str, Any]]:
    """
    여러 소스에서 학술 논문 검색
    
    Args:
        query: 검색 쿼리
        limit: 반환할 최대 결과 수
        sources: 사용할 소스 목록 (기본값: ["semantic_scholar", "google_scholar"])
        
    Returns:
        List[Dict[str, Any]]: 검색 결과 목록
    """
    logger.info(f"학술 논문 검색: '{query}', 최대 {limit}개 결과")
    
    if sources is None:
        sources = ["semantic_scholar", "google_scholar"]
    
    results = []
    errors = []
    
    # 각 소스에서 검색
    for source in sources:
        try:
            if source == "semantic_scholar":
                source_results = search_semantic_scholar(query, limit=limit)
            elif source == "google_scholar":
                source_results = search_google_scholar(query, limit=limit)
            else:
                logger.warning(f"알 수 없는 소스: {source}")
                continue
            
            # 소스 정보 추가
            for result in source_results:
                result["source"] = source
            
            results.extend(source_results)
            
        except Exception as e:
            logger.error(f"{source} 검색 중 오류: {str(e)}")
            errors.append({"source": source, "error": str(e)})
    
    # 결과 정렬 (최신 논문 우선)
    results.sort(key=lambda x: x.get("year", 0), reverse=True)
    
    # 최대 결과 수로 제한
    results = results[:limit]
    
    logger.info(f"총 {len(results)}개 결과 찾음")
    return results


def _generate_mock_results(query: str, limit: int) -> List[Dict[str, Any]]:
    """
    테스트용 가짜 검색 결과 생성 (Semantic Scholar용)
    
    Args:
        query: 검색 쿼리
        limit: 결과 수
        
    Returns:
        List[Dict[str, Any]]: 가짜 검색 결과
    """
    logger.debug(f"가짜 검색 결과 생성: '{query}', {limit}개")
    
    results = []
    current_year = 2023
    
    for i in range(limit):
        # 무작위 연도 (최근 10년)
        year = current_year - random.randint(0, 10)
        
        # 가짜 저자 목록
        authors = [
            {"name": f"Author {j+1}"} for j in range(random.randint(1, 4))
        ]
        
        # 가짜 결과 생성
        result = {
            "paperId": f"mock-paper-{i}",
            "title": f"Research on {query}: Approach {i+1}",
            "abstract": f"This paper explores {query} using novel methods. We present findings that demonstrate significant improvements over previous approaches.",
            "year": year,
            "authors": authors,
            "venue": f"Journal of {query.title()} Research",
            "url": f"https://example.com/papers/{i}",
            "isOpenAccess": random.choice([True, False]),
            "citationCount": random.randint(0, 100),
            "referenceCount": random.randint(10, 50)
        }
        
        results.append(result)
    
    return results


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    reraise=True
)
def download_pdf(url: str, output_path: str) -> str:
    """
    URL에서 PDF 다운로드
    
    Args:
        url: PDF URL
        output_path: 저장할 경로
        
    Returns:
        str: 저장된 PDF 파일 경로
    """
    logger.info(f"PDF 다운로드 중: {url}")
    
    try:
        # 디렉토리 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 실제 다운로드 대신 가짜 PDF 생성 (테스트용)
        with open(output_path, 'w') as f:
            f.write(f"Mock PDF content for {url}\n")
            f.write("This is a placeholder for actual PDF content.\n")
            f.write("In a real implementation, this would be binary PDF data.\n")
        
        logger.info(f"PDF 저장됨: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"PDF 다운로드 오류: {str(e)}")
        raise
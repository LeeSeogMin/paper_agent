import os
import json
import requests
from typing import List, Dict, Any, Optional
from utils.logger import logger

def google_search(query: str, num_results: int = 10) -> List[Dict[str, Any]]:
    """
    Google Custom Search API를 사용하여 검색 수행
    
    Args:
        query: 검색 쿼리
        num_results: 반환할 결과 수 (최대 10)
        
    Returns:
        List[Dict[str, Any]]: 검색 결과 목록
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    
    if not api_key or not cse_id:
        logger.warning("Google API 키 또는 CSE ID가 설정되지 않았습니다.")
        return []
    
    # 결과 수 제한 (API 제한)
    num_results = min(num_results, 10)
    
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cse_id,
        "q": query,
        "num": num_results
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        results = response.json()
        
        if "items" not in results:
            logger.warning(f"검색 결과가 없습니다: {query}")
            return []
        
        formatted_results = []
        for item in results["items"]:
            formatted_results.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "source": "google"
            })
        
        return formatted_results
    
    except Exception as e:
        logger.error(f"Google 검색 오류: {str(e)}")
        return []

def fallback_search(query: str, num_results: int = 10) -> List[Dict[str, Any]]:
    """
    여러 검색 방법을 시도하는 폴백 검색 함수
    
    Args:
        query: 검색 쿼리
        num_results: 반환할 결과 수
        
    Returns:
        List[Dict[str, Any]]: 검색 결과 목록
    """
    # 1. Semantic Scholar API 시도
    try:
        from utils.api_clients import semantic_scholar_search
        results = semantic_scholar_search(query, limit=num_results)
        if results and len(results) > 0:
            logger.info(f"Semantic Scholar 검색 성공: {len(results)} 결과")
            return results
    except Exception as e:
        logger.warning(f"Semantic Scholar 검색 실패: {str(e)}")
    
    # 2. Google 검색 시도
    try:
        results = google_search(query, num_results=num_results)
        if results and len(results) > 0:
            logger.info(f"Google 검색 성공: {len(results)} 결과")
            return results
    except Exception as e:
        logger.warning(f"Google 검색 실패: {str(e)}")
    
    # 3. 모의 결과 반환 (모든 검색이 실패한 경우)
    logger.warning(f"모든 검색 방법 실패, 모의 결과 반환: {query}")
    return [
        {
            "title": f"모의 검색 결과 1: {query}",
            "link": "https://example.com/result1",
            "snippet": "이 결과는 실제 검색 API가 실패하여 생성된 모의 결과입니다.",
            "source": "mock"
        },
        {
            "title": f"모의 검색 결과 2: {query}",
            "link": "https://example.com/result2",
            "snippet": "이 결과는 실제 검색 API가 실패하여 생성된 모의 결과입니다.",
            "source": "mock"
        }
    ]

def academic_search(query: str, num_results: int = 10) -> List[Dict[str, Any]]:
    """
    학술 검색 수행 (여러 소스 통합)
    
    Args:
        query: 검색 쿼리
        num_results: 반환할 결과 수
        
    Returns:
        List[Dict[str, Any]]: 검색 결과 목록
    """
    # 학술 검색 쿼리 개선
    academic_query = f"{query} research paper academic"
    
    # 폴백 검색 사용
    results = fallback_search(academic_query, num_results)
    
    # 결과 필터링 및 정렬 (학술적 관련성 향상)
    # 학술 사이트 도메인 목록
    academic_domains = [
        "scholar.google.com", "arxiv.org", "researchgate.net", 
        "academia.edu", "sciencedirect.com", "ieee.org",
        "acm.org", "springer.com", "nature.com", "science.org",
        "jstor.org", "pubmed.ncbi.nlm.nih.gov", "ssrn.com"
    ]
    
    # 학술 도메인 결과 우선 정렬
    def is_academic(result):
        link = result.get("link", "").lower()
        return any(domain in link for domain in academic_domains)
    
    # 학술 결과 우선 정렬
    results.sort(key=lambda x: (0 if is_academic(x) else 1, 
                               0 if x.get("source") == "semantic_scholar" else 1))
    
    return results[:num_results] 
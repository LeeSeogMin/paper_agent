"""
검색 유틸리티 함수 (academic_search.py로 기능 통합됨)
"""

from utils.academic_search import AcademicSearchManager
from utils.logger import logger

# 호환성을 위한 함수들
def google_search(query: str, num_results: int = 10) -> list:
    """
    Google 검색 (academic_search.py로 기능 이전됨)
    """
    logger.info("search_utils.google_search 함수는 AcademicSearchManager로 이전되었습니다.")
    search_manager = AcademicSearchManager()
    return search_manager.google_search(query, num_results)

def search_arxiv(query: str, max_results: int = 10) -> list:
    """
    arXiv 검색 (academic_search.py로 기능 이전됨)
    """
    logger.info("search_arxiv 함수는 AcademicSearchManager로 이전되었습니다.")
    search_manager = AcademicSearchManager()
    return search_manager.search_arxiv(query, max_results)

def search_crossref(query: str, max_results: int = 10, filter: str = None) -> list:
    """
    Crossref 검색 (academic_search.py로 기능 이전됨)
    """
    logger.info("search_crossref 함수는 AcademicSearchManager로 이전되었습니다.")
    search_manager = AcademicSearchManager()
    return search_manager.search_crossref(query, max_results, filter)

# 누락된 academic_search 함수 추가
def academic_search(query: str, max_results: int = 10, sources: list = None, language: str = None) -> list:
    """
    통합 학술 검색 (academic_search.py로 기능 이전됨)
    
    Args:
        query: 검색 쿼리
        max_results: 최대 결과 수
        sources: 검색 소스 목록
        language: 언어 설정 (새로 추가)
        
    Returns:
        list: 검색 결과 목록
    """
    logger.info("academic_search 함수는 AcademicSearchManager로 이전되었습니다.")
    
    # 지역 임포트로 순환 참조 방지
    from utils.academic_search import AcademicSearchManager
    search_manager = AcademicSearchManager()
    
    # AcademicSearchManager.search 메서드는 다른 형식을 반환할 수 있으므로
    # 원래 academic_search 함수와 호환되는 형식으로 변환
    if sources is None:
        sources = "all"
    else:
        # 문자열 목록을 academic_search.py가 기대하는 형식으로 변환
        sources = ",".join(sources)
    
    try:
        # language 매개변수는 무시 (필요하다면 search 함수에 추가 가능)
        results = search_manager.search_with_fallback(
            query=query,
            limit=max_results
        )
        
        # 원래 함수의 반환 형식으로 변환
        return results.get("results", [])
    except Exception as e:
        logger.error(f"학술 검색 중 오류: {str(e)}")
        return []

# fallback_search 함수 수정
def fallback_search(query: str, max_results: int = 5, language: str = None) -> list:
    """
    대체 검색 메서드 (academic_search.py로 기능 이전됨)
    
    Args:
        query: 검색 쿼리
        max_results: 최대 결과 수
        language: 언어 설정 (기본값: None)
        
    Returns:
        list: 검색 결과 목록
    """
    logger.info("fallback_search 함수는 AcademicSearchManager로 이전되었습니다.")
    
    # 지역 임포트로 순환 참조 방지
    from utils.academic_search import AcademicSearchManager
    
    try:
        # language 매개변수는 일단 무시 (필요하다면 추가 가능)
        search_manager = AcademicSearchManager()
        results = search_manager.search_with_fallback(
            query=query,
            limit=max_results
        )
        return results.get("results", [])
    except Exception as e:
        logger.error(f"대체 검색 중 오류: {str(e)}")
        # 오류 발생 시 빈 결과 반환
        return []

# 다른 함수들도 필요에 따라 래퍼 함수로 구현 
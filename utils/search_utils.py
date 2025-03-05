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
    logger.info(f"search_utils.py의 google_search 함수 호출: query={query}, num_results={num_results}")
    
    # 호출 스택 추적을 위한 로깅
    import traceback
    call_stack = traceback.format_stack()
    logger.debug(f"search_utils.py google_search 호출 스택:\n{''.join(call_stack)}")
    
    try:
        search_manager = AcademicSearchManager()
        logger.info("AcademicSearchManager 인스턴스 생성 완료, google_search_compat 호출 예정")
        results = search_manager.google_search_compat(query, num_results)
        logger.info(f"google_search_compat 호출 완료, 결과 개수: {len(results)}")
        return results
    except Exception as e:
        logger.error(f"search_utils.py Google 검색 중 오류: {str(e)}")
        logger.error(f"오류 상세 정보: {traceback.format_exc()}")
        return []

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
    logger.info(f"academic_search 함수 호출: query={query}, max_results={max_results}, sources={sources}")
    
    # 지역 임포트로 순환 참조 방지
    from utils.academic_search import AcademicSearchManager
    search_manager = AcademicSearchManager()
    
    # 소스 결정
    source = "all"
    if sources:
        if isinstance(sources, list) and len(sources) == 1:
            source = sources[0]
        else:
            # 여러 소스가 지정된 경우 각각 검색하고 결과 합침
            all_results = []
            for src in sources:
                try:
                    src_results = search_manager.search(
                        query=query,
                        source=src,
                        limit=max_results
                    )
                    if src_results.get("results"):
                        all_results.extend(src_results["results"])
                except Exception as e:
                    logger.error(f"{src} 검색 중 오류: {str(e)}")
            return all_results
    
    try:
        # 직접 search 메서드 호출
        results = search_manager.search(
            query=query,
            source=source,
            limit=max_results
        )
        
        # 원래 함수의 반환 형식으로 변환
        return results.get("results", [])
    except Exception as e:
        logger.error(f"학술 검색 중 오류: {str(e)}")
        return []

# 다른 함수들도 필요에 따라 래퍼 함수로 구현 
"""
구글 검색 및 학술 검색 기능 (academic_search.py로 기능 통합됨)
"""

from utils.logger import logger

# academic_search를 지역 변수로 가져와 순환 참조 방지
def google_search(query, num_results=10, language='en'):
    """
    구글 검색을 수행하고 결과를 반환 (academic_search.py로 기능 이전됨)
    """
    logger.info(f"search.py의 google_search 함수 호출: query={query}, num_results={num_results}, language={language}")
    
    # 호출 스택 추적을 위한 로깅
    import traceback
    call_stack = traceback.format_stack()
    logger.debug(f"search.py google_search 호출 스택:\n{''.join(call_stack)}")
    
    try:
        # AcademicSearchManager 사용
        from utils.academic_search import AcademicSearchManager
        search_manager = AcademicSearchManager()
        logger.info("AcademicSearchManager 인스턴스 생성 완료, google_search_compat 호출 예정")
        results = search_manager.google_search_compat(query, num_results, language)
        logger.info(f"google_search_compat 호출 완료, 결과 개수: {len(results)}")
        return results
    except Exception as e:
        logger.error(f"search.py Google 검색 중 오류: {str(e)}")
        logger.error(f"오류 상세 정보: {traceback.format_exc()}")
        return []

def test_academic_search(query):
    """
    학술 검색 기능 테스트 (academic_search.py로 기능 이전됨)
    """
    logger.info("test_academic_search 함수는 AcademicSearchManager로 이전되었습니다.")
    
    # 지역 임포트로 순환 참조 방지
    from utils.academic_search import AcademicSearchManager
    search_manager = AcademicSearchManager()
    return search_manager.test_academic_search(query)

# 누락된 search_academic_resources 함수 추가
def search_academic_resources(query, api_keys=None, max_results=10, format_type="plain"):
    """
    다양한 학술 리소스에서 검색을 수행하고 결과를 반환 (academic_search.py로 기능 이전됨)
    
    Args:
        query: 검색 쿼리
        api_keys: API 키 (사용하지 않음)
        max_results: 최대 결과 수
        format_type: 결과 형식 ("plain", "markdown" 등)
        
    Returns:
        검색 결과 목록 또는 포맷팅된 문자열
    """
    logger.info("search_academic_resources 함수는 AcademicSearchManager로 이전되었습니다.")
    
    # 지역 임포트로 순환 참조 방지
    from utils.academic_search import AcademicSearchManager
    search_manager = AcademicSearchManager()
    
    try:
        # 통합 검색 실행
        results = search_manager.search_with_fallback(
            query=query,
            limit=max_results
        )
        
        # 요청된 형식에 따라 결과 반환
        if format_type.lower() == "markdown":
            return search_manager.format_results_as_markdown(results)
        else:
            return results.get("results", [])
    except Exception as e:
        logger.error(f"학술 리소스 검색 중 오류: {str(e)}")
        return []

# 기존 함수와의 호환성을 위한 wrapper 함수
def search_crossref(query, max_results=10):
    """
    Crossref API를 사용한 학술 논문 검색 (academic_search.py로 기능 이전됨)
    """
    logger.info("search_crossref 함수는 AcademicSearchManager로 이전되었습니다.")
    
    # 지역 임포트로 순환 참조 방지
    from utils.academic_search import AcademicSearchManager
    search_manager = AcademicSearchManager()
    return search_manager.search_crossref(query, max_results)

def search_arxiv(query, max_results=10):
    """
    arXiv API를 사용한 논문 검색 (academic_search.py로 기능 이전됨)
    """
    logger.info("search_arxiv 함수는 AcademicSearchManager로 이전되었습니다.")
    
    # 지역 임포트로 순환 참조 방지
    from utils.academic_search import AcademicSearchManager
    search_manager = AcademicSearchManager()
    return search_manager.search_arxiv(query, max_results)

# 다른 함수들도 필요에 따라 래퍼 함수로 구현
    return formatted_results
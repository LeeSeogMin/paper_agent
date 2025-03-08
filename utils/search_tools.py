"""
논문 검색을 위한 유틸리티 모듈
다양한 학술 검색 서비스(Google Scholar, OpenAlex 등)를 사용하여 논문을 검색하는 기능을 제공합니다.
"""

from utils.serpapi_scholar import ScholarSearchTool
from utils.openalex_api import OpenAlexTool

# Google Scholar 검색 기능
def search_google_scholar(query, num_results=5, year_start=None, year_end=None):
    """
    Google Scholar에서 학술 논문을 검색합니다.
    
    Args:
        query (str): 검색 쿼리
        num_results (int): 반환할 결과 수
        year_start (int, optional): 검색 시작 연도
        year_end (int, optional): 검색 종료 연도
        
    Returns:
        tuple: (원본 결과, 포맷된 결과)
    """
    scholar_tool = ScholarSearchTool()
    results = scholar_tool.search_scholar(
        query=query,
        num_results=num_results,
        year_start=year_start,
        year_end=year_end
    )
    
    formatted_results = scholar_tool.format_results(results)
    return results, formatted_results

# OpenAlex API 검색 기능
def search_openalex(query, limit=10, filter_options=None):
    """
    OpenAlex API를 사용하여 학술 논문을 검색합니다.
    
    Args:
        query (str): 검색 쿼리
        limit (int): 반환할 최대 결과 수
        filter_options (dict, optional): 필터링 옵션
        
    Returns:
        tuple: (원본 결과, 포맷된 결과)
    """
    openalex_tool = OpenAlexTool()
    # 선택적으로 이메일 설정 (Polite Pool 사용)
    # openalex_tool.set_email("your-email@example.com")
    
    results = openalex_tool.search_works(
        query=query,
        limit=limit,
        filter_options=filter_options
    )
    
    formatted_results = openalex_tool.format_paper_results(results)
    return results, formatted_results

# OpenAlex API로 저자 검색 
def search_openalex_authors(author_name, limit=10):
    """
    OpenAlex API를 사용하여 저자를 검색합니다.
    
    Args:
        author_name (str): 저자 이름
        limit (int): 반환할 최대 결과 수
        
    Returns:
        list: 저자 검색 결과
    """
    openalex_tool = OpenAlexTool()
    results = openalex_tool.search_authors(
        query=author_name,
        limit=limit
    )
    return results 
from googleapiclient.discovery import build
from typing import List, Dict
from utils.logger import logger

# Google API 설정
GOOGLE_API_KEY = "AIzaSyD4QKVRA96r0lkIxUV5LrJwI732HA9rhtA"
GOOGLE_CSE_ID = "075b324570149415f"

def google_search(query, num_results=10):
    """
    Google 검색을 수행하고 결과를 반환합니다.
    
    Args:
        query (str): 검색 쿼리
        num_results (int): 반환할 최대 결과 수
        
    Returns:
        List[Dict]: 검색 결과 목록
    """
    logger.info(f"Google 검색 시작: '{query}'")
    
    try:
        # 디버깅을 위한 로그 추가
        logger.debug(f"검색 쿼리 전처리 전: {query}")
        
        # 쿼리 전처리 - 따옴표 제거 및 길이 제한
        if query.startswith('"') and query.endswith('"'):
            query = query[1:-1]
        
        # 쿼리가 너무 길면 잘라내기
        if len(query) > 100:
            query = query[:100]
        
        logger.debug(f"검색 쿼리 전처리 후: {query}")
        
        # 여기에 실제 Google 검색 로직 구현
        # 예: requests를 사용한 검색 또는 서드파티 라이브러리 사용
        
        # 테스트를 위한 더미 결과 반환 (실제 구현에서는 제거)
        dummy_results = [
            {
                "title": f"Sample result for: {query} - 1",
                "url": "https://example.com/1",
                "abstract": "This is a sample abstract for testing purposes."
            },
            {
                "title": f"Sample result for: {query} - 2",
                "url": "https://example.com/2",
                "abstract": "Another sample abstract for testing purposes."
            }
        ]
        
        logger.info(f"Google 검색 결과 {len(dummy_results)}개 찾음")
        return dummy_results
        
    except Exception as e:
        logger.error(f"Google 검색 중 오류 발생: {str(e)}", exc_info=True)
        # 오류 발생 시 빈 리스트 대신 더미 데이터 반환
        return [
            {
                "title": "Error fallback result",
                "url": "https://example.com/error",
                "abstract": f"Search error occurred: {str(e)}. This is a fallback result."
            }
        ]
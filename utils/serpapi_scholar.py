import os
import requests
from typing import List, Dict, Any, Optional

class ScholarSearchTool:
    """
    SerpApi를 사용하여 Google Scholar 검색을 수행하는 도구
    """
    
    def __init__(self):
        self.api_key = os.getenv("SerpApi_key")
        if not self.api_key:
            print("경고: SerpApi_key가 환경변수에 설정되지 않았습니다. Google Scholar 검색이 작동하지 않을 수 있습니다.")
        self.base_url = "https://serpapi.com/search.json"
    
    def search_scholar(self, 
                       query: str, 
                       num_results: int = 5, 
                       year_start: Optional[int] = None,
                       year_end: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Google Scholar 검색을 수행하고 결과를 반환합니다.
        
        Args:
            query: 검색 쿼리
            num_results: 반환할 결과 수 (기본값: 5)
            year_start: 검색 시작 연도 (선택 사항)
            year_end: 검색 종료 연도 (선택 사항)
            
        Returns:
            검색 결과 목록 (제목, 링크, 요약, 저자 등 포함)
        """
        if not self.api_key:
            print("SerpApi_key가 설정되지 않아 Google Scholar 검색을 수행할 수 없습니다.")
            return []
        
        params = {
            "engine": "google_scholar",
            "q": query,
            "api_key": self.api_key,
            "num": num_results
        }
        
        # 연도 필터 추가 (지정된 경우)
        if year_start and year_end:
            params["as_ylo"] = year_start
            params["as_yhi"] = year_end
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for result in data.get("organic_results", []):
                paper_info = {
                    "title": result.get("title", ""),
                    "link": result.get("link", ""),
                    "snippet": result.get("snippet", ""),
                    "publication_info": result.get("publication_info", {}).get("summary", ""),
                    "authors": result.get("publication_info", {}).get("authors", []),
                    "year": result.get("publication_info", {}).get("year", "")
                }
                results.append(paper_info)
            
            return results
        
        except Exception as e:
            print(f"Google Scholar 검색 중 오류 발생: {e}")
            return []
    
    def format_results(self, results: List[Dict[str, Any]]) -> str:
        """
        검색 결과를 읽기 쉬운 문자열로 포맷팅합니다.
        """
        if not results:
            return "검색 결과가 없습니다."
        
        formatted = ""
        for i, result in enumerate(results, 1):
            formatted += f"{i}. {result['title']}\n"
            formatted += f"   링크: {result['link']}\n"
            if result.get('publication_info'):
                formatted += f"   출판 정보: {result['publication_info']}\n"
            if result.get('snippet'):
                formatted += f"   요약: {result['snippet']}\n"
            formatted += "\n"
        
        return formatted 
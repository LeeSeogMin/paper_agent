import requests
from typing import List, Dict, Any, Optional

class OpenAlexTool:
    """
    OpenAlex API를 사용하여 학술 논문을 검색하는 도구
    """
    
    def __init__(self):
        self.base_url = "https://api.openalex.org"
        self.email = None  # 선택적으로 이메일을 설정하여 Polite Pool 사용 가능
    
    def set_email(self, email: str):
        """
        Polite Pool 사용을 위한 이메일 설정
        """
        self.email = email
    
    def search_works(self, 
                    query: str, 
                    limit: int = 10, 
                    filter_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        논문(works) 검색을 수행하고 결과를 반환합니다.
        
        Args:
            query: 검색 쿼리
            limit: 반환할 결과 수 (기본값: 10)
            filter_options: 추가 필터 옵션 (출판 연도, 저자 등)
            
        Returns:
            검색 결과 데이터
        """
        endpoint = f"{self.base_url}/works"
        
        params = {
            "search": query,
            "per_page": limit
        }
        
        # 추가 필터 적용
        if filter_options:
            for key, value in filter_options.items():
                params[key] = value
        
        # Polite Pool 적용
        if self.email:
            endpoint = f"{endpoint}?mailto={self.email}"
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            print(f"OpenAlex API 검색 중 오류 발생: {e}")
            return {"results": []}
    
    def search_authors(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        저자 검색을 수행하고 결과를 반환합니다.
        
        Args:
            query: 검색 쿼리 (저자 이름)
            limit: 반환할 결과 수 (기본값: 10)
            
        Returns:
            검색된 저자 정보
        """
        endpoint = f"{self.base_url}/authors"
        
        params = {
            "search": query,
            "per_page": limit
        }
        
        # Polite Pool 적용
        if self.email:
            endpoint = f"{endpoint}?mailto={self.email}"
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            print(f"OpenAlex API 저자 검색 중 오류 발생: {e}")
            return {"results": []}
    
    def get_paper_details(self, work_id: str) -> Dict[str, Any]:
        """
        특정 논문의 상세 정보를 가져옵니다.
        
        Args:
            work_id: OpenAlex 논문 ID
            
        Returns:
            논문 상세 정보
        """
        endpoint = f"{self.base_url}/works/{work_id}"
        
        # Polite Pool 적용
        if self.email:
            endpoint = f"{endpoint}?mailto={self.email}"
        
        try:
            response = requests.get(endpoint)
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            print(f"OpenAlex API 논문 상세 정보 가져오기 중 오류 발생: {e}")
            return {}
    
    def format_paper_results(self, results: Dict[str, Any]) -> str:
        """
        논문 검색 결과를 읽기 쉬운 문자열로 포맷팅합니다.
        """
        if not results or not results.get("results"):
            return "검색 결과가 없습니다."
        
        formatted = ""
        for i, paper in enumerate(results["results"], 1):
            formatted += f"{i}. {paper.get('title', '제목 없음')}\n"
            
            # 저자 정보
            if paper.get("authorships"):
                authors = [author.get("author", {}).get("display_name", "") 
                          for author in paper["authorships"]]
                formatted += f"   저자: {', '.join(authors)}\n"
            
            # DOI
            if paper.get("doi"):
                formatted += f"   DOI: {paper['doi']}\n"
            
            # 출판 연도
            if paper.get("publication_year"):
                formatted += f"   출판 연도: {paper['publication_year']}\n"
            
            # 초록
            if paper.get("abstract_inverted_index"):
                # OpenAlex는 abstract를 inverted_index 형태로 제공함
                # 여기서는 초록이 있다는 것만 표시
                formatted += f"   초록: 있음\n"
            
            # 저널 정보
            if paper.get("primary_location", {}).get("source", {}).get("display_name"):
                journal = paper["primary_location"]["source"]["display_name"]
                formatted += f"   저널: {journal}\n"
            
            formatted += "\n"
        
        return formatted 
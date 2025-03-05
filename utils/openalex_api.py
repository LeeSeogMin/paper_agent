import requests
from typing import List, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class OpenAlexTool:
    """
    OpenAlex API를 사용하여 학술 논문을 검색하는 도구
    확장된 기능: 다양한 ID 형식 지원, Merged Entity 처리, Dehydrated 객체 지원
    """
    
    # 지원하는 ID 타입 매핑
    ID_TYPES = {
        "works": {
            "doi": "https://doi.org/",
            "pmid": "pmid:",
            "pmcid": "pmcid:",
            "openalex": "https://openalex.org/",
        },
        "authors": {
            "orcid": "https://orcid.org/",
            "openalex": "https://openalex.org/",
        },
        "sources": {
            "issn": "issn:",
            "issn_l": "issn_l:",
            "openalex": "https://openalex.org/",
        },
        "institutions": {
            "ror": "https://ror.org/",
            "grid": "grid:",
            "openalex": "https://openalex.org/",
        },
        "concepts": {
            "wikidata": "https://www.wikidata.org/wiki/",
            "openalex": "https://openalex.org/",
        },
        "publishers": {
            "wikidata": "https://www.wikidata.org/wiki/",
            "openalex": "https://openalex.org/",
        },
    }
    
    def __init__(self):
        self.base_url = "https://api.openalex.org"
        self.email = None  # 선택적으로 이메일을 설정하여 Polite Pool 사용 가능
    
    def set_email(self, email: str):
        """
        Polite Pool 사용을 위한 이메일 설정
        
        Args:
            email (str): 사용자 이메일
        """
        self.email = email
        logger.info(f"OpenAlex Polite Pool 활성화: {email}")
    
    def _build_url(self, endpoint: str) -> str:
        """
        API 요청 URL 생성 (Polite Pool 고려)
        
        Args:
            endpoint: API 엔드포인트
        
        Returns:
            구성된 URL
        """
        url = f"{self.base_url}/{endpoint}"
        
        # Polite Pool 적용
        if self.email:
            if "?" in url:
                url += f"&mailto={self.email}"
            else:
                url += f"?mailto={self.email}"
        
        return url
    
    def _normalize_id(self, entity_type: str, entity_id: str) -> str:
        """
        다양한 형식의 ID를 OpenAlex API에서 사용 가능한 형태로 정규화
        
        Args:
            entity_type: 엔티티 유형 (works, authors 등)
            entity_id: 정규화할 ID
        
        Returns:
            정규화된 ID
        """
        # 이미 URL 형식인 경우
        if entity_id.startswith("http"):
            return entity_id
        
        # namespace:id 형식 처리
        if ":" in entity_id:
            namespace, id_value = entity_id.split(":", 1)
            
            if namespace in self.ID_TYPES.get(entity_type, {}):
                prefix = self.ID_TYPES[entity_type][namespace]
                if prefix.endswith("/"):
                    return f"{prefix}{id_value}"
                else:
                    return f"{prefix}:{id_value}"
        
        # OpenAlex ID 처리 (W123456789 형식)
        first_char = entity_id[0].upper()
        if first_char in "WASICPF":
            # 이미 형식이 맞는 경우
            return entity_id
        
        # 기본 반환 (변환 불가능한 경우)
        return entity_id
    
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
        endpoint = "works"
        
        params = {
            "search": query,
            "per_page": limit
        }
        
        # 추가 필터 적용
        if filter_options:
            for key, value in filter_options.items():
                params[key] = value
        
        try:
            url = self._build_url(endpoint)
            logger.info(f"OpenAlex API 호출: {url}, 매개변수: {params}")
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # 응답 구조 검증 및 로깅
            logger.debug(f"OpenAlex API 응답 구조: {list(data.keys() if isinstance(data, dict) else [])}")
            
            # 결과가 없을 경우 빈 배열 반환
            if not isinstance(data, dict):
                logger.warning(f"OpenAlex 응답이 딕셔너리가 아님: {type(data)}")
                return {"results": []}
                
            if "results" not in data:
                logger.warning(f"OpenAlex 검색 결과 없음: {query}, 응답: {data}")
                return {"results": []}
            
            # 결과 항목 검증
            valid_results = []
            for i, item in enumerate(data.get("results", [])):
                if not isinstance(item, dict):
                    logger.warning(f"OpenAlex 결과 항목 {i}가 딕셔너리가 아님: {type(item)}")
                    continue
                    
                # 중요 필드 존재 여부 확인
                missing_fields = []
                for field in ["title", "authorships", "primary_location"]:
                    if field not in item:
                        missing_fields.append(field)
                
                if missing_fields:
                    logger.warning(f"OpenAlex 결과 항목 {i}에 필드 누락: {missing_fields}")
                    # 누락된 필드 초기화
                    for field in missing_fields:
                        if field == "title":
                            item["title"] = "제목 없음"
                        elif field == "authorships":
                            item["authorships"] = []
                        elif field == "primary_location":
                            item["primary_location"] = {}
                
                valid_results.append(item)
            
            # 검증된 결과로 업데이트
            data["results"] = valid_results
            
            return data
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.error(f"OpenAlex API 속도 제한 초과: {e}")
                return {"results": []}
            elif e.response.status_code in (301, 302):
                logger.info(f"OpenAlex ID 리디렉션 발생: {e.response.headers.get('Location')}")
                return self._handle_redirect(e.response.headers.get('Location', ''))
            else:
                logger.error(f"OpenAlex API HTTP 오류: {e}")
                return {"results": []}
                
        except Exception as e:
            logger.error(f"OpenAlex API 검색 중 오류 발생: {e}")
            return {"results": []}
    
    def _handle_redirect(self, redirect_url: str) -> Dict[str, Any]:
        """
        Merged Entity 리디렉션 처리
        
        Args:
            redirect_url: 리디렉션 URL
            
        Returns:
            리디렉션된 엔티티 데이터
        """
        try:
            if not redirect_url:
                return {"results": []}
                
            response = requests.get(redirect_url)
            response.raise_for_status()
            
            data = response.json()
            # 리디렉션된 데이터를 results 배열로 감싸서 반환
            return {"results": [data]}
            
        except Exception as e:
            logger.error(f"리디렉션 처리 중 오류 발생: {e}")
            return {"results": []}
    
    def get_entity_by_id(self, entity_type: str, entity_id: str) -> Dict[str, Any]:
        """
        ID로 단일 엔티티 조회
        
        Args:
            entity_type: 엔티티 유형 ('works', 'authors', 'sources', 'institutions', 'concepts', 'publishers')
            entity_id: 엔티티 ID (다양한 형식 지원)
            
        Returns:
            엔티티 정보 또는 에러 정보
        """
        normalized_id = self._normalize_id(entity_type, entity_id)
        endpoint = f"{entity_type}/{normalized_id}"
        
        try:
            url = self._build_url(endpoint)
            response = requests.get(url)
            
            # Merged Entity 리디렉션 처리
            if response.status_code in (301, 302):
                redirect_url = response.headers.get('Location')
                logger.info(f"엔티티 리디렉션: {entity_id} → {redirect_url}")
                return self._handle_redirect(redirect_url).get('results', [{}])[0]
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"엔티티 조회 HTTP 오류: {e}")
            return {"error": str(e), "status_code": e.response.status_code}
            
        except Exception as e:
            logger.error(f"엔티티 조회 중 오류 발생: {e}")
            return {"error": str(e)}
    
    def get_paper_details(self, work_id: str, dehydrated: bool = False) -> Dict[str, Any]:
        """
        논문 상세 정보 조회 (다양한 ID 형식 지원)
        
        Args:
            work_id: OpenAlex ID, DOI, PMID 등 다양한 형식 지원
            dehydrated: 경량화된 객체 반환 여부
            
        Returns:
            논문 상세 정보
        """
        # 쿼리 파라미터 구성
        params = {}
        if dehydrated:
            params["select"] = "id,title,publication_year,type,open_access"
            
        try:
            entity = self.get_entity_by_id("works", work_id)
            
            # 오류가 있는 경우
            if "error" in entity:
                return entity
                
            return entity
            
        except Exception as e:
            logger.error(f"논문 상세 정보 조회 중 오류 발생: {e}")
            return {"error": str(e)}
    
    def get_author_details(self, author_id: str, dehydrated: bool = False) -> Dict[str, Any]:
        """
        저자 상세 정보 조회
        
        Args:
            author_id: OpenAlex ID 또는 ORCID ID
            dehydrated: 경량화된 객체 반환 여부
            
        Returns:
            저자 상세 정보
        """
        # 쿼리 파라미터 구성
        params = {}
        if dehydrated:
            params["select"] = "id,display_name,works_count,cited_by_count"
            
        try:
            return self.get_entity_by_id("authors", author_id)
        except Exception as e:
            logger.error(f"저자 상세 정보 조회 중 오류 발생: {e}")
            return {"error": str(e)}
    
    def search_authors(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        저자 검색을 수행하고 결과를 반환합니다.
        
        Args:
            query: 검색 쿼리 (저자 이름)
            limit: 반환할 결과 수 (기본값: 10)
            
        Returns:
            검색된 저자 정보
        """
        endpoint = "authors"
        
        params = {
            "search": query,
            "per_page": limit
        }
        
        try:
            url = self._build_url(endpoint)
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            logger.error(f"OpenAlex API 저자 검색 중 오류 발생: {e}")
            return {"results": []}
    
    def get_related_works(self, work_id: str, limit: int = 10) -> Dict[str, Any]:
        """
        관련 논문 조회
        
        Args:
            work_id: 논문 ID
            limit: 결과 수
            
        Returns:
            관련 논문 목록
        """
        normalized_id = self._normalize_id("works", work_id)
        endpoint = f"works/{normalized_id}/related_works"
        
        params = {
            "per_page": limit
        }
        
        try:
            url = self._build_url(endpoint)
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            logger.error(f"관련 논문 조회 중 오류 발생: {e}")
            return {"results": []}
    
    def format_paper_results(self, results: Dict[str, Any]) -> str:
        """
        논문 검색 결과를 읽기 쉬운 문자열로 포맷팅합니다.
        """
        if not results:
            logger.warning("format_paper_results: 결과가 None 또는 빈 값")
            return "검색 결과가 없습니다."
            
        if not isinstance(results, dict):
            logger.warning(f"format_paper_results: 결과가 딕셔너리가 아님: {type(results)}")
            return "검색 결과 형식 오류"
            
        if "results" not in results:
            logger.warning(f"format_paper_results: 'results' 키가 없음: {list(results.keys())}")
            return "검색 결과가 없습니다."
            
        if not results.get("results"):
            return "검색 결과가 없습니다."
        
        formatted = ""
        for i, paper in enumerate(results["results"], 1):
            if not isinstance(paper, dict):
                logger.warning(f"format_paper_results: 논문 항목 {i}가 딕셔너리가 아님: {type(paper)}")
                continue
                
            formatted += f"{i}. {paper.get('title', '제목 없음')}\n"
            
            # 저자 정보
            authorships = paper.get("authorships")
            if authorships:
                if not isinstance(authorships, list):
                    logger.warning(f"format_paper_results: authorships가 리스트가 아님: {type(authorships)}")
                    authors = []
                else:
                    authors = []
                    for authorship in authorships:
                        if not isinstance(authorship, dict):
                            continue
                        author = authorship.get("author")
                        if not isinstance(author, dict):
                            continue
                        display_name = author.get("display_name", "")
                        if display_name:
                            authors.append(display_name)
                
                if authors:
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
            primary_location = paper.get("primary_location")
            if primary_location and isinstance(primary_location, dict):
                source = primary_location.get("source")
                if source and isinstance(source, dict):
                    display_name = source.get("display_name")
                    if display_name:
                        formatted += f"   저널: {display_name}\n"
            
            formatted += "\n"
        
        return formatted
    
    def extract_abstract_from_inverted_index(self, abstract_inverted_index: Dict[str, List[int]]) -> str:
        """
        OpenAlex의 inverted_index 형태 초록을 일반 텍스트로 변환
        
        Args:
            abstract_inverted_index: OpenAlex 형식의 초록 데이터
            
        Returns:
            텍스트 형식의 초록
        """
        if not abstract_inverted_index:
            return ""
            
        try:
            # inverted_index에서 단어와 위치 추출
            words_positions = []
            for word, positions in abstract_inverted_index.items():
                for pos in positions:
                    words_positions.append((word, pos))
            
            # 위치에 따라 정렬
            words_positions.sort(key=lambda x: x[1])
            
            # 텍스트로 조합
            abstract_text = " ".join([word for word, _ in words_positions])
            return abstract_text
            
        except Exception as e:
            logger.error(f"초록 변환 중 오류 발생: {e}")
            return "초록 변환 오류"
            
    def get_multiple_entities(self, entity_type: str, ids: List[str]) -> Dict[str, Any]:
        """
        여러 엔티티를 한 번에 조회 (filter=id:값 활용)
        
        Args:
            entity_type: 엔티티 유형
            ids: 엔티티 ID 목록
            
        Returns:
            여러 엔티티 정보
        """
        if not ids:
            return {"results": []}
            
        # ID 필터 구성 (최대 50개까지 가능)
        ids = ids[:50]  # API 제한
        id_filter = " | ".join([f"id:{id}" for id in ids])
        
        params = {
            "filter": id_filter,
            "per_page": len(ids)
        }
        
        try:
            url = self._build_url(entity_type)
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"다중 엔티티 조회 중 오류 발생: {e}")
            return {"results": []}
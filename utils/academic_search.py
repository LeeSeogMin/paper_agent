from typing import List, Dict, Any, Optional, Union
import os
import json
import requests
import re
import arxiv
from bs4 import BeautifulSoup
import hashlib
import time
import traceback

from .serpapi_scholar import ScholarSearchTool
from .openalex_api import OpenAlexTool
from utils.logger import logger

class SearchResult:
    """검색 결과를 표준화된 형식으로 저장하는 클래스"""
    
    def __init__(self, title: str, authors: List[str], abstract: str, 
                 url: str, published_date: str = "", 
                 doi: str = "", pdf_url: str = "",
                 citation_count: Optional[int] = None,
                 source: str = ""):
        self.title = title
        self.authors = authors
        self.abstract = abstract
        self.url = url
        self.published_date = published_date
        self.doi = doi
        self.pdf_url = pdf_url
        self.citation_count = citation_count
        self.source = source
    
    def to_dict(self) -> Dict[str, Any]:
        """객체를 딕셔너리로 변환"""
        return {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "url": self.url,
            "published_date": self.published_date,
            "doi": self.doi,
            "pdf_url": self.pdf_url,
            "citation_count": self.citation_count,
            "source": self.source
        }


class AcademicSearchManager:
    """
    다양한 학술 검색 API를 통합하여 관리하는 클래스
    """
    
    def __init__(self):
        self.scholar_tool = ScholarSearchTool()
        self.openalex_tool = OpenAlexTool()
        
        # Google API 키 (환경 변수에서 로드)
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_cse_id = os.getenv("GOOGLE_CSE_ID")
        
        # SerpApi 키
        self.serpapi_key = os.getenv("SerpApi_key")
    
    def search(self, 
              query: str, 
              source: str = "all", 
              limit: int = 10,
              year_start: Optional[int] = None,
              year_end: Optional[int] = None) -> Dict[str, Any]:
        """
        통합 학술 검색 수행
        
        Args:
            query: 검색 쿼리
            source: 검색 소스 ('scholar', 'openalex', 'google', 'arxiv', 'crossref', 'all')
            limit: 반환할 최대 결과 수
            year_start: 검색 시작 연도 (선택 사항)
            year_end: 검색 종료 연도 (선택 사항)
            
        Returns:
            검색 결과 (소스별로 구분)
        """
        logger.info(f"search 메서드 호출: query={query}, source={source}, limit={limit}")
        
        # 호출 스택 추적을 위한 로깅
        import traceback
        call_stack = traceback.format_stack()
        logger.debug(f"search 메서드 호출 스택:\n{''.join(call_stack)}")
        
        # 쿼리 최적화 (특히 OpenAlex 검색용)
        optimized_query = self._optimize_query(query)
        
        # 결과 초기화
        results = {
            "query": query,
            "sources_used": [],
            "results": []
        }
        
        # 검색할 소스 결정
        sources_to_search = []
        if source.lower() == "all":
            sources_to_search = ["scholar", "openalex", "google", "arxiv", "crossref"]
        else:
            sources_to_search = [source.lower()]
        
        # 각 소스별로 독립적으로 검색 실행
        for search_source in sources_to_search:
            try:
                if search_source == "scholar":
                    # Google Scholar 검색 (SerpApi)
                    try:
                        scholar_results = self.scholar_tool.search_scholar(
                            query=optimized_query,
                            num_results=limit,
                            year_start=year_start,
                            year_end=year_end
                        )
                        
                        # 결과 유효성 검사
                        if scholar_results and isinstance(scholar_results, list):
                            results["sources_used"].append("google_scholar")
                            
                            # 형식 통일
                            for item in scholar_results:
                                if not isinstance(item, dict):
                                    logger.warning(f"Google Scholar 검색 결과 항목이 딕셔너리가 아님: {type(item)}")
                                    continue
                                        
                                results["results"].append({
                                    "source": "google_scholar",
                                    "title": item.get("title", ""),
                                    "link": item.get("link", ""),
                                    "abstract": item.get("snippet", ""),
                                    "authors": item.get("publication_info", {}).get("authors", []) if isinstance(item.get("publication_info"), dict) else [],
                                    "year": item.get("publication_info", {}).get("year", "") if isinstance(item.get("publication_info"), dict) else "",
                                    "raw_data": item  # 원본 데이터 보존
                                })
                        elif scholar_results:
                            logger.warning(f"Google Scholar 검색 결과가 리스트가 아님: {type(scholar_results)}")
                    except Exception as e:
                        logger.error(f"Google Scholar 검색 중 오류: {e}")
                        logger.error(f"오류 상세 정보: {traceback.format_exc()}")
                        # 오류가 발생해도 계속 진행
                
                elif search_source == "openalex":
                    # OpenAlex 검색
                    try:
                        filter_options = {}
                        if year_start and year_end:
                            filter_options["from_publication_date"] = f"{year_start}-01-01"
                            filter_options["to_publication_date"] = f"{year_end}-12-31"
                        
                        openalex_data = self.openalex_tool.search_works(
                            query=optimized_query,
                            limit=limit,
                            filter_options=filter_options
                        )
                        
                        # OpenAlex 응답 검증
                        if not openalex_data:
                            logger.warning(f"OpenAlex 검색 결과가 None 또는 빈 값: {query}")
                            continue
                        
                        if isinstance(openalex_data, dict):
                            # 딕셔너리인 경우, 필요한 데이터가 있는 키 확인
                            if "results" in openalex_data:
                                # "results" 키의 값을 리스트로 변환
                                openalex_data = openalex_data["results"]
                                logger.info(f"OpenAlex 검색 결과를 딕셔너리에서 리스트로 변환했습니다.")
                            else:
                                # 필요한 키가 없으면 빈 리스트로 설정
                                logger.warning(f"OpenAlex 검색 결과 딕셔너리에 'results' 키가 없습니다.")
                                openalex_data = []
                        
                        # 이제 openalex_data는 리스트 형태로 보장됨
                        if not isinstance(openalex_data, list):
                            logger.warning(f"OpenAlex 검색 결과가 리스트가 아님: {type(openalex_data)}")
                            continue
                        
                        results["sources_used"].append("openalex")
                        
                        # 결과 처리
                        for item in openalex_data:
                            if not isinstance(item, dict):
                                logger.warning(f"OpenAlex 검색 결과 항목이 딕셔너리가 아님: {type(item)}")
                                continue
                            
                            # 저자 정보 추출
                            authors = []
                            authorships = item.get("authorships", [])
                            if authorships and isinstance(authorships, list):
                                for authorship in authorships:
                                    if isinstance(authorship, dict) and "author" in authorship:
                                        author = authorship.get("author", {})
                                        display_name = author.get("display_name", "")
                                        if display_name:
                                            authors.append(display_name)
                            
                            # 결과 추가
                            primary_location = item.get("primary_location") or {}
                            source_info = {}
                            if primary_location and isinstance(primary_location, dict):
                                source_info = primary_location.get("source") or {}
                                if not isinstance(source_info, dict):
                                    source_info = {}
                            
                            results["results"].append({
                                "source": "openalex",
                                "title": item.get("title", ""),
                                "link": item.get("doi", ""),
                                "abstract": self._extract_abstract_from_openalex(item),
                                "authors": authors,
                                "year": item.get("publication_year", ""),
                                "journal": source_info.get("display_name", ""),
                                "raw_data": item  # 원본 데이터 보존
                            })
                    except Exception as e:
                        logger.error(f"OpenAlex 검색 중 오류: {e}")
                        logger.error(f"오류 상세 정보: {traceback.format_exc()}")
                        # 오류가 발생해도 계속 진행
                
                elif search_source == "google":
                    # 구글 검색 (일반)
                    try:
                        google_results = self.google_search(
                            query=optimized_query,
                            num_results=limit
                        )
                        
                        if google_results:
                            results["sources_used"].append("google")
                            
                            for item in google_results:
                                results["results"].append({
                                    "source": "google",
                                    "title": item.get("title", ""),
                                    "link": item.get("link", ""),
                                    "abstract": item.get("snippet", ""),
                                    "raw_data": item
                                })
                    except Exception as e:
                        logger.error(f"Google 검색 중 오류: {e}")
                        logger.error(f"오류 상세 정보: {traceback.format_exc()}")
                        # 오류가 발생해도 계속 진행
                
                elif search_source == "arxiv":
                    # arXiv 검색
                    try:
                        arxiv_results = self.search_arxiv(
                            query=optimized_query,
                            max_results=limit
                        )
                        
                        if arxiv_results:
                            results["sources_used"].append("arxiv")
                            
                            for item in arxiv_results:
                                results["results"].append({
                                    "source": "arxiv",
                                    "title": item.get("title", ""),
                                    "link": item.get("url", ""),
                                    "abstract": item.get("abstract", ""),
                                    "authors": item.get("authors", []),
                                    "year": item.get("published_date", "")[:4] if item.get("published_date") else "",
                                    "pdf_url": item.get("pdf_url", ""),
                                    "raw_data": item
                                })
                    except Exception as e:
                        logger.error(f"arXiv 검색 중 오류: {e}")
                        logger.error(f"오류 상세 정보: {traceback.format_exc()}")
                        # 오류가 발생해도 계속 진행
                
                elif search_source == "crossref":
                    # Crossref 검색
                    try:
                        crossref_results = self.search_crossref(
                            query=optimized_query,
                            max_results=limit
                        )
                        
                        if crossref_results:
                            results["sources_used"].append("crossref")
                            
                            for item in crossref_results:
                                results["results"].append({
                                    "source": "crossref",
                                    "title": item.get("title", ""),
                                    "link": item.get("url", "") or item.get("doi", ""),
                                    "abstract": item.get("abstract", ""),
                                    "authors": item.get("authors", []),
                                    "year": item.get("published_date", "")[:4] if item.get("published_date") else "",
                                    "pdf_url": item.get("pdf_url", ""),
                                    "doi": item.get("doi", ""),
                                    "raw_data": item
                                })
                    except Exception as e:
                        logger.error(f"Crossref 검색 중 오류: {e}")
                        logger.error(f"오류 상세 정보: {traceback.format_exc()}")
                        # 오류가 발생해도 계속 진행
            
            except Exception as e:
                logger.error(f"{search_source} 검색 중 예상치 못한 오류: {e}")
                logger.error(f"오류 상세 정보: {traceback.format_exc()}")
                # 오류가 발생해도 계속 진행
        
        return results
    
    def _extract_abstract_from_openalex(self, item):
        """OpenAlex 결과에서 초록 추출"""
        if not item:
            logger.debug("_extract_abstract_from_openalex: item이 None 또는 빈 값")
            return ""
        
        if not isinstance(item, dict):
            logger.warning(f"_extract_abstract_from_openalex: item이 딕셔너리가 아님: {type(item)}")
            return ""
        
        abstract_inverted_index = item.get("abstract_inverted_index")
        if not abstract_inverted_index:
            return ""
        
        if not isinstance(abstract_inverted_index, dict):
            logger.warning(f"_extract_abstract_from_openalex: abstract_inverted_index가 딕셔너리가 아님: {type(abstract_inverted_index)}")
            return ""
        
        try:
            # inverted_index에서 단어와 위치 정보 추출
            words = []
            positions = {}
            
            # 각 단어의 위치 정보 수집
            for word, indices in abstract_inverted_index.items():
                if not isinstance(indices, list):
                    continue
                    
                for position in indices:
                    if not isinstance(position, int):
                        continue
                    positions[position] = word
            
            # 위치 순서대로 단어 배열
            for i in sorted(positions.keys()):
                words.append(positions[i])
            
            # 단어들을 공백으로 연결하여 텍스트 생성
            abstract_text = " ".join(words)
            return abstract_text
        except Exception as e:
            logger.error(f"초록 추출 중 오류: {str(e)}")
            return "초록 추출 오류"
    
    def format_search_results(self, results: Dict[str, Any]) -> str:
        """
        통합 검색 결과를 읽기 쉬운 형식으로 변환
        """
        if not results or not results.get("results"):
            return "검색 결과가 없습니다."
        
        formatted = f"검색어: {results['query']}\n"
        formatted += f"사용된 소스: {', '.join(results['sources_used'])}\n\n"
        
        for i, item in enumerate(results["results"], 1):
            # 소스별 아이콘
            source_icon = {
                "google_scholar": "🔍",
                "openalex": "📚",
                "google": "🌐",
                "arxiv": "📑",
                "crossref": "🔗"
            }.get(item["source"], "📄")
            
            formatted += f"{i}. {source_icon} {item['title']}\n"
            
            # 저자
            if item.get("authors"):
                if isinstance(item["authors"], list):
                    formatted += f"   저자: {', '.join(item['authors'])}\n"
                else:
                    formatted += f"   저자: {item['authors']}\n"
            
            # 출판 연도
            if item.get("year"):
                formatted += f"   출판 연도: {item['year']}\n"
            
            # 저널
            if item.get("journal"):
                formatted += f"   저널: {item['journal']}\n"
            
            # 링크
            if item.get("link"):
                formatted += f"   링크: {item['link']}\n"
            
            # DOI
            if item.get("doi"):
                formatted += f"   DOI: {item['doi']}\n"
            
            # 초록/스니펫
            if item.get("abstract"):
                # 초록이 길 경우 줄임
                abstract = item["abstract"]
                if len(abstract) > 200:
                    abstract = abstract[:200] + "..."
                formatted += f"   요약: {abstract}\n"
            
            formatted += "\n"
        
        return formatted
    
    def get_citations_for_rag(self, results: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        RAG 시스템에서 사용할 수 있는 인용 형식으로 결과 변환
        """
        citations = []
        
        for item in results.get("results", []):
            citation = {
                "title": item.get("title", ""),
                "authors": ", ".join(item.get("authors", [])) if isinstance(item.get("authors", []), list) else item.get("authors", ""),
                "year": str(item.get("year", "")),
                "source": item.get("journal", "") or item.get("source", ""),
                "link": item.get("link", ""),
                "snippet": item.get("abstract", ""),
                "citation_text": ""
            }
            
            # APA 형식의 인용 텍스트 생성
            authors_text = citation["authors"]
            if authors_text:
                if "," in authors_text:  # 여러 저자
                    authors_text = authors_text.split(", ")[0] + " et al."
            
            citation["citation_text"] = f"{authors_text or 'Unknown'} ({citation['year'] or 'n.d.'}). {citation['title']}. {citation['source']}."
            
            citations.append(citation)
            
        return citations
    
    # 기존 search.py 및 search_utils.py의 함수들 통합
    
    def google_search(self, query: str, num_results: int = 10, language: str = 'en') -> List[Dict[str, Any]]:
        """
        Google 검색 API를 사용하여 웹 검색 수행
        """
        logger.info(f"google_search 메서드 호출: query={query}, num_results={num_results}, language={language}")
        
        # 호출 스택 추적을 위한 로깅
        import traceback
        call_stack = traceback.format_stack()
        logger.debug(f"google_search 호출 스택:\n{''.join(call_stack)}")
        
        if not self.google_api_key or not self.google_cse_id:
            logger.warning("Google API 키 또는 CSE ID가 설정되지 않았습니다.")
            return []
        
        # 결과 수 제한 (API 제한)
        num_results = min(num_results, 10)
        
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.google_api_key,
            "cx": self.google_cse_id,
            "q": query,
            "num": num_results,
            "lr": f'lang_{language}'
        }
        
        try:
            logger.info(f"Google API 호출: {url}, 매개변수: {params}")
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
            
            logger.info(f"Google 검색 결과: {len(formatted_results)}개 항목 반환")
            return formatted_results
        
        except Exception as e:
            logger.error(f"Google 검색 오류: {str(e)}")
            logger.error(f"오류 상세 정보: {traceback.format_exc()}")
            return []
    
    def search_arxiv(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        arXiv API를 사용하여 학술 논문 검색
        """
        logger.info(f"arXiv 검색: '{query}', 최대 {max_results}개 결과")
        
        try:
            # arXiv API 검색
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            results = []
            for paper in search.results():
                authors = [author.name for author in paper.authors]
                
                result = {
                    'title': paper.title,
                    'authors': authors,
                    'abstract': paper.summary,
                    'url': paper.entry_id,
                    'pdf_url': paper.pdf_url,
                    'published_date': paper.published.strftime('%Y-%m-%d'),
                    'source': 'arXiv'
                }
                
                results.append(result)
            
            logger.info(f"arXiv 검색 결과: {len(results)}개 논문 찾음")
            return results
            
        except Exception as e:
            logger.error(f"arXiv 검색 오류: {str(e)}")
            return []
    
    def search_crossref(self, query: str, max_results: int = 10, filter: str = None) -> List[Dict[str, Any]]:
        """
        Crossref API를 사용하여 학술 논문 검색
        
        Args:
            query: 검색 쿼리
            max_results: 반환할 최대 결과 수
            filter: 추가 필터 옵션
            
        Returns:
            검색 결과 목록
        """
        logger.info(f"Crossref 검색: '{query}', 최대 {max_results}개 결과")
        
        try:
            # Crossref API v1 경로 사용
            url = "https://api.crossref.org/v1/works"
            params = {
                "query": query,
                "rows": max_results,
                "sort": "relevance",
                "select": "DOI,title,author,abstract,URL,published-print,published-online"
            }
            
            if filter:
                params["filter"] = filter
            
            # 사용자 에이전트 헤더 추가 (Crossref 권장)
            headers = {"User-Agent": "ResearchAgent/1.0 (mailto:example@example.com)"}
            
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            results = []
            if "message" in data and "items" in data["message"]:
                for item in data["message"]["items"]:
                    # 데이터 추출 로직 - 기존과 동일
                    title = item.get("title", [""])[0] if "title" in item and len(item["title"]) > 0 else ""
                    
                    # 저자 정보 추출
                    authors = []
                    if "author" in item:
                        for author in item["author"]:
                            name_parts = []
                            if "given" in author:
                                name_parts.append(author["given"])
                            if "family" in author:
                                name_parts.append(author["family"])
                            
                            if name_parts:
                                authors.append(" ".join(name_parts))
                    
                    # 발행일 추출
                    published_date = ""
                    if "published-print" in item and "date-parts" in item["published-print"]:
                        date_parts = item["published-print"]["date-parts"][0]
                        if len(date_parts) >= 1:
                            published_date = str(date_parts[0])
                    elif "published-online" in item and "date-parts" in item["published-online"]:
                        date_parts = item["published-online"]["date-parts"][0]
                        if len(date_parts) >= 1:
                            published_date = str(date_parts[0])
                    
                    # URL 및 DOI 추출
                    url = item.get("URL", "")
                    doi = item.get("DOI", "")
                    
                    # PDF URL 추정
                    pdf_url = f"https://doi.org/{doi}" if doi else url
                    
                    # 추상 추출
                    abstract = ""
                    if "abstract" in item:
                        abstract = item["abstract"]
                    
                    result = {
                        'title': title,
                        'authors': authors,
                        'abstract': abstract,
                        'url': url,
                        'doi': doi,
                        'published_date': published_date,
                        'pdf_url': pdf_url,
                        'source': 'Crossref'
                    }
                    
                    results.append(result)
            
            logger.info(f"Crossref 검색 결과: {len(results)}개 논문 찾음")
            return results
            
        except Exception as e:
            logger.error(f"Crossref 검색 오류: {str(e)}")
            return []
    
    def format_results_as_markdown(self, results: Dict[str, Any]) -> str:
        """
        검색 결과를 마크다운 형식으로 포맷팅
        """
        if not results or not results.get("results"):
            return "# 검색 결과가 없습니다."
        
        markdown = f"# '{results['query']}' 검색 결과\n\n"
        markdown += f"**사용된 소스**: {', '.join(results['sources_used'])}\n\n"
        
        for i, item in enumerate(results["results"], 1):
            source_name = {
                "google_scholar": "Google Scholar",
                "openalex": "OpenAlex",
                "google": "Google",
                "arxiv": "arXiv",
                "crossref": "Crossref"
            }.get(item["source"], item["source"])
            
            markdown += f"## {i}. {item['title']}\n\n"
            markdown += f"**출처**: {source_name}\n\n"
            
            if item.get("authors"):
                if isinstance(item["authors"], list):
                    authors_text = ", ".join(item["authors"])
                else:
                    authors_text = str(item["authors"])
                markdown += f"**저자**: {authors_text}\n\n"
            
            if item.get("year"):
                markdown += f"**출판 연도**: {item['year']}\n\n"
            
            if item.get("journal"):
                markdown += f"**저널**: {item['journal']}\n\n"
            
            if item.get("abstract"):
                markdown += f"**요약**:\n\n> {item['abstract']}\n\n"
            
            if item.get("link"):
                markdown += f"**링크**: [{item['link']}]({item['link']})\n\n"
            
            markdown += "---\n\n"
        
        return markdown
    
    # search.py의 test_academic_search 함수도 통합
    def test_academic_search(self, query: str, max_results: int = 5) -> str:
        """
        학술 검색 기능을 테스트하기 위한 함수
        
        Args:
            query: 검색 쿼리
            max_results: 최대 결과 수
        
        Returns:
            str: 포맷팅된 검색 결과
        """
        # 통합 검색 실행
        results = self.search(
            query=query,
            source="all",
            limit=max_results
        )
        
        # 결과 포맷팅 (마크다운 형식)
        formatted_results = self.format_results_as_markdown(results)
        
        return formatted_results

    # 기존 메서드 외에 호환성을 위한 메서드 추가
    def google_search_compat(self, query: str, num_results: int = 10, language: str = 'en') -> list:
        """
        구글 검색을 수행 (호환성 유지용)
        """
        logger.info(f"google_search_compat 메서드 호출: query={query}, num_results={num_results}, language={language}")
        
        # 호출 스택 추적을 위한 로깅
        import traceback
        call_stack = traceback.format_stack()
        logger.debug(f"google_search_compat 호출 스택:\n{''.join(call_stack)}")
        
        try:
            # 무한 재귀 방지: search 메서드 대신 직접 google_search 메서드 호출
            results = self.google_search(
                query=query,
                num_results=num_results,
                language=language
            )
            
            # 원래 반환 형식에 맞게 조정
            logger.info(f"google_search_compat 결과: {len(results)}개 항목 반환")
            return results
        except Exception as e:
            logger.error(f"google_search_compat 오류: {str(e)}")
            logger.error(f"오류 상세 정보: {traceback.format_exc()}")
            return []

    def search_with_fallback(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        여러 검색 소스를 독립적으로 시도하고 결과를 합침
        
        Args:
            query: 검색 쿼리
            limit: 각 소스당 최대 결과 수
            
        Returns:
            Dict: 통합된 검색 결과
        """
        logger.info(f"search_with_fallback 메서드 호출: query={query}, limit={limit}")
        
        # 결과 초기화
        combined_results = {
            "query": query,
            "sources_used": [],
            "results": []
        }
        
        # 모든 소스 독립적으로 시도
        sources_to_try = ["google", "openalex", "arxiv", "crossref"]
        
        for source in sources_to_try:
            try:
                logger.info(f"{source} 검색 시도 중...")
                source_results = self.search(query=query, source=source, limit=limit)
                
                # 결과가 있으면 통합
                if source_results.get("results"):
                    logger.info(f"{source} 검색 결과: {len(source_results['results'])}개 항목")
                    combined_results["sources_used"].extend(source_results["sources_used"])
                    combined_results["results"].extend(source_results["results"])
                else:
                    logger.info(f"{source} 검색 결과 없음")
            except Exception as e:
                logger.error(f"{source} 검색 중 오류: {e}")
                logger.error(f"오류 상세 정보: {traceback.format_exc()}")
                # 오류가 발생해도 다음 소스로 계속 진행
        
        # 중복 소스 제거
        combined_results["sources_used"] = list(set(combined_results["sources_used"]))
        
        # 결과가 없으면 로그 남김
        if not combined_results["results"]:
            logger.warning(f"모든 소스에서 검색 결과 없음: {query}")
        
        return combined_results

    def _optimize_query(self, query: str) -> str:
        """
        검색 쿼리를 최적화하여 API 호출에 적합하게 변환
        
        Args:
            query: 원본 쿼리
            
        Returns:
            최적화된 쿼리
        """
        # 쿼리가 너무 길면 핵심 키워드만 추출
        if len(query) > 150:
            logger.info(f"쿼리가 너무 깁니다 ({len(query)}자). 최적화를 수행합니다.")
            
            # 1. 불필요한 지시문이나 설명 제거
            patterns_to_remove = [
                r"When summarizing as an academic review,.*",
                r"In particular, organize the discussion.*",
                r"cite the references used.*",
                r"include a bibliography.*",
                r"This search query would.*",
                r"Please provide.*",
                r"I need information about.*",
                r"I'm looking for.*",
                r"Can you find.*",
                r"I want to learn about.*"
            ]
            
            optimized = query
            for pattern in patterns_to_remove:
                optimized = re.sub(pattern, "", optimized, flags=re.IGNORECASE)
            
            # 2. 쿼리에서 중요 키워드 추출 방법 개선
            important_keywords = []
            
            # 2.1 명사구 추출 (2-3단어로 구성된 구문)
            noun_phrases = re.findall(r'\b[A-Za-z][\w-]*(?:\s+[A-Za-z][\w-]*){1,2}\b', optimized)
            
            # 2.2 대문자로 시작하는 단어나 따옴표 안의 구문 (기존 방식)
            capitalized_words = re.findall(r'\b[A-Z][a-zA-Z]*\b', optimized)
            quoted_phrases = re.findall(r'"([^"]*)"', optimized)
            
            # 2.3 숫자가 포함된 중요 식별자 (버전 번호 등)
            identifiers = re.findall(r'\b[A-Za-z]+[\d]+[\w]*\b', optimized)
            
            # 2.4 하이픈으로 연결된 복합어
            compound_words = re.findall(r'\b[\w]+-[\w]+\b', optimized)
            
            # 모든 추출 결과 합치기
            important_keywords.extend(noun_phrases)
            important_keywords.extend(capitalized_words)
            important_keywords.extend(quoted_phrases)
            important_keywords.extend(identifiers)
            important_keywords.extend(compound_words)
            
            # 3. 학술 검색에 중요한 특정 키워드 추가
            academic_terms = [
                "model", "models", "comparison", "analysis", "review", 
                "topic modeling", "LLM", "LLMs", "large language model",
                "algorithm", "method", "approach", "framework",
                "research", "study", "experiment", "evaluation",
                "dataset", "data", "results", "findings",
                "neural", "network", "deep learning", "machine learning",
                "artificial intelligence", "AI", "NLP", "natural language processing",
                "transformer", "attention", "embedding", "fine-tuning",
                "pre-training", "training", "inference", "performance",
                "accuracy", "precision", "recall", "F1",
                "benchmark", "state-of-the-art", "SOTA"
            ]
            
            for term in academic_terms:
                if term.lower() in optimized.lower() and term not in important_keywords:
                    important_keywords.append(term)
            
            # 4. 중복 제거 및 정리
            # 대소문자 구분 없이 중복 제거
            unique_keywords = []
            lowercase_set = set()
            
            for keyword in important_keywords:
                if keyword.lower() not in lowercase_set and len(keyword) > 2:  # 너무 짧은 키워드 제외
                    unique_keywords.append(keyword)
                    lowercase_set.add(keyword.lower())
            
            # 5. 키워드 우선순위 지정 (더 긴 구문 우선)
            unique_keywords.sort(key=len, reverse=True)
            
            # 6. 최종 쿼리 생성 (너무 많은 키워드는 제한)
            if unique_keywords:
                # 최대 10개 키워드로 제한
                final_keywords = unique_keywords[:10]
                final_query = " ".join(final_keywords)
                logger.info(f"최적화된 쿼리: {final_query}")
                return final_query
        
        # 쿼리가 충분히 짧으면 그대로 사용
        return query 
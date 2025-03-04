from typing import List, Dict, Any, Optional, Union
import os
import json
import requests
import re
import arxiv
from bs4 import BeautifulSoup
import hashlib

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
        여러 학술 검색 소스에서 동시에 검색을 수행
        
        Args:
            query: 검색 쿼리
            source: 검색 소스 ('scholar', 'openalex', 'google', 'arxiv', 'crossref', 'all')
            limit: 각 소스당 검색 결과 수
            year_start: 시작 연도 (선택 사항)
            year_end: 종료 연도 (선택 사항)
            
        Returns:
            통합된 검색 결과
        """
        results = {
            "query": query,
            "sources_used": [],
            "results": []
        }
        
        # Google Scholar 검색 (SerpApi)
        if source.lower() in ["scholar", "all"]:
            try:
                scholar_results = self.scholar_tool.search_scholar(
                    query=query,
                    num_results=limit,
                    year_start=year_start,
                    year_end=year_end
                )
                
                if scholar_results:
                    results["sources_used"].append("google_scholar")
                    
                    # 형식 통일
                    for item in scholar_results:
                        results["results"].append({
                            "source": "google_scholar",
                            "title": item.get("title", ""),
                            "link": item.get("link", ""),
                            "abstract": item.get("snippet", ""),
                            "authors": item.get("publication_info", {}).get("authors", []),
                            "year": item.get("publication_info", {}).get("year", ""),
                            "raw_data": item  # 원본 데이터 보존
                        })
            except Exception as e:
                logger.error(f"Google Scholar 검색 중 오류: {e}")
        
        # OpenAlex 검색
        if source.lower() in ["openalex", "all"]:
            try:
                filter_options = {}
                if year_start and year_end:
                    filter_options["from_publication_date"] = f"{year_start}-01-01"
                    filter_options["to_publication_date"] = f"{year_end}-12-31"
                
                openalex_data = self.openalex_tool.search_works(
                    query=query,
                    limit=limit,
                    filter_options=filter_options
                )
                
                if openalex_data and "results" in openalex_data and openalex_data["results"]:
                    results["sources_used"].append("openalex")
                    
                    # 형식 통일
                    for item in openalex_data["results"]:
                        # 저자 정보 추출
                        authors = []
                        if "authorships" in item:
                            for authorship in item["authorships"]:
                                if "author" in authorship and "display_name" in authorship["author"]:
                                    authors.append(authorship["author"]["display_name"])
                        
                        # 결과 추가
                        results["results"].append({
                            "source": "openalex",
                            "title": item.get("title", ""),
                            "link": item.get("doi", ""),
                            "abstract": self._extract_abstract_from_openalex(item),
                            "authors": authors,
                            "year": item.get("publication_year", ""),
                            "journal": item.get("primary_location", {}).get("source", {}).get("display_name", ""),
                            "raw_data": item  # 원본 데이터 보존
                        })
            except Exception as e:
                logger.error(f"OpenAlex 검색 중 오류: {e}")
        
        # 구글 검색 (일반)
        if source.lower() in ["google", "all"]:
            try:
                google_results = self.google_search(
                    query=query,
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
                            "authors": [],
                            "year": "",
                            "raw_data": item
                        })
            except Exception as e:
                logger.error(f"Google 검색 중 오류: {e}")
        
        # arXiv 검색
        if source.lower() in ["arxiv", "all"]:
            try:
                arxiv_results = self.search_arxiv(
                    query=query,
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
        
        # Crossref 검색
        if source.lower() in ["crossref", "all"]:
            try:
                crossref_results = self.search_crossref(
                    query=query,
                    max_results=limit
                )
                
                if crossref_results:
                    results["sources_used"].append("crossref")
                    
                    for item in crossref_results:
                        results["results"].append({
                            "source": "crossref",
                            "title": item.get("title", ""),
                            "link": item.get("url", ""),
                            "abstract": item.get("abstract", ""),
                            "authors": item.get("authors", "").split(", ") if isinstance(item.get("authors"), str) else [],
                            "year": item.get("published_date", "")[:4] if item.get("published_date") else "",
                            "doi": item.get("doi", ""),
                            "raw_data": item
                        })
            except Exception as e:
                logger.error(f"Crossref 검색 중 오류: {e}")
        
        return results
    
    def _extract_abstract_from_openalex(self, item):
        """OpenAlex 결과에서 초록 추출"""
        if "abstract_inverted_index" in item and item["abstract_inverted_index"]:
            # OpenAlex는 inverted_index 형식으로 초록을 제공
            # 간단한 변환만 수행
            return "초록 있음 (추가 처리 필요)"
        return ""
    
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
        """
        logger.info(f"Crossref 검색: '{query}', 최대 {max_results}개 결과")
        
        try:
            # Crossref API 검색
            url = "https://api.crossref.org/works"
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
                    # 데이터 추출 로직
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
                        'authors': ', '.join(authors),
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
                    markdown += f"**저자**: {', '.join(item['authors'])}\n\n"
                else:
                    markdown += f"**저자**: {item['authors']}\n\n"
            
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
    def google_search(self, query: str, num_results: int = 10, language: str = 'en') -> list:
        """
        구글 검색을 수행 (호환성 유지용)
        """
        # 내부적으로 search 메서드 사용
        results = self.search(
            query=query,
            source="google",
            limit=num_results
        )
        
        # 원래 반환 형식에 맞게 조정
        return results.get("results", [])

    def search_with_fallback(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        SerpAPI 실패 시 대체 검색 방법을 사용
        """
        try:
            # 먼저 모든 소스로 검색 시도
            results = self.search(query=query, source="all", limit=limit)
            if not results.get("results"):
                # 결과가 없으면 Google Scholar 제외하고 다시 시도
                logger.info("전체 검색 결과 없음, OpenAlex만 사용하여 재시도")
                return self.search(query=query, source="openalex", limit=limit)
            return results
        except Exception as e:
            logger.warning(f"통합 검색 중 오류: {e}, OpenAlex만 사용하여 재시도")
            return self.search(query=query, source="openalex", limit=limit) 
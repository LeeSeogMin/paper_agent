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
        
        # Google Custom Search API 호출
        service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
        result = service.cse().list(
            q=query,
            cx=GOOGLE_CSE_ID,
            num=num_results
        ).execute()
        
        # 검색 결과 형식 변환
        search_results = []
        if 'items' in result:
            for item in result['items']:
                search_results.append({
                    "title": item.get('title', ''),
                    "url": item.get('link', ''),
                    "abstract": item.get('snippet', '')
                })
        
        logger.info(f"Google 검색 결과 {len(search_results)}개 찾음")
        return search_results
        
    except Exception as e:
        logger.error(f"Google 검색 중 오류 발생: {str(e)}", exc_info=True)
        # 오류 발생 시 빈 리스트 반환
        return []

"""
학술 자료 검색 유틸리티 모듈
"""

import requests
import logging
from typing import List, Dict, Any, Optional
import arxiv
from bs4 import BeautifulSoup

from utils.logger import logger

class SearchResult:
    """검색 결과를 표준화된 형식으로 저장하는 클래스"""
    
    def __init__(self, title: str, authors: List[str], abstract: str, 
                 url: str, published_date: str, source: str, 
                 doi: Optional[str] = None, citation_count: Optional[int] = None):
        self.title = title
        self.authors = authors
        self.abstract = abstract
        self.url = url
        self.published_date = published_date
        self.source = source
        self.doi = doi
        self.citation_count = citation_count
        
    def to_dict(self) -> Dict[str, Any]:
        """SearchResult 객체를 딕셔너리로 변환"""
        return {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "url": self.url,
            "published_date": self.published_date,
            "source": self.source,
            "doi": self.doi,
            "citation_count": self.citation_count
        }
    
    def __str__(self) -> str:
        """SearchResult 객체의 문자열 표현"""
        authors_str = ", ".join(self.authors) if self.authors else "저자 정보 없음"
        return f"제목: {self.title}\n저자: {authors_str}\n출처: {self.source}\nURL: {self.url}"

def search_crossref(query: str, rows: int = 10) -> List[SearchResult]:
    """
    Crossref API를 사용하여 학술 자료 검색
    
    Args:
        query: 검색 쿼리
        rows: 반환할 결과 수
        
    Returns:
        SearchResult 객체 리스트
    """
    logger.info(f"Crossref에서 검색: {query}")
    url = f"https://api.crossref.org/works?query={query}&rows={rows}"
    headers = {
        "User-Agent": "ResearchPaperAssistant/1.0 (mailto:contact@example.com)"  # Crossref 권장사항
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get('message', {}).get('items', []):
            # 필요한 데이터 추출
            title = item.get('title', ['제목 없음'])[0] if item.get('title') else '제목 없음'
            
            authors = []
            for author in item.get('author', []):
                name_parts = []
                if 'given' in author:
                    name_parts.append(author['given'])
                if 'family' in author:
                    name_parts.append(author['family'])
                authors.append(' '.join(name_parts) if name_parts else '이름 없음')
            
            abstract = item.get('abstract', '초록 없음')
            doi = item.get('DOI')
            url = f"https://doi.org/{doi}" if doi else ''
            
            # 날짜 정보 처리
            published_date = '날짜 정보 없음'
            if 'published' in item and item['published'] and 'date-parts' in item['published']:
                date_parts = item['published']['date-parts'][0]
                if len(date_parts) >= 3:
                    published_date = f"{date_parts[0]}-{date_parts[1]:02d}-{date_parts[2]:02d}"
                elif len(date_parts) >= 2:
                    published_date = f"{date_parts[0]}-{date_parts[1]:02d}"
                elif len(date_parts) >= 1:
                    published_date = f"{date_parts[0]}"
            
            results.append(SearchResult(
                title=title,
                authors=authors,
                abstract=abstract,
                url=url,
                published_date=published_date,
                source="Crossref",
                doi=doi,
                citation_count=item.get('is-referenced-by-count')
            ))
        
        return results
    except Exception as e:
        logger.error(f"Crossref 검색 중 오류 발생: {str(e)}")
        return []

def search_arxiv(query: str, max_results: int = 10) -> List[SearchResult]:
    """
    arXiv API를 사용하여 학술 자료 검색
    
    Args:
        query: 검색 쿼리
        max_results: 반환할 최대 결과 수
        
    Returns:
        SearchResult 객체 리스트
    """
    logger.info(f"arXiv에서 검색: {query}")
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        for item in search.results():
            # 저자 목록 추출
            authors = [author.name for author in item.authors]
            
            results.append(SearchResult(
                title=item.title,
                authors=authors,
                abstract=item.summary,
                url=item.pdf_url,
                published_date=item.published.strftime('%Y-%m-%d') if item.published else '날짜 정보 없음',
                source="arXiv",
                doi=None,  # arXiv는 DOI 정보가 없음
                citation_count=None  # arXiv는 인용 수 정보가 없음
            ))
        
        return results
    except Exception as e:
        logger.error(f"arXiv 검색 중 오류 발생: {str(e)}")
        return []

def search_core(query: str, api_key: str, limit: int = 10) -> List[SearchResult]:
    """
    CORE API를 사용하여 학술 자료 검색
    
    Args:
        query: 검색 쿼리
        api_key: CORE API 키
        limit: 반환할 결과 수
        
    Returns:
        SearchResult 객체 리스트
    """
    logger.info(f"CORE에서 검색: {query}")
    url = f"https://core.ac.uk/api-v2/search/{query}"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"limit": limit}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get('data', []):
            # 저자 목록 추출 (CORE API 응답 형식에 따라 조정 필요)
            authors = item.get('authors', [])
            if isinstance(authors, str):
                authors = [author.strip() for author in authors.split(',')]
            
            results.append(SearchResult(
                title=item.get('title', '제목 없음'),
                authors=authors,
                abstract=item.get('description', '초록 없음'),
                url=item.get('downloadUrl') or item.get('url', ''),
                published_date=item.get('publishedDate', '날짜 정보 없음'),
                source="CORE",
                doi=item.get('doi'),
                citation_count=None
            ))
        
        return results
    except Exception as e:
        logger.error(f"CORE 검색 중 오류 발생: {str(e)}")
        return []

def search_google_scholar(query: str, num_results: int = 10) -> List[SearchResult]:
    """
    Google Scholar에서 웹 스크래핑을 통한 검색 (주의: Google의 이용약관 확인 필요)
    
    참고: Google은 스크래핑을 제한할 수 있으므로 학술 목적으로만 조심스럽게 사용
    
    Args:
        query: 검색 쿼리
        num_results: 반환할 결과 수
        
    Returns:
        SearchResult 객체 리스트
    """
    logger.warning("Google Scholar 스크래핑은 이용약관을 확인하고 제한적으로 사용하세요.")
    
    try:
        # 실제 구현에서는 헤더, 지연 시간, 프록시 등을 고려해야 함
        url = f"https://scholar.google.com/scholar?q={query}&num={num_results}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # BeautifulSoup을 사용한 파싱 로직 (Google Scholar의 HTML 구조에 따라 조정 필요)
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        
        # 이 부분은 Google Scholar의 실제 HTML 구조에 맞게 수정해야 함
        # 아래는 예시일 뿐입니다
        for item in soup.select('.gs_ri')[:num_results]:
            title_elem = item.select_one('.gs_rt')
            title = title_elem.text if title_elem else '제목 없음'
            
            url_elem = title_elem.select_one('a') if title_elem else None
            url = url_elem['href'] if url_elem and 'href' in url_elem.attrs else ''
            
            authors_elem = item.select_one('.gs_a')
            authors_text = authors_elem.text if authors_elem else ''
            # 저자와 연도 정보 추출 (형식: "저자1, 저자2... - 출처, 연도")
            authors_parts = authors_text.split(' - ')[0] if ' - ' in authors_text else ''
            authors = [author.strip() for author in authors_parts.split(',')]
            
            abstract_elem = item.select_one('.gs_rs')
            abstract = abstract_elem.text if abstract_elem else '초록 없음'
            
            # 발행 연도 추출 (예시 로직)
            year_match = re.search(r'\b(19|20)\d{2}\b', authors_text)
            published_date = year_match.group(0) if year_match else '연도 미상'
            
            results.append(SearchResult(
                title=title,
                authors=authors,
                abstract=abstract,
                url=url,
                published_date=published_date,
                source="Google Scholar",
                doi=None,
                citation_count=None
            ))
        
        return results
    except Exception as e:
        logger.error(f"Google Scholar 검색 중 오류 발생: {str(e)}")
        return []

def search_dblp(query: str, max_results: int = 10) -> List[SearchResult]:
    """
    DBLP API를 사용하여 컴퓨터 과학 분야 학술 자료 검색
    
    Args:
        query: 검색 쿼리
        max_results: 반환할 최대 결과 수
        
    Returns:
        SearchResult 객체 리스트
    """
    logger.info(f"DBLP에서 검색: {query}")
    url = f"https://dblp.org/search/publ/api?q={query}&format=json&h={max_results}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        results = []
        hits = data.get('result', {}).get('hits', {}).get('hit', [])
        
        for hit in hits:
            info = hit.get('info', {})
            
            # 저자 정보 추출
            authors = []
            if 'authors' in info and 'author' in info['authors']:
                author_list = info['authors']['author']
                if isinstance(author_list, list):
                    for author in author_list:
                        authors.append(author.get('text', ''))
                else:
                    authors.append(author_list.get('text', ''))
            
            # 제목 및 URL
            title = info.get('title', '제목 없음')
            url = info.get('url', '')
            
            # 발행 정보
            venue = info.get('venue', '')
            year = info.get('year', '')
            published_date = year if year else '날짜 정보 없음'
            
            # DBLP는 초록 정보를 제공하지 않음
            abstract = f"[DBLP는 초록 정보를 제공하지 않습니다. 출처: {venue}]"
            
            results.append(SearchResult(
                title=title,
                authors=authors,
                abstract=abstract,
                url=url,
                published_date=published_date,
                source="DBLP",
                doi=None,
                citation_count=None
            ))
        
        return results
    except Exception as e:
        logger.error(f"DBLP 검색 중 오류 발생: {str(e)}")
        return []

def search_academic_resources(query: str, api_keys: Dict[str, str] = None, max_results: int = 10) -> Dict[str, List[SearchResult]]:
    """
    여러 소스에서 학술 자료를 통합 검색
    
    Args:
        query: 검색 쿼리
        api_keys: 각 API에 필요한 키 딕셔너리
        max_results: 각 소스별 최대 결과 수
        
    Returns:
        소스별 SearchResult 객체 리스트 딕셔너리
    """
    if api_keys is None:
        api_keys = {}
    
    logger.info(f"통합 학술 자료 검색: {query}")
    results = {}
    
    # Crossref 검색
    results["crossref"] = search_crossref(query, rows=max_results)
    
    # arXiv 검색
    results["arxiv"] = search_arxiv(query, max_results=max_results)
    
    # DBLP 검색 추가
    results["dblp"] = search_dblp(query, max_results=max_results)
    
    # CORE 검색 (API 키가 있는 경우)
    if "core" in api_keys:
        results["core"] = search_core(query, api_keys["core"], limit=max_results)
    
    # Google Scholar 검색 삭제 (주석 처리)
    # results["google_scholar"] = search_google_scholar(query, num_results=max_results)
    
    return results

def format_search_results(results: Dict[str, List[SearchResult]], format_type: str = "text") -> str:
    """
    검색 결과를 지정된 형식으로 포맷팅
    
    Args:
        results: 검색 결과 딕셔너리
        format_type: 출력 형식 ("text", "markdown", "html" 등)
        
    Returns:
        포맷팅된 검색 결과 문자열
    """
    if format_type == "markdown":
        output = "# 학술 검색 결과\n\n"
        
        for source, source_results in results.items():
            if source_results:
                output += f"## {source.capitalize()} ({len(source_results)}개 결과)\n\n"
                
                for i, result in enumerate(source_results):
                    output += f"### {i+1}. {result.title}\n\n"
                    output += f"**저자**: {', '.join(result.authors)}\n\n"
                    output += f"**발행일**: {result.published_date}\n\n"
                    output += f"**요약**: {result.abstract[:300]}{'...' if len(result.abstract) > 300 else ''}\n\n"
                    
                    if result.doi:
                        output += f"**DOI**: {result.doi}\n\n"
                    
                    output += f"**링크**: [{result.url}]({result.url})\n\n"
                    
                    if result.citation_count is not None:
                        output += f"**인용 횟수**: {result.citation_count}\n\n"
                    
                    output += "---\n\n"
            else:
                output += f"## {source.capitalize()}\n\n결과 없음\n\n"
        
        return output
    else:  # 기본 텍스트 형식
        output = "학술 검색 결과:\n\n"
        
        for source, source_results in results.items():
            if source_results:
                output += f"{source.upper()} ({len(source_results)}개 결과):\n"
                
                for i, result in enumerate(source_results):
                    output += f"{i+1}. {result.title}\n"
                    output += f"   저자: {', '.join(result.authors)}\n"
                    output += f"   발행일: {result.published_date}\n"
                    output += f"   요약: {result.abstract[:200]}{'...' if len(result.abstract) > 200 else ''}\n"
                    output += f"   링크: {result.url}\n\n"
            else:
                output += f"{source.upper()}: 결과 없음\n\n"
        
        return output

# 필요한 추가 모듈 import
import re

# 예시: 학술 검색 테스트를 위한 함수
def test_academic_search(query):
    """
    학술 검색 기능을 테스트하기 위한 함수
    
    Args:
        query (str): 검색 쿼리
    
    Returns:
        str: 포맷팅된 검색 결과
    """
    # API 키 설정 (CORE API 키가 있는 경우)
    api_keys = {
        # "core": "YOUR_CORE_API_KEY"  # 실제 사용 시 주석 해제 및 키 입력
    }
    
    # 통합 검색 실행
    results = search_academic_resources(query, api_keys, max_results=5)
    
    # 결과 포맷팅 (마크다운 형식)
    formatted_results = format_search_results(results, format_type="markdown")
    
    return formatted_results
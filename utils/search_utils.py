import os
import json
import requests
import arxiv
from typing import List, Dict, Any, Optional
from utils.logger import logger
from utils.search import google_search, search_crossref, search_arxiv

def google_search(query: str, num_results: int = 10) -> List[Dict[str, Any]]:
    """
    Google Custom Search API를 사용하여 검색 수행
    
    Args:
        query: 검색 쿼리
        num_results: 반환할 결과 수 (최대 10)
        
    Returns:
        List[Dict[str, Any]]: 검색 결과 목록
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    
    if not api_key or not cse_id:
        logger.warning("Google API 키 또는 CSE ID가 설정되지 않았습니다.")
        return []
    
    # 결과 수 제한 (API 제한)
    num_results = min(num_results, 10)
    
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cse_id,
        "q": query,
        "num": num_results
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

def fallback_search(query: str, max_results: int = 10, language: str = 'en') -> List[Dict[str, Any]]:
    """
    대체 검색 수행 (구글 검색, 학술 논문만 필터링)
    
    Args:
        query: 검색 쿼리
        max_results: 최대 검색 결과 수
        language: 검색 언어 (기본값: 영어)
        
    Returns:
        List[Dict]: 검색 결과 목록
    """
    logger.info(f"대체 검색 수행: '{query}', 최대 {max_results}개 결과, 언어: {language}")
    
    # 검색 쿼리에 학술 논문 관련 키워드 추가
    academic_query = f"{query} filetype:pdf academic paper research journal"
    
    try:
        # 구글 검색 수행
        results = google_search(academic_query, num_results=max_results*2)  # 필터링 후 충분한 결과를 얻기 위해 2배로 검색
        
        # 학술 논문 및 PDF 링크만 필터링
        filtered_results = []
        for result in results:
            url = result.get('url', '').lower()
            title = result.get('title', '').lower()
            snippet = result.get('snippet', '').lower()
            
            # PDF 파일 여부 확인
            is_pdf = url.endswith('.pdf') or 'pdf' in url
            
            # 학술적 출처 확인 (도메인 또는 키워드 기반)
            academic_domains = ['.edu', '.ac.', '.gov', '.org', 'research', 'journal', 'conference', 
                               'proceedings', 'arxiv', 'ieee', 'acm', 'springer', 'sciencedirect']
            is_academic = any(domain in url for domain in academic_domains)
            
            # 학술 키워드 포함 여부
            academic_keywords = ['paper', 'journal', 'research', 'proceedings', 'conference', 'study', 
                               'analysis', 'publication', 'article']
            has_academic_keywords = any(keyword in title or keyword in snippet for keyword in academic_keywords)
            
            if is_pdf and (is_academic or has_academic_keywords):
                # PDF URL 필드 추가
                result['pdf_url'] = url if is_pdf else None
                filtered_results.append(result)
        
        return filtered_results[:max_results]
        
    except Exception as e:
        logger.error(f"대체 검색 오류: {str(e)}")
        return []

def academic_search(query: str, max_results: int = 10, language: str = 'en') -> List[Dict[str, Any]]:
    """
    학술 검색 수행 (언어 필터 추가)
    
    Args:
        query: 검색 쿼리
        max_results: 최대 검색 결과 수
        language: 검색 언어 (기본값: 영어)
        
    Returns:
        List[Dict]: 검색 결과 목록
    """
    logger.info(f"학술 논문 검색: '{query}', 최대 {max_results}개 결과, 언어: {language}")
    
    results = []
    
    # arXiv 검색
    try:
        arxiv_results = search_arxiv(query, max_results=max_results)
        # PDF 링크가 있는 결과만 유지
        arxiv_results = [r for r in arxiv_results if r.get('pdf_url')]
        results.extend(arxiv_results)
    except Exception as e:
        logger.error(f"arXiv 검색 오류: {str(e)}")
    
    # Crossref 검색
    try:
        crossref_results = search_crossref(query, max_results=max_results, filter="has-full-text:true")
        # PDF 링크가 있는 결과만 유지
        crossref_results = [r for r in crossref_results if r.get('pdf_url')]
        results.extend(crossref_results)
    except Exception as e:
        logger.error(f"Crossref 검색 오류: {str(e)}")
    
    # 중복 제거 및 최대 결과 수 제한
    unique_results = []
    seen_titles = set()
    
    for result in results:
        title = result.get('title', '').lower()
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique_results.append(result)
    
    return unique_results[:max_results]

def search_arxiv(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    arXiv에서 학술 논문 검색
    
    Args:
        query: 검색 쿼리
        max_results: 최대 결과 수
        
    Returns:
        List[Dict]: 검색 결과 목록
    """
    logger.info(f"arXiv 검색: '{query}'")
    
    try:
        # arXiv API 검색
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        for paper in search.results():
            # PDF URL 추출
            pdf_url = paper.pdf_url
            
            # 저자 정보 추출
            authors = [author.name for author in paper.authors]
            
            result = {
                'title': paper.title,
                'authors': ', '.join(authors),
                'abstract': paper.summary,
                'url': paper.entry_id,
                'published_date': paper.published.strftime('%Y-%m-%d'),
                'pdf_url': pdf_url,
                'source': 'arXiv'
            }
            
            results.append(result)
        
        return results
        
    except Exception as e:
        logger.error(f"arXiv 검색 오류: {str(e)}")
        return []

def search_crossref(query: str, max_results: int = 10, filter: str = None) -> List[Dict[str, Any]]:
    """
    Crossref에서 학술 논문 검색
    
    Args:
        query: 검색 쿼리
        max_results: 최대 결과 수
        filter: 필터 문자열
        
    Returns:
        List[Dict]: 검색 결과 목록
    """
    logger.info(f"Crossref 검색: '{query}'")
    
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
                # 기본 데이터 추출
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
                
                # PDF URL 추정 (Crossref는 직접 PDF URL을 제공하지 않음)
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
        
        return results
        
    except Exception as e:
        logger.error(f"Crossref 검색 오류: {str(e)}")
        return [] 
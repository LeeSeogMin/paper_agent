import os
from typing import List, Dict, Any, Optional
from .academic_search import AcademicSearchManager

class RAGEnhancer:
    """
    RAG(Retrieval-Augmented Generation) 시스템에 학술 검색 기능을 통합하는 클래스
    """
    
    def __init__(self):
        self.search_manager = AcademicSearchManager()
    
    def enhance_prompt_with_research(self, 
                                    topic: str, 
                                    base_prompt: str,
                                    num_sources: int = 5) -> str:
        """
        주어진 주제에 대한 학술 검색 결과로 프롬프트를 강화
        
        Args:
            topic: 검색할 주제
            base_prompt: 기본 프롬프트
            num_sources: 추가할 소스 수
        
        Returns:
            강화된 프롬프트
        """
        # 학술 검색 수행
        search_results = self.search_manager.search(
            query=topic,
            source="all",
            limit=num_sources
        )
        
        # 인용 정보 추출
        citations = self.search_manager.get_citations_for_rag(search_results)
        
        # 프롬프트에 추가할 컨텍스트 구성
        context = "## 관련 학술 자료:\n\n"
        
        for i, citation in enumerate(citations, 1):
            context += f"{i}. {citation['citation_text']}\n"
            if citation['snippet']:
                snippet = citation['snippet']
                if len(snippet) > 300:  # 스니펫이 너무 길면 자름
                    snippet = snippet[:300] + "..."
                context += f"   요약: {snippet}\n"
            context += "\n"
        
        # 강화된 프롬프트 구성
        enhanced_prompt = f"{base_prompt}\n\n{context}\n"
        enhanced_prompt += "위 학술 자료의 정보를 참고하여 응답해주세요."
        
        return enhanced_prompt
    
    def get_research_summary(self, topic: str, limit: int = 10) -> str:
        """
        주제에 대한 연구 요약 생성
        
        Args:
            topic: 검색할 주제
            limit: 검색 결과 수
            
        Returns:
            연구 요약 문자열
        """
        # 학술 검색 수행
        search_results = self.search_manager.search(
            query=topic,
            source="all",
            limit=limit
        )
        
        # 검색 결과 포맷팅
        formatted_results = self.search_manager.format_search_results(search_results)
        
        return formatted_results 
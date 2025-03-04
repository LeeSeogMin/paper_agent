import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# 환경 변수 로드
load_dotenv()

from utils.academic_search import AcademicSearchManager
from utils.rag_integration import RAGEnhancer

def run_demo():
    # 검색 관리자 초기화
    search_manager = AcademicSearchManager()
    
    # 검색 주제 설정
    search_topic = "retrieval augmented generation"
    
    print(f"검색 주제: {search_topic}")
    print("=" * 50)
    
    # Google Scholar 검색
    print("\n1. Google Scholar 검색:")
    scholar_results = search_manager.search(
        query=search_topic,
        source="scholar",
        limit=3
    )
    print(search_manager.format_search_results(scholar_results))
    
    # OpenAlex 검색
    print("\n2. OpenAlex 검색:")
    openalex_results = search_manager.search(
        query=search_topic,
        source="openalex",
        limit=3
    )
    print(search_manager.format_search_results(openalex_results))
    
    # 통합 검색
    print("\n3. 통합 검색:")
    combined_results = search_manager.search(
        query=search_topic,
        source="all",
        limit=3
    )
    print(search_manager.format_search_results(combined_results))
    
    # RAG 통합 데모
    print("\n4. RAG 프롬프트 강화 데모:")
    rag_enhancer = RAGEnhancer()
    base_prompt = f"'{search_topic}'에 대한 기술 동향과 최신 발전 상황을 설명해주세요."
    
    enhanced_prompt = rag_enhancer.enhance_prompt_with_research(
        topic=search_topic,
        base_prompt=base_prompt,
        num_sources=3
    )
    
    print("\n기본 프롬프트:")
    print(base_prompt)
    print("\n강화된 프롬프트:")
    print(enhanced_prompt)

if __name__ == "__main__":
    run_demo() 
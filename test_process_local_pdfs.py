import os
import sys
import json
from utils.pdf_processor import process_local_pdfs
from utils.logger import logger

def test_process_local_pdfs():
    """process_local_pdfs 함수 테스트"""
    # 로컬 PDF 파일 처리 테스트
    print("로컬 PDF 파일 처리 테스트 시작...")
    
    try:
        # 처리 실행
        processed_papers = process_local_pdfs(local_dir="data/local")
        
        # 결과 출력
        print(f"처리된 논문 수: {len(processed_papers)}")
        
        # 처리된 논문 정보 출력 (처음 3개만)
        for i, paper in enumerate(processed_papers[:3]):
            print(f"\n논문 {i+1}:")
            print(f"  ID: {paper['id']}")
            print(f"  제목: {paper['title']}")
            print(f"  저자: {', '.join(paper['authors'])}")
            print(f"  초록: {paper['abstract'][:100]}..." if paper['abstract'] else "  초록: 없음")
            print(f"  벡터화 여부: {paper['vectorized']}")
        
        # 벡터 데이터베이스 확인
        from utils.vector_db import list_vector_dbs
        vector_dbs = list_vector_dbs()
        print(f"\n벡터 데이터베이스 목록: {vector_dbs}")
        
        # research_papers 데이터베이스 확인
        if "research_papers" in vector_dbs:
            print("research_papers 데이터베이스가 성공적으로 생성되었습니다.")
        else:
            print("research_papers 데이터베이스가 생성되지 않았습니다.")
        
    except Exception as e:
        print(f"테스트 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    test_process_local_pdfs() 
import os
import sys
import json
from utils.pdf_processor import PDFProcessor, process_local_pdfs
from utils.logger import logger

def test_pdf_processor():
    """PDF 처리 기능 테스트"""
    # 로컬 PDF 파일 경로 설정
    local_dir = "data/local"
    
    # PDF 파일 목록 확인
    pdf_files = [f for f in os.listdir(local_dir) if f.lower().endswith('.pdf')]
    print(f"발견된 PDF 파일: {len(pdf_files)}")
    
    if not pdf_files:
        print("테스트할 PDF 파일이 없습니다.")
        return
    
    # 첫 번째와 두 번째 PDF 파일만 테스트
    test_files = pdf_files[:2]
    print(f"테스트할 파일: {test_files}")
    
    # PDF 프로세서 초기화
    pdf_processor = PDFProcessor(use_llm=True)
    
    for pdf_file in test_files:
        pdf_path = os.path.join(local_dir, pdf_file)
        print(f"\n===== 파일 테스트: {pdf_file} =====")
        
        try:
            # 1. 텍스트 추출 테스트
            print("\n1. 텍스트 추출 테스트:")
            text = pdf_processor.extract_text_from_pdf(pdf_path)
            print(f"추출된 텍스트 길이: {len(text)}")
            print(f"텍스트 미리보기: {text[:200]}...")
            
            # 2. 메타데이터 추출 테스트
            print("\n2. 메타데이터 추출 테스트:")
            metadata = pdf_processor.extract_metadata_from_pdf(pdf_path)
            print(f"추출된 메타데이터: {json.dumps(metadata, indent=2, ensure_ascii=False)}")
            
            # 3. 섹션 추출 테스트
            print("\n3. 섹션 추출 테스트:")
            sections = pdf_processor.extract_sections_from_pdf(pdf_path)
            print(f"추출된 섹션 수: {len(sections)}")
            for i, section in enumerate(sections[:2]):  # 처음 2개 섹션만 출력
                print(f"  섹션 {i+1}: {section['title']}")
                print(f"  내용 미리보기: {section['content'][:100]}...")
            
            # 4. 전체 처리 테스트
            print("\n4. 전체 처리 테스트:")
            result = pdf_processor.process_pdf(pdf_path)
            print(f"처리 성공: {result['success']}")
            if result['success']:
                print(f"메타데이터 소스: {result['metadata']['source']}")
                print(f"섹션 수: {len(result['sections'])}")
            
        except Exception as e:
            print(f"테스트 중 오류 발생: {str(e)}")
    
    # 벡터 데이터베이스 처리 테스트
    print("\n===== 벡터 데이터베이스 처리 테스트 =====")
    try:
        processed_papers = process_local_pdfs(local_dir=local_dir)
        print(f"처리된 논문 수: {len(processed_papers)}")
        for paper in processed_papers[:2]:  # 처음 2개만 출력
            print(f"  ID: {paper['id']}")
            print(f"  제목: {paper['title']}")
            print(f"  벡터화 여부: {paper['vectorized']}")
    except Exception as e:
        print(f"벡터 데이터베이스 처리 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    test_pdf_processor() 
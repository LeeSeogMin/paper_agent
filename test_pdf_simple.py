import os
import sys
import json
from utils.pdf_processor import PDFProcessor, process_local_pdfs
from utils.logger import logger

def test_simple_pdf_processor():
    """간단한 PDF 처리 테스트"""
    # 로컬 PDF 파일 경로 설정
    local_dir = "data/local"
    
    # PDF 파일 목록 확인
    pdf_files = [f for f in os.listdir(local_dir) if f.lower().endswith('.pdf')]
    print(f"발견된 PDF 파일: {len(pdf_files)}")
    
    if not pdf_files:
        print("테스트할 PDF 파일이 없습니다.")
        return
    
    # 첫 번째 PDF 파일만 테스트
    test_file = pdf_files[0]
    pdf_path = os.path.join(local_dir, test_file)
    print(f"테스트 파일: {test_file}")
    
    # PDF 프로세서 초기화
    pdf_processor = PDFProcessor(use_llm=True)
    
    try:
        # 메타데이터 추출 테스트
        print("\n메타데이터 추출 테스트:")
        metadata = pdf_processor.extract_metadata_from_pdf(pdf_path)
        print(f"추출된 메타데이터: {json.dumps(metadata, indent=2, ensure_ascii=False)}")
        
        # 메타데이터 유효성 검사
        if not metadata.get("title") or not metadata.get("authors"):
            print("메타데이터 추출 실패: 제목 또는 저자 정보 누락")
        else:
            print("메타데이터 추출 성공")
        
    except Exception as e:
        print(f"테스트 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    test_simple_pdf_processor() 
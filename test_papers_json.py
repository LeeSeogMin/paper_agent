"""
papers.json 파일 생성 테스트 스크립트
"""

import os
import sys
import json
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# 필요한 모듈 임포트
from utils.logger import logger, configure_logging
from utils.pdf_processor import process_local_pdfs
from models.research import ResearchMaterial

# 로깅 설정
configure_logging()

def save_research_materials_to_json(materials, file_path='data/papers.json'):
    """
    연구 자료를 JSON 파일로 저장
    """
    # 디렉토리 확인
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # 자료 변환
    materials_data = []
    for material in materials:
        if hasattr(material, 'dict'):
            material_dict = material.dict()
        elif isinstance(material, dict):
            material_dict = material
        else:
            print(f"예상치 못한 자료 유형: {type(material)}")
            continue
        
        materials_data.append(material_dict)
    
    # JSON 파일 저장
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(materials_data, f, ensure_ascii=False, indent=2)
    
    print(f"연구 자료가 {file_path}에 저장되었습니다.")

def main():
    """
    메인 함수
    """
    print("로컬 PDF 처리 및 papers.json 파일 생성 테스트 시작")
    
    # 로컬 PDF 처리
    local_papers = process_local_pdfs(local_dir="data/local", vector_db_path="data/vector_db")
    print(f"처리된 로컬 PDF 파일 수: {len(local_papers)}")
    
    # 연구 자료 생성
    materials = []
    for paper in local_papers:
        # 간단한 관련성 점수 (테스트용)
        relevance_score = 0.8
        
        # 연구 자료 생성
        material = {
            "id": paper.get("id", ""),
            "title": paper.get("title", ""),
            "authors": paper.get("authors", []),
            "year": paper.get("year", ""),
            "abstract": paper.get("abstract", ""),
            "content": "",
            "url": "",
            "pdf_url": "",
            "local_path": paper.get("pdf_path", ""),
            "relevance_score": relevance_score,
            "evaluation": "테스트 평가",
            "query_id": "test",
            "citation_count": 0,
            "venue": "",
            "source": "local"
        }
        
        materials.append(material)
    
    # JSON 파일로 저장
    save_research_materials_to_json(materials)
    
    # 저장된 파일 확인
    if os.path.exists('data/papers.json'):
        file_size = os.path.getsize('data/papers.json')
        print(f"papers.json 파일이 생성되었습니다. 파일 크기: {file_size} 바이트")
        
        # 파일 내용 확인
        with open('data/papers.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"저장된 연구 자료 수: {len(data)}")
    else:
        print("papers.json 파일이 생성되지 않았습니다.")

if __name__ == "__main__":
    main() 
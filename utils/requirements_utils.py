"""
사용자 요구사항 처리를 위한 유틸리티 모듈
논문 작성을 위한 사용자 요구사항을 파일에서 로드하거나 대화형으로 입력받는 기능을 제공합니다.
"""

import os
from typing import Dict, Any
from utils.logger import logger

def load_user_requirements(file_path: str = None) -> Dict[str, Any]:
    """
    사용자 요구사항을 파일에서 로드하거나 대화형으로 입력받습니다.
    
    Args:
        file_path (str, optional): 요구사항 파일 경로
        
    Returns:
        Dict[str, Any]: 사용자 요구사항 딕셔너리
    """
    requirements = {}
    
    if file_path and os.path.exists(file_path):
        logger.info(f"파일에서 사용자 요구사항 로드: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        current_key = None
        current_value = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.endswith(':'):
                # 이전 키-값 쌍 저장
                if current_key and current_key not in ["constraints", "resources"]:  # 제한사항과 참고자료 키 제외
                    requirements[current_key] = '\n'.join(current_value).strip()
                
                # 새 키 설정
                current_key = line[:-1].lower()
                current_value = []
            else:
                if current_key and current_key not in ["constraints", "resources"]:
                    current_value.append(line)
        
        # 마지막 키-값 쌍 저장
        if current_key and current_key not in ["constraints", "resources"]:
            requirements[current_key] = '\n'.join(current_value).strip()
    else:
        logger.info("대화형으로 사용자 요구사항 입력 받기")
        print("\n=== 논문 작성 요구사항 입력 ===")
        
        requirements["topic"] = input("연구 주제: ")
        requirements["task"] = requirements["topic"]  # task 필드 추가
        
        print("\n논문 유형 (선택):")
        print("1. 문헌 리뷰")
        print("2. 실험 연구")
        print("3. 사례 연구")
        print("4. 이론적 분석")
        print("5. 기타")
        paper_type = input("선택 (1-5): ")
        
        if paper_type == "1":
            requirements["paper_type"] = "literature_review"
        elif paper_type == "2":
            requirements["paper_type"] = "experimental_research"
        elif paper_type == "3":
            requirements["paper_type"] = "case_study"
        elif paper_type == "4":
            requirements["paper_type"] = "theoretical_analysis"
        elif paper_type == "5":
            requirements["paper_type"] = input("논문 유형 직접 입력: ")
        
        requirements["additional_instructions"] = input("\n추가 지시사항 (없으면 Enter): ")
        
        print("\n=== 입력 완료 ===\n")
    
    return requirements 
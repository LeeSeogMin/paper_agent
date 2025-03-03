"""
논문 작성 AI 에이전트 시스템의 메인 애플리케이션 모듈
"""

import os
import argparse
from typing import Dict, Any
import sys
from pathlib import Path

from utils.logger import logger, configure_logging
from agents.coordinator_agent import CoordinatorAgent
from config.settings import OUTPUT_DIR
from config.api_keys import check_required_api_keys
from utils import ensure_directories_exist

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def parse_arguments():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description="논문 작성 AI 에이전트 시스템")
    parser.add_argument("--verbose", action="store_true", help="상세 로깅 활성화")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR, help="출력 디렉토리 경로")
    parser.add_argument("--requirements", type=str, help="사용자 요구사항 파일 경로")
    return parser.parse_args()

def load_user_requirements(file_path: str = None) -> Dict[str, Any]:
    """사용자 요구사항 로드"""
    requirements = {
        "constraints": [],  # 기본 제약사항 추가
        "resources": [],    # 기본 리소스 추가
    }
    
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
                if current_key:
                    requirements[current_key] = '\n'.join(current_value).strip()
                
                # 새 키 설정
                current_key = line[:-1].lower()
                current_value = []
            else:
                if current_key:
                    current_value.append(line)
        
        # 마지막 키-값 쌍 저장
        if current_key:
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
        
        # 제약사항 입력 추가
        print("\n제약사항 (선택사항, 완료시 빈 입력):")
        while True:
            constraint = input("제약사항: ").strip()
            if not constraint:
                break
            requirements["constraints"].append(constraint)
            
        # 참고자료 입력 추가
        print("\n참고자료 (선택사항, 완료시 빈 입력):")
        while True:
            resource = input("참고자료 URL 또는 DOI: ").strip()
            if not resource:
                break
            requirements["resources"].append(resource)
        
        requirements["additional_instructions"] = input("\n추가 지시사항 (없으면 Enter): ")
        
        print("\n=== 입력 완료 ===\n")
    
    return requirements

def main():
    """메인 애플리케이션 함수"""
    # 필요한 API 키 확인
    api_keys_available, missing_keys = check_required_api_keys()
    if not api_keys_available:
        logger.error(f"Cannot start application. Missing required API keys: {', '.join(missing_keys)}")
        logger.error("Please set the required API keys in your environment or .env file.")
        return
    
    # 필요한 디렉토리 생성
    ensure_directories_exist()
    
    # 인수 파싱
    args = parse_arguments()
    
    # 로깅 설정
    log_level = "DEBUG" if args.verbose else "INFO"
    configure_logging(log_level=log_level)
    
    # 사용자 요구사항 로드
    user_requirements = load_user_requirements(args.requirements)
    
    # 총괄 에이전트 초기화
    coordinator = CoordinatorAgent(verbose=args.verbose)
    
    # 사용자 요구사항 처리
    logger.info("총괄 에이전트에 사용자 요구사항 전달")
    try:
        result = coordinator.process_user_requirements(user_requirements)
        
        # 결과 검증 추가
        if not result.get("outline") or not result.get("content"):
            logger.error("논문 생성 실패: 필수 결과물이 생성되지 않았습니다")
            return 1
            
        # 결과 출력
        logger.info(f"처리 결과: {result.get('status', 'unknown')}")
        
        if result.get("status") == "completed":
            logger.info("논문 작성 프로세스 완료")
            # 결과물 저장
            output_path = Path(args.output) / f"paper_{result.get('id', 'output')}.md"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result.get("content"))
            logger.info(f"논문이 저장되었습니다: {output_path}")
        else:
            logger.warning("논문 작성 프로세스 실패 또는 미완료")
            return 1
            
    except Exception as e:
        logger.error(f"논문 작성 중 오류 발생: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
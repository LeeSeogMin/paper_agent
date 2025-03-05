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
from utils.serpapi_scholar import ScholarSearchTool
from utils.openalex_api import OpenAlexTool

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

def main():
    """메인 애플리케이션 함수"""
    # 필요한 API 키 확인
    api_keys_available, missing_keys = check_required_api_keys()
    if not api_keys_available:
        logger.error(f"Cannot start application. Missing required API keys: {', '.join(missing_keys)}")
        logger.error("Please set the required API keys in your environment or .env file.")
        return 1
    
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
        # 총괄 에이전트가 사용자 요구사항을 분석하여 작업 계획 수립
        logger.info("총괄 에이전트가 사용자 요구사항 분석 중...")
        
        # 1. 연구 계획 수립
        study_plan = coordinator.create_study_plan(user_requirements)
        logger.info("연구 계획 수립 완료")
        
        # 2. 보고서 양식 결정
        report_format = coordinator.create_report_format(user_requirements)
        logger.info("보고서 양식 결정 완료")
        
        # 3. 각 에이전트에게 계획 전달
        logger.info("조사 에이전트에게 연구 계획 전달 중...")
        coordinator.delegate_to_research_agent(study_plan)
        
        logger.info("작성 에이전트에게 보고서 양식 전달 중...")
        coordinator.delegate_to_writing_agent(report_format)
        
        # 4. 전체 프로세스 실행
        logger.info("논문 작성 프로세스 시작...")
        result = coordinator.process_user_requirements(user_requirements)
        
        # 결과 검증
        if not result.get("content"):
            logger.error("논문 생성 실패: 결과물이 생성되지 않았습니다")
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

# Google Scholar 검색 예시
def search_google_scholar(query, num_results=5, year_start=None, year_end=None):
    scholar_tool = ScholarSearchTool()
    results = scholar_tool.search_scholar(
        query=query,
        num_results=num_results,
        year_start=year_start,
        year_end=year_end
    )
    
    formatted_results = scholar_tool.format_results(results)
    return results, formatted_results

# OpenAlex API 검색 예시
def search_openalex(query, limit=10, filter_options=None):
    openalex_tool = OpenAlexTool()
    # 선택적으로 이메일 설정 (Polite Pool 사용)
    # openalex_tool.set_email("your-email@example.com")
    
    results = openalex_tool.search_works(
        query=query,
        limit=limit,
        filter_options=filter_options
    )
    
    formatted_results = openalex_tool.format_paper_results(results)
    return results, formatted_results

# OpenAlex API로 저자 검색 예시
def search_openalex_authors(author_name, limit=10):
    openalex_tool = OpenAlexTool()
    results = openalex_tool.search_authors(
        query=author_name,
        limit=limit
    )
    return results

if __name__ == "__main__":
    exit(main())
#!/usr/bin/env python
"""
논문 작성 AI 에이전트 시스템 실행 스크립트
웹 인터페이스 또는 콘솔 모드로 시스템을 실행할 수 있습니다.
"""

import os
import sys
import argparse
import traceback
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# 로깅 설정
from utils.logger import logger, configure_logging
from utils import ensure_directories_exist
from config.api_keys import check_required_api_keys

def parse_arguments():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description="논문 작성 AI 에이전트 시스템")
    
    # 모드 선택 인수
    parser.add_argument("--mode", type=str, choices=["web", "console"], default="console",
                      help="실행 모드 (web: 웹 인터페이스, console: 콘솔 애플리케이션)")
    
    # 공통 인수
    parser.add_argument("--verbose", action="store_true", help="상세 로깅 활성화")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                      default="INFO", help="로그 레벨 설정")
    
    # 웹 모드 인수
    parser.add_argument("--port", type=int, default=8080, help="웹 서버 포트 (웹 모드 전용)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="웹 서버 호스트 (웹 모드 전용)")
    parser.add_argument("--debug", action="store_true", help="Flask 디버그 모드 활성화 (웹 모드 전용)")
    
    # 콘솔 모드 인수
    parser.add_argument("--topic", type=str, help="연구 주제 (콘솔 모드 전용)")
    parser.add_argument("--paper-type", type=str, default="literature_review", 
                      choices=["literature_review", "experimental_research", "case_study", "theoretical_analysis"],
                      help="논문 유형 (콘솔 모드 전용)")
    parser.add_argument("--output-format", type=str, default="markdown",
                      choices=["markdown", "docx", "pdf", "html"],
                      help="출력 형식 (콘솔 모드 전용)")
    parser.add_argument("--requirements", type=str, help="사용자 요구사항 파일 경로 (콘솔 모드 전용)")
    
    return parser.parse_args()

def run_web_app(args):
    """웹 애플리케이션 실행"""
    try:
        from web.app import app
        
        logger.info(f"웹 애플리케이션을 {args.host}:{args.port}에서 시작합니다")
        app.run(host=args.host, port=args.port, debug=args.debug)
        return 0
    except Exception as e:
        logger.error(f"웹 애플리케이션 실행 중 오류 발생: {str(e)}")
        if args.verbose:
            logger.error(traceback.format_exc())
        return 1

def run_console_app(args):
    """콘솔 애플리케이션 실행"""
    try:
        # 사용자 요구사항 로드
        from utils.requirements_utils import load_user_requirements
        user_requirements = load_user_requirements(args.requirements)
        
        # topic 인수가 제공된 경우 적용
        if args.topic:
            user_requirements["topic"] = args.topic
            user_requirements["task"] = args.topic
            
        # paper_type 인수가 제공된 경우 적용
        if args.paper_type:
            user_requirements["paper_type"] = args.paper_type
            
        logger.info(f"주제 '{user_requirements.get('topic', '지정되지 않음')}'에 대한 논문 작성을 시작합니다")
        
        # 총괄 에이전트 초기화 및 워크플로우 시작
        from agents.coordinator_agent import CoordinatorAgent
        coordinator = CoordinatorAgent(verbose=args.verbose)
        
        # 전체 프로세스 실행
        logger.info("논문 작성 프로세스 시작...")
        result = coordinator.process_user_requirements(user_requirements)
        
        # 결과 검증
        if not result or not result.get("content"):
            logger.error("논문 생성 실패: 결과물이 생성되지 않았습니다")
            return 1
            
        # 결과 출력
        logger.info(f"처리 결과: {result.get('status', 'unknown')}")
        
        if result.get("status") == "completed":
            logger.info("논문 작성 프로세스 완료")
            # 결과물 저장
            from config.settings import OUTPUT_DIR
            output_path = Path(OUTPUT_DIR) / f"paper_{result.get('id', 'output')}.md"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result.get("content"))
            logger.info(f"논문이 저장되었습니다: {output_path}")
            return 0
        else:
            logger.warning("논문 작성 프로세스 실패 또는 미완료")
            return 1
    except Exception as e:
        logger.error(f"콘솔 애플리케이션 실행 중 오류 발생: {str(e)}")
        if args.verbose:
            logger.error(traceback.format_exc())
        return 1

def main():
    """메인 함수"""
    # 인수 파싱
    args = parse_arguments()
    
    # 로깅 설정
    log_level = "DEBUG" if args.verbose else args.log_level
    configure_logging(log_level=log_level)
    
    # 필요한 API 키 확인
    api_keys_available, missing_keys = check_required_api_keys()
    if not api_keys_available:
        logger.error(f"필수 API 키가 누락되어 애플리케이션을 시작할 수 없습니다: {', '.join(missing_keys)}")
        logger.error("환경 변수 또는 .env 파일에 필요한 API 키를 설정하세요.")
        return 1
    
    # 필요한 디렉토리 생성
    ensure_directories_exist()
    
    # 선택된 모드에 따라 실행
    if args.mode == "web":
        return run_web_app(args)
    else:  # console 모드
        return run_console_app(args)

if __name__ == "__main__":
    exit(main()) 
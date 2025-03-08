#!/usr/bin/env python
"""
vectorize_materials 함수를 테스트하는 스크립트
"""

import os
import sys
import json
import logging
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# 로깅 설정
from utils.logger import logger, configure_logging

# 필요한 모듈 임포트
from agents.research_agent import ResearchAgent
from models.research import ResearchMaterial

def main():
    # 로깅 설정
    configure_logging(log_level="DEBUG")
    logger.info("vectorize_materials 테스트 시작")
    
    # ResearchAgent 초기화
    agent = ResearchAgent(model_name="gpt-4o-mini", temperature=0.2, verbose=True)
    
    # 테스트용 ResearchMaterial 생성
    test_materials = [
        ResearchMaterial(
            id="test1",
            title="토픽 모델링의 역사",
            authors=["김연구", "이학자"],
            year=2020,
            abstract="이 논문은 토픽 모델링의 역사를 다룹니다.",
            url="https://example.com/paper1",
            content="토픽 모델링은 문서 집합에서 주제를 추출하는 기술입니다. LDA(Latent Dirichlet Allocation)는 가장 널리 사용되는 토픽 모델링 알고리즘 중 하나입니다. " * 20
        ),
        ResearchMaterial(
            id="test2",
            title="토픽 모델링의 최신 동향",
            authors=["박학자", "최연구"],
            year=2021,
            abstract="이 논문은 토픽 모델링의 최신 동향을 살펴봅니다.",
            url="https://example.com/paper2",
            content="최근 토픽 모델링 연구는 신경망 기반 방법론으로 발전하고 있습니다. BERTopic과 같은 방법론은 BERT와 같은 사전 훈련된 언어 모델을 활용합니다. " * 20
        )
    ]
    
    # vectorize_materials 함수 테스트
    logger.info(f"총 {len(test_materials)}개 연구 자료 벡터화 테스트")
    result = agent.vectorize_materials(test_materials)
    
    logger.info(f"벡터화 결과: {result}")
    logger.info("테스트 완료")

if __name__ == "__main__":
    main() 
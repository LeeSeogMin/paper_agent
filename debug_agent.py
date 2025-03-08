import os
import logging
import traceback
from agents.research_agent import ResearchAgent
from models.research import ResearchMaterial
from utils.vector_db import process_and_vectorize_paper, search_vector_db, vectorize_content

# 로깅 설정
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        # ResearchAgent 인스턴스 생성
        agent = ResearchAgent(model_name="gpt-4o-mini", verbose=True)
        
        # 테스트용 연구 주제
        topic = "Topic models for topic modeling tasks are highly diverse. In particular, there are statistical topic models that emerged before the advent of Large Language Models (LLMs) and semantic-based topic models that developed after the introduction of LLMs. I aim to examine in detail the history of these topic models, as well as the characteristics, methods, and significance of each model."
        
        logger.info(f"연구 주제: {topic}")
        
        # 전체 프로세스 실행
        run_full_process(agent, topic)
        
    except Exception as e:
        logger.error(f"전체 프로세스 중 오류 발생: {str(e)}")
        traceback.print_exc()

def run_full_process(agent, topic):
    # 1. 자료 수집 단계
    logger.info("1. 연구 자료 수집 시작...")
    materials = collect_materials(agent, topic)
    if not materials:
        logger.error("자료 수집에 실패했습니다.")
        return
    
    # 2. 자료 강화 단계
    logger.info("2. 연구 자료 강화 시작...")
    enriched_materials = enrich_materials(agent, materials)
    if not enriched_materials:
        logger.error("자료 강화에 실패했습니다.")
        # 원본 자료라도 저장
        agent.save_research_materials_to_json(materials)
        return
    
    # 3. 벡터화 및 벡터 DB 저장 단계
    logger.info("3. 자료 벡터화 및 DB 저장 시작...")
    vectorize_materials(enriched_materials)
    
    # 4. 자료 분석 단계
    logger.info("4. 연구 자료 분석 시작...")
    analysis = analyze_materials(agent, enriched_materials, topic)
    if not analysis:
        logger.error("자료 분석에 실패했습니다.")
        return
    
    # 5. 보고서 작성 단계
    logger.info("5. 문헌 리뷰 보고서 작성 시작...")
    report = agent.generate_report(topic, analysis, enriched_materials)
    if not report:
        logger.error("보고서 작성에 실패했습니다.")
        return
    
    logger.info("모든 단계가 성공적으로 완료되었습니다.")

def collect_materials(agent, topic):
    try:
        materials = agent.collect_research_materials(
            topic, 
            max_queries=3,
            results_per_source=5,
            final_result_count=7
        )
        
        logger.info(f"수집된 자료 수: {len(materials)}")
        return materials
        
    except Exception as e:
        logger.error(f"자료 수집 중 오류 발생: {str(e)}")
        traceback.print_exc()
        return None

def enrich_materials(agent, materials):
    try:
        enriched_materials = agent.enrich_research_materials(materials)
        logger.info(f"강화된 자료 수: {len(enriched_materials)}")
        
        # JSON으로 저장
        logger.info("자료를 JSON으로 저장 중...")
        agent.save_research_materials_to_json(enriched_materials)
        logger.info("자료 저장 완료")
        
        return enriched_materials
        
    except Exception as e:
        logger.error(f"자료 강화 중 오류 발생: {str(e)}")
        traceback.print_exc()
        return None

def vectorize_materials(materials):
    try:
        vectorized_count = 0
        for material in materials:
            try:
                # PDF URL이 있는 경우 PDF 벡터화
                if material.pdf_url:
                    logger.info(f"PDF 벡터화 중: {material.title}")
                    process_and_vectorize_paper(material.pdf_url)
                    vectorized_count += 1
                # 콘텐츠가 있는 경우 텍스트 벡터화
                elif material.content and len(material.content) > 100:
                    logger.info(f"콘텐츠 벡터화 중: {material.title}")
                    vectorize_content(
                        content=material.content,
                        title=material.title,
                        material_id=material.id
                    )
                    vectorized_count += 1
                else:
                    logger.warning(f"벡터화 불가: {material.title} - 콘텐츠 또는 PDF URL 필요")
            except Exception as e:
                logger.warning(f"자료 벡터화 중 오류 발생, 건너뜁니다: {str(e)}")
                continue
                
        logger.info(f"벡터화 완료: {vectorized_count}/{len(materials)} 자료")
        return True
        
    except Exception as e:
        logger.error(f"벡터화 과정 중 오류 발생: {str(e)}")
        traceback.print_exc()
        return False

def analyze_materials(agent, materials, topic):
    try:
        logger.info("연구 자료 분석 중...")
        analysis = agent.analyze_research_materials(materials, topic)
        if not analysis:
            logger.error("연구 자료 분석에 실패했습니다.")
            return None
            
        logger.info("연구 자료 분석 완료")
        return analysis
        
    except Exception as e:
        logger.error(f"자료 분석 중 오류 발생: {str(e)}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main() 
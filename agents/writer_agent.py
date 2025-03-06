from typing import List
from utils.logger import logger
from models.research import ResearchMaterial
from models.paper import Paper, Section, PaperMetadata, PaperSection

class WriterAgent:
    def run(self, topic: str, research_materials: List[ResearchMaterial] = None, **kwargs):
        # 연구 자료가 없을 경우 기본 내용 생성
        if not research_materials:
            logger.warning("연구 자료 없이 기본 템플릿 생성")
            return self.generate_basic_template(topic)
        
        # 기존 로직 유지
        return self.write_paper(topic, research_materials, **kwargs)

    def generate_basic_template(self, topic: str) -> Paper:
        """기본 템플릿 생성 메서드 추가"""
        return Paper(
            metadata=PaperMetadata(
                title=f"{topic} 연구 보고서",
                authors=["AI Writer"],
                abstract=f"본 보고서는 {topic}에 대한 기초적인 조사 결과를 담고 있습니다.",
                keywords=[topic, "연구", "보고서"]
            ),
            sections=[
                PaperSection(
                    section_id="intro",
                    title="서론",
                    content=f"본 보고서는 {topic}에 대한 기초적인 조사를 목적으로 합니다."
                ),
                PaperSection(
                    section_id="conclusion",
                    title="결론",
                    content="추가 연구가 필요합니다."
                )
            ],
            references=[]
        )

    def write_paper(self, topic: str, research_materials: List[ResearchMaterial], **kwargs):
        # Implementation of write_paper method
        pass

    # Add any other necessary methods here 
"""
총괄 에이전트 모듈
전체 논문 작성 프로세스를 조정하고 관리하는 에이전트입니다.
"""

import os
import re
import json
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from pydantic import BaseModel, Field

from langchain_core.runnables import RunnableConfig
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser

from config.settings import OUTPUT_DIR
from utils.logger import logger
from models.paper import Paper, PaperOutline
from models.research import ResearchMaterial, ResearchSummary
from models.state import PaperWorkflowState
from agents.base import BaseAgent
from agents.researcher import ResearchAgent
from agents.writer import WriterAgent
from agents.editor import EditorAgent, StyleGuide
from graphs.paper_writing import PaperWritingGraph


class ResearchPlan(BaseModel):
    """연구 계획 형식"""
    topic: str = Field(description="연구 주제")
    research_questions: List[str] = Field(description="연구 질문 목록")
    expected_outcomes: List[str] = Field(description="예상되는 결과")
    rationale: str = Field(description="연구 계획 근거")


class ProjectStatus(BaseModel):
    """프로젝트 상태 형식"""
    status: str = Field(description="현재 진행 상태")
    completed_steps: List[str] = Field(description="완료된 단계")
    current_step: str = Field(description="현재 진행 중인 단계")
    next_steps: List[str] = Field(description="다음 단계")
    issues: Optional[List[str]] = Field(description="현재 이슈", default=None)
    progress_percentage: float = Field(description="전체 진행률 (0-100)")


class CoordinatorAgent(BaseAgent[PaperWorkflowState]):
    """논문 작성 프로세스를 총괄하는 에이전트"""

    def __init__(
        self,
        name: str = "총괄 에이전트",
        description: str = "논문 작성 프로세스 조정 및 관리",
        verbose: bool = False
    ):
        """
        CoordinatorAgent 초기화

        Args:
            name (str, optional): 에이전트 이름. 기본값은 "총괄 에이전트"
            description (str, optional): 에이전트 설명. 기본값은 "논문 작성 프로세스 조정 및 관리"
            verbose (bool, optional): 상세 로깅 활성화 여부. 기본값은 False
        """
        super().__init__(name, description, verbose=verbose)
        
        # 출력 디렉토리 생성
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # 프롬프트 초기화
        self._init_prompts()
        
        # 워크플로우 그래프 초기화
        self.graph = PaperWritingGraph(name="논문 작성 워크플로우")
        
        # 상태 저장소 초기화
        self.workflow_state = None
        
        logger.info(f"{self.name} 초기화 완료")

    def _init_prompts(self) -> None:
        """프롬프트와 체인 초기화"""
        # 연구 계획 파서 초기화
        self.research_plan_parser = PydanticOutputParser(pydantic_object=ResearchPlan)
        
        # 상태 파서 초기화
        self.status_parser = PydanticOutputParser(pydantic_object=ProjectStatus)
        
        # 연구 계획 체인 초기화
        self.research_plan_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template="""당신은 학술 연구 계획 전문가입니다. 
                주어진 주제에 대한 체계적인 연구 계획을 작성해 주세요.
                
                주제: {topic}
                
                연구 계획에는 다음이 포함되어야 합니다:
                1. 주제
                2. 주요 연구 질문 (3-5개)
                3. 예상되는 연구 결과
                4. 연구 계획의 근거
                
                {format_instructions}
                """,
                input_variables=["topic", "format_instructions"],
            ),
            verbose=self.verbose
        )
        
        # 논문 요약 체인 초기화
        self.summary_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template="""다음 논문을 요약해 주세요:
                
                제목: {title}
                
                논문 내용:
                {content}
                
                300단어 이내로 핵심 내용, 주요 발견, 그리고 결론을 요약해 주세요.
                """,
                input_variables=["title", "content"],
            ),
            verbose=self.verbose
        )
        
        logger.debug("총괄 에이전트 프롬프트 및 체인 초기화 완료")

    def create_research_plan(self, topic: str) -> ResearchPlan:
        """
        연구 주제에 대한 연구 계획을 생성합니다.

        Args:
            topic (str): 연구 주제

        Returns:
            ResearchPlan: 생성된 연구 계획
        """
        logger.info(f"주제 '{topic}'에 대한 연구 계획 생성 중...")
        
        try:
            format_instructions = self.research_plan_parser.get_format_instructions()
            
            result = self.research_plan_chain.invoke({
                "topic": topic,
                "format_instructions": format_instructions
            })
            
            research_plan = self.research_plan_parser.parse(result["text"])
            
            logger.info(f"연구 계획 생성 완료: {len(research_plan.research_questions)}개 연구 질문 포함")
            return research_plan
            
        except Exception as e:
            logger.error(f"연구 계획 생성 중 오류 발생: {str(e)}")
            
            # 오류 발생 시 기본 계획 반환
            return ResearchPlan(
                topic=topic,
                research_questions=["이 주제의 주요 개념은 무엇인가?", "이 주제의 현재 연구 동향은 어떠한가?"],
                expected_outcomes=["주제에 대한 포괄적인 이해", "향후 연구 방향 제안"],
                rationale="기본 연구 계획"
            )

    def start_workflow(
        self, 
        topic: str,
        template_name: Optional[str] = None,
        style_guide: Optional[str] = None,
        citation_style: Optional[str] = None,
        output_format: str = "markdown",
        verbose: bool = False
    ) -> PaperWorkflowState:
        """
        논문 작성 워크플로우를 시작합니다.

        Args:
            topic (str): 논문 주제
            template_name (Optional[str], optional): 논문 템플릿 이름
            style_guide (Optional[str], optional): 스타일 가이드 이름
            citation_style (Optional[str], optional): 인용 스타일
            output_format (str, optional): 출력 형식. 기본값은 "markdown"
            verbose (bool, optional): 상세 로깅 활성화 여부. 기본값은 False

        Returns:
            PaperWorkflowState: 워크플로우 상태
        """
        logger.info(f"주제 '{topic}'에 대한 논문 작성 워크플로우 시작")
        
        # 연구 계획 생성
        research_plan = self.create_research_plan(topic)
        
        # 워크플로우 실행
        workflow_state = self.graph.run_workflow(
            topic=topic,
            template_name=template_name,
            style_guide=style_guide,
            citation_style=citation_style,
            output_format=output_format,
            verbose=verbose
        )
        
        # 상태 저장
        self.workflow_state = workflow_state
        
        logger.info(f"워크플로우 실행 완료: 상태 - {workflow_state.status}")
        return workflow_state

    def get_workflow_status(self) -> ProjectStatus:
        """
        현재 워크플로우 상태를 가져옵니다.

        Returns:
            ProjectStatus: 프로젝트 상태
        """
        if not self.workflow_state:
            return ProjectStatus(
                status="not_started",
                completed_steps=[],
                current_step="none",
                next_steps=["start_workflow"],
                progress_percentage=0.0
            )
        
        state = self.workflow_state
        status = state.status
        
        # 완료된 단계와 현재 단계 결정
        completed_steps = []
        current_step = ""
        next_steps = []
        progress_percentage = 0.0
        
        if status == "initialized":
            current_step = "initialization"
            next_steps = ["research"]
            progress_percentage = 10.0
        elif status == "research_completed":
            completed_steps = ["initialization", "research"]
            current_step = "writing_preparation"
            next_steps = ["writing"]
            progress_percentage = 30.0
        elif status == "writing_completed":
            completed_steps = ["initialization", "research", "writing"]
            current_step = "editing_preparation"
            next_steps = ["editing"]
            progress_percentage = 60.0
        elif status == "editing_completed":
            completed_steps = ["initialization", "research", "writing", "editing"]
            current_step = "completion"
            next_steps = []
            progress_percentage = 100.0
        
        # 이슈 확인
        issues = None
        if state.error:
            issues = [state.error]
        
        return ProjectStatus(
            status=status,
            completed_steps=completed_steps,
            current_step=current_step,
            next_steps=next_steps,
            issues=issues,
            progress_percentage=progress_percentage
        )

    def get_research_summary(self) -> Optional[str]:
        """
        현재 연구 요약을 가져옵니다.

        Returns:
            Optional[str]: 연구 요약 또는 None
        """
        if not self.workflow_state or not self.workflow_state.research_summary:
            return None
        
        research_summary = self.workflow_state.research_summary
        
        summary_text = f"## 연구 주제: {research_summary.topic}\n\n"
        
        # 주요 발견 추가
        summary_text += "### 주요 발견\n\n"
        for finding in research_summary.key_findings:
            summary_text += f"- {finding}\n"
        
        # 수집된 자료 추가
        summary_text += f"\n### 수집된 자료 ({len(research_summary.collected_materials)}개)\n\n"
        for i, material in enumerate(research_summary.collected_materials, 1):
            summary_text += f"{i}. **{material.title}**"
            if material.authors:
                summary_text += f" - {', '.join(material.authors)}"
            if material.year:
                summary_text += f" ({material.year})"
            summary_text += f"\n   {material.summary[:150]}...\n\n"
        
        # 연구 격차 및 다음 단계 추가
        if research_summary.gaps:
            summary_text += "### 식별된 연구 격차\n\n"
            for gap in research_summary.gaps:
                summary_text += f"- {gap}\n"
        
        if research_summary.next_steps:
            summary_text += "\n### 권장 다음 단계\n\n"
            for step in research_summary.next_steps:
                summary_text += f"- {step}\n"
        
        return summary_text

    def get_paper_summary(self) -> Optional[str]:
        """
        현재 논문 요약을 가져옵니다.

        Returns:
            Optional[str]: 논문 요약 또는 None
        """
        if not self.workflow_state or not self.workflow_state.paper:
            return None
        
        paper = self.workflow_state.paper
        
        # 논문 내용 추출
        paper_content = f"# {paper.title}\n\n"
        paper_content += f"## 초록\n\n{paper.abstract}\n\n"
        
        for section in paper.sections:
            paper_content += f"## {section.title}\n\n"
            # 섹션 내용을 요약을 위해 최대 1000자로 제한
            content_preview = section.content[:1000] + "..." if len(section.content) > 1000 else section.content
            paper_content += f"{content_preview}\n\n"
        
        try:
            # 요약 생성
            result = self.summary_chain.invoke({
                "title": paper.title,
                "content": paper_content
            })
            
            summary = result["text"]
            
            # 추가 메타데이터 포함
            full_summary = f"## 논문 요약: {paper.title}\n\n"
            full_summary += f"{summary}\n\n"
            
            # 섹션 구조 추가
            full_summary += "### 논문 구조\n\n"
            for i, section in enumerate(paper.sections, 1):
                full_summary += f"{i}. {section.title}\n"
            
            # 참고 문헌 정보 추가
            if paper.references:
                full_summary += f"\n### 참고 문헌 ({len(paper.references)}개)\n\n"
            
            return full_summary
            
        except Exception as e:
            logger.error(f"논문 요약 생성 중 오류 발생: {str(e)}")
            
            # 기본 요약 반환
            return f"논문 제목: {paper.title}\n섹션 수: {len(paper.sections)}\n참고 문헌 수: {len(paper.references)}"

    def handle_user_feedback(self, feedback: str) -> Dict[str, Any]:
        """
        사용자 피드백을 처리합니다.

        Args:
            feedback (str): 사용자 피드백

        Returns:
            Dict[str, Any]: 피드백 처리 결과
        """
        logger.info("사용자 피드백 처리 중...")
        
        if not self.workflow_state or not self.workflow_state.paper:
            return {
                "success": False,
                "message": "현재 작업 중인 논문이 없습니다."
            }
        
        try:
            # 피드백 분석 (실제로는 더 복잡한 로직이 필요할 수 있음)
            is_positive = any(keyword in feedback.lower() for keyword in ["좋", "만족", "훌륭", "좋아"])
            needs_revision = any(keyword in feedback.lower() for keyword in ["수정", "고치", "변경", "개선", "추가"])
            
            result = {
                "success": True,
                "is_positive": is_positive,
                "needs_revision": needs_revision,
                "feedback": feedback
            }
            
            # 피드백 저장
            if not self.workflow_state.metadata:
                self.workflow_state.metadata = {}
            
            if "feedback_history" not in self.workflow_state.metadata:
                self.workflow_state.metadata["feedback_history"] = []
            
            self.workflow_state.metadata["feedback_history"].append({
                "timestamp": self._get_timestamp(),
                "feedback": feedback,
                "analysis": result
            })
            
            logger.info("사용자 피드백 처리 완료")
            return result
            
        except Exception as e:
            logger.error(f"피드백 처리 중 오류 발생: {str(e)}")
            return {
                "success": False,
                "message": f"피드백 처리 중 오류 발생: {str(e)}"
            }

    def _get_timestamp(self) -> str:
        """현재 타임스탬프를 반환합니다."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    def run(
        self, 
        input_data: Dict[str, Any], 
        config: Optional[RunnableConfig] = None
    ) -> PaperWorkflowState:
        """
        총괄 에이전트를 실행합니다.

        Args:
            input_data (Dict[str, Any]): 입력 데이터 (주제 및 설정 포함)
            config (Optional[RunnableConfig], optional): 실행 구성

        Returns:
            PaperWorkflowState: 워크플로우 상태
        """
        logger.info("총괄 에이전트 실행 중...")
        
        # 입력 데이터 파싱
        topic = input_data.get("topic", "")
        if not topic:
            raise ValueError("논문 주제가 제공되지 않았습니다.")
        
        template_name = input_data.get("template_name")
        style_guide = input_data.get("style_guide")
        citation_style = input_data.get("citation_style")
        output_format = input_data.get("output_format", "markdown")
        verbose = input_data.get("verbose", False)
        
        # 워크플로우 시작
        workflow_state = self.start_workflow(
            topic=topic,
            template_name=template_name,
            style_guide=style_guide,
            citation_style=citation_style,
            output_format=output_format,
            verbose=verbose
        )
        
        # 상태 업데이트
        self.update_state({
            "topic": topic,
            "workflow_state": workflow_state
        })
        
        logger.info("총괄 에이전트 실행 완료")
        return workflow_state
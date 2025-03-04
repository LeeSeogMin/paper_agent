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
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

from config.settings import OUTPUT_DIR
from utils.logger import logger
from models.paper import Paper, PaperOutline
from models.research import ResearchMaterial, ResearchSummary
from models.state import PaperWorkflowState
from agents.base import BaseAgent
from agents.research_agent import ResearchAgent
from agents.writing_agent import WriterAgent
from agents.editing_agent import EditorAgent, StyleGuide
from prompts.agent_prompts import (
    AGENT_PLANNING_PROMPT,
    AGENT_COMMUNICATION_PROMPT,
    AGENT_ERROR_HANDLING_PROMPT,
    AGENT_SELF_EVALUATION_PROMPT
)

# 타입 힌트만을 위한 import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from graphs.paper_writing import PaperWritingGraph


class WorkflowRunner:
    """워크플로우 실행을 위한 래퍼 클래스"""
    def __init__(self, name: str):
        self.name = name
        self._graph = None

    def _init_graph(self):
        if self._graph is None:
            from graphs.paper_writing import PaperWritingGraph
            self._graph = PaperWritingGraph(name=self.name)
        return self._graph

    def run_workflow(self, **kwargs) -> PaperWorkflowState:
        graph = self._init_graph()
        return graph.run_workflow(**kwargs)


class ResearchPlan(BaseModel):
    """문제 해결 계획 형식"""
    problem_statement: str = Field(description="명확히 정의된 문제 진술")
    analysis_approach: List[str] = Field(description="문제 분석을 위한 접근 방법")
    required_data_sources: List[str] = Field(description="필요한 데이터 및 자료 유형")
    implementation_steps: List[str] = Field(description="단계별 실행 계획")
    expected_outcomes: List[str] = Field(description="예상되는 결과물")


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
        """CoordinatorAgent 초기화"""
        super().__init__(name, description, verbose=verbose)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self._init_prompts()
        self.workflow_runner = WorkflowRunner("논문 작성 워크플로우")
        self.workflow_state = None
        
        # 필요한 에이전트들 초기화
        self._init_agents()
        
        logger.info(f"{self.name} 초기화 완료")

    def _init_prompts(self) -> None:
        """프롬프트와 체인 초기화"""
        self.research_plan_parser = PydanticOutputParser(pydantic_object=ResearchPlan)
        self.status_parser = PydanticOutputParser(pydantic_object=ProjectStatus)
        
        # 기존 프롬프트 대신 agent_prompts.py의 프롬프트 활용
        self.research_plan_chain = LLMChain(
            llm=self.llm,
            prompt=AGENT_PLANNING_PROMPT,
            verbose=self.verbose
        )
        
        self.summary_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template="""다음 논문을 요약해 주세요:
                
                제목: {title}
                
                논문 내용:
                {content}
                
                300단어 이내로 핵심 내용, 주요 발견, 그리고 결론을 요약해 주세요.""",
                input_variables=["title", "content"],
            ),
            verbose=self.verbose
        )
        
        # 에러 처리 체인 추가
        self.error_handling_chain = LLMChain(
            llm=self.llm,
            prompt=AGENT_ERROR_HANDLING_PROMPT,
            verbose=self.verbose
        )
        
        logger.debug("총괄 에이전트 프롬프트 및 체인 초기화 완료")

    def _init_agents(self):
        """필요한 에이전트들 초기화"""
        from agents import ResearchAgent, WriterAgent, EditorAgent, ReviewAgent
        
        self.research_agent = ResearchAgent(verbose=self.verbose)
        self.writer_agent = WriterAgent(verbose=self.verbose)
        self.editor_agent = EditorAgent(verbose=self.verbose)
        self.review_agent = ReviewAgent(verbose=self.verbose)
        
        logger.info("총괄 에이전트: 모든 하위 에이전트 초기화 완료")

    def create_research_plan(self, user_question: str) -> ResearchPlan:
        """사용자 질문 기반 문제 해결 계획 생성"""
        logger.info(f"질문 '{user_question}'에 대한 문제 해결 계획 수립 중...")
        try:
            format_instructions = self.research_plan_parser.get_format_instructions()
            result = self.research_plan_chain.invoke({
                "task": user_question,
                "constraints": "학술적 정확성, 시간 효율성, 자원 가용성",
                "resources": "학술 데이터베이스, 연구 자료, AI 분석 도구",
                "format_instructions": format_instructions
            })
            research_plan = self.research_plan_parser.parse(result["text"])
            logger.info(f"문제 해결 계획 수립 완료: {len(research_plan.implementation_steps)}개 실행 단계")
            return research_plan
        except Exception as e:
            logger.error(f"계획 수립 실패: {str(e)}")
            # 에러 처리 체인 활용
            error_analysis = self.error_handling_chain.invoke({
                "task": "연구 계획 수립",
                "error": str(e),
                "context": f"사용자 질문: {user_question}"
            })
            logger.info(f"에러 분석: {error_analysis['text']}")
            
            # 기본 계획 반환
            return ResearchPlan(
                problem_statement=user_question,
                analysis_approach=["기본 문헌 분석", "관련 데이터 수집"],
                required_data_sources=["학술 논문", "신뢰할 수 있는 온라인 자료"],
                implementation_steps=["1. 초기 자료 수집", "2. 핵심 주제 분석", "3. 종합 보고서 작성"],
                expected_outcomes=["질문에 대한 체계적인 답변", "추가 연구 방향 제시"]
            )

    def start_workflow(
        self,
        user_question: str,
        template_name: Optional[str] = None,
        style_guide: Optional[str] = None,
        citation_style: Optional[str] = None,
        output_format: str = "markdown",
        verbose: bool = False
    ) -> PaperWorkflowState:
        """사용자 질문 기반 워크플로우 시작"""
        logger.info(f"질문 '{user_question}'에 대한 워크플로우 시작")
        research_plan = self.create_research_plan(user_question)
        
        workflow_state = self.workflow_runner.run_workflow(
            user_question=user_question,
            template_name=template_name,
            style_guide=style_guide,
            citation_style=citation_style,
            output_format=output_format,
            verbose=verbose
        )
        self.workflow_state = workflow_state
        logger.info(f"워크플로우 실행 완료: 상태 - {workflow_state.status}")
        return workflow_state

    def get_workflow_status(self) -> ProjectStatus:
        """현재 워크플로우 상태를 가져오기"""
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
        issues = [state.error] if state.error else None
        return ProjectStatus(
            status=status,
            completed_steps=completed_steps,
            current_step=current_step,
            next_steps=next_steps,
            issues=issues,
            progress_percentage=progress_percentage
        )

    def get_research_summary(self) -> Optional[str]:
        """현재 연구 요약을 가져오기"""
        if not self.workflow_state or not self.workflow_state.research_summary:
            return None
        research_summary = self.workflow_state.research_summary
        summary_text = f"## 연구 주제: {research_summary.topic}\n\n"
        summary_text += "### 주요 발견\n\n"
        for finding in research_summary.key_findings:
            summary_text += f"- {finding}\n"
        summary_text += f"\n### 수집된 자료 ({len(research_summary.collected_materials)}개)\n\n"
        for i, material in enumerate(research_summary.collected_materials, 1):
            summary_text += f"{i}. **{material.title}**"
            if material.authors:
                summary_text += f" - {', '.join(material.authors)}"
            if material.year:
                summary_text += f" ({material.year})"
            summary_text += f"\n   {material.summary[:150]}...\n\n"
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
        """현재 논문 요약을 가져오기"""
        if not self.workflow_state or not self.workflow_state.paper:
            return None
        paper = self.workflow_state.paper
        paper_content = f"# {paper.title}\n\n"
        paper_content += f"## 초록\n\n{paper.abstract}\n\n"
        for section in paper.sections:
            paper_content += f"## {section.title}\n\n"
            content_preview = section.content[:1000] + "..." if len(section.content) > 1000 else section.content
            paper_content += f"{content_preview}\n\n"
        try:
            result = self.summary_chain.invoke({"title": paper.title, "content": paper_content})
            summary = result["text"]
            full_summary = f"## 논문 요약: {paper.title}\n\n{summary}\n\n"
            full_summary += "### 논문 구조\n\n"
            for i, section in enumerate(paper.sections, 1):
                full_summary += f"{i}. {section.title}\n"
            if paper.references:
                full_summary += f"\n### 참고 문헌 ({len(paper.references)}개)\n\n"
            return full_summary
        except Exception as e:
            logger.error(f"논문 요약 생성 중 오류 발생: {str(e)}")
            return f"논문 제목: {paper.title}\n섹션 수: {len(paper.sections)}\n참고 문헌 수: {len(paper.references)}"

    def handle_user_feedback(self, feedback: str) -> Dict[str, Any]:
        """사용자 피드백 처리"""
        logger.info("사용자 피드백 처리 중...")
        if not self.workflow_state or not self.workflow_state.paper:
            return {"success": False, "message": "현재 작업 중인 논문이 없습니다."}
        try:
            is_positive = any(keyword in feedback.lower() for keyword in ["좋", "만족", "훌륭", "좋아"])
            needs_revision = any(keyword in feedback.lower() for keyword in ["수정", "고치", "변경", "개선", "추가"])
            result = {
                "success": True,
                "is_positive": is_positive,
                "needs_revision": needs_revision,
                "feedback": feedback
            }
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
            return {"success": False, "message": f"피드백 처리 중 오류 발생: {str(e)}"}

    def _get_timestamp(self) -> str:
        """현재 타임스탬프 반환"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def run(
        self,
        input_data: Dict[str, Any],
        config: Optional[RunnableConfig] = None
    ) -> PaperWorkflowState:
        """총괄 에이전트 실행"""
        logger.info("총괄 에이전트 실행 중...")
        user_question = input_data.get("user_question", "")
        if not user_question:
            raise ValueError("사용자 질문이 제공되지 않았습니다.")
        workflow_state = self.start_workflow(
            user_question=user_question,
            template_name=input_data.get("template_name"),
            style_guide=input_data.get("style_guide"),
            citation_style=input_data.get("citation_style"),
            output_format=input_data.get("output_format", "markdown"),
            verbose=input_data.get("verbose", False)
        )
        self.update_state({"user_question": user_question, "workflow_state": workflow_state})
        logger.info("총괄 에이전트 실행 완료")
        return workflow_state

    def execute_research_plan(self, research_plan: ResearchPlan, research_materials: List[ResearchMaterial]) -> Dict[str, Any]:
        """Execute a research plan using available materials"""
        
        # 연구 계획에서 필요한 작업 유형 결정
        if "literature review" in research_plan.implementation_steps[0].lower():
            writing_task = {
                "task_type": "literature_review",
                "task_description": research_plan.problem_statement,
                "additional_context": {
                    "format": "thematic",
                    "focus_areas": research_plan.analysis_approach
                }
            }
        elif "full analysis" in research_plan.implementation_steps[0].lower():
            writing_task = {
                "task_type": "full_paper",
                "task_description": research_plan.problem_statement,
                "additional_context": {
                    "template": "analytical",
                    "key_questions": research_plan.expected_outcomes
                }
            }
        else:
            writing_task = {
                "task_type": "custom",
                "task_description": research_plan.implementation_steps[0],
                "additional_context": {
                    "expected_outcome": research_plan.expected_outcomes[0]
                }
            }
        
        # WriterAgent 호출
        writer_result = self.writer_agent.process_writing_task(
            task_type=writing_task["task_type"],
            task_description=writing_task["task_description"],
            research_materials=research_materials,
            additional_context=writing_task["additional_context"]
        )
        
        return {
            "research_plan": research_plan.dict(),
            "writing_result": writer_result if isinstance(writer_result, dict) else writer_result.dict(),
            "status": "completed"
        }

    def handle_error(self, task: str, error: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """에러 처리 및 분석"""
        try:
            error_analysis = self.error_handling_chain.invoke({
                "task": task,
                "error": str(error),
                "context": json.dumps(context)
            })
            return {
                "success": False,
                "error": str(error),
                "analysis": error_analysis["text"],
                "recovery_suggested": True if "해결 방법" in error_analysis["text"] else False
            }
        except Exception as e:
            logger.error(f"에러 처리 중 추가 오류 발생: {str(e)}")
            return {
                "success": False,
                "error": str(error),
                "analysis": "에러 분석 실패",
                "recovery_suggested": False
            }

    def process_user_requirements(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        사용자 요구사항 처리
        
        Args:
            requirements: 사용자 요구사항 딕셔너리
            
        Returns:
            Dict[str, Any]: 처리 결과
        """
        logger.info("사용자 요구사항 처리 시작")
        
        try:
            # 요구사항 검증
            required_fields = ["topic"]
            for field in required_fields:
                if field not in requirements or not requirements[field]:
                    raise ValueError(f"필수 필드 누락: {field}")
            
            # 연구 계획 수립
            research_plan = self.create_research_plan(requirements["topic"])
            
            # 연구 수행
            research_materials = self.research_agent.conduct_research(
                research_plan.problem_statement,
                research_plan.required_data_sources
            )
            
            # 논문 유형에 따른 작업 설정
            paper_type = requirements.get("paper_type", "general")
            
            if paper_type == "literature_review":
                writing_task = {
                    "task_type": "literature_review",
                    "task_description": "Create a comprehensive literature review",
                    "additional_context": {
                        "focus_areas": research_plan.analysis_approach,
                        "key_themes": research_plan.expected_outcomes
                    }
                }
            elif paper_type == "experimental_research":
                writing_task = {
                    "task_type": "methodology",
                    "task_description": "Design an experimental methodology",
                    "additional_context": {
                        "research_questions": research_plan.problem_statement,
                        "expected_outcomes": research_plan.expected_outcomes
                    }
                }
            else:
                writing_task = {
                    "task_type": "custom",
                    "task_description": requirements.get("research_question", research_plan.problem_statement),
                    "additional_context": {
                        "additional_instructions": requirements.get("additional_instructions", ""),
                        "expected_outcome": research_plan.expected_outcomes[0] if research_plan.expected_outcomes else ""
                    }
                }
            
            # 작성 에이전트 호출
            writer_result = self.writer_agent.process_writing_task(
                task_type=writing_task["task_type"],
                task_description=writing_task["task_description"],
                research_materials=research_materials,
                additional_context=writing_task["additional_context"]
            )
            
            # 결과 반환
            return {
                "status": "completed",
                "research_plan": research_plan.dict(),
                "writing_result": writer_result if isinstance(writer_result, dict) else writer_result.dict(),
                "requirements": requirements
            }
        
        except Exception as e:
            logger.error(f"사용자 요구사항 처리 중 오류 발생: {str(e)}")
            return self.handle_error(
                task="사용자 요구사항 처리",
                error=str(e),
                context={"requirements": requirements}
            )

    def handle_writing_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """작성 작업 처리"""
        logger.info(f"작성 작업 처리 중: {task.get('task_type', 'unknown')}")
        
        try:
            # 작성 에이전트 초기화
            if not hasattr(self, 'writer_agent') or self.writer_agent is None:
                from agents import WriterAgent
                self.writer_agent = WriterAgent(verbose=self.verbose)
            
            # 작업 처리
            result = self.writer_agent.process_writing_task(task)
            
            # 결과 타입 확인 및 처리
            if isinstance(result, Paper):
                # Paper 객체인 경우
                return {
                    "task_type": task.get("task_type", "unknown"),
                    "paper": result.dict(),
                    "status": "completed"
                }
            elif isinstance(result, dict):
                # 딕셔너리인 경우
                return result
            else:
                # 예상치 못한 타입인 경우
                return {
                    "task_type": task.get("task_type", "unknown"),
                    "error": f"예상치 못한 결과 타입: {type(result)}",
                    "status": "failed"
                }
            
        except Exception as e:
            logger.error(f"작성 작업 처리 중 오류 발생: {str(e)}")
            return {
                "task_type": task.get("task_type", "unknown"),
                "error": str(e),
                "status": "failed"
            }
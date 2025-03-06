"""
총괄 에이전트 모듈
전체 논문 작성 프로세스를 조정하고 관리하는 에이전트입니다.
"""

import os
import re
import json
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from pydantic import BaseModel, Field

from langchain_core.runnables import RunnableConfig, RunnableSequence
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

from utils.academic_search import AcademicSearchManager
from utils.rag_integration import RAGEnhancer

# Add global paper requirements
PAPER_REQUIREMENTS = {
    "language": "English",
    "citation_required": True,
    "bibliography_required": True,
    "vector_db_based": True
}


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
        
        self.search_manager = AcademicSearchManager()
        self.rag_enhancer = RAGEnhancer()
        
        # Set global paper requirements
        self.paper_requirements = PAPER_REQUIREMENTS
        logger.info(f"Coordinator agent initialized with paper requirements: {self.paper_requirements}")

    def _init_prompts(self) -> None:
        """Initialize prompt templates with paper requirements"""
        # Add paper requirements to planning prompt
        self.planning_prompt_template = PromptTemplate(
            template=AGENT_PLANNING_PROMPT.template + "\n\nIMPORTANT REQUIREMENTS:\n1. All papers and reports must be written in English.\n2. All content must be based on the vector database content.\n3. Every claim or statement must include proper citations.\n4. All papers and reports must include a complete bibliography/references section at the end.",
            input_variables=AGENT_PLANNING_PROMPT.input_variables
        )
        
        # Add paper requirements to communication prompt
        self.communication_prompt_template = PromptTemplate(
            template=AGENT_COMMUNICATION_PROMPT.template + "\n\nIMPORTANT: Remind all agents that papers and reports must be in English, based on vector database content, with proper citations and a bibliography.",
            input_variables=AGENT_COMMUNICATION_PROMPT.input_variables
        )
        
        # Add paper requirements to error handling prompt
        self.error_handling_prompt_template = PromptTemplate(
            template=AGENT_ERROR_HANDLING_PROMPT.template + "\n\nWhen handling errors, ensure that paper requirements (English language, vector DB content, citations, bibliography) are still met.",
            input_variables=AGENT_ERROR_HANDLING_PROMPT.input_variables
        )
        
        # Add paper requirements to self-evaluation prompt
        self.self_evaluation_prompt_template = PromptTemplate(
            template=AGENT_SELF_EVALUATION_PROMPT.template + "\n\nEvaluation criteria must include checking that papers are in English, based on vector database content, with proper citations and a bibliography.",
            input_variables=AGENT_SELF_EVALUATION_PROMPT.input_variables
        )
        
        # Initialize chains using the new LangChain API with RunnableSequence
        from langchain_core.runnables import RunnableSequence
        
        self.planning_chain = RunnableSequence(
            first=self.planning_prompt_template,
            last=self.llm
        )
        
        self.communication_chain = RunnableSequence(
            first=self.communication_prompt_template,
            last=self.llm
        )
        
        self.error_handling_chain = RunnableSequence(
            first=self.error_handling_prompt_template,
            last=self.llm
        )
        
        self.self_evaluation_chain = RunnableSequence(
            first=self.self_evaluation_prompt_template,
            last=self.llm
        )
        
        # 연구 계획 프롬프트 템플릿 초기화
        self.research_plan_prompt_template = PromptTemplate(
            template="연구 주제: {topic}\n작업 유형: {task}\n가용 자원: {resources}\n제약 사항: {constraints}\n\n위 정보를 바탕으로 상세한 연구 계획을 작성해주세요.",
            input_variables=["topic", "task", "resources", "constraints"]
        )
        
        # 연구 계획 체인 초기화
        self.research_plan_chain = RunnableSequence(
            first=self.research_plan_prompt_template,
            last=self.llm
        )
        
        logger.info("Coordinator agent prompts initialized with paper requirements")

    def _init_agents(self):
        """Initialize agent instances"""
        logger.info("에이전트 초기화 중...")
        
        # 에이전트 임포트
        from agents import ResearchAgent, WriterAgent, EditorAgent  # ReviewAgent 제거
        
        # 연구 에이전트 초기화
        self.research_agent = ResearchAgent()
        
        # 작성 에이전트 초기화
        self.writer_agent = WriterAgent()
        
        # 편집 에이전트 초기화
        self.editor_agent = EditorAgent()
        
        # 리뷰 에이전트 초기화 (필요하다면 다른 방식으로 처리)
        # self.review_agent = ReviewAgent()  # 주석 처리 또는 제거
        
        logger.info("에이전트 초기화 완료")

    def create_research_plan(self, topic, task_type):
        """
        연구 계획을 생성합니다.
        """
        try:
            # 필요한 모든 입력 키 제공
            inputs = {
                "topic": topic,
                "task": task_type,
                "resources": [],
                "constraints": ""
            }
            # 새로운 LangChain API 사용
            result = self.research_plan_chain.invoke(inputs)
            # result는 {"output": AIMessage} 형식이므로 content 속성을 추출
            if isinstance(result, dict) and "output" in result:
                if hasattr(result["output"], "content"):
                    return result["output"].content
                else:
                    return str(result["output"])
            elif hasattr(result, "content"):
                return result.content
            else:
                return str(result)
        except Exception as e:
            logger.error(f"연구 계획 생성 중 오류: {str(e)}")
            # 오류 발생 시 기본 템플릿 반환
            return f"# 연구 계획: {topic}\n\n## 목표\n\n{topic}에 대한 연구 수행\n\n## 단계\n\n1. 문헌 조사\n2. 분석\n3. 정리"

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
        논문 작성 워크플로우 시작
        
        Args:
            topic: 논문 주제
            template_name: 논문 템플릿 이름
            style_guide: 스타일 가이드 이름
            citation_style: 인용 스타일
            output_format: 출력 형식
            verbose: 상세 로깅 여부
            
        Returns:
            PaperWorkflowState: 워크플로우 상태
        """
        logger.info(f"Starting workflow for question: {topic}")
        
        # 워크플로우 설정 - PaperWritingGraph.run_workflow에서 지원하는 매개변수만 포함
        workflow_config = {
            "topic": topic,
            "template_name": template_name or "academic",
            "style_guide": style_guide or "Standard Academic",
            "citation_style": citation_style or "APA",
            "output_format": output_format,
            "verbose": verbose
        }
        
        # Run the workflow
        self.workflow_state = self.workflow_runner.run_workflow(**workflow_config)
        
        return self.workflow_state

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

    def run(self, topic, paper_type, constraints=None, references=None, instructions=None):
        """
        논문 작성 프로세스 실행
        
        Args:
            topic: 연구 주제
            paper_type: 논문 유형
            constraints: 제약사항
            references: 참고자료
            instructions: 추가 지시사항
            
        Returns:
            Dict: 생성된 논문 및 메타데이터
        """
        try:
            logger.info(f"사용자 요구사항 처리 시작: {topic}")
            
            # 1. 사용자 요청 분석
            logger.info("사용자 요청 분석 중...")
            request_analysis = self._analyze_user_request(topic, paper_type, constraints, instructions)
            
            # 2. 조사 계획 생성 및 조사 에이전트에게 전달
            logger.info("조사 계획 생성 중...")
            investigation_plan = self._create_investigation_plan(request_analysis, topic)
            
            # 조사 에이전트에게 조사 계획 전달
            logger.info("조사 에이전트에게 조사 계획 전달")
            investigation_success = self.delegate_to_research_agent(investigation_plan)
            
            if not investigation_success:
                logger.error("조사 계획 전달 실패")
                return {"error": "조사 계획 전달 실패"}
            
            # 3. 조사 에이전트 실행 및 결과 검증
            logger.info("조사 에이전트 실행 중...")
            investigation_results = self.research_agent.run(
                task="collect_materials",
                topic=topic,
                research_plan=investigation_plan,
                max_queries=5,
                results_per_source=10
            )
            
            # 조사 결과 검증
            logger.info("조사 결과 검증 중...")
            investigation_valid, investigation_feedback = self._validate_investigation_results(investigation_results, topic)
            
            # 조사 결과가 타당하지 않으면 다시 조사 지시
            max_investigation_attempts = 3
            investigation_attempts = 1
            
            while not investigation_valid and investigation_attempts < max_investigation_attempts:
                logger.info(f"조사 결과 타당하지 않음. 피드백: {investigation_feedback}")
                logger.info(f"조사 재시도 ({investigation_attempts}/{max_investigation_attempts})...")
                
                # 피드백을 반영하여 조사 계획 수정
                investigation_plan = self._revise_investigation_plan(investigation_plan, investigation_feedback)
                
                # 수정된 계획으로 다시 조사
                self.delegate_to_research_agent(investigation_plan)
                investigation_results = self.research_agent.run(
                    task="collect_materials",
                    topic=topic,
                    research_plan=investigation_plan,
                    max_queries=5,
                    results_per_source=10
                )
                
                # 결과 재검증
                investigation_valid, investigation_feedback = self._validate_investigation_results(investigation_results, topic)
                investigation_attempts += 1
            
            if not investigation_valid:
                logger.warning(f"최대 조사 시도 횟수 도달. 최선의 결과로 진행합니다.")
            
            # 4. 연구 계획 생성 및 연구 에이전트에게 전달
            logger.info("연구 계획 생성 중...")
            research_plan = self._create_research_plan(request_analysis, topic, investigation_results)
            
            # 연구 에이전트에게 연구 계획 전달
            logger.info("연구 에이전트에게 연구 계획 전달")
            # 연구 에이전트 실행
            research_materials = self.research_agent.run(
                task="analyze_materials",
                topic=topic,
                research_plan=research_plan,
                materials=investigation_results
            )
            
            # 연구 결과 검증
            logger.info("연구 결과 검증 중...")
            research_valid, research_feedback = self._validate_research_results(research_materials, topic)
            
            # 연구 결과가 타당하지 않으면 다시 연구 지시
            max_research_attempts = 3
            research_attempts = 1
            
            while not research_valid and research_attempts < max_research_attempts:
                logger.info(f"연구 결과 타당하지 않음. 피드백: {research_feedback}")
                logger.info(f"연구 재시도 ({research_attempts}/{max_research_attempts})...")
                
                # 피드백을 반영하여 연구 계획 수정
                research_plan = self._revise_research_plan(research_plan, research_feedback)
                
                # 수정된 계획으로 다시 연구
                research_materials = self.research_agent.run(
                    task="analyze_materials",
                    topic=topic,
                    research_plan=research_plan,
                    materials=investigation_results
                )
                
                # 결과 재검증
                research_valid, research_feedback = self._validate_research_results(research_materials, topic)
                research_attempts += 1
            
            if not research_valid:
                logger.warning(f"최대 연구 시도 횟수 도달. 최선의 결과로 진행합니다.")
            
            # 5. 보고서 양식 생성 및 편집 에이전트에게 전달
            logger.info("보고서 양식 생성 중...")
            report_format = self._create_report_format(request_analysis, topic, paper_type)
            
            # 편집 에이전트에게 보고서 양식 전달
            logger.info("편집 에이전트에게 보고서 양식 전달")
            format_success = self.delegate_to_writing_agent(report_format)
            
            if not format_success:
                logger.error("보고서 양식 전달 실패")
                return {"error": "보고서 양식 전달 실패"}
            
            # 6. 작성 에이전트 실행
            logger.info("작성 에이전트 실행 중...")
            paper_draft = self.writer_agent.run(
                task="write_paper",
                topic=topic,
                paper_type=paper_type,
                research_materials=research_materials,
                report_format=report_format
            )
            
            # 작성 결과 검증
            logger.info("작성 결과 검증 중...")
            writing_valid, writing_feedback = self._validate_writing_results(paper_draft, topic, report_format)
            
            # 작성 결과가 타당하지 않으면 다시 작성 지시
            max_writing_attempts = 3
            writing_attempts = 1
            
            while not writing_valid and writing_attempts < max_writing_attempts:
                logger.info(f"작성 결과 타당하지 않음. 피드백: {writing_feedback}")
                logger.info(f"작성 재시도 ({writing_attempts}/{max_writing_attempts})...")
                
                # 피드백을 반영하여 작성 지시 수정
                revised_report_format = self._revise_report_format(report_format, writing_feedback)
                
                # 수정된 양식으로 다시 작성
                self.delegate_to_writing_agent(revised_report_format)
                paper_draft = self.writer_agent.run(
                    task="write_paper",
                    topic=topic,
                    paper_type=paper_type,
                    research_materials=research_materials,
                    report_format=revised_report_format
                )
                
                # 결과 재검증
                writing_valid, writing_feedback = self._validate_writing_results(paper_draft, topic, revised_report_format)
                writing_attempts += 1
                report_format = revised_report_format
            
            if not writing_valid:
                logger.warning(f"최대 작성 시도 횟수 도달. 최선의 결과로 진행합니다.")
            
            # 7. 편집 에이전트 실행
            logger.info("편집 에이전트 실행 중...")
            edited_paper = self.editor_agent.run(
                task="edit_paper",
                paper_draft=paper_draft,
                report_format=report_format,
                citation_style=report_format.get("citation_style", "APA")
            )
            
            # 편집 결과 검증
            logger.info("편집 결과 검증 중...")
            editing_valid, editing_feedback = self._validate_editing_results(edited_paper, paper_draft, report_format)
            
            # 편집 결과가 타당하지 않으면 다시 편집 지시
            max_editing_attempts = 3
            editing_attempts = 1
            
            while not editing_valid and editing_attempts < max_editing_attempts:
                logger.info(f"편집 결과 타당하지 않음. 피드백: {editing_feedback}")
                logger.info(f"편집 재시도 ({editing_attempts}/{max_editing_attempts})...")
                
                # 피드백을 반영하여 편집 지시 수정
                revised_editing_instructions = self._revise_editing_instructions(report_format, editing_feedback)
                
                # 수정된 지시로 다시 편집
                edited_paper = self.editor_agent.run(
                    task="edit_paper",
                    paper_draft=paper_draft,
                    report_format=revised_editing_instructions,
                    citation_style=revised_editing_instructions.get("citation_style", "APA")
                )
                
                # 결과 재검증
                editing_valid, editing_feedback = self._validate_editing_results(edited_paper, paper_draft, revised_editing_instructions)
                editing_attempts += 1
            
            if not editing_valid:
                logger.warning(f"최대 편집 시도 횟수 도달. 최선의 결과로 진행합니다.")
            
            # 8. 최종 결과 반환
            logger.info("논문 작성 프로세스 완료")
            return {
                "paper": edited_paper,
                "topic": topic,
                "paper_type": paper_type,
                "research_materials": research_materials,
                "metadata": {
                    "investigation_plan": investigation_plan,
                    "research_plan": research_plan,
                    "report_format": report_format
                }
            }
            
        except Exception as e:
            logger.error(f"논문 생성 실패: {str(e)}", exc_info=True)
            return {"error": str(e)}

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
        writer_result = self.writer_agent.process_writing_task({
            "task_type": writing_task["task_type"],
            "task_description": writing_task["task_description"],
            "materials": research_materials,
            "additional_context": writing_task["additional_context"]
        })
        
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
        """
        logger.info("사용자 요구사항 처리 시작")
        
        try:
            # 요구사항 검증
            required_fields = ["topic"]
            for field in required_fields:
                if field not in requirements or not requirements[field]:
                    raise ValueError(f"필수 필드 누락: {field}")
            
            # 요구사항 분석 및 작업 유형 결정
            task_type = self._determine_task_type(requirements)
            logger.info(f"결정된 작업 유형: {task_type}")
            
            # 로컬 PDF 파일 수 확인
            local_pdf_count = self._count_local_pdf_files()
            logger.info(f"로컬 PDF 파일 수: {local_pdf_count}")
            
            # 연구 계획 수립
            research_plan_text = self.create_research_plan(requirements["topic"], task_type)
            
            # 로컬 PDF 파일이 30개 이상인 경우 조사 단계 건너뛰기
            if local_pdf_count >= 30:
                logger.info("충분한 로컬 PDF 파일이 있어 추가 조사를 건너뜁니다.")
                # 기존 로컬 자료만 활용하여 연구 수행
                research_materials = self._process_existing_materials(requirements["topic"])
            else:
                # 주제와 관련된 학술 자료 검색
                if "topic" in requirements:
                    try:
                        logger.info(f"주제에 대한 학술 검색 수행: {requirements['topic']}")
                        research_summary = self.rag_enhancer.get_research_summary(
                            topic=requirements['topic'],
                            limit=5
                        )
                        
                        # 검색 결과를 요구사항에 추가
                        requirements["research_context"] = research_summary
                        logger.info("학술 검색 결과가 요구사항에 추가되었습니다")
                    except Exception as e:
                        logger.warning(f"학술 검색 중 오류 발생: {e}")
                
                # 조사 에이전트를 통한 연구 수행
                research_materials = self.research_agent.run(
                    topic=requirements["topic"]
                )
            
            # 작업 유형에 맞는 작성 작업 구성
            writing_task = self._create_writing_task(task_type, requirements, research_plan_text, research_materials)
            
            # 작성 작업 실행
            writer_result = self.process_writing_task(task_type, writing_task)
            
            # 결과 형식 확인 및 조정
            result = self._format_final_result(writer_result, requirements, research_materials)
            
            return result
        except Exception as e:
            logger.error(f"사용자 요구사항 처리 중 오류 발생: {str(e)}")
            return self.handle_error(
                task="사용자 요구사항 처리",
                error=str(e),
                context={"requirements": requirements}
            )

    def _count_local_pdf_files(self) -> int:
        """
        로컬 PDF 파일 수를 계산합니다.
        
        Returns:
            int: 로컬 PDF 파일 수
        """
        import os
        from pathlib import Path
        
        # 확인할 디렉토리 목록
        directories = [
            "data/local",  # 로컬 폴더
            "data/pdfs"    # 이전에 다운받은 PDF 폴더
        ]
        
        total_pdf_count = 0
        
        for directory in directories:
            if os.path.exists(directory):
                # 디렉토리 내 PDF 파일 수 계산
                pdf_files = list(Path(directory).glob("**/*.pdf"))
                total_pdf_count += len(pdf_files)
                logger.info(f"{directory} 디렉토리에서 {len(pdf_files)}개의 PDF 파일을 찾았습니다.")
        
        return total_pdf_count

    def _process_existing_materials(self, topic: str) -> Dict[str, Any]:
        """
        기존 로컬 PDF 파일을 처리하여 연구 자료로 변환합니다.
        
        Args:
            topic: 연구 주제
            
        Returns:
            Dict[str, Any]: 처리된 연구 자료
        """
        logger.info(f"기존 로컬 PDF 파일을 '{topic}' 주제에 맞게 처리합니다.")
        
        try:
            # 연구 에이전트 초기화 (필요한 경우)
            if not hasattr(self, 'research_agent') or self.research_agent is None:
                from agents.research_agent import ResearchAgent
                self.research_agent = ResearchAgent(verbose=self.verbose)
            
            # 로컬 자료만 사용하도록 설정된 연구 계획 생성
            local_research_plan = {
                "search_strategy": {
                    "search_scope": "local_only",  # 로컬 자료만 사용
                    "min_papers": 30,
                    "queries": [topic]
                }
            }
            
            # 연구 에이전트에게 로컬 자료만 수집하도록 지시
            research_materials = self.research_agent.collect_research_materials(
                topic=topic,
                research_plan=local_research_plan,
                max_queries=1,
                results_per_source=30,
                final_result_count=30
            )
            
            # 수집된 자료 보강
            enriched_materials = self.research_agent.enrich_research_materials(research_materials)
            
            logger.info(f"기존 로컬 PDF 파일에서 {len(enriched_materials)}개의 연구 자료를 처리했습니다.")
            
            # 연구 자료를 적절한 형식으로 변환
            return {
                "materials": enriched_materials,
                "outline": self._generate_outline_from_materials(topic, enriched_materials),
                "source": "local_only"
            }
        except Exception as e:
            logger.error(f"기존 자료 처리 중 오류 발생: {str(e)}", exc_info=True)
            # 오류 발생 시 빈 결과 반환
            return {
                "materials": [],
                "outline": {"title": topic, "sections": []},
                "source": "local_only",
                "error": str(e)
            }

    def _generate_outline_from_materials(self, topic: str, materials: List[Any]) -> Dict[str, Any]:
        """
        연구 자료를 바탕으로 논문 개요를 생성합니다.
        
        Args:
            topic: 연구 주제
            materials: 연구 자료 목록
            
        Returns:
            Dict[str, Any]: 생성된 개요
        """
        logger.info(f"'{topic}' 주제에 대한 개요 생성 중...")
        
        try:
            # 자료 요약 추출
            summaries = []
            for material in materials[:10]:  # 상위 10개 자료만 사용
                if hasattr(material, 'summary') and material.summary:
                    summaries.append(material.summary)
                elif hasattr(material, 'abstract') and material.abstract:
                    summaries.append(material.abstract[:500])  # 초록 일부만 사용
            
            # LLM을 사용하여 개요 생성
            prompt = f"""
            다음 연구 자료를 바탕으로 '{topic}' 주제에 대한 학술 논문 개요를 생성해주세요.
            
            연구 자료 요약:
            {' '.join(summaries[:5])}
            
            개요는 다음 형식으로 JSON 형태로 반환해주세요:
            {{
                "title": "논문 제목",
                "sections": [
                    {{"title": "섹션 제목", "content": "섹션 내용 요약"}},
                    ...
                ]
            }}
            
            학술 논문의 표준 구조(서론, 방법론, 결과, 논의, 결론 등)를 따라주세요.
            """
            
            response = self.llm.invoke(prompt)
            
            # 응답에서 JSON 추출
            import re
            import json
            
            # JSON 블록 찾기
            json_match = re.search(r"\{[\s\S]*\}", response.content)
            if json_match:
                try:
                    outline = json.loads(json_match.group(0))
                    logger.info(f"개요 생성 완료: {outline.get('title', '제목 없음')}")
                    return outline
                except json.JSONDecodeError:
                    logger.warning("JSON 파싱 실패, 기본 개요 사용")
            
            # 파싱 실패 시 기본 개요 반환
            return {
                "title": f"Research on {topic}",
                "sections": [
                    {"title": "Introduction", "content": "Introduction to the topic."},
                    {"title": "Literature Review", "content": "Review of existing literature."},
                    {"title": "Methodology", "content": "Research methodology."},
                    {"title": "Results", "content": "Research findings."},
                    {"title": "Discussion", "content": "Discussion of results."},
                    {"title": "Conclusion", "content": "Conclusion and future work."}
                ]
            }
        except Exception as e:
            logger.error(f"개요 생성 중 오류 발생: {str(e)}", exc_info=True)
            # 오류 발생 시 기본 개요 반환
            return {
                "title": f"Research on {topic}",
                "sections": [
                    {"title": "Introduction", "content": ""},
                    {"title": "Literature Review", "content": ""},
                    {"title": "Methodology", "content": ""},
                    {"title": "Results", "content": ""},
                    {"title": "Discussion", "content": ""},
                    {"title": "Conclusion", "content": ""}
                ]
            }

    def _determine_task_type(self, requirements: Dict[str, Any]) -> str:
        """
        사용자 요구사항을 분석하여 작업 유형 결정
        """
        # 명시적으로 paper_type이 지정된 경우
        paper_type = requirements.get("paper_type", "").lower()
        if paper_type:
            if "literature" in paper_type or "review" in paper_type:
                return "literature_review"
            elif "method" in paper_type or "experimental" in paper_type:
                return "methodology"
            elif "case" in paper_type:
                return "case_study"
            elif "theoretical" in paper_type or "analysis" in paper_type:
                return "theoretical_analysis"
        
        # 작업 내용에서 키워드 탐색
        topic = requirements.get("topic", "").lower()
        additional_instructions = requirements.get("additional_instructions", "").lower()
        combined_text = f"{topic} {additional_instructions}"
        
        # 키워드 기반 작업 유형 추론
        if any(kw in combined_text for kw in ["review", "literature", "previous work", "existing research"]):
            return "literature_review"
        elif any(kw in combined_text for kw in ["method", "methodology", "approach", "experiment"]):
            return "methodology"
        elif any(kw in combined_text for kw in ["result", "analysis", "finding", "data"]):
            return "results_analysis"
        elif any(kw in combined_text for kw in ["conclusion", "summary", "implication"]):
            return "conclusion"
        elif any(kw in combined_text for kw in ["abstract", "overview"]):
            return "abstract"
        
        # 기본값은 전체 논문 작성
        return "full_paper"

    def _create_investigation_plan(self, request_analysis, topic):
        """
        사용자 요청 분석을 바탕으로 조사 계획을 생성합니다.
        
        Args:
            request_analysis: 사용자 요청 분석 결과
            topic: 연구 주제
            
        Returns:
            Dict: 조사 계획
        """
        logger.info(f"조사 계획 생성: 주제={topic}")
        
        try:
            # 조사 계획 구성
            investigation_plan = {
                "topic": topic,
                "search_strategy": {
                    "search_scope": "all",  # 기본값은 모든 소스 검색
                    "min_papers": 10,  # 최소 필요 논문 수
                    "queries": []  # 검색 쿼리 목록
                },
                "focus_keywords": request_analysis.get("keywords", [topic]),
                "research_questions": request_analysis.get("research_questions", [f"{topic}에 대한 최신 연구는 무엇인가?"]),
                "required_sources": request_analysis.get("required_sources", ["학술 논문", "연구 보고서"]),
                "search_depth": 3,  # 기본 검색 깊이
                "max_sources": 20,  # 최대 소스 수
                "time_constraints": "최근 5년",  # 시간적 제약
                "language_preferences": ["한국어", "영어"],  # 언어 선호도
                "evaluation_criteria": {
                    "relevance_threshold": 0.6,  # 관련성 임계값
                    "quality_metrics": ["인용 수", "저널 영향력", "최신성"]
                }
            }
            
            # 검색 쿼리 생성
            keywords = request_analysis.get("keywords", [topic])
            for keyword in keywords:
                investigation_plan["search_strategy"]["queries"].append(keyword)
            
            # 연구 질문을 기반으로 추가 쿼리 생성
            research_questions = request_analysis.get("research_questions", [])
            for question in research_questions:
                # 질문에서 핵심 키워드 추출
                question_keywords = question.split()
                if len(question_keywords) > 3:
                    # 긴 질문에서는 주요 명사만 추출
                    query = " ".join([word for word in question_keywords if len(word) > 1 and word.lower() not in ["무엇", "어떻게", "왜", "언제", "어디서", "누구", "의", "에", "을", "를", "이", "가"]])
                    if query and query not in investigation_plan["search_strategy"]["queries"]:
                        investigation_plan["search_strategy"]["queries"].append(query)
            
            # 중복 제거 및 최대 5개 쿼리로 제한
            investigation_plan["search_strategy"]["queries"] = list(set(investigation_plan["search_strategy"]["queries"]))[:5]
            
            logger.info(f"조사 계획 생성 완료: {len(investigation_plan['search_strategy']['queries'])}개 검색 쿼리 생성")
            return investigation_plan
            
        except Exception as e:
            logger.error(f"조사 계획 생성 중 오류: {str(e)}")
            # 오류 발생 시 기본 조사 계획 반환
            return {
                "topic": topic,
                "search_strategy": {
                    "search_scope": "all",
                    "min_papers": 10,
                    "queries": [topic]
                },
                "focus_keywords": [topic],
                "research_questions": [f"{topic}에 대한 최신 연구는 무엇인가?"],
                "required_sources": ["학술 논문", "연구 보고서"],
                "search_depth": 3,
                "max_sources": 20
            }

    def _create_writing_task(self, task_type: str, requirements: Dict[str, Any], 
                            research_plan: str, research_materials: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a writing task based on the research plan and materials
        
        Args:
            task_type: Type of writing task
            requirements: User requirements
            research_plan: Research plan
            research_materials: Research materials
            
        Returns:
            Writing task configuration
        """
        logger.info(f"Creating {task_type} writing task")
        
        # Base writing task
        writing_task = {
            "task_type": task_type,
            "topic": requirements.get("topic", ""),
            "research_plan": research_plan,
            "research_materials": research_materials,
            "requirements": requirements,
            # Add global paper requirements
            "language": self.paper_requirements["language"],
            "citation_required": self.paper_requirements["citation_required"],
            "bibliography_required": self.paper_requirements["bibliography_required"],
            "vector_db_based": self.paper_requirements["vector_db_based"]
        }
        
        # Add task-specific configurations
        if task_type == "academic_paper":
            writing_task.update({
                "paper_type": requirements.get("paper_type", "research"),
                "citation_style": requirements.get("citation_style", "APA"),
                "target_length": requirements.get("target_length", 3000),
                "template": requirements.get("template", "academic")
            })
        elif task_type == "report":
            writing_task.update({
                "report_type": requirements.get("report_type", "technical"),
                "target_audience": requirements.get("target_audience", "professional"),
                "format": requirements.get("format", "standard"),
                "include_visuals": requirements.get("include_visuals", True)
            })
        elif task_type == "literature_review":
            writing_task.update({
                "focus_area": requirements.get("focus_area", "comprehensive"),
                "chronological": requirements.get("chronological", False),
                "critical_analysis": requirements.get("critical_analysis", True)
            })
        
        logger.info(f"Created {task_type} writing task with paper requirements")
        return writing_task

    def process_writing_task(self, task_type: str, writing_task: Dict[str, Any]) -> Dict[str, Any]:
        """
        작성 작업 처리 및 결과 검토
        """
        logger.info(f"작성 작업 처리: {task_type}")
        
        # WriterAgent가 없으면 초기화
        if not hasattr(self, 'writer_agent') or self.writer_agent is None:
            from agents import WriterAgent
            self.writer_agent = WriterAgent(verbose=self.verbose)
        
        # 작업 처리
        writer_result = self.writer_agent.process_writing_task(writing_task)
        
        # 결과 검토
        review_result = self._review_writing_result(task_type, writing_task, writer_result)
        
        # 재작업 필요 여부 확인
        if review_result["needs_revision"]:
            logger.info(f"작성 결과 재작업 필요: {review_result['revision_reason']}")
            
            # 수정 지시사항 추가
            revised_task = writing_task.copy()
            revised_task["revision_instructions"] = review_result["revision_instructions"]
            revised_task["previous_attempt"] = writer_result.get("content", "")
            
            # 재작업 요청
            return self._request_revision(revised_task)
        
        return writer_result

    def _review_writing_result(self, task_type: str, original_task: Dict[str, Any], 
                             writing_result: Dict[str, Any]) -> Dict[str, bool]:
        """
        작성 결과 검토
        
        Args:
            task_type: 작업 유형
            original_task: 원래 작업 지시사항
            writing_result: 작성 결과
            
        Returns:
            Dict: 검토 결과 (needs_revision, revision_reason, revision_instructions)
        """
        # 기본 검토 결과
        review = {
            "needs_revision": False,
            "revision_reason": "",
            "revision_instructions": ""
        }
        
        # 결과가 딕셔너리가 아니거나 콘텐츠가 없는 경우
        if not isinstance(writing_result, dict) or not writing_result.get("content"):
            review["needs_revision"] = True
            review["revision_reason"] = "결과물 형식 오류 또는 내용 부재"
            review["revision_instructions"] = "작업 유형에 맞는 적절한 내용을 생성해주세요."
            return review
        
        content = writing_result.get("content", "")
        
        # 작업 유형별 검토
        if task_type == "literature_review":
            # 문헌 검토 검토 로직
            if len(content.split()) < 300:  # 너무 짧은 경우
                review["needs_revision"] = True
                review["revision_reason"] = "문헌 검토가 너무 짧습니다."
                review["revision_instructions"] = "더 포괄적인 문헌 검토를 작성해주세요. 최소 500단어 이상 필요합니다."
            
            # 참고 문헌 인용 여부 확인
            if "[" not in content and "(" not in content:
                review["needs_revision"] = True
                review["revision_reason"] = "참고 문헌 인용이 없습니다."
                review["revision_instructions"] = "문헌 검토에 적절한 인용을 포함해주세요."
        
        elif task_type == "methodology":
            # 방법론 검토 로직
            if "data collection" not in content.lower() and "method" not in content.lower():
                review["needs_revision"] = True
                review["revision_reason"] = "방법론 설명이 부족합니다."
                review["revision_instructions"] = "데이터 수집 방법과 분석 방법을 더 자세히 설명해주세요."
        
        elif task_type == "results_analysis":
            # 결과 분석 검토 로직
            if "finding" not in content.lower() and "result" not in content.lower():
                review["needs_revision"] = True
                review["revision_reason"] = "결과 및 분석이 명확하지 않습니다."
                review["revision_instructions"] = "주요 연구 결과를 명확히 제시하고 분석을 강화해주세요."
        
        elif task_type == "full_paper":
            # 전체 논문 검토 로직
            required_sections = ["abstract", "introduction", "conclusion", "reference"]
            missing_sections = [s for s in required_sections if s.lower() not in content.lower()]
            
            if missing_sections:
                review["needs_revision"] = True
                review["revision_reason"] = f"논문에 필수 섹션이 누락되었습니다: {', '.join(missing_sections)}"
                review["revision_instructions"] = f"누락된 섹션을 추가해주세요: {', '.join(missing_sections)}"
        
        # 일반적인 품질 검토
        if len(content) > 0 and len(content.split()) < 100:
            review["needs_revision"] = True
            review["revision_reason"] = "콘텐츠가 너무 짧습니다."
            review["revision_instructions"] = "더 자세한 내용을 작성해주세요. 현재 내용이 충분하지 않습니다."
        
        return review

    def _request_revision(self, revised_task: Dict[str, Any]) -> Dict[str, Any]:
        """
        작성 에이전트에게 재작업 요청
        
        Args:
            revised_task: 수정 지시사항이 포함된 작업
            
        Returns:
            Dict: 재작업 결과
        """
        logger.info("작성 에이전트에 재작업 요청")
        
        # 재작업 요청을 위한 특수 작업 유형 설정
        revised_task["task_type"] = "revision"
        
        # WriterAgent가 "revision" 작업 유형을 처리할 수 있도록 하기 위한 코드 추가 필요
        # (WriterAgent의 process_writing_task 메서드 수정 필요)
        
        # 재작업 요청
        revised_result = self.writer_agent.process_writing_task(revised_task)
        
        # 결과 확인
        if isinstance(revised_result, dict) and revised_result.get("status") == "completed":
            logger.info("재작업 완료")
            return revised_result
        else:
            # 재작업이 실패한 경우 원래 결과 반환하고 경고 로그
            logger.warning("재작업 실패, 원래 결과 사용")
            return {
                "task_type": revised_task.get("task_type", "unknown"),
                "content": revised_task.get("previous_attempt", ""),
                "status": "completed",
                "revision_failed": True
            }

    def _format_final_result(self, writer_result: Dict[str, Any], 
                            requirements: Dict[str, Any], 
                            research_materials: Dict[str, Any]) -> Dict[str, Any]:
        """
        최종 결과 포맷팅
        """
        # 기본 결과 템플릿
        result = {
            "status": "completed",
            "id": "paper_" + requirements["topic"][:10].replace(" ", "_").lower()
        }
        
        # 작성 결과 통합
        if isinstance(writer_result, dict):
            if "content" in writer_result:
                result["content"] = writer_result["content"]
            elif "section_content" in writer_result:
                result["content"] = writer_result["section_content"]
            elif isinstance(writer_result.get("paper"), dict):
                result["content"] = writer_result["paper"].get("content", "")
            else:
                # 다른 필드 탐색
                for key, value in writer_result.items():
                    if isinstance(value, str) and len(value) > 100:
                        result["content"] = value
                        break
        elif isinstance(writer_result, str):
            result["content"] = writer_result
        
        # 개요 추가
        result["outline"] = research_materials.get("outline", "")
        
        # 결과 확인
        if not result.get("content"):
            logger.warning("생성된 콘텐츠가 없습니다")
            result["content"] = f"# {requirements['topic']}\n\n내용을 생성하는 중 오류가 발생했습니다."
            result["status"] = "partial"
        
        return result

    def create_study_plan(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        연구 계획을 수립합니다. (사용자 요구사항 기반)
        
        Args:
            requirements: 사용자 요구사항
            
        Returns:
            Dict: 연구 계획 (에이전트에게 전달될 형식)
        """
        topic = requirements.get("topic", "")
        paper_type = requirements.get("paper_type", "")
        additional_instructions = requirements.get("additional_instructions", "")
        
        logger.info(f"'{topic}'에 대한 연구 계획 수립 중...")
        
        try:
            # LLM을 활용한 연구 계획 생성
            prompt = f"""
            Create a detailed study plan for the following research topic:
            
            Topic: {topic}
            Paper Type: {paper_type}
            Additional Instructions: {additional_instructions}
            
            Your study plan should include:
            1. Main research questions to be addressed
            2. Key hypotheses or arguments to explore
            3. Theoretical framework or perspective to be used
            4. Potential challenges and limitations
            5. Expected outcomes or contributions
            
            Format the plan as a structured JSON object with these sections.
            """
            
            # 연구 계획 생성
            response = self.llm.invoke(prompt)
            
            # 응답에서 JSON 추출 시도
            study_plan = self._extract_json_from_response(response.content)
            
            # JSON 파싱 실패 시 텍스트 형식으로 처리
            if not study_plan:
                study_plan = {
                    "research_questions": ["What are the key aspects of " + topic + "?"],
                    "hypotheses": ["The research will reveal important insights about " + topic],
                    "framework": "Standard academic analysis",
                    "challenges": ["Limited existing research"],
                    "expected_outcomes": ["Better understanding of " + topic]
                }
            
            # 연구 계획에 원본 주제 추가
            study_plan["topic"] = topic
            study_plan["paper_type"] = paper_type
            
            logger.info(f"연구 계획 수립 완료: {', '.join(study_plan.get('research_questions', [])[:1])}...")
            return study_plan
            
        except Exception as e:
            logger.error(f"연구 계획 수립 중 오류: {str(e)}")
            # 기본 연구 계획 반환
            return {
                "topic": topic,
                "research_questions": ["What are the key aspects of " + topic + "?"],
                "expected_outcomes": ["Better understanding of " + topic]
            }

    def create_report_format(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        보고서 형식을 결정합니다.
        
        Args:
            requirements: 사용자 요구사항
            
        Returns:
            Dict: 보고서 형식 지침
        """
        topic = requirements.get("topic", "")
        paper_type = requirements.get("paper_type", "")
        task_type = self._determine_task_type(requirements)
        
        # 기본 보고서 형식
        report_format = {
            "topic": topic,
            "task_type": task_type,
            "sections": [],
            "citation_style": "APA",
            "formatting_guidelines": {
                "headings": "## for main sections, ### for subsections",
                "citations": "Use parenthetical citations (Author, Year)",
                "figures": "Include descriptive captions",
                "tables": "Number and title all tables",
                "emphasis": "Use **bold** for emphasis sparingly"
            }
        }
        
        # 작업 유형에 따른 섹션 구성
        if task_type == "literature_review":
            report_format["sections"] = [
                "Introduction",
                "Methodology of Literature Review",
                "Key Findings from Literature",
                "Gaps in Current Research",
                "Discussion",
                "Conclusion",
                "References"
            ]
        elif task_type == "methodology":
            report_format["sections"] = [
                "Introduction",
                "Research Design",
                "Data Collection Methods",
                "Analysis Approach",
                "Ethical Considerations",
                "Limitations",
                "References"
            ]
        elif task_type == "results_analysis":
            report_format["sections"] = [
                "Introduction",
                "Results Overview",
                "Detailed Findings",
                "Statistical Analysis",
                "Discussion",
                "Conclusion",
                "References"
            ]
        elif task_type == "full_paper":
            report_format["sections"] = [
                "Abstract",
                "Introduction",
                "Literature Review",
                "Methodology",
                "Results",
                "Discussion",
                "Conclusion",
                "References"
            ]
        
        logger.info(f"보고서 형식 결정 완료: {task_type} 유형, {len(report_format['sections'])} 섹션")
        return report_format

    def delegate_to_research_agent(self, research_plan: Dict[str, Any]) -> bool:
        """
        조사 에이전트에게 조사 계획을 전달합니다.
        
        Args:
            research_plan: 조사 계획
            
        Returns:
            bool: 성공 여부
        """
        if not hasattr(self, 'research_agent') or self.research_agent is None:
            try:
                from agents.research_agent import ResearchAgent
                self.research_agent = ResearchAgent(verbose=self.verbose)
            except Exception as e:
                logger.error(f"조사 에이전트 초기화 실패: {str(e)}")
                return False
        
        logger.info("조사 에이전트에게 조사 계획 전달")
        try:
            # 조사 에이전트의 설정 업데이트
            self.research_agent.update_config({
                "search_depth": research_plan.get("search_depth", 3),
                "max_sources": research_plan.get("max_sources", 10),
                "focus_keywords": research_plan.get("focus_keywords", []),
                "research_questions": research_plan.get("research_questions", [])
            })
            
            logger.info("조사 에이전트 설정 업데이트 완료")
            return True
        except Exception as e:
            logger.error(f"조사 에이전트에게 계획 전달 중 오류: {str(e)}")
            return False

    def delegate_to_writing_agent(self, report_format: Dict[str, Any]) -> bool:
        """
        작성 에이전트에게 보고서 형식을 전달합니다.
        
        Args:
            report_format: 보고서 형식
            
        Returns:
            bool: 성공 여부
        """
        if not hasattr(self, 'writer_agent') or self.writer_agent is None:
            try:
                from agents.writing_agent import WriterAgent
                self.writer_agent = WriterAgent(verbose=self.verbose)
            except Exception as e:
                logger.error(f"작성 에이전트 초기화 실패: {str(e)}")
                return False
        
        logger.info("작성 에이전트에게 보고서 형식 전달")
        try:
            # 작성 에이전트의 설정 업데이트
            self.writer_agent.update_template_config({
                "sections": report_format.get("sections", []),
                "citation_style": report_format.get("citation_style", "APA"),
                "formatting": report_format.get("formatting_guidelines", {})
            })
            
            logger.info("작성 에이전트 설정 업데이트 완료")
            return True
        except Exception as e:
            logger.error(f"작성 에이전트에게 형식 전달 중 오류: {str(e)}")
            return False

    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """
        LLM 응답에서 JSON 형식의 데이터를 추출합니다.
        
        Args:
            response_text: LLM 응답 텍스트
            
        Returns:
            Dict: 추출된 JSON 데이터
        """
        import re
        import json
        
        # JSON 블록 찾기 (```json과 ``` 사이)
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 중괄호로 둘러싸인 블록 찾기
            json_match = re.search(r"\{[\s\S]*\}", response_text)
            if json_match:
                json_str = json_match.group(0)
            else:
                return {}
        
        # JSON 파싱 시도
        try:
            return json.loads(json_str)
        except Exception:
            # 파싱 실패시 빈 딕셔너리 반환
            return {}

    def _analyze_user_request(self, topic, paper_type, constraints=None, instructions=None):
        """
        사용자 요청을 분석하여 요구사항을 파악합니다.
        
        Args:
            topic: 연구 주제
            paper_type: 논문 유형
            constraints: 제약사항
            instructions: 추가 지시사항
            
        Returns:
            Dict: 분석된 요구사항
        """
        logger.info(f"사용자 요청 분석: 주제={topic}, 유형={paper_type}")
        
        try:
            # 프롬프트 생성
            prompt = f"""
            다음 연구 주제와 논문 유형에 대한 요구사항을 분석해주세요:
            
            주제: {topic}
            논문 유형: {paper_type}
            
            제약사항: {constraints if constraints else '없음'}
            추가 지시사항: {instructions if instructions else '없음'}
            
            다음 항목을 포함하여 JSON 형식으로 응답해주세요:
            1. 핵심 키워드 (keywords): 연구에 필요한 핵심 키워드 목록
            2. 연구 질문 (research_questions): 답변해야 할 주요 연구 질문 목록
            3. 필요한 자료 유형 (required_sources): 필요한 자료 유형 목록 (예: 학술 논문, 통계 자료, 사례 연구 등)
            4. 연구 범위 (scope): 연구의 범위와 한계
            5. 방법론 (methodology): 적합한 연구 방법론 제안
            6. 예상 결과물 (expected_outcomes): 예상되는 결과물 설명
            7. 보고서 구조 (report_structure): 적합한 보고서 구조 제안
            """
            
            # LLM을 사용하여 요청 분석
            response = self.llm.invoke(prompt)
            
            # JSON 추출
            analysis = self._extract_json_from_response(response.content)
            
            # 기본값 설정
            if "keywords" not in analysis:
                analysis["keywords"] = [topic]
            
            if "research_questions" not in analysis:
                analysis["research_questions"] = [f"{topic}에 대한 최신 연구는 무엇인가?"]
            
            if "required_sources" not in analysis:
                analysis["required_sources"] = ["학술 논문", "연구 보고서"]
            
            if "scope" not in analysis:
                analysis["scope"] = f"{topic}에 관한 최근 5년간의 연구"
            
            if "methodology" not in analysis:
                analysis["methodology"] = "문헌 조사 및 분석"
            
            if "expected_outcomes" not in analysis:
                analysis["expected_outcomes"] = [f"{topic}에 대한 종합적인 이해와 분석"]
            
            if "report_structure" not in analysis:
                analysis["report_structure"] = ["서론", "본론", "결론"]
            
            logger.info(f"사용자 요청 분석 완료: {len(analysis['keywords'])}개 키워드, {len(analysis['research_questions'])}개 연구 질문 식별")
            return analysis
            
        except Exception as e:
            logger.error(f"사용자 요청 분석 중 오류: {str(e)}")
            # 오류 발생 시 기본 분석 결과 반환
            return {
                "keywords": [topic],
                "research_questions": [f"{topic}에 대한 최신 연구는 무엇인가?"],
                "required_sources": ["학술 논문", "연구 보고서"],
                "scope": f"{topic}에 관한 최근 5년간의 연구",
                "methodology": "문헌 조사 및 분석",
                "expected_outcomes": [f"{topic}에 대한 종합적인 이해와 분석"],
                "report_structure": ["서론", "본론", "결론"]
            }

    def _validate_investigation_results(self, investigation_results, topic):
        """
        조사 결과의 타당성을 검증합니다.
        
        Args:
            investigation_results: 조사 결과
            topic: 연구 주제
            
        Returns:
            Tuple[bool, str]: (타당성 여부, 피드백)
        """
        logger.info(f"조사 결과 검증: {len(investigation_results) if investigation_results else 0}개 자료")
        
        try:
            # 결과가 없는 경우
            if not investigation_results or len(investigation_results) == 0:
                return False, "조사 결과가 없습니다. 검색 범위를 확장하거나 다른 키워드를 사용해 보세요."
            
            # 결과가 너무 적은 경우
            if len(investigation_results) < 3:
                return False, f"조사 결과가 부족합니다 ({len(investigation_results)}개). 최소 3개 이상의 자료가 필요합니다."
            
            # 주제 관련성 검증
            relevant_count = 0
            for material in investigation_results:
                if hasattr(material, 'relevance_score') and material.relevance_score >= 0.6:
                    relevant_count += 1
            
            if relevant_count < len(investigation_results) * 0.5:
                return False, f"관련성 높은 자료가 부족합니다 ({relevant_count}/{len(investigation_results)}). 더 정확한 키워드로 검색해 보세요."
            
            # 자료의 다양성 검증
            sources = set()
            years = set()
            for material in investigation_results:
                if hasattr(material, 'source'):
                    sources.add(material.source)
                if hasattr(material, 'year') and material.year:
                    years.add(material.year)
            
            if len(sources) < 2:
                return False, "자료 출처가 다양하지 않습니다. 다양한 소스에서 자료를 수집해 보세요."
            
            if len(years) < 2:
                return False, "자료의 출판 연도가 다양하지 않습니다. 다양한 시기의 자료를 수집해 보세요."
            
            # 최신 자료 포함 여부 검증
            current_year = datetime.now().year
            has_recent = False
            for material in investigation_results:
                if hasattr(material, 'year') and material.year and material.year >= current_year - 3:
                    has_recent = True
                    break
            
            if not has_recent:
                return False, "최신 자료(최근 3년 이내)가 포함되어 있지 않습니다. 최신 연구 자료를 추가해 보세요."
            
            # 모든 검증을 통과한 경우
            return True, "조사 결과가 타당합니다."
            
        except Exception as e:
            logger.error(f"조사 결과 검증 중 오류: {str(e)}")
            # 오류 발생 시 기본적으로 통과시킴
            return True, f"검증 중 오류가 발생했으나 진행합니다: {str(e)}"
    
    def _revise_investigation_plan(self, original_plan, feedback):
        """
        피드백을 바탕으로 조사 계획을 수정합니다.
        
        Args:
            original_plan: 원래 조사 계획
            feedback: 피드백
            
        Returns:
            Dict: 수정된 조사 계획
        """
        logger.info(f"조사 계획 수정: {feedback}")
        
        try:
            # 원본 계획 복사
            revised_plan = original_plan.copy()
            
            # 피드백에 따른 수정
            if "검색 범위를 확장" in feedback or "자료가 없습니다" in feedback or "자료가 부족합니다" in feedback:
                # 검색 범위 확장
                revised_plan["search_strategy"]["search_scope"] = "all"
                revised_plan["search_depth"] = min(revised_plan.get("search_depth", 3) + 1, 5)
                revised_plan["max_sources"] = min(revised_plan.get("max_sources", 20) + 10, 50)
                
                # 최소 논문 수 감소
                revised_plan["search_strategy"]["min_papers"] = max(revised_plan["search_strategy"].get("min_papers", 10) - 2, 3)
                
                # 관련성 임계값 낮추기
                if "evaluation_criteria" in revised_plan:
                    revised_plan["evaluation_criteria"]["relevance_threshold"] = max(
                        revised_plan["evaluation_criteria"].get("relevance_threshold", 0.6) - 0.1, 
                        0.4
                    )
            
            if "더 정확한 키워드" in feedback or "관련성 높은 자료가 부족" in feedback:
                # 주제에서 추가 키워드 추출
                topic = revised_plan.get("topic", "")
                
                # 프롬프트 생성
                prompt = f"""
                다음 주제에 대한 더 구체적인 검색 키워드를 5개 생성해주세요:
                
                주제: {topic}
                
                피드백: {feedback}
                
                JSON 형식으로 응답해주세요:
                {{
                    "keywords": ["키워드1", "키워드2", ...]
                }}
                """
                
                # LLM을 사용하여 키워드 생성
                response = self.llm.invoke(prompt)
                
                # JSON 추출
                result = self._extract_json_from_response(response.content)
                
                if "keywords" in result and result["keywords"]:
                    # 기존 쿼리와 새 키워드 결합
                    existing_queries = set(revised_plan["search_strategy"].get("queries", []))
                    for keyword in result["keywords"]:
                        if keyword not in existing_queries:
                            existing_queries.add(keyword)
                    
                    # 최대 7개 쿼리로 제한
                    revised_plan["search_strategy"]["queries"] = list(existing_queries)[:7]
                    
                    # 포커스 키워드도 업데이트
                    revised_plan["focus_keywords"] = list(set(revised_plan.get("focus_keywords", []) + result["keywords"]))
            
            if "자료 출처가 다양하지 않습니다" in feedback:
                # 다양한 소스 추가
                if "required_sources" not in revised_plan or len(revised_plan["required_sources"]) < 3:
                    revised_plan["required_sources"] = list(set(revised_plan.get("required_sources", []) + ["학술 논문", "연구 보고서", "학위 논문", "컨퍼런스 자료", "서적"]))
            
            if "출판 연도가 다양하지 않습니다" in feedback or "최신 자료" in feedback:
                # 시간 범위 확장
                revised_plan["time_constraints"] = "최근 10년"
                
                # 최신 자료 강조
                if "search_strategy" in revised_plan:
                    current_year = datetime.now().year
                    revised_plan["search_strategy"]["queries"].append(f"{revised_plan['topic']} {current_year-1}")
                    revised_plan["search_strategy"]["queries"].append(f"{revised_plan['topic']} {current_year-2}")
                    
                    # 중복 제거 및 최대 7개 쿼리로 제한
                    revised_plan["search_strategy"]["queries"] = list(set(revised_plan["search_strategy"]["queries"]))[:7]
            
            logger.info(f"조사 계획 수정 완료: {len(revised_plan['search_strategy']['queries'])}개 검색 쿼리")
            return revised_plan
            
        except Exception as e:
            logger.error(f"조사 계획 수정 중 오류: {str(e)}")
            # 오류 발생 시 원본 계획 반환
            return original_plan

    def _create_research_plan(self, request_analysis, topic, investigation_results):
        """
        사용자 요청 분석과 조사 결과를 바탕으로 연구 계획을 생성합니다.
        
        Args:
            request_analysis: 사용자 요청 분석 결과
            topic: 연구 주제
            investigation_results: 조사 결과
            
        Returns:
            Dict: 연구 계획
        """
        logger.info(f"연구 계획 생성: 주제={topic}, 자료={len(investigation_results) if investigation_results else 0}개")
        
        try:
            # 조사 결과에서 주요 키워드 추출
            keywords = set(request_analysis.get("keywords", [topic]))
            for material in investigation_results:
                if hasattr(material, 'title'):
                    # 제목에서 키워드 추출
                    title_words = material.title.split()
                    for word in title_words:
                        if len(word) > 1 and word.lower() not in ["the", "and", "or", "of", "in", "on", "at", "by", "for", "with", "about", "to", "from"]:
                            keywords.add(word)
            
            # 연구 계획 구성
            research_plan = {
                "topic": topic,
                "objective": f"{topic}에 대한 종합적인 이해와 분석",
                "methodology": request_analysis.get("methodology", "문헌 조사 및 분석"),
                "key_questions": request_analysis.get("research_questions", [f"{topic}에 대한 최신 연구는 무엇인가?"]),
                "analysis_framework": {
                    "approach": "주제별 분석",
                    "dimensions": ["현황", "동향", "이슈", "전망"],
                    "comparison_criteria": ["방법론", "결과", "한계점"]
                },
                "key_materials": [],
                "expected_outcomes": request_analysis.get("expected_outcomes", [f"{topic}에 대한 종합적인 이해와 분석"]),
                "timeline": {
                    "analysis": "1일",
                    "synthesis": "1일",
                    "writing": "1일"
                }
            }
            
            # 주요 자료 선정 (최대 5개)
            if investigation_results:
                sorted_materials = sorted(investigation_results, key=lambda x: getattr(x, 'relevance_score', 0), reverse=True)
                for material in sorted_materials[:5]:
                    if hasattr(material, 'id') and hasattr(material, 'title'):
                        research_plan["key_materials"].append({
                            "id": material.id,
                            "title": material.title,
                            "reason": "높은 관련성"
                        })
            
            logger.info(f"연구 계획 생성 완료: {len(research_plan['key_materials'])}개 주요 자료 선정")
            return research_plan
            
        except Exception as e:
            logger.error(f"연구 계획 생성 중 오류: {str(e)}")
            # 오류 발생 시 기본 연구 계획 반환
            return {
                "topic": topic,
                "objective": f"{topic}에 대한 종합적인 이해와 분석",
                "methodology": "문헌 조사 및 분석",
                "key_questions": [f"{topic}에 대한 최신 연구는 무엇인가?"],
                "analysis_framework": {
                    "approach": "주제별 분석",
                    "dimensions": ["현황", "동향", "이슈", "전망"]
                },
                "key_materials": [],
                "expected_outcomes": [f"{topic}에 대한 종합적인 이해와 분석"]
            }
    
    def _validate_research_results(self, research_materials, topic):
        """
        연구 결과의 타당성을 검증합니다.
        
        Args:
            research_materials: 연구 결과
            topic: 연구 주제
            
        Returns:
            Tuple[bool, str]: (타당성 여부, 피드백)
        """
        logger.info(f"연구 결과 검증: {len(research_materials) if research_materials else 0}개 자료")
        
        try:
            # 결과가 없는 경우
            if not research_materials or len(research_materials) == 0:
                return False, "연구 결과가 없습니다. 다시 분석을 시도해 보세요."
            
            # 내용 및 요약 검증
            missing_content = 0
            missing_summary = 0
            for material in research_materials:
                if not hasattr(material, 'content') or not material.content:
                    missing_content += 1
                if not hasattr(material, 'summary') or not material.summary:
                    missing_summary += 1
            
            if missing_content > len(research_materials) * 0.5:
                return False, f"많은 자료({missing_content}/{len(research_materials)})에 내용이 없습니다. 내용 추출을 다시 시도해 보세요."
            
            if missing_summary > len(research_materials) * 0.5:
                return False, f"많은 자료({missing_summary}/{len(research_materials)})에 요약이 없습니다. 요약 생성을 다시 시도해 보세요."
            
            # 주제 관련성 검증
            low_relevance = 0
            for material in research_materials:
                if hasattr(material, 'relevance_score') and material.relevance_score < 0.6:
                    low_relevance += 1
            
            if low_relevance > len(research_materials) * 0.3:
                return False, f"관련성이 낮은 자료({low_relevance}/{len(research_materials)})가 많습니다. 더 관련성 높은 자료를 선별해 보세요."
            
            # 모든 검증을 통과한 경우
            return True, "연구 결과가 타당합니다."
            
        except Exception as e:
            logger.error(f"연구 결과 검증 중 오류: {str(e)}")
            # 오류 발생 시 기본적으로 통과시킴
            return True, f"검증 중 오류가 발생했으나 진행합니다: {str(e)}"
    
    def _revise_research_plan(self, original_plan, feedback):
        """
        피드백을 바탕으로 연구 계획을 수정합니다.
        
        Args:
            original_plan: 원래 연구 계획
            feedback: 피드백
            
        Returns:
            Dict: 수정된 연구 계획
        """
        logger.info(f"연구 계획 수정: {feedback}")
        
        try:
            # 원본 계획 복사
            revised_plan = original_plan.copy()
            
            # 피드백에 따른 수정
            if "내용이 없습니다" in feedback:
                # 내용 추출 강화
                revised_plan["content_extraction"] = {
                    "method": "hybrid",
                    "fallback_strategy": "title_based",
                    "retry_count": 3
                }
            
            if "요약이 없습니다" in feedback:
                # 요약 생성 강화
                revised_plan["summary_generation"] = {
                    "method": "extractive_then_abstractive",
                    "length": "medium",
                    "focus_on_key_findings": True
                }
            
            if "관련성이 낮은 자료" in feedback:
                # 관련성 기준 강화
                revised_plan["relevance_criteria"] = {
                    "threshold": 0.7,
                    "prioritize_recent": True,
                    "keyword_match_weight": 0.8
                }
                
                # 주요 자료 재선정 지시
                revised_plan["reselect_materials"] = True
            
            logger.info(f"연구 계획 수정 완료")
            return revised_plan
            
        except Exception as e:
            logger.error(f"연구 계획 수정 중 오류: {str(e)}")
            # 오류 발생 시 원본 계획 반환
            return original_plan

    def _create_report_format(self, request_analysis, topic, paper_type):
        """
        사용자 요청 분석을 바탕으로 보고서 양식을 생성합니다.
        
        Args:
            request_analysis: 사용자 요청 분석 결과
            topic: 연구 주제
            paper_type: 논문 유형
            
        Returns:
            Dict: 보고서 양식
        """
        logger.info(f"보고서 양식 생성: 주제={topic}, 유형={paper_type}")
        
        try:
            # 보고서 구조 결정
            structure = request_analysis.get("report_structure", ["서론", "본론", "결론"])
            
            # 논문 유형에 따른 기본 섹션 설정
            if paper_type.lower() in ["academic", "research", "학술", "연구"]:
                sections = ["초록", "서론", "선행 연구", "연구 방법", "결과", "논의", "결론", "참고문헌"]
            elif paper_type.lower() in ["review", "survey", "리뷰", "조사"]:
                sections = ["초록", "서론", "문헌 조사", "주요 발견", "동향 분석", "향후 전망", "결론", "참고문헌"]
            elif paper_type.lower() in ["report", "보고서"]:
                sections = ["요약", "서론", "현황 분석", "주요 이슈", "해결 방안", "결론", "참고자료"]
            else:
                # 기본 구조
                sections = ["서론", "본론", "결론", "참고문헌"]
            
            # 보고서 양식 구성
            report_format = {
                "title": f"{topic}에 관한 {paper_type}",
                "sections": sections,
                "citation_style": "APA",  # 기본 인용 스타일
                "formatting_guidelines": {
                    "font": "맑은 고딕",
                    "font_size": "11pt",
                    "line_spacing": 1.5,
                    "margins": "2.5cm"
                },
                "section_guidelines": {}
            }
            
            # 각 섹션별 가이드라인 추가
            for section in sections:
                if section.lower() in ["초록", "abstract", "요약", "summary"]:
                    report_format["section_guidelines"][section] = "연구의 목적, 방법, 결과, 결론을 간략히 요약 (200-300단어)"
                elif section.lower() in ["서론", "introduction"]:
                    report_format["section_guidelines"][section] = "연구 배경, 목적, 연구 질문 제시"
                elif section.lower() in ["선행 연구", "literature review", "문헌 조사"]:
                    report_format["section_guidelines"][section] = "관련 선행 연구 검토 및 분석"
                elif section.lower() in ["연구 방법", "methodology", "methods"]:
                    report_format["section_guidelines"][section] = "연구 방법, 자료 수집 및 분석 방법 설명"
                elif section.lower() in ["결과", "results", "findings", "주요 발견"]:
                    report_format["section_guidelines"][section] = "연구 결과 제시 (표, 그림 활용 가능)"
                elif section.lower() in ["논의", "discussion", "동향 분석"]:
                    report_format["section_guidelines"][section] = "연구 결과의 의미와 시사점 논의"
                elif section.lower() in ["결론", "conclusion"]:
                    report_format["section_guidelines"][section] = "연구 요약, 한계점, 향후 연구 방향 제시"
                elif section.lower() in ["참고문헌", "references", "참고자료"]:
                    report_format["section_guidelines"][section] = "인용된 모든 자료의 출처 (APA 형식)"
                else:
                    report_format["section_guidelines"][section] = f"{section} 내용 작성"
            
            logger.info(f"보고서 양식 생성 완료: {len(sections)}개 섹션")
            return report_format
            
        except Exception as e:
            logger.error(f"보고서 양식 생성 중 오류: {str(e)}")
            # 오류 발생 시 기본 보고서 양식 반환
            return {
                "title": f"{topic}에 관한 보고서",
                "sections": ["서론", "본론", "결론", "참고문헌"],
                "citation_style": "APA",
                "formatting_guidelines": {
                    "font": "맑은 고딕",
                    "font_size": "11pt",
                    "line_spacing": 1.5
                }
            }
    
    def _validate_writing_results(self, paper_draft, topic, report_format):
        """
        Validate the writing results against requirements
        
        Args:
            paper_draft: Draft paper
            topic: Paper topic
            report_format: Report format
            
        Returns:
            Validation results
        """
        logger.info(f"Validating writing results for topic: {topic}")
        
        # Create validation criteria
        validation_criteria = [
            f"The paper addresses the topic: {topic}",
            f"The paper follows the specified format: {report_format.get('format', 'standard')}",
            "The paper is well-structured with clear sections",
            "The content is coherent and logically organized",
            "The paper is written in English",
            "All claims are supported by citations",
            "The paper includes a complete bibliography/references section",
            "The content is based on the vector database materials"
        ]
        
        # Prepare validation prompt
        validation_prompt = f"""
        Evaluate the following paper draft against these criteria:
        
        Paper Topic: {topic}
        
        Validation Criteria:
        {chr(10).join([f"- {criterion}" for criterion in validation_criteria])}
        
        Paper Draft:
        {paper_draft}
        
        For each criterion, provide a pass/fail assessment and brief explanation.
        Also provide an overall assessment and recommendations for improvement.
        """
        
        # Get validation results
        validation_result = self.llm.invoke(validation_prompt)
        
        # Extract validation details
        validation_details = {
            "overall_assessment": "Needs revision",  # Default
            "criteria_results": {},
            "recommendations": []
        }
        
        # Parse validation result
        try:
            # Extract overall assessment
            overall_pattern = r"Overall\s+Assessment:?\s*(.*?)(?:\n|$)"
            overall_match = re.search(overall_pattern, validation_result, re.IGNORECASE)
            if overall_match:
                validation_details["overall_assessment"] = overall_match.group(1).strip()
            
            # Extract criteria results
            for criterion in validation_criteria:
                key = criterion.split(":")[0].strip().lower().replace(" ", "_")
                pattern = f"{re.escape(criterion)}:?\\s*(Pass|Fail)\\s*(.*?)(?:\\n\\n|$)"
                match = re.search(pattern, validation_result, re.IGNORECASE | re.DOTALL)
                if match:
                    validation_details["criteria_results"][key] = {
                        "status": match.group(1).strip(),
                        "explanation": match.group(2).strip()
                    }
            
            # Extract recommendations
            recommendations_pattern = r"Recommendations:?\s*(.*?)(?:\n\n|$)"
            recommendations_match = re.search(recommendations_pattern, validation_result, re.IGNORECASE | re.DOTALL)
            if recommendations_match:
                recommendations_text = recommendations_match.group(1).strip()
                validation_details["recommendations"] = [r.strip() for r in recommendations_text.split("\n-") if r.strip()]
        
        except Exception as e:
            logger.error(f"Error parsing validation result: {str(e)}")
            validation_details["error"] = str(e)
        
        # Check if paper meets the requirements
        meets_requirements = (
            validation_details.get("criteria_results", {}).get("the_paper_is_written_in_english", {}).get("status", "Fail") == "Pass" and
            validation_details.get("criteria_results", {}).get("all_claims_are_supported_by_citations", {}).get("status", "Fail") == "Pass" and
            validation_details.get("criteria_results", {}).get("the_paper_includes_a_complete_bibliography/references_section", {}).get("status", "Fail") == "Pass" and
            validation_details.get("criteria_results", {}).get("the_content_is_based_on_the_vector_database_materials", {}).get("status", "Fail") == "Pass"
        )
        
        validation_details["meets_requirements"] = meets_requirements
        
        logger.info(f"Writing validation completed. Meets requirements: {meets_requirements}")
        return validation_details
    
    def _revise_report_format(self, original_format, feedback):
        """
        피드백을 바탕으로 보고서 양식을 수정합니다.
        
        Args:
            original_format: 원래 보고서 양식
            feedback: 피드백
            
        Returns:
            Dict: 수정된 보고서 양식
        """
        logger.info(f"보고서 양식 수정: {feedback}")
        
        try:
            # 원본 양식 복사
            revised_format = original_format.copy()
            
            # 피드백에 따른 수정
            if "작성된 내용이 너무 짧습니다" in feedback:
                # 각 섹션별 최소 길이 지정
                revised_format["section_min_length"] = {}
                for section in revised_format.get("sections", []):
                    if section.lower() in ["초록", "abstract", "요약", "summary"]:
                        revised_format["section_min_length"][section] = 200
                    elif section.lower() in ["서론", "introduction"]:
                        revised_format["section_min_length"][section] = 500
                    elif section.lower() in ["본론", "body", "선행 연구", "연구 방법", "결과", "논의"]:
                        revised_format["section_min_length"][section] = 1000
                    elif section.lower() in ["결론", "conclusion"]:
                        revised_format["section_min_length"][section] = 300
                    else:
                        revised_format["section_min_length"][section] = 500
                
                # 전체 최소 길이 지정
                revised_format["min_total_length"] = 3000
            
            if "섹션이 누락되었습니다" in feedback:
                # 누락된 섹션 강조
                missing_sections = re.findall(r'다음 섹션이 누락되었습니다: (.*?)\.', feedback)
                if missing_sections:
                    sections = missing_sections[0].split(', ')
                    revised_format["required_sections"] = sections
                    
                    # 각 섹션별 가이드라인 강화
                    for section in sections:
                        if section in revised_format.get("section_guidelines", {}):
                            revised_format["section_guidelines"][section] += " (필수 섹션)"
            
            if "주제와의 연관성을 강화" in feedback:
                # 주제 강조 지시 추가
                revised_format["emphasis"] = {
                    "topic": True,
                    "keywords": True
                }
                
                # 서론 가이드라인 강화
                if "서론" in revised_format.get("section_guidelines", {}):
                    revised_format["section_guidelines"]["서론"] += " 주제를 명확히 제시하고 연구의 중요성을 강조해 주세요."
            
            if "인용이 포함되어 있지 않습니다" in feedback:
                # 인용 지침 강화
                revised_format["citation_requirements"] = {
                    "min_citations": 5,
                    "style": revised_format.get("citation_style", "APA"),
                    "include_in_sections": ["서론", "선행 연구", "본론", "논의"]
                }
                
                # 참고문헌 가이드라인 강화
                if "참고문헌" in revised_format.get("section_guidelines", {}):
                    revised_format["section_guidelines"]["참고문헌"] += " 최소 5개 이상의 참고문헌을 포함해 주세요."
            
            logger.info(f"보고서 양식 수정 완료")
            return revised_format
            
        except Exception as e:
            logger.error(f"보고서 양식 수정 중 오류: {str(e)}")
            # 오류 발생 시 원본 양식 반환
            return original_format
    
    def _validate_editing_results(self, edited_paper, paper_draft, report_format):
        """
        편집 결과의 타당성을 검증합니다.
        
        Args:
            edited_paper: 편집된 논문
            paper_draft: 원본 논문 초안
            report_format: 보고서 양식
            
        Returns:
            Tuple[bool, str]: (타당성 여부, 피드백)
        """
        logger.info(f"편집 결과 검증")
        
        try:
            # 편집 결과가 없는 경우
            if not edited_paper or not isinstance(edited_paper, dict) or "content" not in edited_paper:
                return False, "편집된 결과가 없습니다. 다시 편집을 시도해 보세요."
            
            edited_content = edited_paper.get("content", "")
            original_content = paper_draft.get("content", "") if paper_draft and isinstance(paper_draft, dict) else ""
            
            # 내용이 너무 짧아진 경우
            if len(edited_content) < len(original_content) * 0.8:
                return False, f"편집 후 내용이 너무 많이 줄었습니다 (원본: {len(original_content)}자, 편집: {len(edited_content)}자). 중요 내용이 누락되지 않았는지 확인해 주세요."
            
            # 형식 지침 준수 여부 검증
            formatting_issues = []
            
            # 인용 스타일 검증
            citation_style = report_format.get("citation_style", "APA").upper()
            if citation_style == "APA":
                # APA 스타일 인용 패턴
                if not re.search(r'\([A-Za-z가-힣]+,?\s+\d{4}\)', edited_content):
                    formatting_issues.append("APA 형식의 인용이 올바르게 적용되지 않았습니다.")
            
            # 참고문헌 섹션 검증
            if not re.search(r'참고문헌|References', edited_content, re.IGNORECASE):
                formatting_issues.append("참고문헌 섹션이 없거나 올바르게 포맷되지 않았습니다.")
            
            if formatting_issues:
                return False, f"다음 형식 문제가 발견되었습니다: {'; '.join(formatting_issues)}. 형식 지침에 맞게 수정해 주세요."
            
            # 모든 검증을 통과한 경우
            return True, "편집 결과가 타당합니다."
            
        except Exception as e:
            logger.error(f"편집 결과 검증 중 오류: {str(e)}")
            # 오류 발생 시 기본적으로 통과시킴
            return True, f"검증 중 오류가 발생했으나 진행합니다: {str(e)}"
    
    def _revise_editing_instructions(self, report_format, feedback):
        """
        피드백을 바탕으로 편집 지시를 수정합니다.
        
        Args:
            report_format: 보고서 양식
            feedback: 피드백
            
        Returns:
            Dict: 수정된 편집 지시
        """
        logger.info(f"편집 지시 수정: {feedback}")
        
        try:
            # 원본 양식 복사
            revised_instructions = report_format.copy()
            
            # 피드백에 따른 수정
            if "내용이 너무 많이 줄었습니다" in feedback:
                # 내용 보존 지시 추가
                revised_instructions["editing_guidelines"] = {
                    "preserve_content": True,
                    "focus_on_formatting": True,
                    "minimal_content_changes": True
                }
            
            if "인용이 올바르게 적용되지 않았습니다" in feedback:
                # 인용 스타일 강화
                revised_instructions["citation_enforcement"] = {
                    "style": revised_instructions.get("citation_style", "APA"),
                    "strict": True,
                    "check_all_references": True
                }
            
            if "참고문헌 섹션이 없거나" in feedback:
                # 참고문헌 섹션 강화
                revised_instructions["references_section"] = {
                    "required": True,
                    "format": "APA",
                    "alphabetical_order": True
                }
                
                # 참고문헌 섹션 가이드라인 강화
                if "section_guidelines" in revised_instructions and "참고문헌" in revised_instructions["section_guidelines"]:
                    revised_instructions["section_guidelines"]["참고문헌"] += " (필수 섹션, APA 형식으로 알파벳 순서로 정렬)"
            
            logger.info(f"편집 지시 수정 완료")
            return revised_instructions
            
        except Exception as e:
            logger.error(f"편집 지시 수정 중 오류: {str(e)}")
            # 오류 발생 시 원본 지시 반환
            return report_format
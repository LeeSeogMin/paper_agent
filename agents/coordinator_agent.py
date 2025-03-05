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

from utils.academic_search import AcademicSearchManager
from utils.rag_integration import RAGEnhancer


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
        
        logger.info(f"{self.name} 초기화 완료")

    def _init_prompts(self) -> None:
        """프롬프트와 체인 초기화"""
        self.research_plan_parser = PydanticOutputParser(pydantic_object=ResearchPlan)
        self.status_parser = PydanticOutputParser(pydantic_object=ProjectStatus)
        
        # 기존 프롬프트 대신 agent_prompts.py의 프롬프트 활용
        self.research_plan_chain = LLMChain(
            llm=self.llm,
            prompt=AGENT_PLANNING_PROMPT,
            output_key="text",
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
            output_key="text",
            verbose=self.verbose
        )
        
        logger.debug("총괄 에이전트 프롬프트 및 체인 초기화 완료")

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
            result = self.research_plan_chain.run({
                "topic": topic,
                "task": task_type,  # 누락된 'task' 키
                "resources": [],    # 누락된 'resources' 키
                "constraints": ""   # 누락된 'constraints' 키
            })
            return result
        except Exception as e:
            logger.error(f"연구 계획 생성 중 오류: {str(e)}")
            # 오류 발생 시 기본 템플릿 반환
            return f"# 연구 계획: {topic}\n\n## 목표\n\n{topic}에 대한 연구 수행\n\n## 단계\n\n1. 문헌 조사\n2. 분석\n3. 정리"

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
        research_plan = self.create_research_plan(user_question, "literature_review")
        
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
            logger.info(f"사용자 요구사항 처리 시작")
            
            # 1. 연구 계획 생성 (중앙집중형 모델의 핵심)
            research_plan = self.create_research_plan(topic, paper_type, constraints, references)
            
            # 추가 지시사항 처리
            if instructions:
                self.process_instructions(instructions, research_plan)
            
            # 2. 연구 자료 수집 (연구 계획 전달)
            materials = self.research_agent.run(
                task="collect_materials",
                topic=topic,
                research_plan=research_plan,  # 연구 계획 전달
                max_queries=5,
                results_per_source=10,
                constraints=constraints
            )
            
            # 자료가 충분한지 확인
            if not materials or len(materials) < 3:
                # 계획 수정 - 로컬만 검색했다면 외부 검색 허용
                if research_plan["search_strategy"].get("search_scope") == "local_only":
                    logger.info("로컬 자료 부족, 외부 검색으로 확장")
                    research_plan["search_strategy"]["search_scope"] = "all"
                    # 다시 자료 수집
                    materials = self.research_agent.run(
                        task="collect_materials",
                        topic=topic,
                        research_plan=research_plan,
                        max_queries=5
                    )
            
            # 3. 논문 작성 (연구 계획 전달)
            # ... (이하 다른 에이전트에도 연구 계획 전달)
            
            # 4. 최종 논문 통합 및 반환
            # ... (기존 코드)
            
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

    def _create_writing_task(self, task_type: str, requirements: Dict[str, Any], 
                            research_plan: str, research_materials: Dict[str, Any]) -> Dict[str, Any]:
        """
        작업 유형에 따른 작성 작업 구성
        """
        topic = requirements.get("topic", "")
        additional_instructions = requirements.get("additional_instructions", "")
        
        # 기본 작업 템플릿
        writing_task = {
            "task_type": task_type,
            "topic": topic,
            "materials": research_materials.get("materials", []),
            "additional_context": {
                "research_plan": research_plan,
                "additional_instructions": additional_instructions
            }
        }
        
        # 작업 유형에 따른 추가 필드
        if task_type == "literature_review":
            writing_task["content"] = {
                "research_materials": research_materials.get("materials", []),
                "topic": topic,
                "outline": research_materials.get("outline", "")
            }
        elif task_type == "methodology":
            writing_task["additional_context"]["research_question"] = requirements.get("research_question", topic)
        elif task_type == "results_analysis":
            # 분석할 데이터 추가
            writing_task["data"] = research_materials.get("analysis_data", {})
        elif task_type == "custom":
            # 사용자 정의 프롬프트 구성
            writing_task["prompt"] = additional_instructions or f"Write about {topic}"
            writing_task["context"] = {
                "topic": topic,
                "materials": research_materials.get("materials", []),
                "additional_info": requirements.get("research_context", "")
            }
        
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
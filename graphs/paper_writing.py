"""
논문 작성 워크플로우 그래프
전체 논문 작성 프로세스를 관리하는 LangGraph 워크플로우입니다.
"""

from typing import Dict, Any, List, Optional, Callable, Union, Literal, cast

from langchain_core.runnables import RunnableConfig

from config.settings import DEFAULT_TEMPLATE
from utils.logger import logger
from models.state import PaperWorkflowState
from models.paper import Paper
from agents.research_agent import ResearchAgent
from agents.writing_agent import WriterAgent
from agents.editing_agent import EditorAgent, StyleGuide
from graphs.base import BaseGraph
from agents.review_agent import ReviewAgent


def research_node(state: PaperWorkflowState) -> PaperWorkflowState:
    """
    연구 노드 함수
    
    Args:
        state: 워크플로우 상태
        
    Returns:
        PaperWorkflowState: 업데이트된 상태
    """
    logger.info(f"연구 노드 실행 중: {state.topic}")
    
    # 연구 에이전트 초기화
    agent = ResearchAgent(verbose=state.verbose)
    
    # 연구 수행
    research_summary = agent.run(state.topic)
    
    # 상태 업데이트
    state.research_summary = research_summary
    state.research_materials = research_summary.collected_materials
    state.status = "research_completed"
    
    logger.info(f"연구 노드 완료: {len(state.research_materials)}개 자료 수집됨")
    return state


def writing_node(state: PaperWorkflowState) -> PaperWorkflowState:
    """
    작성 노드 함수
    
    Args:
        state: 워크플로우 상태
        
    Returns:
        PaperWorkflowState: 업데이트된 상태
    """
    logger.info(f"작성 노드 실행 중: {state.topic}")
    
    # 작성 에이전트 초기화
    agent = WriterAgent(verbose=state.verbose)
    
    # 템플릿 이름 확인
    template_name = state.template_name or DEFAULT_TEMPLATE
    
    # 논문 작성
    paper = agent.run(
        state.topic,
        state.research_materials,
        template_name=template_name
    )
    
    # 상태 업데이트
    state.paper = paper
    state.status = "writing_completed"
    
    logger.info(f"작성 노드 완료: {paper.title}, {len(paper.sections)}개 섹션")
    return state


def editing_node(state: PaperWorkflowState) -> PaperWorkflowState:
    """
    편집 노드 함수
    
    Args:
        state: 워크플로우 상태
        
    Returns:
        PaperWorkflowState: 업데이트된 상태
    """
    logger.info(f"편집 노드 실행 중: {state.paper.title}")
    
    # 편집 에이전트 초기화
    agent = EditorAgent(verbose=state.verbose)
    
    # 스타일 가이드 생성
    style_guide = StyleGuide(
        name=state.style_guide or "Standard Academic",
        citation_style=state.citation_style or "APA",
        formatting_rules=[
            "명확하고 간결한 문장 사용",
            "학술적 어조 유지",
            "능동태 우선 사용",
            "일관된 용어 사용"
        ],
        language_preferences=[
            "제3인칭 시점 사용",
            "전문 용어 적절히 사용",
            "불필요한 수식어 제거"
        ]
    )
    
    # 논문 편집
    edited_paper = agent.run(
        state.paper,
        style_guide=style_guide,
        output_format=state.output_format or "markdown"
    )
    
    # 상태 업데이트
    state.paper = edited_paper
    state.status = "editing_completed"
    
    # 리뷰 정보 추출
    if edited_paper.metadata and "review" in edited_paper.metadata:
        state.review = edited_paper.metadata["review"]
    
    # 포맷팅된 파일 경로 추출
    if edited_paper.metadata and "formatted_file" in edited_paper.metadata:
        state.output_file = edited_paper.metadata["formatted_file"]
    
    logger.info(f"편집 노드 완료: {edited_paper.title}")
    return state


def decide_next_step(state: PaperWorkflowState) -> str:
    """
    다음 단계 결정 함수
    
    Args:
        state: 워크플로우 상태
        
    Returns:
        str: 다음 단계 노드 이름
    """
    status = state.status
    
    if status == "initialized":
        # 초기 상태에서는 항상 연구부터 시작
        return "research"
    
    elif status == "research_completed":
        # 연구가 완료되면 작성 단계로 이동
        if state.research_materials and len(state.research_materials) > 0:
            return "writing"
        else:
            # 연구 자료가 없으면 완료 처리
            state.error = "연구 자료를 찾을 수 없습니다."
            return "END"
    
    elif status == "writing_completed":
        # 작성이 완료되면 편집 단계로 이동
        if state.paper and state.paper.sections:
            return "editing"
        else:
            # 논문이 비어있으면 완료 처리
            state.error = "논문 작성에 실패했습니다."
            return "END"
    
    elif status == "editing_completed":
        # 편집이 완료되면 종료
        if state.output_file:
            state.message = f"논문이 성공적으로 작성되었습니다: {state.output_file}"
        else:
            state.message = "논문 작성이 완료되었습니다."
        return "END"
    
    else:
        # 알 수 없는 상태
        state.error = f"알 수 없는 워크플로우 상태: {status}"
        return "END"


class PaperWritingGraph(BaseGraph[PaperWorkflowState]):
    """논문 작성 워크플로우 그래프"""
    
    def __init__(self, name: str = "논문 작성 워크플로우"):
        """
        논문 작성 그래프 초기화
        
        Args:
            name: 그래프 이름
        """
        description = "연구, 작성, 편집 단계를 포함하는 전체 논문 작성 프로세스"
        super().__init__(name, description, PaperWorkflowState)
        
        # 노드 추가
        self.add_node("research", research_node)
        self.add_node("writing", writing_node)
        self.add_node("editing", editing_node)
        
        # 진입점 설정
        self.set_entry_point("research")
        
        # 조건부 엣지 추가
        self.add_conditional_edges(
            "research",
            decide_next_step,
            ["writing", "END"]
        )
        
        self.add_conditional_edges(
            "writing",
            decide_next_step,
            ["editing", "END"]
        )
        
        self.add_conditional_edges(
            "editing",
            decide_next_step,
            ["END"]
        )
        
        logger.info("논문 작성 워크플로우 그래프 구성 완료")
    
    def create_initial_state(
        self, 
        topic: str,
        template_name: Optional[str] = None,
        style_guide: Optional[str] = None,
        citation_style: Optional[str] = None,
        output_format: Optional[str] = None,
        verbose: bool = False
    ) -> PaperWorkflowState:
        """
        초기 상태 생성
        
        Args:
            topic: 논문 주제
            template_name: 논문 템플릿 이름
            style_guide: 스타일 가이드 이름
            citation_style: 인용 스타일
            output_format: 출력 형식
            verbose: 상세 로깅 여부
            
        Returns:
            PaperWorkflowState: 초기 워크플로우 상태
        """
        return PaperWorkflowState(
            topic=topic,
            template_name=template_name,
            style_guide=style_guide,
            citation_style=citation_style,
            output_format=output_format,
            verbose=verbose,
            status="initialized"
        )
    
    def run_workflow(
        self,
        topic: str,
        template_name: Optional[str] = None,
        style_guide: Optional[str] = None,
        citation_style: Optional[str] = None,
        output_format: str = "markdown",
        verbose: bool = False,
        config: Optional[RunnableConfig] = None
    ) -> PaperWorkflowState:
        """
        워크플로우 실행
        
        Args:
            topic: 논문 주제
            template_name: 논문 템플릿 이름
            style_guide: 스타일 가이드 이름
            citation_style: 인용 스타일
            output_format: 출력 형식
            verbose: 상세 로깅 여부
            config: 실행 설정
            
        Returns:
            PaperWorkflowState: 최종 워크플로우 상태
        """
        logger.info(f"논문 작성 워크플로우 시작: 주제 '{topic}'")
        
        # 초기 상태 생성
        initial_state = self.create_initial_state(
            topic=topic,
            template_name=template_name,
            style_guide=style_guide,
            citation_style=citation_style,
            output_format=output_format,
            verbose=verbose
        )
        
        # 그래프 실행
        final_state = self.run(initial_state, config)
        
        # 결과 기록
        if final_state.error:
            logger.error(f"워크플로우 오류: {final_state.error}")
        else:
            logger.info(f"워크플로우 완료: {final_state.message or '성공'}")
            if final_state.output_file:
                logger.info(f"출력 파일: {final_state.output_file}")
        
        return final_state 

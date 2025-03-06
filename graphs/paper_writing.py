"""
논문 작성 워크플로우 그래프
전체 논문 작성 프로세스를 관리하는 LangGraph 워크플로우입니다.
"""

from typing import Dict, Any, List, Optional, Callable, Union, Literal, cast
import concurrent.futures

from langchain_core.runnables import RunnableConfig

from config.settings import DEFAULT_TEMPLATE, OUTPUT_DIR
from utils.logger import logger
from models.state import PaperWorkflowState
from models.paper import Paper
from models.research import ResearchMaterial
from agents import (
    ResearchAgent,
    WriterAgent,
    EditorAgent,
    CoordinatorAgent
)
from agents.editing_agent import StyleGuide
from graphs.base import BaseWorkflowGraph


def research_node(state: PaperWorkflowState, research_agent=None) -> PaperWorkflowState:
    """
    연구 노드 함수
    
    Args:
        state: 워크플로우 상태
        research_agent: 사용할 연구 에이전트 (테스트용)
        
    Returns:
        PaperWorkflowState: 업데이트된 상태
    """
    logger.info(f"연구 노드 실행 중: {state.topic}")
    
    try:
        # 연구 에이전트 초기화
        if research_agent is None:
            # state.verbose가 없으면 기본값 False 사용
            verbose = getattr(state, 'verbose', False)
            research_agent = ResearchAgent(verbose=verbose)
        
        # 연구 수행
        research_summary = research_agent.run(state.topic)
        
        # 상태 업데이트
        state.research_summary = research_summary
        
        # 검색 상태에 따라 처리
        if isinstance(research_summary, dict) and research_summary.get("status") == "error":
            # 에러 처리
            error_message = research_summary.get("message", "연구 과정에서 오류가 발생했습니다.")
            logger.warning(f"연구 자료 수집 실패: {error_message}")
            state.research_materials = []
            state.status = "research_completed"  # 오류 발생해도 진행
        elif isinstance(research_summary, dict) and "materials" in research_summary:
            # 성공 케이스
            state.research_materials = research_summary["materials"]
            state.status = "research_completed"
            logger.info(f"연구 노드 완료: {len(state.research_materials)}개 자료 수집됨")
        else:
            # 예상치 못한 형식
            logger.warning(f"예상하지 못한 연구 결과 형식: {type(research_summary)}")
            state.research_materials = []  # 빈 목록으로 초기화
            state.status = "research_completed"  # 일단 계속 진행
            logger.info("연구 노드 완료: 0개 자료 수집됨")
    
    except Exception as e:
        logger.error(f"연구 노드 실행 중 오류 발생: {str(e)}", exc_info=True)
        state.error = f"연구 단계 오류: {str(e)}"
        state.status = "research_failed"
        state.research_materials = []  # 예외 발생 시 빈 목록으로 초기화
    
    return state


def writing_node(state: PaperWorkflowState, writer_agent=None) -> PaperWorkflowState:
    """
    작성 노드 함수
    
    Args:
        state: 워크플로우 상태
        writer_agent: 사용할 작성 에이전트 (테스트용)
        
    Returns:
        PaperWorkflowState: 업데이트된 상태
    """
    logger.info(f"작성 노드 실행 중: {state.topic}")
    
    try:
        # 작성 에이전트 초기화
        if writer_agent is None:
            # state.verbose가 없으면 기본값 False 사용
            verbose = getattr(state, 'verbose', False)
            writer_agent = WriterAgent(verbose=verbose)
        
        # 템플릿 이름 확인
        template_name = state.template_name or DEFAULT_TEMPLATE
        
        # 논문 작성
        paper = writer_agent.run(
            state.topic,
            state.research_materials,
            template_name=template_name
        )
        
        # 상태 업데이트
        state.paper = paper
        state.status = "writing_completed"
        
        logger.info(f"작성 노드 완료: {paper.title}, {len(paper.sections)}개 섹션")
    
    except Exception as e:
        logger.error(f"작성 노드 실행 중 오류 발생: {str(e)}", exc_info=True)
        state.error = f"작성 단계 오류: {str(e)}"
        state.status = "writing_failed"
    
    return state


def editing_node(state: PaperWorkflowState, editor_agent=None) -> PaperWorkflowState:
    """
    논문 편집 노드
    
    Args:
        state: 워크플로우 상태
        editor_agent: 사용할 편집 에이전트 (테스트용)
        
    Returns:
        PaperWorkflowState: 업데이트된 상태
    """
    logger.info(f"편집 노드 실행 중: {state.paper.title}")
    
    try:
        # 편집 에이전트 초기화
        if editor_agent is None:
            # state.verbose가 없으면 기본값 False 사용
            verbose = getattr(state, 'verbose', False)
            editor_agent = EditorAgent(verbose=verbose)
        
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
        edited_paper = editor_agent.run(
            state.paper,
            style_guide=style_guide,
            output_format=state.output_format or "markdown"
        )
        
        # 상태 업데이트
        state.paper = edited_paper
        state.status = "editing_completed"
        
        # 리뷰 필요 여부 설정 (기본값: True)
        state.needs_review = True
        
        # 리뷰 정보 추출
        if edited_paper.metadata and "review" in edited_paper.metadata:
            state.review = edited_paper.metadata["review"]
        
        # 포맷팅된 파일 경로 추출
        if edited_paper.metadata and "formatted_file" in edited_paper.metadata:
            state.output_file = edited_paper.metadata["formatted_file"]
        
        logger.info(f"편집 노드 완료: {edited_paper.title}")
    
    except Exception as e:
        logger.error(f"편집 노드 실행 중 오류 발생: {str(e)}", exc_info=True)
        state.error = f"편집 단계 오류: {str(e)}"
        state.status = "editing_failed"
    
    return state


def review_node(state: PaperWorkflowState, review_agent=None) -> PaperWorkflowState:
    """
    논문 리뷰 노드
    
    Args:
        state: 워크플로우 상태
        review_agent: 사용할 리뷰 에이전트 (테스트용)
        
    Returns:
        PaperWorkflowState: 업데이트된 상태
    """
    logger.info(f"리뷰 노드 실행 중: {state.paper.title}")
    
    try:
        # 리뷰 에이전트가 없으므로 간단한 리뷰 결과 생성
        review_result = {
            "score": 8.5,  # 기본 점수
            "comments": ["논문이 잘 작성되었습니다."],
            "strengths": ["논리적 구성", "명확한 설명"],
            "weaknesses": [],
            "suggestions": []
        }
        
        # 상태 업데이트
        state.review_result = review_result
        state.status = "review_completed"
        
        # 수정 필요 여부 결정
        if review_result.get("score", 0) < 7.0:  # 7점 미만이면 수정 필요
            state.needs_revision = True
            state.revision_comments = review_result.get("comments", [])
        else:
            state.needs_revision = False
        
        logger.info(f"리뷰 노드 완료: 점수 {review_result.get('score', 'N/A')}/10")
    
    except Exception as e:
        logger.error(f"리뷰 노드 실행 중 오류 발생: {str(e)}", exc_info=True)
        state.error = f"리뷰 단계 오류: {str(e)}"
        state.status = "review_failed"
    
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
        if state.research_materials:
            logger.info(f"연구 자료 {len(state.research_materials)}개 사용")
        else:
            logger.warning("연구 자료 없이 기본 템플릿으로 진행")
        return "writing"  # 무조건 작성 단계로 진행
    
    elif status == "writing_completed":
        # 작성이 완료되면 편집 단계로 이동
        if state.paper and state.paper.sections:
            return "editing"
        else:
            # 논문이 비어있으면 완료 처리
            state.error = "논문 작성에 실패했습니다."
            return "END"
    
    elif status == "editing_completed":
        # 편집이 완료되면 리뷰 단계로 이동
        if state.needs_review:
            return "review"
        else:
            # 리뷰가 필요 없으면 종료
            if state.output_file:
                state.message = f"논문이 성공적으로 작성되었습니다: {state.output_file}"
            else:
                state.message = "논문 작성이 완료되었습니다."
            return "END"
    
    elif status == "review_completed":
        # 리뷰가 완료되면 수정 필요 여부에 따라 다음 단계 결정
        if state.needs_revision:
            # 수정이 필요하면 편집 단계로 돌아감
            state.status = "editing_needed"  # 상태 변경
            return "editing"
        else:
            # 수정이 필요 없으면 종료
            if state.output_file:
                state.message = f"논문이 성공적으로 작성되었습니다: {state.output_file}"
            else:
                state.message = "논문 작성이 완료되었습니다."
            return "END"
    
    elif status in ["research_failed", "writing_failed", "editing_failed", "review_failed"]:
        # 오류 발생 시 종료
        if not state.error:
            state.error = f"워크플로우 오류: {status}"
        return "END"
    
    else:
        # 알 수 없는 상태
        state.error = f"알 수 없는 워크플로우 상태: {status}"
        return "END"


class PaperWritingGraph(BaseWorkflowGraph[PaperWorkflowState]):
    """논문 작성 워크플로우 그래프"""
    
    def __init__(self, name: str = "논문 작성 워크플로우"):
        """
        논문 작성 그래프 초기화
        
        Args:
            name: 그래프 이름
        """
        description = "연구, 작성, 편집, 리뷰 단계를 포함하는 전체 논문 작성 프로세스"
        super().__init__(name, description, PaperWorkflowState)
        
        # 에이전트 미리 초기화 (선택적)
        self.research_agent = None
        self.writer_agent = None
        self.editor_agent = None
        
        # 노드 추가
        self.add_node("research", research_node)
        self.add_node("writing", writing_node)
        self.add_node("editing", editing_node)
        self.add_node("review", review_node)
        
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
            ["review", "END"]
        )
        
        self.add_conditional_edges(
            "review",
            decide_next_step,
            ["editing", "END"]
        )
        
        logger.info("논문 작성 워크플로우 그래프 구성 완료")
    
    def initialize_agents(self, verbose: bool = False) -> None:
        """
        에이전트 초기화
        
        Args:
            verbose: 상세 로깅 여부
        """
        self.research_agent = ResearchAgent(verbose=verbose)
        self.writer_agent = WriterAgent(verbose=verbose)
        self.editor_agent = EditorAgent(verbose=verbose)
        
        logger.info("모든 에이전트 초기화 완료")
    
    def create_initial_state(
        self, 
        topic: str,
        template_name: Optional[str] = None,
        style_guide: Optional[str] = None,
        citation_style: Optional[str] = None,
        output_format: Optional[str] = None,
        paper_format: str = "standard",
        additional_instructions: Optional[str] = None,
        uploaded_file: Optional[str] = None,
        verbose: bool = False
    ) -> PaperWorkflowState:
        """
        초기 워크플로우 상태 생성
        
        Args:
            topic: 논문 주제
            template_name: 논문 템플릿 이름
            style_guide: 스타일 가이드 이름
            citation_style: 인용 스타일
            output_format: 출력 형식
            paper_format: 논문 형식 (standard, literature_review 등)
            additional_instructions: 추가 지시사항
            uploaded_file: 업로드된 파일 경로
            verbose: 상세 로깅 여부
            
        Returns:
            PaperWorkflowState: 초기 워크플로우 상태
        """
        # 기본값 설정
        template_name = template_name or "academic"
        style_guide = style_guide or "Standard Academic"
        citation_style = citation_style or "APA"
        output_format = output_format or "markdown"
        
        # 에이전트 초기화
        self.initialize_agents(verbose=verbose)
        
        # 초기 상태 생성
        return PaperWorkflowState(
            topic=topic,
            template_name=template_name,
            style_guide=style_guide,
            citation_style=citation_style,
            output_format=output_format,
            paper_format=paper_format,
            additional_instructions=additional_instructions,
            uploaded_file=uploaded_file,
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
        paper_format: str = "standard",
        additional_instructions: Optional[str] = None,
        uploaded_file: Optional[str] = None,
        verbose: bool = False,
        config: Optional[RunnableConfig] = None,
        save_state_path: Optional[str] = None
    ) -> PaperWorkflowState:
        """
        워크플로우 실행
        
        Args:
            topic: 논문 주제
            template_name: 논문 템플릿 이름
            style_guide: 스타일 가이드 이름
            citation_style: 인용 스타일
            output_format: 출력 형식
            paper_format: 논문 형식 (standard, literature_review 등)
            additional_instructions: 추가 지시사항
            uploaded_file: 업로드된 파일 경로
            verbose: 상세 로깅 여부
            config: 실행 설정
            save_state_path: 상태 저장 경로 (None이면 저장하지 않음)
            
        Returns:
            PaperWorkflowState: 최종 워크플로우 상태
        """
        logger.info(f"논문 작성 워크플로우 시작: 주제 '{topic}'")
        
        # 에이전트 초기화
        if verbose:
            self.initialize_agents(verbose=verbose)
        
        # 초기 상태 생성
        initial_state = self.create_initial_state(
            topic=topic,
            template_name=template_name,
            style_guide=style_guide,
            citation_style=citation_style,
            output_format=output_format,
            paper_format=paper_format,
            additional_instructions=additional_instructions,
            uploaded_file=uploaded_file,
            verbose=verbose
        )
        
        # 그래프 실행
        final_state = self.run(initial_state, config)
        
        # 상태 저장 (선택적)
        if save_state_path:
            self.save_state(final_state, save_state_path)
        
        # 결과 기록
        if final_state.error:
            logger.error(f"워크플로우 오류: {final_state.error}")
        else:
            logger.info(f"워크플로우 완료: {final_state.message or '성공'}")
            if final_state.output_file:
                logger.info(f"출력 파일: {final_state.output_file}")
        
        return final_state
    
    def run_parallel_research(self, topic: str, max_workers: int = 3) -> List[ResearchMaterial]:
        """
        병렬 연구 수행
        
        Args:
            topic: 연구 주제
            max_workers: 최대 작업자 수
            
        Returns:
            List[ResearchMaterial]: 수집된 연구 자료 목록
        """
        logger.info(f"병렬 연구 시작: 주제 '{topic}'")
        
        # 연구 에이전트 초기화
        if self.research_agent is None:
            self.research_agent = ResearchAgent()
        
        # 검색 쿼리 생성
        queries = self.research_agent.generate_search_queries(topic)
        
        # 병렬 검색 실행
        all_materials = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_query = {
                executor.submit(self.research_agent.search_for_papers, query): query 
                for query in queries
            }
            
            for future in concurrent.futures.as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    results = future.result()
                    logger.info(f"쿼리 '{query}'에 대한 검색 결과: {len(results)}개")
                    
                    # 검색 결과를 ResearchMaterial로 변환
                    materials = [
                        self.research_agent.create_research_material(result)
                        for result in results
                    ]
                    all_materials.extend(materials)
                    
                except Exception as e:
                    logger.error(f"쿼리 '{query}' 처리 중 오류: {str(e)}")
        
        logger.info(f"병렬 연구 완료: {len(all_materials)}개 자료 수집")
        return all_materials
    
    def visualize_workflow(self, output_path: Optional[str] = None) -> str:
        """
        워크플로우 시각화
        
        Args:
            output_path: 출력 파일 경로
            
        Returns:
            str: 시각화 파일 경로
        """
        return self.visualize(output_path) 

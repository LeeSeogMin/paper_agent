"""
기본 그래프 클래스
모든 워크플로우 그래프의 기반 클래스입니다.
"""

from typing import Dict, Any, List, Optional, Type, TypeVar, Generic, Callable

from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import StateGraph, END

from models.state import WorkflowState
from utils.logger import logger


# 상태 타입 정의
T = TypeVar('T', bound=WorkflowState)


class BaseGraph(Generic[T]):
    """모든 워크플로우 그래프의 기본 클래스"""
    
    def __init__(
        self,
        name: str,
        description: str,
        state_type: Type[T]
    ):
        """
        기본 그래프 초기화
        
        Args:
            name: 그래프 이름
            description: 그래프 설명
            state_type: 그래프 상태 타입
        """
        self.name = name
        self.description = description
        self.state_type = state_type
        self.graph = StateGraph(state_type)
        
        # 노드 및 엣지 초기화
        self.nodes: Dict[str, Callable] = {}
        
        logger.info(f"그래프 '{name}' 초기화됨: {description}")
    
    def add_node(self, name: str, node: Callable) -> None:
        """
        그래프에 노드 추가
        
        Args:
            name: 노드 이름
            node: 노드 함수
        """
        self.nodes[name] = node
        self.graph.add_node(name, node)
        logger.info(f"노드 '{name}' 추가됨")
    
    def set_entry_point(self, node_name: str) -> None:
        """
        그래프 진입점 설정
        
        Args:
            node_name: 진입점 노드 이름
        """
        self.graph.set_entry_point(node_name)
        logger.info(f"진입점 설정됨: {node_name}")
    
    def add_edge(self, start: str, end: str) -> None:
        """
        두 노드 사이에 직접 엣지 추가
        
        Args:
            start: 시작 노드 이름
            end: 끝 노드 이름
        """
        self.graph.add_edge(start, end)
        logger.info(f"엣지 추가됨: {start} -> {end}")
    
    def add_conditional_edges(
        self, 
        source: str, 
        condition_func: Callable[[T], str],
        destinations: List[str]
    ) -> None:
        """
        조건에 따른 엣지 추가
        
        Args:
            source: 소스 노드 이름
            condition_func: 조건 함수 (어떤 노드로 이동할지 결정)
            destinations: 가능한 목적지 노드 이름 목록
        """
        if "END" in destinations:
            # END가 목적지에 포함된 경우 대체
            destinations = [d if d != "END" else END for d in destinations]
            
        self.graph.add_conditional_edges(
            source,
            condition_func,
            destinations
        )
        logger.info(f"조건부 엣지 추가됨: {source} -> {', '.join(str(d) for d in destinations)}")
    
    def compile(self) -> Runnable:
        """
        그래프 컴파일
        
        Returns:
            Runnable: 컴파일된 그래프
        """
        compiled_graph = self.graph.compile()
        logger.info(f"그래프 '{self.name}' 컴파일됨")
        return compiled_graph
    
    def run(self, initial_state: T, config: Optional[RunnableConfig] = None) -> T:
        """
        그래프 실행
        
        Args:
            initial_state: 초기 상태
            config: 실행 설정
            
        Returns:
            T: 최종 상태
        """
        compiled_graph = self.compile()
        final_state = compiled_graph.invoke(initial_state, config)
        logger.info(f"그래프 '{self.name}' 실행 완료")
        return final_state
    
    def __str__(self) -> str:
        """그래프 문자열 표현"""
        return f"{self.name}: {self.description} (노드: {len(self.nodes)}개)" 
"""
기본 에이전트 클래스
모든 에이전트의 기본 기능을 정의한 추상 클래스입니다.
"""

import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Callable, TypeVar, Generic

from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from config.settings import OPENAI_MODEL, TEMPERATURE
from utils.logger import logger


# 에이전트 출력 타입 정의
T = TypeVar('T')


class BaseAgent(Generic[T], ABC):
    """
    모든 에이전트의 기본 기능을 정의한 추상 기본 클래스.
    각 에이전트는 이 클래스를 상속하여 구체적인 기능을 구현해야 합니다.
    """

    def __init__(
        self,
        name: str,
        description: str,
        model: str = OPENAI_MODEL,
        temperature: float = TEMPERATURE,
        verbose: bool = False
    ):
        """
        BaseAgent 클래스 초기화

        Args:
            name (str): 에이전트 이름
            description (str): 에이전트 설명
            model (str, optional): 사용할 LLM 모델. 기본값은 settings.py에서 정의된 OPENAI_MODEL
            temperature (float, optional): LLM 생성 온도. 기본값은 settings.py에서 정의된 TEMPERATURE
            verbose (bool, optional): 상세 로깅 활성화 여부. 기본값은 False
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.model = model
        self.temperature = temperature
        self.verbose = verbose
        self.memory: List[BaseMessage] = []
        self.state: Dict[str, Any] = {}
        
        # LLM 초기화
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            verbose=self.verbose
        )
        
        logger.info(f"에이전트 초기화: {self.name} (ID: {self.id})")

    def reset(self) -> None:
        """
        에이전트 상태를 초기화합니다. 메모리와 상태를 지웁니다.
        """
        self.memory = []
        self.state = {}
        logger.debug(f"에이전트 {self.name} 초기화됨")

    def add_message(self, message: BaseMessage) -> None:
        """
        에이전트 메모리에 메시지를 추가합니다.

        Args:
            message (BaseMessage): 추가할 메시지
        """
        self.memory.append(message)
        if self.verbose:
            logger.debug(f"메시지 추가됨: {message.type}: {message.content[:50]}...")

    def add_human_message(self, content: str) -> None:
        """
        에이전트 메모리에 사용자 메시지를 추가합니다.

        Args:
            content (str): 메시지 내용
        """
        message = HumanMessage(content=content)
        self.add_message(message)
        if self.verbose:
            logger.debug(f"사용자 메시지 추가됨: {content[:50]}...")

    def add_ai_message(self, content: str) -> None:
        """
        에이전트 메모리에 AI 메시지를 추가합니다.

        Args:
            content (str): 메시지 내용
        """
        message = AIMessage(content=content)
        self.add_message(message)
        if self.verbose:
            logger.debug(f"AI 메시지 추가됨: {content[:50]}...")

    def update_state(self, new_state: Dict[str, Any]) -> None:
        """
        에이전트 상태를 업데이트합니다.

        Args:
            new_state (Dict[str, Any]): 새로운 상태 정보
        """
        self.state.update(new_state)
        if self.verbose:
            logger.debug(f"상태 업데이트됨: {new_state}")
            
    def get_state(self) -> Dict[str, Any]:
        """
        현재 에이전트 상태를 반환합니다.

        Returns:
            Dict[str, Any]: 현재 상태
        """
        return self.state

    @abstractmethod
    def run(self, input_data: Any, config: Optional[RunnableConfig] = None) -> T:
        """
        에이전트를 실행하고 결과를 반환합니다.
        모든 하위 클래스는 이 메서드를 구현해야 합니다.

        Args:
            input_data (Any): 에이전트에 제공할 입력 데이터
            config (Optional[RunnableConfig], optional): 실행 구성

        Returns:
            T: 에이전트 출력 타입에 해당하는 결과
        """
        pass

    def __str__(self) -> str:
        """
        에이전트 정보를 문자열로 반환합니다.

        Returns:
            str: 에이전트 정보 문자열
        """
        return f"{self.name} (ID: {self.id}): {self.description}" 
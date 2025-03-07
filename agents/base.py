"""
Base Agent Class
An abstract class that defines the basic functionality for all agents.
"""

import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Callable, TypeVar, Generic

from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from config.settings import (
    TEMPERATURE, OPENAI_MODEL, XAI_MODEL, ANTHROPIC_MODEL,
    USE_OPENAI_API, USE_XAI_API, USE_ANTHROPIC_API
)
from utils.logger import logger
from utils.xai_client import ChatXAI
from utils.anthropic_client import ChatAnthropic


# Define agent output type
T = TypeVar('T')


class BaseAgent(Generic[T], ABC):
    """
    Abstract base class defining the basic functionality for all agents.
    Each agent must inherit from this class and implement its specific functionality.
    """

    def __init__(
        self,
        name: str,
        description: str,
        verbose: bool = False
    ):
        """
        Initialize the BaseAgent class

        Args:
            name (str): Agent name
            description (str): Agent description
            verbose (bool, optional): Enable detailed logging. Defaults to False
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.verbose = verbose
        self.memory: List[BaseMessage] = []
        self.state: Dict[str, Any] = {}
        
        # Initialize LLM based on selected API
        if USE_OPENAI_API:
            logger.info(f"Using OpenAI API with model: {OPENAI_MODEL}")
            self.llm = ChatOpenAI(
                model=OPENAI_MODEL,
                temperature=TEMPERATURE,
                verbose=self.verbose
            )
        elif USE_XAI_API:
            logger.info(f"Using xAI API with model: {XAI_MODEL}")
            self.llm = ChatXAI(
                model=XAI_MODEL,
                temperature=TEMPERATURE,
                verbose=self.verbose
            )
        elif USE_ANTHROPIC_API:
            logger.info(f"Using Anthropic API with model: {ANTHROPIC_MODEL}")
            self.llm = ChatAnthropic(
                model=ANTHROPIC_MODEL,
                temperature=TEMPERATURE,
                verbose=self.verbose
            )
        else:
            logger.warning("No API selected. Using Anthropic API as default.")
            self.llm = ChatAnthropic(
                model=ANTHROPIC_MODEL,
                temperature=TEMPERATURE,
                verbose=self.verbose
            )
        
        logger.info(f"Agent initialized: {self.name} (ID: {self.id})")

    def reset(self) -> None:
        """
        Reset agent state. Clears memory and state.
        """
        self.memory = []
        self.state = {}
        logger.debug(f"Agent {self.name} reset")

    def add_message(self, message: BaseMessage) -> None:
        """
        Add a message to the agent's memory.

        Args:
            message (BaseMessage): Message to add
        """
        self.memory.append(message)
        if self.verbose:
            logger.debug(f"Message added: {message.type}: {message.content[:50]}...")

    def add_human_message(self, content: str) -> None:
        """
        Add a human message to the agent's memory.

        Args:
            content (str): Message content
        """
        message = HumanMessage(content=content)
        self.add_message(message)
        if self.verbose:
            logger.debug(f"Human message added: {content[:50]}...")

    def add_ai_message(self, content: str) -> None:
        """
        Add an AI message to the agent's memory.

        Args:
            content (str): Message content
        """
        message = AIMessage(content=content)
        self.add_message(message)
        if self.verbose:
            logger.debug(f"AI message added: {content[:50]}...")

    def update_state(self, new_state: Dict[str, Any]) -> None:
        """
        Update the agent's state.

        Args:
            new_state (Dict[str, Any]): New state information
        """
        self.state.update(new_state)
        if self.verbose:
            logger.debug(f"State updated: {new_state}")
            
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current agent state.

        Returns:
            Dict[str, Any]: Current state
        """
        return self.state

    @abstractmethod
    def run(self, input_data: Any, config: Optional[RunnableConfig] = None) -> T:
        """
        Run the agent and return results.
        All subclasses must implement this method.

        Args:
            input_data (Any): Input data to provide to the agent
            config (Optional[RunnableConfig], optional): Run configuration

        Returns:
            T: Result corresponding to the agent's output type
        """
        pass

    def __str__(self) -> str:
        """
        Return agent information as a string.

        Returns:
            str: Agent information string
        """
        return f"{self.name} (ID: {self.id}): {self.description}"
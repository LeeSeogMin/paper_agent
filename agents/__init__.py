"""
Agents package initialization.

This package contains various agent implementations for the paper writing AI system.
Each agent specializes in a specific aspect of the paper writing process, such as
research, writing, editing, and coordination.
"""

from agents.base import BaseAgent

# 순환 참조를 피하기 위해 lazy imports 사용
def get_coordinator_agent():
    from agents.coordinator_agent import CoordinatorAgent
    return CoordinatorAgent

def get_research_agent():
    from agents.research_agent import ResearchAgent
    return ResearchAgent

def get_writer_agent():
    from agents.writing_agent import WriterAgent
    return WriterAgent

def get_editor_agent():
    from agents.editing_agent import EditorAgent
    return EditorAgent

def get_review_agent():
    from agents.review_agent import ReviewAgent
    return ReviewAgent

def get_user_interaction_agent():
    from agents.user_interaction_agent import UserInteractionAgent
    return UserInteractionAgent

def get_data_processing_agent():
    from agents.data_processing_agent import DataProcessingAgent
    return DataProcessingAgent

# 외부에서 사용할 클래스들을 __all__에 정의
__all__ = [
    'BaseAgent',
    'get_coordinator_agent',
    'get_research_agent',
    'get_writer_agent',
    'get_editor_agent',
    'get_review_agent',
    'get_user_interaction_agent',
    'get_data_processing_agent'
]
"""
Agent module initialization
Provides consistent imports for all agent classes
"""

from agents.base import BaseAgent
from agents.research_agent import ResearchAgent
from agents.writing_agent import WriterAgent
from agents.editing_agent import EditorAgent
from agents.data_processing_agent import DataProcessingAgent
from agents.user_interaction_agent import UserInteractionAgent
from agents.review_agent import ReviewAgent
from agents.coordinator_agent import CoordinatorAgent

__all__ = [
    'BaseAgent',
    'ResearchAgent',
    'WriterAgent',
    'EditorAgent',
    'DataProcessingAgent',
    'UserInteractionAgent',
    'ReviewAgent',
    'CoordinatorAgent'
]
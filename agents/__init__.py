"""
Agent modules initialization
"""

# Import agents
from agents.base import BaseAgent
from agents.research_agent import ResearchAgent
from agents.writing_agent import WriterAgent
from agents.editing_agent import EditorAgent
from agents.coordinator_agent import CoordinatorAgent

__all__ = [
    'BaseAgent',
    'ResearchAgent',
    'WriterAgent',
    'EditorAgent',
    'CoordinatorAgent'
]
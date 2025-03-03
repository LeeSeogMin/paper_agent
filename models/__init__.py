"""
Models package initialization.

This package contains data models and state representations used throughout the paper writing AI agent system.
It includes models for representing paper structure, workflow state, and other data structures
needed for the paper writing process.
"""

from models.paper import (
    Paper,
    PaperSection,
    PaperMetadata,
    Reference,
    Citation,
    Figure,
    Table
)

from models.state import (
    WorkflowState,
    ResearchState,
    WritingState,
    EditingState
)

__all__ = [
    # Paper models
    "Paper",
    "PaperSection",
    "PaperMetadata",
    "Reference",
    "Citation",
    "Figure",
    "Table",
    
    # State models
    "WorkflowState",
    "ResearchState",
    "WritingState",
    "EditingState"
]

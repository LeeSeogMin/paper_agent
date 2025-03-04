"""
Research models module.

This module contains data models for representing research information,
including search queries, research materials, and analysis results.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class Author(BaseModel):
    """Model representing an author of a research paper"""
    name: str
    affiliation: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "John Smith",
                "affiliation": "University of Research"
            }
        }


class SearchQuery(BaseModel):
    """Model representing a search query for research"""
    id: str
    text: str
    rationale: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "q1",
                "text": "graph neural networks survey",
                "rationale": "To find overview papers on GNNs",
                "timestamp": "2023-06-01T00:00:00"
            }
        }


class ResearchMaterial(BaseModel):
    """Model representing a research material such as a paper"""
    id: str
    title: str
    authors: List[Any]  # Can be list of strings or Author objects
    year: Optional[int] = None
    abstract: str = ""
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    relevance_score: float = 0.0
    evaluation: Optional[str] = None
    query_id: Optional[str] = None
    content: str = ""
    summary: str = ""
    citation_count: Optional[int] = None
    venue: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "paper_1",
                "title": "A Survey of Graph Neural Networks",
                "authors": ["John Smith", "Jane Doe"],
                "year": 2020,
                "abstract": "This paper provides a comprehensive survey...",
                "url": "https://example.org/papers/gnn-survey",
                "pdf_url": "https://example.org/papers/gnn-survey.pdf",
                "relevance_score": 0.92,
                "evaluation": "Highly relevant survey paper on GNNs",
                "query_id": "q1",
                "citation_count": 450,
                "venue": "Journal of AI Research"
            }
        }


class ResearchAnalysis(BaseModel):
    """Model representing analysis of research materials"""
    topic: str
    key_findings: List[str] = []
    themes: List[str] = []
    gaps: List[str] = []
    methodologies: List[str] = []
    recommendations: List[str] = []
    materials: List[str] = []  # IDs of research materials used
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "topic": "Graph Neural Networks for Topic Modeling",
                "key_findings": [
                    "GNNs can effectively model document relationships",
                    "Attention mechanisms improve performance in topic extraction"
                ],
                "themes": [
                    "Graph-based text representation",
                    "Neural topic modeling"
                ],
                "gaps": [
                    "Limited research on scalability for large document collections"
                ],
                "methodologies": [
                    "Graph construction from document similarity",
                    "Message passing for information propagation"
                ],
                "recommendations": [
                    "Explore hierarchical graph structures for multi-level topic modeling"
                ],
                "materials": ["paper_1", "paper_2", "paper_3"],
                "timestamp": "2023-06-10T00:00:00"
            }
        }


class ResearchSummary(BaseModel):
    """Model representing a summary of research findings"""
    topic: str
    key_findings: List[str] = Field(
        default_factory=list,
        description="주요 연구 결과 목록"
    )
    collected_materials: List[ResearchMaterial] = Field(
        default_factory=list,
        description="수집된 연구 자료 목록"
    )
    gaps: Optional[List[str]] = Field(
        default=None,
        description="식별된 연구 격차"
    )
    next_steps: Optional[List[str]] = Field(
        default=None,
        description="권장되는 다음 연구 단계"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="연구 요약 생성 시간"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "topic": "AI in Academic Writing",
                "key_findings": [
                    "AI tools significantly improve writing efficiency",
                    "Language models can assist in literature review"
                ],
                "collected_materials": [
                    {
                        "id": "paper1",
                        "title": "AI Writing Assistants",
                        "authors": ["John Smith"],
                        "summary": "Overview of AI writing tools..."
                    }
                ],
                "gaps": [
                    "Limited research on AI's impact on academic style",
                    "Need for better evaluation metrics"
                ],
                "next_steps": [
                    "Conduct comparative analysis of AI writing tools",
                    "Develop academic style guidelines for AI assistance"
                ],
                "timestamp": "2023-06-01T00:00:00"
            }
        }
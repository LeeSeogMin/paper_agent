"""
Research models module.

This module contains data models for representing research information,
including search queries, research materials, and analysis results.
"""

import re
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pydantic import Field, validator

from models.base import VersionedModel, I18nModel


class Author(VersionedModel):
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


class SearchQuery(VersionedModel):
    """Model representing a search query for research"""
    id: str
    text: str
    rationale: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "q1",
                "text": "graph neural networks survey",
                "rationale": "To find overview papers on GNNs"
            }
        }


class ResearchMaterial(I18nModel):
    """Model representing a research material such as a paper"""
    id: str
    title: str
    authors: List[Union[str, Author]] = Field(
        default_factory=list,
        description="List of authors (can be strings or Author objects)"
    )
    year: Optional[int] = None
    abstract: str = Field(
        default="",
        description="Abstract of the paper (empty if not available)"
    )
    url: Optional[str] = Field(
        default=None,
        description="URL to the paper"
    )
    pdf_url: Optional[str] = Field(
        default=None,
        description="URL to the PDF version of the paper"
    )
    pdf_path: Optional[str] = Field(
        default=None,
        description="Local path to the PDF file"
    )
    relevance_score: float = Field(
        default=0.0,
        description="Relevance score (0.0-1.0) for the research topic"
    )
    evaluation: Optional[str] = Field(
        default=None,
        description="Evaluation notes about the material"
    )
    query_id: Optional[str] = Field(
        default=None,
        description="ID of the query that found this material"
    )
    content: Optional[str] = Field(
        default=None,
        description="Full content of the material (None if not extracted yet)"
    )
    summary: Optional[str] = Field(
        default=None,
        description="Summary of the material (None if not summarized yet)"
    )
    citation_count: Optional[int] = Field(
        default=None,
        description="Number of citations"
    )
    venue: Optional[str] = Field(
        default=None,
        description="Publication venue"
    )
    source: str = Field(
        default="unknown",
        description="Source of the material (e.g., 'Semantic Scholar', 'Google Scholar')"
    )
    related_documents: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description="Related documents from vector database"
    )
    
    @validator('authors', pre=True)
    def validate_authors(cls, v):
        """Convert string authors to Author objects if needed"""
        if isinstance(v, list):
            result = []
            for author in v:
                if isinstance(author, str):
                    result.append(author)
                elif isinstance(author, dict):
                    result.append(Author(**author))
                else:
                    result.append(author)
            return result
        return v
    
    def has_content(self) -> bool:
        """Check if the material has content extracted"""
        return self.content is not None and len(self.content) > 0
    
    def has_summary(self) -> bool:
        """Check if the material has been summarized"""
        return self.summary is not None and len(self.summary) > 0
    
    def get_citation(self) -> str:
        """Generate a citation string for this material"""
        author_text = ""
        if self.authors:
            if isinstance(self.authors[0], str):
                author_text = ", ".join(self.authors)
            else:
                author_text = ", ".join([a.name for a in self.authors])
        
        year_text = f" ({self.year})" if self.year else ""
        title_text = f". {self.title}" if self.title else ""
        venue_text = f". {self.venue}" if self.venue else ""
        
        return f"{author_text}{year_text}{title_text}{venue_text}"
    
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
                "venue": "Journal of AI Research",
                "source": "Semantic Scholar"
            }
        }


class ResearchAnalysis(VersionedModel):
    """Model representing analysis of research materials"""
    topic: str
    key_findings: List[str] = Field(default_factory=list)
    themes: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    methodologies: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    materials: List[str] = Field(default_factory=list)  # IDs of research materials used
    
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
                "materials": ["paper_1", "paper_2", "paper_3"]
            }
        }


class ResearchSummary(I18nModel):
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
        default_factory=list,
        description="식별된 연구 격차"
    )
    next_steps: Optional[List[str]] = Field(
        default_factory=list,
        description="권장되는 다음 연구 단계"
    )
    
    def get_material_by_id(self, material_id: str) -> Optional[ResearchMaterial]:
        """Get a research material by ID"""
        for material in self.collected_materials:
            if material.id == material_id:
                return material
        return None
    
    def get_top_materials(self, count: int = 5) -> List[ResearchMaterial]:
        """Get top materials by relevance score"""
        sorted_materials = sorted(
            self.collected_materials, 
            key=lambda m: m.relevance_score, 
            reverse=True
        )
        return sorted_materials[:count]
    
    def add_material(self, material: ResearchMaterial) -> 'ResearchSummary':
        """Add a research material to the collection"""
        # Check if material with same ID already exists
        for i, existing in enumerate(self.collected_materials):
            if existing.id == material.id:
                # Replace existing material
                self.collected_materials[i] = material
                return self
        
        # Add new material
        self.collected_materials.append(material)
        return self
    
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
                ]
            }
        }
"""
Paper model module
Data models that define paper structure.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class Citation(BaseModel):
    """Model representing citation information in a paper"""
    text: str
    reference_id: str
    page: Optional[int] = None
    context: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "text": "According to Smith (2020), AI systems...",
                "reference_id": "smith2020advances",
                "page": 42,
                "context": "Discussing the limitations of current AI systems"
            }
        }

class Reference(BaseModel):
    """Model representing reference information"""
    id: str
    title: str
    authors: List[str]
    year: int
    venue: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    abstract: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "id": "smith2020advances",
                "title": "Advances in AI Research",
                "authors": ["John Smith", "Jane Doe"],
                "year": 2020,
                "venue": "Journal of AI Research",
                "doi": "10.1234/jair.2020.123",
                "url": "https://example.org/papers/smith2020advances",
                "abstract": "This paper discusses recent advances in AI research..."
            }
        }

class Figure(BaseModel):
    """Model representing figure information in a paper"""
    figure_id: str
    caption: str
    description: str
    path: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "figure_id": "fig1",
                "caption": "Figure 1: System Architecture",
                "description": "Overview of the proposed system architecture showing components and connections",
                "path": "figures/system_architecture.png",
                "data": {"type": "architecture", "components": ["UI", "API", "Database"]}
            }
        }

class Table(BaseModel):
    """Model representing table information in a paper"""
    table_id: str
    caption: str
    headers: List[str]
    rows: List[List[Any]]
    notes: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "table_id": "tab1",
                "caption": "Table 1: Performance Results",
                "headers": ["Model", "Accuracy", "F1 Score"],
                "rows": [
                    ["Model A", 0.92, 0.90],
                    ["Model B", 0.94, 0.93],
                    ["Model C", 0.91, 0.89]
                ],
                "notes": "All models trained on the same dataset with 5-fold cross validation"
            }
        }

class PaperSection(BaseModel):
    """Model representing a paper section"""
    section_id: str
    title: str
    content: str
    subsections: Optional[List['PaperSection']] = None
    figures: Optional[List[Figure]] = None
    tables: Optional[List[Table]] = None
    citations: Optional[List[Citation]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "section_id": "introduction",
                "title": "Introduction",
                "content": "This paper presents a novel approach to...",
                "subsections": [],
                "figures": [],
                "tables": [],
                "citations": []
            }
        }

class PaperMetadata(BaseModel):
    """Model representing paper metadata"""
    title: str
    authors: List[str]
    abstract: str
    keywords: List[str]
    date: datetime = Field(default_factory=datetime.now)
    journal: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    doi: Optional[str] = None
    template: str = "default"
    language: str = "english"
    
    class Config:
        schema_extra = {
            "example": {
                "title": "Novel Approaches to Academic Paper Writing with AI",
                "authors": ["John Smith", "Jane Doe"],
                "abstract": "This paper discusses innovative methods...",
                "keywords": ["AI", "academic writing", "automation"],
                "date": "2023-06-01T00:00:00",
                "journal": "Journal of AI Research",
                "volume": "14",
                "issue": "2",
                "doi": "10.1234/jair.2023.123",
                "template": "ieee",
                "language": "english"
            }
        }

class Paper(BaseModel):
    """Complete paper model"""
    metadata: PaperMetadata
    sections: List[PaperSection]
    references: List[Reference]
    acknowledgements: Optional[str] = None
    appendices: Optional[List[PaperSection]] = None
    version: str = "1.0"
    status: str = "draft"
    last_updated: datetime = Field(default_factory=datetime.now)
    
    class Config:
        schema_extra = {
            "example": {
                "metadata": {
                    "title": "Novel Approaches to Academic Paper Writing with AI",
                    "authors": ["John Smith", "Jane Doe"],
                    "abstract": "This paper discusses innovative methods...",
                    "keywords": ["AI", "academic writing", "automation"],
                    "date": "2023-06-01T00:00:00",
                    "template": "ieee",
                    "language": "english"
                },
                "sections": [
                    {
                        "section_id": "introduction",
                        "title": "Introduction",
                        "content": "This paper presents a novel approach to..."
                    }
                ],
                "references": [
                    {
                        "id": "smith2020advances",
                        "title": "Advances in AI Research",
                        "authors": ["John Smith", "Jane Doe"],
                        "year": 2020
                    }
                ],
                "version": "1.0",
                "status": "draft"
            }
        }

class PaperOutline(BaseModel):
    """Model representing the outline/structure of a paper"""
    title: str
    sections: List[Dict[str, Any]] = Field(
        description="List of sections with their titles and brief descriptions"
    )
    target_length: Optional[int] = Field(
        default=None,
        description="Target length of the paper in words"
    )
    key_points: Optional[List[str]] = Field(
        default=None,
        description="Key points to be covered in the paper"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "title": "Novel Approaches to Academic Paper Writing with AI",
                "sections": [
                    {
                        "title": "Introduction",
                        "description": "Background and motivation of the study"
                    },
                    {
                        "title": "Literature Review",
                        "description": "Review of existing approaches"
                    }
                ],
                "target_length": 5000,
                "key_points": [
                    "Current challenges in academic writing",
                    "AI-based solutions",
                    "Future directions"
                ]
            }
        }
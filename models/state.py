"""
State models for the paper writing workflow.

This module defines the state objects used to track the progress and state of the research, writing, and review process.
"""

from typing import Dict, List, Optional, Any, Union, ClassVar, TypeVar, Generic
from datetime import datetime
from pydantic import BaseModel, Field, validator

from models.base import StateTransitionModel
from models.paper import Paper, PaperOutline
from models.research import ResearchMaterial, ResearchSummary

class ResearchState(StateTransitionModel['ResearchState']):
    """Model representing the state of the research phase"""
    topic: str
    query_history: List[str] = Field(default_factory=list)
    research_notes: Dict[str, Any] = Field(default_factory=dict)
    sources_collected: List[Dict[str, Any]] = Field(default_factory=list)
    key_findings: List[str] = Field(default_factory=list)
    outline: Optional[Dict[str, Any]] = None
    
    # Status constants
    INITIALIZED: ClassVar[str] = "initialized"
    IN_PROGRESS: ClassVar[str] = "in_progress"
    COLLECTING: ClassVar[str] = "collecting_sources"
    ANALYZING: ClassVar[str] = "analyzing_sources"
    COMPLETED: ClassVar[str] = "completed"
    FAILED: ClassVar[str] = "failed"
    
    def start(self) -> 'ResearchState':
        """Start the research phase"""
        self.validate_transition(self.status, [self.INITIALIZED])
        return self.transition(self.IN_PROGRESS)
    
    def start_collecting(self) -> 'ResearchState':
        """Start collecting sources"""
        self.validate_transition(self.status, [self.INITIALIZED, self.IN_PROGRESS])
        return self.transition(self.COLLECTING)
    
    def add_source(self, source: Dict[str, Any]) -> 'ResearchState':
        """Add a collected source"""
        self.sources_collected.append(source)
        return self
    
    def start_analyzing(self) -> 'ResearchState':
        """Start analyzing sources"""
        self.validate_transition(self.status, [self.COLLECTING, self.IN_PROGRESS])
        return self.transition(self.ANALYZING)
    
    def add_finding(self, finding: str) -> 'ResearchState':
        """Add a key finding"""
        self.key_findings.append(finding)
        return self
    
    def complete(self, outline: Optional[Dict[str, Any]] = None) -> 'ResearchState':
        """Complete the research phase"""
        if outline:
            self.outline = outline
        return self.transition(self.COMPLETED)
    
    def fail(self, error: str) -> 'ResearchState':
        """Mark the research phase as failed"""
        return self.transition(self.FAILED, error)

class WritingState(StateTransitionModel['WritingState']):
    """Model representing the state of the writing phase"""
    outline: Dict[str, Any]
    current_section: Optional[str] = None
    completed_sections: List[str] = Field(default_factory=list)
    draft_content: Dict[str, str] = Field(default_factory=dict)
    version: int = 1
    
    # Status constants
    INITIALIZED: ClassVar[str] = "initialized"
    IN_PROGRESS: ClassVar[str] = "in_progress"
    SECTION_WRITING: ClassVar[str] = "writing_section"
    REVIEWING: ClassVar[str] = "reviewing"
    COMPLETED: ClassVar[str] = "completed"
    FAILED: ClassVar[str] = "failed"
    
    def start(self) -> 'WritingState':
        """Start the writing phase"""
        self.validate_transition(self.status, [self.INITIALIZED])
        return self.transition(self.IN_PROGRESS)
    
    def start_section(self, section_id: str) -> 'WritingState':
        """Start writing a section"""
        self.validate_transition(self.status, [self.INITIALIZED, self.IN_PROGRESS])
        self.current_section = section_id
        return self.transition(self.SECTION_WRITING)
    
    def complete_section(self, section_id: str, content: str) -> 'WritingState':
        """Complete a section"""
        if self.current_section != section_id:
            raise ValueError(f"Current section is {self.current_section}, not {section_id}")
        
        self.draft_content[section_id] = content
        self.completed_sections.append(section_id)
        self.current_section = None
        return self.transition(self.IN_PROGRESS)
    
    def start_review(self) -> 'WritingState':
        """Start reviewing the draft"""
        self.validate_transition(self.status, [self.IN_PROGRESS])
        return self.transition(self.REVIEWING)
    
    def complete(self) -> 'WritingState':
        """Complete the writing phase"""
        self.validate_transition(self.status, [self.IN_PROGRESS, self.REVIEWING])
        return self.transition(self.COMPLETED)
    
    def fail(self, error: str) -> 'WritingState':
        """Mark the writing phase as failed"""
        return self.transition(self.FAILED, error)

class EditingState(StateTransitionModel['EditingState']):
    """Model representing the state of the editing phase"""
    draft_paper: Paper
    edits_made: List[Dict[str, Any]] = Field(default_factory=list)
    style_guide: str
    citation_style: str
    format: str
    
    # Status constants
    INITIALIZED: ClassVar[str] = "initialized"
    IN_PROGRESS: ClassVar[str] = "in_progress"
    GRAMMAR_CHECK: ClassVar[str] = "grammar_check"
    STYLE_CHECK: ClassVar[str] = "style_check"
    CITATION_CHECK: ClassVar[str] = "citation_check"
    FORMATTING: ClassVar[str] = "formatting"
    COMPLETED: ClassVar[str] = "completed"
    FAILED: ClassVar[str] = "failed"
    
    def start(self) -> 'EditingState':
        """Start the editing phase"""
        self.validate_transition(self.status, [self.INITIALIZED])
        return self.transition(self.IN_PROGRESS)
    
    def start_grammar_check(self) -> 'EditingState':
        """Start grammar check"""
        self.validate_transition(self.status, [self.INITIALIZED, self.IN_PROGRESS])
        return self.transition(self.GRAMMAR_CHECK)
    
    def start_style_check(self) -> 'EditingState':
        """Start style check"""
        self.validate_transition(self.status, [self.GRAMMAR_CHECK, self.IN_PROGRESS])
        return self.transition(self.STYLE_CHECK)
    
    def start_citation_check(self) -> 'EditingState':
        """Start citation check"""
        self.validate_transition(self.status, [self.STYLE_CHECK, self.IN_PROGRESS])
        return self.transition(self.CITATION_CHECK)
    
    def start_formatting(self) -> 'EditingState':
        """Start formatting"""
        self.validate_transition(self.status, [self.CITATION_CHECK, self.IN_PROGRESS])
        return self.transition(self.FORMATTING)
    
    def add_edit(self, edit_type: str, section: str, description: str) -> 'EditingState':
        """Add an edit"""
        self.edits_made.append({
            "type": edit_type,
            "section": section,
            "description": description,
            "timestamp": datetime.now()
        })
        return self
    
    def complete(self) -> 'EditingState':
        """Complete the editing phase"""
        return self.transition(self.COMPLETED)
    
    def fail(self, error: str) -> 'EditingState':
        """Mark the editing phase as failed"""
        return self.transition(self.FAILED, error)

class ReviewState(StateTransitionModel['ReviewState']):
    """Model representing the state of the review phase"""
    paper: Paper
    feedback: List[Dict[str, Any]] = Field(default_factory=list)
    revision_needed: bool = False
    
    # Status constants
    INITIALIZED: ClassVar[str] = "initialized"
    IN_PROGRESS: ClassVar[str] = "in_progress"
    COMPLETED: ClassVar[str] = "completed"
    FAILED: ClassVar[str] = "failed"
    
    def start(self) -> 'ReviewState':
        """Start the review phase"""
        self.validate_transition(self.status, [self.INITIALIZED])
        return self.transition(self.IN_PROGRESS)
    
    def add_feedback(self, section: str, comment: str, severity: str) -> 'ReviewState':
        """Add review feedback"""
        self.feedback.append({
            "section": section,
            "comment": comment,
            "severity": severity,
            "timestamp": datetime.now()
        })
        
        # If any critical feedback, mark as needing revision
        if severity == "critical":
            self.revision_needed = True
            
        return self
    
    def complete(self) -> 'ReviewState':
        """Complete the review phase"""
        self.validate_transition(self.status, [self.IN_PROGRESS])
        return self.transition(self.COMPLETED)
    
    def fail(self, error: str) -> 'ReviewState':
        """Mark the review phase as failed"""
        return self.transition(self.FAILED, error)

class WorkflowState(StateTransitionModel['WorkflowState']):
    """Model representing the overall workflow state"""
    topic: str
    template_name: str
    style_guide: str
    citation_style: str
    output_format: str
    research_state: Optional[ResearchState] = None
    writing_state: Optional[WritingState] = None
    editing_state: Optional[EditingState] = None
    review_state: Optional[ReviewState] = None
    paper: Optional[Paper] = None
    output_file: Optional[str] = None
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    current_stage: str = "initialized"
    
    # Stage constants
    STAGE_INITIALIZED: ClassVar[str] = "initialized"
    STAGE_RESEARCH: ClassVar[str] = "research"
    STAGE_WRITING: ClassVar[str] = "writing"
    STAGE_EDITING: ClassVar[str] = "editing"
    STAGE_REVIEW: ClassVar[str] = "review"
    STAGE_COMPLETED: ClassVar[str] = "completed"
    STAGE_FAILED: ClassVar[str] = "failed"
    
    def start_research(self) -> 'WorkflowState':
        """Start the research stage"""
        self.validate_transition(self.current_stage, [self.STAGE_INITIALIZED])
        self.research_state = ResearchState(topic=self.topic)
        self.research_state.start()
        self.current_stage = self.STAGE_RESEARCH
        return self
    
    def complete_research(self, research_summary: ResearchSummary) -> 'WorkflowState':
        """Complete the research stage and prepare for writing"""
        self.validate_transition(self.current_stage, [self.STAGE_RESEARCH])
        
        if not self.research_state:
            raise ValueError("Research state not initialized")
        
        # Update research state
        self.research_state.complete()
        
        # Create outline from research summary
        outline = {
            "title": f"Paper on {self.topic}",
            "sections": [
                {"title": "Introduction", "content": ""},
                {"title": "Literature Review", "content": ""},
                {"title": "Methodology", "content": ""},
                {"title": "Results", "content": ""},
                {"title": "Discussion", "content": ""},
                {"title": "Conclusion", "content": ""}
            ],
            "key_points": research_summary.key_findings
        }
        
        # Initialize writing state
        self.writing_state = WritingState(outline=outline)
        self.current_stage = self.STAGE_WRITING
        return self
    
    def start_writing(self) -> 'WorkflowState':
        """Start the writing stage"""
        self.validate_transition(self.current_stage, [self.STAGE_RESEARCH])
        
        if not self.writing_state:
            if not self.research_state or not self.research_state.outline:
                raise ValueError("Research not completed, no outline available")
            
            self.writing_state = WritingState(outline=self.research_state.outline)
        
        self.writing_state.start()
        self.current_stage = self.STAGE_WRITING
        return self
    
    def complete_writing(self, paper: Paper) -> 'WorkflowState':
        """Complete the writing stage and prepare for editing"""
        self.validate_transition(self.current_stage, [self.STAGE_WRITING])
        
        if not self.writing_state:
            raise ValueError("Writing state not initialized")
        
        # Update writing state
        self.writing_state.complete()
        
        # Store the paper
        self.paper = paper
        
        # Initialize editing state
        self.editing_state = EditingState(
            draft_paper=paper,
            style_guide=self.style_guide,
            citation_style=self.citation_style,
            format=self.output_format
        )
        
        self.current_stage = self.STAGE_EDITING
        return self
    
    def start_editing(self) -> 'WorkflowState':
        """Start the editing stage"""
        self.validate_transition(self.current_stage, [self.STAGE_WRITING])
        
        if not self.paper:
            raise ValueError("No paper available for editing")
        
        if not self.editing_state:
            self.editing_state = EditingState(
                draft_paper=self.paper,
                style_guide=self.style_guide,
                citation_style=self.citation_style,
                format=self.output_format
            )
        
        self.editing_state.start()
        self.current_stage = self.STAGE_EDITING
        return self
    
    def complete_editing(self, edited_paper: Paper) -> 'WorkflowState':
        """Complete the editing stage and prepare for review"""
        self.validate_transition(self.current_stage, [self.STAGE_EDITING])
        
        if not self.editing_state:
            raise ValueError("Editing state not initialized")
        
        # Update editing state
        self.editing_state.complete()
        
        # Update the paper
        self.paper = edited_paper
        
        # Initialize review state
        self.review_state = ReviewState(paper=edited_paper)
        
        self.current_stage = self.STAGE_REVIEW
        return self
    
    def start_review(self) -> 'WorkflowState':
        """Start the review stage"""
        self.validate_transition(self.current_stage, [self.STAGE_EDITING])
        
        if not self.paper:
            raise ValueError("No paper available for review")
        
        if not self.review_state:
            self.review_state = ReviewState(paper=self.paper)
        
        self.review_state.start()
        self.current_stage = self.STAGE_REVIEW
        return self
    
    def complete_review(self, needs_revision: bool = False) -> 'WorkflowState':
        """Complete the review stage"""
        self.validate_transition(self.current_stage, [self.STAGE_REVIEW])
        
        if not self.review_state:
            raise ValueError("Review state not initialized")
        
        # Update review state
        self.review_state.complete()
        
        if needs_revision:
            # Go back to editing
            self.current_stage = self.STAGE_EDITING
        else:
            # Complete the workflow
            self.current_stage = self.STAGE_COMPLETED
            self.end_time = datetime.now()
        
        return self
    
    def complete_workflow(self, output_file: Optional[str] = None) -> 'WorkflowState':
        """Complete the entire workflow"""
        self.validate_transition(self.current_stage, [self.STAGE_REVIEW, self.STAGE_EDITING])
        
        if output_file:
            self.output_file = output_file
        
        self.current_stage = self.STAGE_COMPLETED
        self.end_time = datetime.now()
        return self
    
    def fail_workflow(self, error: str) -> 'WorkflowState':
        """Mark the workflow as failed"""
        self.error = error
        self.current_stage = self.STAGE_FAILED
        self.end_time = datetime.now()
        return self
    
    @validator('current_stage')
    def validate_stage(cls, v):
        """Validate that the current stage is one of the defined stages"""
        valid_stages = [
            cls.STAGE_INITIALIZED, cls.STAGE_RESEARCH, cls.STAGE_WRITING,
            cls.STAGE_EDITING, cls.STAGE_REVIEW, cls.STAGE_COMPLETED,
            cls.STAGE_FAILED
        ]
        if v not in valid_stages:
            raise ValueError(f"Invalid stage: {v}. Must be one of {valid_stages}")
        return v

# For backward compatibility
PaperWorkflowState = WorkflowState
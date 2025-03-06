"""
Academic Paper Editing Agent Module
Handles proofreading, style enforcement, and formatting of research papers.
"""

import os
import re
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from pydantic import BaseModel, Field, validator

from langchain_core.runnables import RunnableConfig, RunnableSequence
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

from config.settings import OUTPUT_DIR
from utils.logger import logger
from models.paper import Paper, PaperSection, Reference
from prompts.paper_prompts import (
    PAPER_EDITING_PROMPT,
    PAPER_REVIEW_PROMPT,
    REFERENCE_FORMATTING_PROMPT
)
from agents.base import BaseAgent


class EditingTask(BaseModel):
    """Editing task definition"""
    paper_id: str = Field(description="ID of the paper to edit")
    style_guide: Optional[str] = Field(default="Standard Academic", description="Style guide to apply")
    citation_style: Optional[str] = Field(default="APA", description="Citation style")
    focus_areas: Optional[List[str]] = Field(default_factory=list, description="Focus areas for editing")
    output_format: Optional[str] = Field(default="markdown", description="Output format")
    paper_format: Optional[str] = Field(default="standard", description="Paper format (standard, literature_review, custom)")
    
    @validator('paper_id')
    def paper_id_must_not_be_empty(cls, v):
        if not v:
            raise ValueError('paper_id is required')
        return v
    
    @validator('style_guide', 'citation_style', 'output_format', 'paper_format')
    def provide_defaults_for_empty_strings(cls, v, values, **kwargs):
        if not v:
            field_name = kwargs['field'].name
            if field_name == 'style_guide':
                return "Standard Academic"
            elif field_name == 'citation_style':
                return "APA"
            elif field_name == 'output_format':
                return "markdown"
            elif field_name == 'paper_format':
                return "standard"
        return v


class StyleGuide(BaseModel):
    """Style guide specification format"""
    name: str = Field(description="Name of the style guide")
    rules: List[str] = Field(description="List of style rules")
    examples: Dict[str, str] = Field(description="Example correct/incorrect usages")


class ReviewResult(BaseModel):
    """Paper review result format"""
    overall_rating: int = Field(description="Overall quality score (1-10)")
    strengths: List[str] = Field(description="Strengths of the paper")
    weaknesses: List[str] = Field(description="Areas needing improvement")
    suggestions: List[str] = Field(description="Specific improvement suggestions")
    grammar_issues: List[str] = Field(description="Identified grammatical issues")
    structure_comments: str = Field(description="Comments on paper structure")


class EditorAgent(BaseAgent[Paper]):
    """Expert academic paper editor agent"""

    def __init__(
        self,
        name: str = "Academic Editor",
        description: str = "Specializes in technical paper editing and review",
        verbose: bool = False
    ):
        """
        Initialize the EditorAgent
        
        Args:
            name (str): Agent identifier
            description (str): Role description
            verbose (bool): Enable debug logging
        """
        super().__init__(name, description, verbose=verbose)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self._init_prompts()
        self.current_paper = None
        logger.info(f"{self.name} initialized")

    def _init_prompts(self) -> None:
        """Initialize prompt templates for editing tasks"""
        # Add English language and citation requirements to editing prompts
        self.editing_prompt_template = PromptTemplate(
            template=PAPER_EDITING_PROMPT.template + "\n\nIMPORTANT REQUIREMENTS:\n1. The paper must be written in English.\n2. All content must be based on the vector database content.\n3. Every claim or statement must include proper citations.\n4. The paper must include a complete bibliography/references section at the end.",
            input_variables=PAPER_EDITING_PROMPT.input_variables
        )
        
        self.review_prompt_template = PromptTemplate(
            template=PAPER_REVIEW_PROMPT.template + "\n\nIMPORTANT: Check that the paper is written in English, includes proper citations for all claims, and has a complete bibliography.",
            input_variables=PAPER_REVIEW_PROMPT.input_variables
        )
        
        self.reference_formatting_prompt_template = PromptTemplate(
            template=REFERENCE_FORMATTING_PROMPT.template + "\n\nEnsure all references are properly formatted according to the specified citation style.",
            input_variables=REFERENCE_FORMATTING_PROMPT.input_variables
        )
        
        # Initialize chains using RunnableSequence
        self.editing_chain = RunnableSequence(
            first=self.editing_prompt_template,
            last=self.llm
        )
        
        self.review_chain = RunnableSequence(
            first=self.review_prompt_template,
            last=self.llm
        )
        
        self.reference_formatting_chain = RunnableSequence(
            first=self.reference_formatting_prompt_template,
            last=self.llm
        )
        
        logger.info("Editor agent prompts initialized with English language and citation requirements")

    def create_style_guide(self, style_name: str = "Standard Academic") -> StyleGuide:
        """
        Create a style guide based on the specified style name
        
        Args:
            style_name: Name of the style guide to create
            
        Returns:
            StyleGuide object
        """
        # Standard Academic style guide with English language requirement
        if style_name.lower() == "standard academic":
            return StyleGuide(
                name="Standard Academic English",
                rules=[
                    "Use formal academic English throughout the paper",
                    "Write in the third person perspective",
                    "Use active voice when possible",
                    "Avoid contractions and colloquialisms",
                    "Use precise and specific language",
                    "Maintain consistent terminology throughout",
                    "All papers and reports must be written in English",
                    "All claims must be supported by citations",
                    "Include a complete bibliography/references section"
                ],
                examples={
                    "formal_language": "Correct: 'The research demonstrates that...' | Incorrect: 'The research shows that...'",
                    "third_person": "Correct: 'The authors argue that...' | Incorrect: 'I believe that...'",
                    "active_voice": "Correct: 'The researchers conducted experiments...' | Incorrect: 'Experiments were conducted...'",
                    "precise_language": "Correct: 'The temperature increased by 5.2 degrees' | Incorrect: 'The temperature went up a lot'",
                    "citation": "Correct: 'This approach has shown promising results (Smith et al., 2020).' | Incorrect: 'This approach has shown promising results.'"
                }
            )
        # Add other style guides as needed
        else:
            # Default to standard academic with a note about the requested style
            return StyleGuide(
                name=f"Modified {style_name}",
                rules=[
                    f"Follow {style_name} conventions where applicable",
                    "Use formal academic English throughout the paper",
                    "All papers and reports must be written in English",
                    "All claims must be supported by citations",
                    "Include a complete bibliography/references section"
                ],
                examples={}
            )

    def edit_paper(self, paper: Paper, style_guide: StyleGuide, paper_format: str = "standard") -> Paper:
        """
        Edit a paper according to the specified style guide
        
        Args:
            paper: Paper to edit
            style_guide: Style guide to apply
            paper_format: Paper format type
            
        Returns:
            Edited paper
        """
        logger.info(f"Editing paper: {paper.metadata.title} with style guide: {style_guide.name}, format: {paper_format}")
        
        # Prepare the editing prompt with style guide and paper content
        editing_input = {
            "paper_content": self._paper_to_text(paper),
            "style_guide": style_guide.name,
            "style_rules": "\n".join([f"- {rule}" for rule in style_guide.rules]),
            "paper_format": paper_format,
            "editing_type": "Full paper editing",
            "language": "English"  # Explicitly specify English
        }
        
        # Run the editing chain
        editing_result = self.editing_chain.invoke(editing_input)
        
        # Parse the edited paper
        edited_paper = self._parse_edited_paper(editing_result, paper)
        
        # Ensure the paper has a references section
        if not any(section.title.lower() in ["references", "bibliography"] for section in edited_paper.sections):
            # If no references section exists, add one
            logger.warning("No references section found in edited paper. Adding one.")
            self._add_references_section(edited_paper)
        
        # Verify all sections are in English
        for section in edited_paper.sections:
            if not self._is_english(section.content):
                logger.warning(f"Section '{section.title}' may not be in English. Translating.")
                section.content = self._translate_to_english(section.content)
        
        logger.info(f"Paper editing completed: {edited_paper.metadata.title}")
        return edited_paper
    
    def _is_english(self, text: str) -> bool:
        """
        Check if text is primarily in English
        
        Args:
            text: Text to check
            
        Returns:
            True if text is primarily in English, False otherwise
        """
        # Simple heuristic: check if common English words are present
        english_words = ["the", "and", "of", "to", "in", "is", "that", "for", "it", "with"]
        text_lower = text.lower()
        
        # Count occurrences of common English words
        english_word_count = sum(1 for word in english_words if f" {word} " in f" {text_lower} ")
        
        # If at least 3 common English words are present, consider it English
        return english_word_count >= 3
    
    def _translate_to_english(self, text: str) -> str:
        """
        Translate text to English
        
        Args:
            text: Text to translate
            
        Returns:
            Translated text
        """
        # Use LLM to translate
        translation_prompt = f"""
        Translate the following text to academic English:
        
        {text}
        
        Translated text:
        """
        
        translation_result = self.llm.invoke(translation_prompt)
        return translation_result
    
    def _add_references_section(self, paper: Paper) -> None:
        """
        Add a references section to a paper
        
        Args:
            paper: Paper to add references section to
        """
        # Create a references section
        references_section = PaperSection(
            section_id="references",
            title="References",
            content=self._format_references_content(paper.references)
        )
        
        # Add to paper
        paper.sections.append(references_section)
    
    def _format_references_content(self, references: List[Reference]) -> str:
        """
        Format references as text
        
        Args:
            references: List of references
            
        Returns:
            Formatted references text
        """
        if not references:
            return "No references."
        
        formatted_refs = []
        for ref in references:
            authors = ", ".join(ref.authors) if ref.authors else "Unknown"
            year = ref.year if ref.year else "n.d."
            title = ref.title if ref.title else "Untitled"
            venue = ref.venue if ref.venue else ""
            
            formatted_ref = f"{authors} ({year}). {title}."
            if venue:
                formatted_ref += f" {venue}."
            if ref.doi:
                formatted_ref += f" DOI: {ref.doi}"
            elif ref.url:
                formatted_ref += f" Retrieved from {ref.url}"
            
            formatted_refs.append(formatted_ref)
        
        return "\n\n".join(formatted_refs)

    def format_references(self, references: List[Reference], citation_style: str = "APA") -> List[str]:
        """
        Format references according to the specified citation style
        
        Args:
            references: List of references to format
            citation_style: Citation style to use
            
        Returns:
            List of formatted reference strings
        """
        logger.info(f"Formatting {len(references)} references in {citation_style} style")
        
        if not references:
            return ["No references available."]
        
        # Prepare the reference formatting prompt
        formatting_input = {
            "references": json.dumps([ref.dict() for ref in references], indent=2),
            "citation_style": citation_style
        }
        
        # Run the reference formatting chain
        formatting_result = self.reference_formatting_chain.invoke(formatting_input)
        
        # Split the result into individual references
        formatted_refs = [ref.strip() for ref in formatting_result.split("\n\n") if ref.strip()]
        
        logger.info(f"Formatted {len(formatted_refs)} references in {citation_style} style")
        return formatted_refs

    def save_formatted_paper(self, paper: Paper, output_format: str = "markdown") -> str:
        """
        Save the formatted paper to a file
        
        Args:
            paper: Paper to format and save
            output_format: Output format (markdown, latex, docx)
            
        Returns:
            Path to the saved file
        """
        logger.info(f"Saving paper in {output_format} format: {paper.metadata.title}")
        
        # Ensure the paper is in English
        if not self._is_english(paper.sections[0].content if paper.sections else ""):
            logger.warning("Paper may not be in English. Converting to English.")
            for section in paper.sections:
                if not self._is_english(section.content):
                    section.content = self._translate_to_english(section.content)
        
        # Ensure the paper has a references section
        if not any(section.title.lower() in ["references", "bibliography"] for section in paper.sections):
            logger.warning("No references section found. Adding one.")
            self._add_references_section(paper)
        
        # Format the paper based on the requested output format
        if output_format.lower() == "markdown":
            formatted_content = self._format_as_markdown(paper)
            file_ext = "md"
        elif output_format.lower() == "latex":
            formatted_content = self._format_as_latex(paper)
            file_ext = "tex"
        elif output_format.lower() == "docx":
            # For docx, we'll save as markdown and note that conversion would happen externally
            formatted_content = self._format_as_markdown(paper)
            file_ext = "md"
            logger.info("Note: DOCX format requires external conversion from markdown")
        else:
            # Default to markdown
            formatted_content = self._format_as_markdown(paper)
            file_ext = "md"
            logger.warning(f"Unknown format '{output_format}'. Defaulting to markdown.")
        
        # Create a filename based on the paper title
        safe_title = re.sub(r'[^\w\s-]', '', paper.metadata.title).strip().lower()
        safe_title = re.sub(r'[-\s]+', '-', safe_title)
        timestamp = self._get_timestamp()
        filename = f"{safe_title}-{timestamp}.{file_ext}"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        # Save the formatted content to the file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(formatted_content)
        
        logger.info(f"Paper saved to: {filepath}")
        return filepath

    def _parse_edited_paper(self, edited_content: str, original_paper: Paper) -> Paper:
        """
        편집된 논문 내용을 파싱하여 Paper 객체로 변환합니다.

        Args:
            edited_content (str): 편집된 논문 내용
            original_paper (Paper): 원본 논문

        Returns:
            Paper: 파싱된 논문 객체
        """
        try:
            # 제목 추출
            title_match = re.search(r"#\s+(.+?)(?:\n|$)", edited_content)
            title = title_match.group(1).strip() if title_match else original_paper.title
            
            # 섹션 추출
            sections = []
            section_pattern = re.compile(r"##\s+(.+?)\n\n([\s\S]+?)(?=\n##|\n#|$)")
            
            for match in section_pattern.finditer(edited_content):
                section_title = match.group(1).strip()
                section_content = match.group(2).strip()
                
                # 참고 문헌 섹션은 건너뜀
                if "References" in section_title:
                    continue
                
                # 기존 섹션에서 인용 정보 가져오기
                citations = []
                for original_section in original_paper.sections:
                    if original_section.title == section_title:
                        citations = original_section.citations
                        break
                
                sections.append(PaperSection(
                    title=section_title,
                    content=section_content,
                    citations=citations
                ))
            
            # 편집된 논문 객체 생성
            edited_paper = Paper(
                title=title,
                topic=original_paper.topic,
                sections=sections,
                references=original_paper.references,
                template_name=original_paper.template_name,
                edit_history=original_paper.edit_history.copy()
            )
            
            return edited_paper
            
        except Exception as e:
            logger.error(f"편집된 논문 파싱 중 오류 발생: {str(e)}")
            return original_paper

    def _get_timestamp(self) -> str:
        """현재 타임스탬프를 반환합니다."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def run(
        self,
        paper: Paper,
        style_guide_name: str = "Standard Academic",
        output_format: str = "markdown",
        paper_format: str = "standard",
        citation_style: str = "APA"
    ) -> Paper:
        """
        Run the editor agent to edit a paper.

        Args:
            paper (Paper): Paper to edit
            style_guide_name (str, optional): Style guide name to apply. Defaults to "Standard Academic".
            output_format (str, optional): Output format. Defaults to "markdown".
            paper_format (str, optional): Paper format. Defaults to "standard".
            citation_style (str, optional): Citation style. Defaults to "APA".

        Returns:
            Paper: Edited paper
        """
        logger.info(f"Running editor agent on: {paper.title}")
        
        try:
            # 스타일 가이드 이름이 비어있으면 기본값 사용
            if not style_guide_name:
                style_guide_name = "Standard Academic"
                logger.warning("스타일 가이드가 지정되지 않아 기본값을 사용합니다.")
            
            # 출력 형식이 비어있으면 기본값 사용
            if not output_format:
                output_format = "markdown"
                logger.warning("출력 형식이 지정되지 않아 기본값을 사용합니다.")
            
            # 1. 스타일 가이드 생성
            style_guide = self.create_style_guide(style_guide_name)
            
            # 2. 논문 편집
            edited_paper = self.edit_paper(paper, style_guide, paper_format)
            
            # 3. 논문 리뷰
            review_result = self.review_paper(edited_paper)
            
            # 리뷰 결과를 논문 상태에 저장
            edited_paper.metadata = edited_paper.metadata or {}
            edited_paper.metadata["review"] = review_result
            
            # 4. 논문 포맷팅
            formatted_file = self.save_formatted_paper(edited_paper, output_format)
            edited_paper.metadata["formatted_file"] = formatted_file
            
            # 4. Format references (using citation_style)
            if edited_paper.references:
                formatted_references = self.format_references(edited_paper.references, citation_style)
                edited_paper.metadata["formatted_references"] = formatted_references
            
            logger.info(f"편집 에이전트 실행 완료: {paper.title}, 평점: {review_result['overall_rating']}/10")
            return edited_paper
            
        except Exception as e:
            logger.error(f"편집 에이전트 실행 중 오류 발생: {str(e)}")
            
            # 오류 발생 시 원본 논문 반환
            paper.edit_history.append({
                "timestamp": self._get_timestamp(),
                "changes": ["편집 중 오류 발생"],
                "style_guide": style_guide.name
            })
            return paper 

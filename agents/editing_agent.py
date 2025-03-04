"""
Academic Paper Editing Agent Module
Handles proofreading, style enforcement, and formatting of research papers.
"""

import os
import re
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from pydantic import BaseModel, Field, validator

from langchain_core.runnables import RunnableConfig
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
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
    """편집 작업 정의"""
    paper_id: str = Field(description="편집할 논문 ID")
    style_guide: Optional[str] = Field(default="Standard Academic", description="적용할 스타일 가이드")
    citation_style: Optional[str] = Field(default="APA", description="인용 스타일")
    focus_areas: Optional[List[str]] = Field(default_factory=list, description="중점 편집 영역")
    output_format: Optional[str] = Field(default="markdown", description="출력 형식")
    
    @validator('paper_id')
    def paper_id_must_not_be_empty(cls, v):
        if not v:
            raise ValueError('paper_id는 필수 항목입니다')
        return v
    
    @validator('style_guide', 'citation_style', 'output_format')
    def provide_defaults_for_empty_strings(cls, v, values, **kwargs):
        if not v:
            field_name = kwargs['field'].name
            if field_name == 'style_guide':
                return "Standard Academic"
            elif field_name == 'citation_style':
                return "APA"
            elif field_name == 'output_format':
                return "markdown"
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
        """Initialize LLM prompts and processing chains"""
        self.editing_parser = PydanticOutputParser(pydantic_object=EditingTask)
        self.review_parser = PydanticOutputParser(pydantic_object=ReviewResult)

        # Paper editing prompt
        PAPER_EDITING_PROMPT = PromptTemplate(
            template="""Act as an expert academic editor. Improve this paper:
            {content}
            
            Style Guide:
            {style_guide}
            
            Output Requirements:
            {format_instructions}
            
            Key Focus Areas:
            1. Technical accuracy preservation
            2. Style guide compliance
            3. Clarity enhancement
            4. Structural optimization""",
            input_variables=["content", "style_guide", "format_instructions"],
        )

        # Paper review prompt  
        PAPER_REVIEW_PROMPT = PromptTemplate(
            template="""Critique this academic paper:
            {paper_content}
            
            Evaluation Criteria:
            1. Technical rigor
            2. Presentation clarity  
            3. Argument structure
            4. Contribution significance
            5. Literature integration
            
            {format_instructions}""",
            input_variables=["paper_content", "format_instructions"],
        )

        # Reference formatting prompt
        REFERENCE_FORMATTING_PROMPT = PromptTemplate(
            template="""Format references in {citation_style} style:
            {references}
            
            Required Elements:
            - Author names
            - Publication year
            - Title formatting
            - Source details
            - DOI/URL if available""",
            input_variables=["citation_style", "references"],
        )

        # Initialize processing chains
        self.editing_chain = LLMChain(
            llm=self.llm,
            prompt=PAPER_EDITING_PROMPT,
            verbose=self.verbose
        )
        
        # 리뷰 체인 초기화
        self.review_chain = LLMChain(
            llm=self.llm,
            prompt=PAPER_REVIEW_PROMPT,
            verbose=self.verbose
        )
        
        # 참고 문헌 형식 체인 초기화
        self.reference_chain = LLMChain(
            llm=self.llm,
            prompt=REFERENCE_FORMATTING_PROMPT,
            verbose=self.verbose
        )
        
        logger.debug("편집 에이전트 프롬프트 및 체인 초기화 완료")

    def create_style_guide(self, style_name: str = "Standard Academic") -> StyleGuide:
        """Generate style guide specifications"""
        logger.info(f"Compiling {style_name} style guide")
        
        if style_name == "APA":
            return StyleGuide(
                name="APA 7th Edition",
                rules=[
                    "Use active voice where appropriate",
                    "Maintain past tense for methodology",
                    "Apply title case for headings",
                    "Format in-text citations as (Author, Year)",
                    "Include DOI for digital sources"
                ],
                examples={
                    "Passive voice": "The experiment was conducted → Researchers conducted the experiment",
                    "Citation": "(Smith et al., 2020) → (Smith, 2020; Johnson & Lee, 2019)"
                }
            )
        # ... other style guides

    def edit_paper(self, paper: Paper, style_guide: StyleGuide) -> Paper:
        """Execute comprehensive paper editing"""
        logger.info(f"Editing paper: {paper.title}")
        try:
            paper_content = self._compile_paper_content(paper)
            result = self.editing_chain.invoke({
                "content": paper_content,
                "style_guide": style_guide.json(),
                "format_instructions": self.editing_parser.get_format_instructions()
            })
            return self._process_edits(result, paper, style_guide)
        except Exception as e:
            logger.error(f"Editing failed: {str(e)}")
            return self._handle_edit_error(paper, style_guide)

    def review_paper(self, paper: Paper) -> Dict[str, Any]:
        """Conduct thorough paper evaluation"""
        logger.info(f"Reviewing paper: {paper.title}")
        try:
            paper_content = self._compile_paper_content(paper)
            result = self.review_chain.invoke({
                "paper_content": paper_content,
                "format_instructions": self.review_parser.get_format_instructions()
            })
            return self._parse_review_results(result)
        except Exception as e:
            logger.error(f"Review failed: {str(e)}")
            return self._default_review_response()

    def format_references(self, references: List[Reference], citation_style: str = "APA") -> List[str]:
        """
        참고 문헌을 지정된 인용 스타일로 형식화합니다.

        Args:
            references (List[Reference]): 참고 문헌 목록
            citation_style (str, optional): 인용 스타일. 기본값은 "APA"

        Returns:
            List[str]: 형식화된 참고 문헌 목록
        """
        logger.info(f"{len(references)}개의 참고 문헌을 '{citation_style}' 스타일로 형식화 중...")
        
        try:
            # 참고 문헌 정보 준비
            references_info = []
            for ref in references:
                ref_info = {
                    "title": ref.title,
                    "authors": ref.authors,
                    "year": ref.year,
                    "source": ref.source,
                    "url": ref.url
                }
                references_info.append(ref_info)
            
            references_json = json.dumps(references_info, ensure_ascii=False)
            
            # 참고 문헌 형식화 수행
            result = self.reference_chain.invoke({
                "references": references_json,
                "citation_style": citation_style
            })
            
            # 결과 파싱
            formatted_references = []
            for line in result["text"].strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("```"):
                    formatted_references.append(line)
            
            logger.info(f"{len(formatted_references)}개의 참고 문헌 형식화 완료")
            return formatted_references
            
        except Exception as e:
            logger.error(f"참고 문헌 형식화 중 오류 발생: {str(e)}")
            
            # 오류 발생 시 기본 형식 반환
            return [f"{ref.title}. {', '.join(ref.authors) if ref.authors else '알 수 없음'}. ({ref.year}). {ref.source}." 
                   for ref in references]

    def save_formatted_paper(self, paper: Paper, output_format: str = "markdown") -> str:
        """
        편집된 논문을 지정된 형식으로 저장합니다.

        Args:
            paper (Paper): 저장할 논문
            output_format (str, optional): 출력 형식. 기본값은 "markdown"

        Returns:
            str: 저장된 파일 경로
        """
        # 파일 이름 생성
        safe_title = re.sub(r'[^\w\s-]', '', paper.title).strip().lower()
        safe_title = re.sub(r'[-\s]+', '-', safe_title)
        
        if output_format == "markdown":
            filename = f"{safe_title}.md"
            file_path = os.path.join(OUTPUT_DIR, filename)
            
            try:
                # 마크다운 형식으로 논문 내용 생성
                content = f"# {paper.title}\n\n"
                
                for section in paper.sections:
                    content += f"## {section.title}\n\n{section.content}\n\n"
                
                # 참고 문헌 섹션 추가
                if paper.references:
                    content += "## References\n\n"
                    for i, ref in enumerate(paper.references, 1):
                        authors = ", ".join(ref.authors) if ref.authors else "Unknown"
                        content += f"{i}. {ref.title}. {authors}. {ref.year}. {ref.source}.\n\n"
                
                # 파일 저장
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logger.info(f"논문 '{paper.title}' Markdown 형식으로 저장됨: {file_path}")
                return file_path
                
            except Exception as e:
                logger.error(f"논문 저장 중 오류 발생: {str(e)}")
                return ""
                
        elif output_format == "latex":
            filename = f"{safe_title}.tex"
            file_path = os.path.join(OUTPUT_DIR, filename)
            
            try:
                # LaTeX 형식으로 논문 내용 생성
                content = "\\documentclass{article}\n"
                content += "\\usepackage[utf8]{inputenc}\n"
                content += "\\usepackage{natbib}\n"
                content += "\\usepackage{graphicx}\n"
                content += "\\usepackage{hyperref}\n\n"

                content += f"\\title{{{paper.title}}}\n"

                # 저자 정보 (없으므로 기본값 사용)
                content += "\\author{AI Paper Writer}\n"
                content += "\\date{\\today}\n\n"

                content += "\\begin{document}\n\n"
                content += "\\maketitle\n\n"

                # 초록 섹션 (있는 경우)
                abstract_section = next((s for s in paper.sections if s.title.lower() == "초록" or s.title.lower() == "abstract"), None)
                if abstract_section:
                    content += "\\begin{abstract}\n"
                    content += abstract_section.content + "\n"
                    content += "\\end{abstract}\n\n"

                # 목차
                content += "\\tableofcontents\n\\newpage\n\n"

                # 각 섹션
                for section in paper.sections:
                    # 초록은 이미 처리했으므로 건너뜀
                    if section.title.lower() == "초록" or section.title.lower() == "abstract":
                        continue
                        
                    content += f"\\section{{{section.title}}}\n\n"
                    
                    # LaTeX 특수 문자 이스케이프
                    section_content = section.content
                    section_content = section_content.replace("_", "\\_")
                    section_content = section_content.replace("%", "\\%")
                    section_content = section_content.replace("&", "\\&")
                    section_content = section_content.replace("#", "\\#")
                    
                    content += section_content + "\n\n"
                
                # 참고 문헌
                if paper.references:
                    content += "\\begin{thebibliography}{99}\n\n"
                    
                    for i, ref in enumerate(paper.references, 1):
                        authors = ", ".join(ref.authors) if ref.authors else "Unknown"
                        content += f"\\bibitem{{{i}}}\n"
                        content += f"{authors} ({ref.year}). {ref.title}. {ref.source}.\n\n"
                    
                    content += "\\end{thebibliography}\n\n"
                
                content += "\\end{document}"
                
                # 파일 저장
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logger.info(f"논문 '{paper.title}' LaTeX 형식으로 저장됨: {file_path}")
                return file_path
                
            except Exception as e:
                logger.error(f"논문 LaTeX 형식 저장 중 오류 발생: {str(e)}")
                return ""
        
        else:
            logger.error(f"지원되지 않는 출력 형식: {output_format}")
            return ""

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
        output_format: str = "markdown"
    ) -> Paper:
        """
        편집 에이전트를 실행하여 논문을 편집합니다.

        Args:
            paper (Paper): 편집할 논문
            style_guide_name (str, optional): 적용할 스타일 가이드 이름. Defaults to "Standard Academic".
            output_format (str, optional): 출력 형식. Defaults to "markdown".

        Returns:
            Paper: 편집된 논문
        """
        logger.info(f"편집 에이전트 실행 중: {paper.title}")
        
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
            edited_paper = self.edit_paper(paper, style_guide)
            
            # 3. 논문 리뷰
            review_result = self.review_paper(edited_paper)
            
            # 리뷰 결과를 논문 상태에 저장
            edited_paper.metadata = edited_paper.metadata or {}
            edited_paper.metadata["review"] = review_result
            
            # 4. 논문 포맷팅
            formatted_file = self.save_formatted_paper(edited_paper, output_format)
            edited_paper.metadata["formatted_file"] = formatted_file
            
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

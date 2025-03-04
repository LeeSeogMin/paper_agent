"""
Writer Agent Module
An agent that writes academic papers based on research materials.
"""

import os
import re
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from pydantic import BaseModel, Field

from langchain_core.runnables import RunnableConfig
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser

from config.settings import OUTPUT_DIR, MAX_SECTION_TOKENS
from config.templates import get_template
from utils.logger import logger
from models.paper import Paper, PaperSection, Reference
from models.research import ResearchMaterial
from prompts.paper_prompts import (
    PAPER_OUTLINE_PROMPT,
    PAPER_SECTION_PROMPT,
    PAPER_EDITING_PROMPT,
    LITERATURE_REVIEW_PROMPT,
    METHODOLOGY_PROMPT,
    RESEARCH_SUMMARY_PROMPT,
    ANALYSIS_PROMPT,
    CUSTOM_WRITING_PROMPT
)
from agents.base import BaseAgent


class OutlineTask(BaseModel):
    """Paper outline writing task"""
    title: str = Field(description="Paper title")
    main_points: List[str] = Field(description="List of main points to be addressed in the paper")
    sections: List[Dict[str, Any]] = Field(description="List of paper sections (with titles and descriptions)")


class SectionTask(BaseModel):
    """Paper section writing task"""
    section_title: str = Field(description="Section title")
    section_content: str = Field(description="Written section content")
    citations: List[str] = Field(description="List of references cited in the section")


class EditTask(BaseModel):
    """Paper editing task"""
    edited_content: str = Field(description="Edited content")
    changes_made: List[str] = Field(description="List of applied changes")


class WriterAgent(BaseAgent[Union[Paper, Dict[str, Any]]]):
    """Flexible content writing agent that responds to coordinator instructions"""

    def __init__(
        self,
        name: str = "Writing Agent",
        description: str = "Content Writing Expert",
        verbose: bool = False
    ):
        """
        Initialize WriterAgent

        Args:
            name (str, optional): Agent name. Defaults to "Writing Agent"
            description (str, optional): Agent description. Defaults to "Content Writing Expert"
            verbose (bool, optional): Enable detailed logging. Defaults to False
        """
        super().__init__(name, description, verbose=verbose)
        
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Initialize prompts
        self._init_prompts()
        
        # Currently working paper
        self.current_paper = None
        
        # 추가: 작업 유형별 템플릿 초기화
        self.task_templates = {
            "full_paper": PAPER_SECTION_PROMPT,
            "literature_review": LITERATURE_REVIEW_PROMPT,
            "methodology": METHODOLOGY_PROMPT,
            "research_summary": RESEARCH_SUMMARY_PROMPT,
            "analysis": ANALYSIS_PROMPT,
            "custom": CUSTOM_WRITING_PROMPT
        }
        
        logger.info(f"{self.name} initialized with flexible task support")

    def _init_prompts(self) -> None:
        """Initialize prompts and chains"""
        # Initialize outline task parser
        self.outline_parser = PydanticOutputParser(pydantic_object=OutlineTask)
        
        # Initialize section task parser
        self.section_parser = PydanticOutputParser(pydantic_object=SectionTask)
        
        # Initialize edit task parser
        self.edit_parser = PydanticOutputParser(pydantic_object=EditTask)
        
        # Initialize outline writing chain
        self.outline_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=PAPER_OUTLINE_PROMPT,
                input_variables=["topic", "paper_type", "research_materials", "format_instructions"],
            ),
            verbose=self.verbose
        )
        
        # Initialize section writing chain
        self.section_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=PAPER_SECTION_PROMPT,
                input_variables=["paper_title", "section_title", "section_purpose", "paper_outline", "research_materials", "format_instructions"],
            ),
            verbose=self.verbose
        )
        
        # Initialize editing chain
        self.edit_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=PAPER_EDITING_PROMPT,
                input_variables=["editing_type", "style_guide", "paper_content", "format_instructions"],
            ),
            verbose=self.verbose
        )
        
        logger.debug("Writer agent prompts and chains initialized")

    def create_paper_outline(
        self, 
        topic: str, 
        research_materials: List[ResearchMaterial],
        template_name: str = "academic"
    ) -> Dict[str, Any]:
        """
        Create a paper outline.

        Args:
            topic (str): Paper topic
            research_materials (List[ResearchMaterial]): List of research materials
            template_name (str, optional): Paper template name. Defaults to "academic"

        Returns:
            Dict[str, Any]: Generated paper outline
        """
        logger.info(f"Creating paper outline for topic '{topic}'...")
        
        try:
            # Get template
            template = get_template(template_name)
            
            # Generate research material summary
            research_summary = "\n\n".join([
                f"Title: {material.title}\n"
                f"Authors: {', '.join(material.authors) if material.authors else 'Unknown'}\n"
                f"Summary: {material.summary}"
                for material in research_materials[:5]  # Use only top 5 materials
            ])
            
            # Generate outline
            format_instructions = self.outline_parser.get_format_instructions()
            
            result = self.outline_chain.invoke({
                "topic": topic,
                "paper_type": template_name,
                "research_materials": research_summary,
                "format_instructions": format_instructions
            })
            
            # Parse result
            outline_task = self.outline_parser.parse(result["text"])
            
            # Extract references
            references = []
            for material in research_materials:
                ref = Reference(
                    id=material.id,
                    title=material.title,
                    authors=material.authors,
                    year=material.year,
                    venue=material.venue,
                    url=material.url
                )
                references.append(ref)
            
            # Create paper outline object
            paper_outline = {
                "title": outline_task.title,
                "topic": topic,
                "main_points": outline_task.main_points,
                "sections": outline_task.sections,
                "references": references,
                "template_name": template_name
            }
            
            logger.info(f"Paper outline created: '{outline_task.title}' ({len(outline_task.sections)} sections)")
            return paper_outline
            
        except Exception as e:
            logger.error(f"Error creating paper outline: {str(e)}")
            
            # Return default outline on error
            default_sections = [
                {"title": "Introduction", "description": "Research background and purpose"},
                {"title": "Related Work", "description": "Review of existing research"},
                {"title": "Methodology", "description": "Description of research methods"},
                {"title": "Results", "description": "Presentation of research results"},
                {"title": "Discussion", "description": "Interpretation and meaning of results"},
                {"title": "Conclusion", "description": "Research summary and future directions"}
            ]
            
            return {
                "title": f"Research on {topic}",
                "topic": topic,
                "main_points": ["Basic research on the topic"],
                "sections": default_sections,
                "references": [],
                "template_name": template_name
            }

    def write_section(
        self, 
        section: Dict[str, Any], 
        research_materials: List[ResearchMaterial],
        outline: Dict[str, Any],
        previous_sections: Optional[List[PaperSection]] = None
    ) -> PaperSection:
        """
        Write a paper section.

        Args:
            section (Dict[str, Any]): Section information
            research_materials (List[ResearchMaterial]): List of research materials
            outline (Dict[str, Any]): Paper outline
            previous_sections (Optional[List[PaperSection]], optional): List of previously written sections

        Returns:
            PaperSection: Written paper section
        """
        section_title = section["title"]
        logger.info(f"Writing section '{section_title}'...")
        
        try:
            # Generate research materials text
            research_materials_text = "\n\n".join([
                f"Title: {material.title}\n"
                f"Authors: {', '.join(material.authors) if material.authors else 'Unknown'}\n"
                f"Summary: {material.summary}\n"
                f"Content Sample: {material.content[:500]}..." if material.content else ""
                for material in research_materials[:5]  # Use only top 5 materials
            ])
            
            # Generate previous sections information
            previous_sections_text = ""
            if previous_sections and len(previous_sections) > 0:
                previous_sections_text = "\n\n".join([
                    f"Section: {prev_section.title}\nContent: {prev_section.content[:300]}..."
                    for prev_section in previous_sections[-2:]  # Use only the most recent 2 sections
                ])
            
            # Prepare section information
            paper_outline = {
                "title": outline["title"],
                "main_points": outline["main_points"],
                "sections": [s["title"] for s in outline["sections"]]
            }
            
            # Write section
            format_instructions = self.section_parser.get_format_instructions()
            
            result = self.section_chain.invoke({
                "paper_title": outline["title"],
                "section_title": section_title,
                "section_purpose": section.get("description", ""),
                "paper_outline": json.dumps(paper_outline, ensure_ascii=False),
                "research_materials": research_materials_text,
                "format_instructions": format_instructions
            })
            
            # Parse result
            section_task = self.section_parser.parse(result["text"])
            
            # Handle citations
            citations = section_task.citations
            
            # Create PaperSection object
            paper_section = PaperSection(
                title=section_title,
                content=section_task.section_content,
                citations=citations
            )
            
            logger.info(f"Section '{section_title}' written ({len(section_task.section_content)} characters)")
            return paper_section
            
        except Exception as e:
            logger.error(f"Error writing section '{section_title}': {str(e)}")
            
            # Return default section on error
            return PaperSection(
                title=section_title,
                content=f"This section covers '{section_title}'. An error occurred during writing.",
                citations=[]
            )

    def edit_paper(self, paper: Paper) -> Paper:
        """
        Edit the entire paper.

        Args:
            paper (Paper): Paper to edit

        Returns:
            Paper: Edited paper
        """
        logger.info(f"Editing paper '{paper.title}'...")
        
        try:
            # Generate full paper content
            paper_content = f"# {paper.title}\n\n"
            
            for section in paper.sections:
                paper_content += f"## {section.title}\n\n{section.content}\n\n"
            
            # Add references section
            if paper.references:
                paper_content += "## References\n\n"
                for i, ref in enumerate(paper.references, 1):
                    authors = ", ".join(ref.authors) if ref.authors else "Unknown"
                    paper_content += f"{i}. {ref.title}. {authors}. {ref.year}. {ref.venue}.\n"
            
            # Perform editing
            format_instructions = self.edit_parser.get_format_instructions()
            
            result = self.edit_chain.invoke({
                "editing_type": "Comprehensive Editing",
                "style_guide": "Academic",
                "paper_content": paper_content,
                "format_instructions": format_instructions
            })
            
            # Parse result
            edit_task = self.edit_parser.parse(result["text"])
            
            # Parse edited paper
            edited_paper = self._parse_edited_paper(edit_task.edited_content, paper)
            
            # Record changes
            edited_paper.edit_history.append({
                "timestamp": self._get_timestamp(),
                "changes": edit_task.changes_made
            })
            
            logger.info(f"Paper '{paper.title}' edited ({len(edit_task.changes_made)} changes)")
            return edited_paper
            
        except Exception as e:
            logger.error(f"Error editing paper: {str(e)}")
            
            # Return original paper on error
            paper.edit_history.append({
                "timestamp": self._get_timestamp(),
                "changes": ["Error occurred during editing"]
            })
            return paper

    def edit_section(self, section: PaperSection) -> PaperSection:
        """
        Edit an individual section.

        Args:
            section (PaperSection): Section to edit

        Returns:
            PaperSection: Edited section
        """
        logger.info(f"Editing section '{section.title}'...")
        
        try:
            # Prepare section content
            section_content = f"## {section.title}\n\n{section.content}"
            
            # Perform editing
            format_instructions = self.edit_parser.get_format_instructions()
            
            result = self.edit_chain.invoke({
                "editing_type": "Section Editing",
                "style_guide": "Academic",
                "paper_content": section_content,
                "format_instructions": format_instructions
            })
            
            # Parse result
            edit_task = self.edit_parser.parse(result["text"])
            
            # Extract title and content from edited content
            edited_content = edit_task.edited_content
            
            # Extract title
            title_match = re.search(r"##\s+(.+?)(?:\n|$)", edited_content)
            edited_title = title_match.group(1).strip() if title_match else section.title
            
            # Extract content (excluding title)
            content_match = re.search(r"##\s+.+?\n\n([\s\S]+)$", edited_content)
            edited_content = content_match.group(1).strip() if content_match else edited_content
            
            # Create edited section
            edited_section = PaperSection(
                title=edited_title,
                content=edited_content,
                citations=section.citations
            )
            
            logger.info(f"Section '{section.title}' edited")
            return edited_section
            
        except Exception as e:
            logger.error(f"Error editing section '{section.title}': {str(e)}")
            
            # Return original section on error
            return section

    def _parse_edited_paper(self, edited_content: str, original_paper: Paper) -> Paper:
        """
        Parse edited paper content into a Paper object.

        Args:
            edited_content (str): Edited paper content
            original_paper (Paper): Original paper

        Returns:
            Paper: Parsed paper object
        """
        try:
            # Extract title
            title_match = re.search(r"#\s+(.+?)(?:\n|$)", edited_content)
            title = title_match.group(1).strip() if title_match else original_paper.title
            
            # Extract sections
            sections = []
            section_pattern = re.compile(r"##\s+(.+?)\n\n([\s\S]+?)(?=\n##|\n# |$)")
            
            for match in section_pattern.finditer(edited_content):
                section_title = match.group(1).strip()
                section_content = match.group(2).strip()
                
                # Skip references section
                if "References" in section_title:
                    continue
                
                # Get citations from original section
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
            
            # Create edited paper object
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
            logger.error(f"Error parsing edited paper: {str(e)}")
            return original_paper

    def save_paper(self, paper: Paper, filename: Optional[str] = None) -> str:
        """
        Save the paper to a file.

        Args:
            paper (Paper): Paper to save
            filename (Optional[str], optional): Filename. Defaults to None (auto-generated)

        Returns:
            str: Path to the saved file
        """
        # Generate filename
        if not filename:
            safe_title = re.sub(r'[^\w\s-]', '', paper.title).strip().lower()
            safe_title = re.sub(r'[-\s]+', '-', safe_title)
            filename = f"{safe_title}.md"
        
        # Create file path
        file_path = os.path.join(OUTPUT_DIR, filename)
        
        try:
            # Generate paper content
            content = f"# {paper.title}\n\n"
            
            for section in paper.sections:
                content += f"## {section.title}\n\n{section.content}\n\n"
            
            # Add references section
            if paper.references:
                content += "## References\n\n"
                for i, ref in enumerate(paper.references, 1):
                    authors = ", ".join(ref.authors) if ref.authors else "Unknown"
                    content += f"{i}. {ref.title}. {authors}. {ref.year}. {ref.venue}.\n"
            
            # Save file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Paper '{paper.title}' saved: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving paper: {str(e)}")
            return ""

    def _get_timestamp(self) -> str:
        """Return current timestamp."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def run(
        self, 
        topic: str, 
        research_materials: List[ResearchMaterial],
        template_name: str = "academic",
        config: Optional[RunnableConfig] = None
    ) -> Paper:
        """
        Run the writer agent and generate a paper.

        Args:
            topic (str): Paper topic
            research_materials (List[ResearchMaterial]): List of research materials
            template_name (str, optional): Paper template name. Defaults to "academic"
            config (Optional[RunnableConfig], optional): Run configuration

        Returns:
            Paper: Generated paper
        """
        logger.info(f"Running writer agent: topic '{topic}'")
        
        try:
            # Reset state
            self.reset()
            
            # 1. Create paper outline
            outline = self.create_paper_outline(topic, research_materials, template_name)
            
            # 2. Write each section
            sections = []
            previous_sections = []
            
            for section_info in outline["sections"]:
                # Write section
                section = self.write_section(
                    section=section_info,
                    research_materials=research_materials,
                    outline=outline,
                    previous_sections=previous_sections
                )
                
                sections.append(section)
                previous_sections.append(section)
            
            # 3. Create paper object
            paper = Paper(
                title=outline["title"],
                topic=topic,
                sections=sections,
                references=outline["references"],
                template_name=template_name,
                edit_history=[]
            )
            
            # 4. Edit paper
            edited_paper = self.edit_paper(paper)
            
            # 5. Save paper
            file_path = self.save_paper(edited_paper)
            
            # Update state
            self.update_state({
                "topic": topic,
                "paper": edited_paper.dict(),
                "output_file": file_path
            })
            
            self.current_paper = edited_paper
            
            logger.info(f"Writer agent execution completed: paper '{edited_paper.title}' generated")
            return edited_paper
            
        except Exception as e:
            logger.error(f"Error in writer agent execution: {str(e)}")
            
            # Return empty paper on error
            empty_paper = Paper(
                title=f"Research on {topic}",
                topic=topic,
                sections=[
                    PaperSection(
                        title="Error",
                        content="An error occurred during paper generation.",
                        citations=[]
                    )
                ],
                references=[],
                template_name=template_name,
                edit_history=[]
            )
            
            return empty_paper

    def process_writing_task(self, task: Dict[str, Any]) -> Union[Paper, Dict[str, Any]]:
        """
        특정 작성 작업을 처리합니다.

        Args:
            task (Dict[str, Any]): 작업 정의

        Returns:
            Union[Paper, Dict[str, Any]]: 작업 결과 (Paper 또는 결과 딕셔너리)
        """
        logger.info(f"작성 작업 처리 중: {task.get('task_type', 'unknown')}")
        
        try:
            task_type = task.get("task_type", "")
            
            if task_type == "full_paper":
                # 전체 논문 작성
                topic = task.get("topic", "")
                materials = task.get("materials", [])
                template_name = task.get("template_name", "academic")
                
                if not topic:
                    raise ValueError("논문 주제가 제공되지 않았습니다.")
                
                return self.run(topic, materials, template_name)
                
            elif task_type == "section":
                # 특정 섹션 작성
                section_title = task.get("section_title", "")
                section_content = task.get("section_content", "")
                materials = task.get("materials", [])
                
                if not section_title:
                    raise ValueError("섹션 제목이 제공되지 않았습니다.")
                
                return {
                    "task_type": "section",
                    "section_title": section_title,
                    "section_content": self._write_section(section_title, section_content, materials),
                    "status": "completed"
                }
                
            # 다른 작업 유형 처리...
                
            else:
                raise ValueError(f"지원되지 않는 작업 유형: {task_type}")
                
        except Exception as e:
            logger.error(f"작성 작업 처리 중 오류 발생: {str(e)}")
            return {
                "task_type": task.get("task_type", "unknown"),
                "error": str(e),
                "status": "failed"
            }

    # 새로운 메소드: 문헌 리뷰 작성
    def _write_literature_review(self, topic: str, materials: List[ResearchMaterial], format: str = "chronological") -> Dict[str, Any]:
        """Generate a focused literature review"""
        logger.info(f"문헌 리뷰 작성 중: 주제 '{topic}', 형식 '{format}'")
        
        try:
            # 자료 정렬 (연대순 또는 주제별)
            if format == "chronological":
                # 연도별로 정렬
                sorted_materials = sorted(materials, key=lambda m: m.year if m.year else 0)
            elif format == "thematic":
                # 주제별로 그룹화 (여기서는 간단히 구현)
                sorted_materials = materials
            else:
                sorted_materials = materials
            
            # 문헌 리뷰 프롬프트 생성
            prompt = f"""
            다음 연구 자료들을 바탕으로 '{topic}'에 관한 문헌 리뷰를 작성해주세요.
            
            형식: {format} (연대순 또는 주제별)
            
            연구 자료:
            """
            
            for i, material in enumerate(sorted_materials):
                prompt += f"\n{i+1}. {material.title} ({material.year})"
                prompt += f"\n   저자: {', '.join(material.authors)}"
                prompt += f"\n   요약: {material.abstract[:200]}..." if material.abstract else ""
                prompt += f"\n   키워드: {', '.join(material.keywords)}" if material.keywords else ""
                prompt += "\n"
            
            # LLM을 사용하여 문헌 리뷰 생성
            messages = [
                SystemMessage(content="당신은 학술 논문 작성을 돕는 전문가입니다. 제공된 연구 자료를 바탕으로 체계적인 문헌 리뷰를 작성해주세요."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            literature_review = response.content
            
            # 결과 반환
            return {
                "topic": topic,
                "format": format,
                "content": literature_review,
                "sources": [m.id for m in materials],
                "word_count": len(literature_review.split())
            }
            
        except Exception as e:
            logger.error(f"문헌 리뷰 작성 중 오류 발생: {str(e)}")
            return {
                "topic": topic,
                "format": format,
                "content": f"문헌 리뷰 생성 중 오류가 발생했습니다: {str(e)}",
                "sources": [m.id for m in materials],
                "word_count": 0
            }

    # 새로운 메소드: 연구 요약 작성
    def _write_research_summary(self, materials: List[ResearchMaterial], focus: str = "key_findings") -> Dict[str, Any]:
        """Generate a research summary with specified focus"""
        logger.info(f"연구 요약 작성 중: 초점 '{focus}'")
        
        try:
            # 요약 초점에 따른 프롬프트 조정
            focus_prompts = {
                "key_findings": "각 연구의 주요 발견과 결론에 초점을 맞추세요.",
                "methodology": "각 연구에서 사용된 방법론과 연구 설계에 초점을 맞추세요.",
                "gaps": "연구 분야의 현재 지식 격차와 향후 연구 방향에 초점을 맞추세요.",
                "comparison": "연구들 간의 유사점과 차이점을 비교하며 요약하세요."
            }
            
            focus_instruction = focus_prompts.get(focus, focus_prompts["key_findings"])
            
            # 요약 프롬프트 생성
            prompt = f"""
            다음 연구 자료들을 요약해주세요. {focus_instruction}
            
            연구 자료:
            """
            
            for i, material in enumerate(materials):
                prompt += f"\n{i+1}. {material.title} ({material.year})"
                prompt += f"\n   저자: {', '.join(material.authors)}"
                prompt += f"\n   요약: {material.abstract[:200]}..." if material.abstract else ""
                prompt += "\n"
            
            # LLM을 사용하여 연구 요약 생성
            messages = [
                SystemMessage(content="당신은 학술 연구 요약 전문가입니다. 제공된 연구 자료를 명확하고 간결하게 요약해주세요."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            research_summary = response.content
            
            # 결과 반환
            return {
                "focus": focus,
                "content": research_summary,
                "sources": [m.id for m in materials],
                "word_count": len(research_summary.split())
            }
            
        except Exception as e:
            logger.error(f"연구 요약 작성 중 오류 발생: {str(e)}")
            return {
                "focus": focus,
                "content": f"연구 요약 생성 중 오류가 발생했습니다: {str(e)}",
                "sources": [m.id for m in materials],
                "word_count": 0
            }
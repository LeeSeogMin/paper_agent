"""
Writer Agent Module
An agent that writes academic papers based on research materials.
"""

import os
import re
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from pydantic import BaseModel, Field

from langchain_core.runnables import RunnableConfig, RunnableSequence
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import OpenAI

from config.settings import OUTPUT_DIR, MAX_SECTION_TOKENS
from config.templates import get_template
from utils.logger import logger
from models.paper import Paper, PaperSection, Reference
from models.research import ResearchMaterial
from prompts.paper_prompts import (
    PAPER_OUTLINE_PROMPT,
    PAPER_SECTION_PROMPT,
    PAPER_EDITING_PROMPT,
    METHODOLOGY_PROMPT,
    RESEARCH_SUMMARY_PROMPT,
    ANALYSIS_PROMPT,
    CUSTOM_WRITING_PROMPT,
    PAPER_CONCLUSION_PROMPT,
    LITERATURE_REVIEW_PROMPT
)
from agents.base import BaseAgent
from utils.rag_integration import RAGEnhancer


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
        
        # Task templates initialization
        self.task_templates = {
            "full_paper": PAPER_SECTION_PROMPT,
            "literature_review": LITERATURE_REVIEW_PROMPT,
            "methodology": METHODOLOGY_PROMPT,
            "research_summary": RESEARCH_SUMMARY_PROMPT,
            "analysis": ANALYSIS_PROMPT,
            "custom": CUSTOM_WRITING_PROMPT
        }
        
        # Initialize prompts
        self._init_prompts()
        
        # Currently working paper
        self.current_paper = None
        
        self.rag_enhancer = RAGEnhancer()
        
        logger.info(f"{self.name} initialized with flexible task support")

    def _init_prompts(self) -> None:
        """Initialize prompt templates for paper writing tasks"""
        # Add English language requirement to all prompts
        self.outline_prompt_template = PromptTemplate(
            template=PAPER_OUTLINE_PROMPT.template + "\n\nIMPORTANT: The outline must be in English.",
            input_variables=PAPER_OUTLINE_PROMPT.input_variables
        )
        
        self.section_prompt_template = PromptTemplate(
            template=PAPER_SECTION_PROMPT.template + "\n\nIMPORTANT: The section must be written in English. All content must be based on the provided research materials. Every claim or statement must include proper citations to the research materials.",
            input_variables=PAPER_SECTION_PROMPT.input_variables
        )
        
        self.editing_prompt_template = PromptTemplate(
            template=PAPER_EDITING_PROMPT.template + "\n\nIMPORTANT: The edited content must be in English. Ensure all claims are properly cited and all citations are included in the references section.",
            input_variables=PAPER_EDITING_PROMPT.input_variables
        )
        
        # Update all task templates to require English and proper citations
        for task_type, template in self.task_templates.items():
            self.task_templates[task_type] = PromptTemplate(
                template=template.template + "\n\nIMPORTANT: All content must be written in English. All content must be based on the provided research materials. Every claim or statement must include proper citations to the research materials.",
                input_variables=template.input_variables
            )
        
        # Initialize parsers
        self.outline_parser = PydanticOutputParser(pydantic_object=OutlineTask)
        self.section_parser = PydanticOutputParser(pydantic_object=SectionTask)
        self.edit_parser = PydanticOutputParser(pydantic_object=EditTask)
        
        # Initialize chains using RunnableSequence
        self.outline_chain = RunnableSequence(
            first=self.outline_prompt_template,
            last=self.llm
        )
        
        self.section_chain = RunnableSequence(
            first=self.section_prompt_template,
            last=self.llm
        )
        
        self.editing_chain = RunnableSequence(
            first=self.editing_prompt_template,
            last=self.llm
        )
        
        # Initialize task chains
        self.task_chains = {}
        for task_type, template in self.task_templates.items():
            self.task_chains[task_type] = RunnableSequence(
                first=template,
                last=self.llm
            )
        
        logger.info("Writing agent prompts initialized with English language and citation requirements")

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
            
            # 섹션 제목을 기반으로 관련 학술 자료 검색
            enhanced_prompt = self.rag_enhancer.enhance_prompt_with_research(
                topic=f"{outline['title']} {section_title}",
                base_prompt=format_instructions,
                num_sources=3
            )
            
            response = self.section_chain.invoke({
                "paper_title": outline["title"],
                "section_title": section_title,
                "section_purpose": section.get("description", ""),
                "paper_outline": json.dumps(paper_outline, ensure_ascii=False),
                "research_materials": research_materials_text,
                "format_instructions": enhanced_prompt
            })
            
            # Parse result
            section_task = self.section_parser.parse(response.content)
            
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
            
            # 다양한 작업 유형에 따른 처리
            if task_type == "full_paper":
                # 전체 논문 작성
                topic = task.get("topic", "")
                materials = task.get("materials", [])
                template_name = task.get("template_name", "academic")
                
                if not topic:
                    raise ValueError("논문 주제가 제공되지 않았습니다.")
                
                return self.run(topic, materials, template_name)
                
            elif task_type == "literature_review":
                # 문헌 검토만 작성
                content = task.get("content", {})
                return self.write_literature_review(content)
                
            elif task_type == "methodology":
                # 방법론 섹션만 작성
                topic = task.get("topic", "")
                context = task.get("additional_context", {})
                return self.write_methodology_section(topic, context)
                
            elif task_type == "results_analysis":
                # 결과 분석 섹션만 작성
                data = task.get("data", {})
                context = task.get("additional_context", {})
                return self.write_results_analysis(data, context)
                
            elif task_type == "conclusion":
                # 결론만 작성
                content = task.get("content", "")
                return self.write_conclusion(content)
                
            elif task_type == "abstract":
                # 초록만 작성
                topic = task.get("topic", "")
                summary = task.get("summary", "")
                return self.write_abstract(topic, summary)
                
            elif task_type == "custom":
                # 사용자 정의 작성 작업
                prompt = task.get("prompt", "")
                context = task.get("context", {})
                return self.custom_writing_task(prompt, context)
                
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
                
            elif task_type == "revision":
                # 재작업 요청 처리
                previous_content = task.get("previous_attempt", "")
                revision_instructions = task.get("revision_instructions", "")
                original_task_type = task.get("original_task_type", "unknown")
                
                return self.revise_content(previous_content, revision_instructions, original_task_type)
                
            else:
                raise ValueError(f"지원되지 않는 작업 유형: {task_type}")
                
        except Exception as e:
            logger.error(f"작성 작업 처리 중 오류 발생: {str(e)}")
            return {
                "task_type": task.get("task_type", "unknown"),
                "error": str(e),
                "status": "failed"
            }

    def write_literature_review(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """문헌 리뷰 작성"""
        prompt = f"""
        Write a comprehensive literature review based on the following research materials:
        
        {content.get('research_materials', '')}
        
        Topic: {content.get('topic', 'No specific topic provided')}
        Outline: {content.get('outline', 'No outline provided')}
        
        Please include the following elements:
        1. Major research trends and development history
        2. Explanation of key concepts and models
        3. Important research findings and discoveries
        4. Limitations of current research and future research directions
        
        Write in an academic and objective style.
        
        {self._get_citation_guidelines()}
        """
        
        response = self.llm.invoke(prompt)
        review_content = response.content
        
        # Return response in appropriate format
        return {
            "task_type": "literature_review",
            "content": review_content,
            "outline": content.get('outline', ''),  # Include original outline
            "status": "completed"
        }

    def _get_citation_guidelines(self, style="APA") -> str:
        """인용 가이드라인을 반환합니다.
        
        Args:
            style: 인용 스타일 (기본값: APA)
            
        Returns:
            str: 인용 가이드라인 텍스트
        """
        if style.upper() == "APA":
            return """Citation Guidelines (APA Style):
1. Parenthetical citation: Include both the author(s) and year in parentheses at the end of the sentence.
   Example: "Research has shown that artificial intelligence significantly impacts workplace dynamics (Kim & Park, 2023)."

2. Narrative citation: Include the author(s) in the narrative with the year in parentheses.
   Example: "Kim and Park (2023) found that artificial intelligence transforms traditional workplace hierarchies."

3. For three or more authors, use the first author's name followed by 'et al.' and the year.
   Example: "Machine learning applications continue to evolve rapidly (Smith et al., 2022)."
"""
        else:
            # 다른 인용 스타일에 대한 가이드라인 추가 가능
            return "Please use appropriate academic citations."

    def write_conclusion(self, content: str) -> Dict[str, Any]:
        """결론 작성"""
        
        prompt = f"""
        Write a conclusion based on the following research results:
        
        {content}
        
        Please write in an academic and objective style in English.
        
        {self._get_citation_guidelines()}
        
        Please use appropriate citations throughout the conclusion to support your statements.
        """
        
        response = self.llm.invoke(prompt)
        return response.content

    def write_methodology_section(self, topic: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """방법론 섹션 작성"""
        # 연구 질문 또는 가설 추출
        research_question = context.get("research_question", topic)
        expected_outcomes = context.get("expected_outcomes", [])
        
        prompt = f"""
        Write a methodology section for the following research topic:
        
        Research Topic: {topic}
        
        Research Question: {research_question}
        
        Expected outcomes: {', '.join(expected_outcomes) if isinstance(expected_outcomes, list) else expected_outcomes}
        
        Please describe:
        1. The research approach and design
        2. Data collection methods
        3. Analysis techniques
        4. Any limitations of the methodology
        
        Write in an academic and objective style.
        """
        
        response = self.llm.invoke(prompt)
        
        return {
            "task_type": "methodology",
            "content": response.content,
            "status": "completed"
        }

    def write_results_analysis(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """결과 분석 섹션 작성"""
        topic = context.get("topic", "")
        
        prompt = f"""
        Write a results and analysis section based on the following data:
        
        Topic: {topic}
        
        Data summary: {json.dumps(data, ensure_ascii=False)}
        
        Please include:
        1. Clear presentation of the key findings
        2. Analysis of patterns or trends in the data
        3. Interpretation of the results in relation to the research question
        4. Visual representations of data (described in text)
        
        Write in an academic and objective style.
        """
        
        response = self.llm.invoke(prompt)
        
        return {
            "task_type": "results_analysis",
            "content": response.content,
            "status": "completed"
        }

    def write_abstract(self, topic: str, summary: str) -> Dict[str, Any]:
        """논문 초록 작성"""
        prompt = f"""
        Write an abstract for a research paper on the following topic:
        
        Topic: {topic}
        
        Research Summary: {summary}
        
        The abstract should:
        1. Be approximately 200-250 words
        2. Clearly state the purpose of the research
        3. Briefly describe the methodology
        4. Summarize key findings
        5. State the main conclusion and implications
        
        Write in an academic and objective style.
        """
        
        response = self.llm.invoke(prompt)
        
        return {
            "task_type": "abstract",
            "content": response.content,
            "status": "completed"
        }

    def custom_writing_task(self, prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """사용자 정의 작성 작업"""
        # 컨텍스트에서 관련 정보 추출
        topic = context.get("topic", "")
        materials = context.get("materials", [])
        additional_info = context.get("additional_info", "")
        
        # 기본 프롬프트 확장
        enhanced_prompt = f"""
        {prompt}
        
        Topic: {topic}
        
        Additional Information: {additional_info}
        
        Please write in an academic and objective style.
        """
        
        # 참고 자료가 있으면 추가
        if materials:
            material_text = "\n\n".join([
                f"Material {i+1}:\nTitle: {m.get('title', '')}\nContent: {m.get('content', '')}"
                for i, m in enumerate(materials[:3])  # 처음 3개만 사용
            ])
            enhanced_prompt += f"\n\nReference Materials:\n{material_text}"
        
        response = self.llm.invoke(enhanced_prompt)
        
        return {
            "task_type": "custom",
            "content": response.content,
            "status": "completed"
        }

    def generate_academic_paper(self, topic, research_data, outline, references, **kwargs):
        """
        Generate a complete academic paper based on research data and outline
        
        Args:
            topic: Paper topic
            research_data: Research materials to use
            outline: Paper outline structure
            references: Reference materials
            **kwargs: Additional parameters
            
        Returns:
            Complete academic paper
        """
        logger.info(f"Generating academic paper on topic: {topic}")
        
        # Ensure we have a reference ID mapping for citations
        ref_id_map = {}
        for i, ref in enumerate(references):
            ref_id = ref.get('id', f'ref_{i+1}')
            ref_id_map[ref_id] = i
        
        # Get citation style
        citation_style = kwargs.get('citation_style', 'APA')
        
        # Generate each section based on the outline
        sections = []
        for section_info in outline.get('sections', []):
            section_type = section_info.get('type', 'standard')
            
            # Generate the section content
            section_result = self.generate_section(
                section_type=section_type,
                topic=topic,
                research_data=research_data,
                section_info=section_info,
                ref_id_map=ref_id_map,
                citation_style=citation_style,
                **kwargs
            )
            
            # Add to sections list
            sections.append(section_result)
        
        # Format references section
        references_section = {
            'title': 'References',
            'content': self.format_references(references, citation_style),
            'type': 'references'
        }
        
        # Add references section to the paper
        sections.append(references_section)
        
        # Assemble the complete paper
        paper = {
            'title': outline.get('title', topic),
            'sections': sections,
            'references': references
        }
        
        logger.info(f"Academic paper generated with {len(sections)} sections including references")
        return paper

    def format_references(self, references, citation_style="APA"):
        """
        Format references according to the specified citation style
        
        Args:
            references: List of reference objects
            citation_style: Citation style to use (APA, MLA, Chicago, etc.)
            
        Returns:
            Formatted references as a string
        """
        logger.info(f"Formatting {len(references)} references in {citation_style} style")
        
        if not references:
            return "No references."
        
        formatted_refs = []
        
        for ref in references:
            if citation_style.upper() == "APA":
                # APA style formatting
                authors = ref.get('authors', [])
                if not authors:
                    author_text = "Unknown"
                elif len(authors) == 1:
                    author_text = f"{authors[0]}"
                elif len(authors) == 2:
                    author_text = f"{authors[0]} & {authors[1]}"
                else:
                    author_text = f"{authors[0]} et al."
                
                year = ref.get('year', 'n.d.')
                title = ref.get('title', 'Untitled')
                venue = ref.get('venue', '')
                url = ref.get('url', '')
                doi = ref.get('doi', '')
                
                ref_text = f"{author_text} ({year}). {title}."
                if venue:
                    ref_text += f" {venue}."
                if doi:
                    ref_text += f" DOI: {doi}"
                elif url:
                    ref_text += f" Retrieved from {url}"
                
                formatted_refs.append(ref_text)
            
            elif citation_style.upper() == "MLA":
                # MLA style formatting
                authors = ref.get('authors', [])
                if not authors:
                    author_text = "Unknown"
                elif len(authors) == 1:
                    author_text = f"{authors[0]}"
                elif len(authors) == 2:
                    author_text = f"{authors[0]} and {authors[1]}"
                else:
                    author_text = f"{authors[0]} et al."
                
                title = ref.get('title', 'Untitled')
                venue = ref.get('venue', '')
                year = ref.get('year', 'n.d.')
                url = ref.get('url', '')
                
                ref_text = f"{author_text}. \"{title}\"."
                if venue:
                    ref_text += f" {venue},"
                ref_text += f" {year}."
                if url:
                    ref_text += f" {url}."
                
                formatted_refs.append(ref_text)
            
            else:
                # Default formatting for other styles
                authors = ref.get('authors', [])
                author_text = ", ".join(authors) if authors else "Unknown"
                title = ref.get('title', 'Untitled')
                year = ref.get('year', 'n.d.')
                
                ref_text = f"{author_text}. {title}. {year}."
                formatted_refs.append(ref_text)
        
        # Join all formatted references with line breaks
        return "\n\n".join(formatted_refs)

    def generate_section(self, section_type, topic, research_data, section_info, ref_id_map, citation_style, **kwargs):
        """
        Generate a specific section of an academic paper
        
        Args:
            section_type: Type of section to generate
            topic: Paper topic
            research_data: Research materials
            section_info: Section information
            ref_id_map: Reference ID mapping
            citation_style: Citation style to use
            **kwargs: Additional parameters
            
        Returns:
            Generated section content
        """
        logger.info(f"Generating {section_type} section: {section_info.get('title', 'Untitled')}")
        
        # Prepare research materials for the prompt
        formatted_research = self._format_ref_list_for_prompt(research_data, ref_id_map)
        
        # Get section title and description
        section_title = section_info.get('title', 'Untitled Section')
        section_desc = section_info.get('description', '')
        
        # Base prompt variables
        prompt_vars = {
            'topic': topic,
            'section_title': section_title,
            'section_description': section_desc,
            'research_materials': formatted_research,
            'citation_style': self._get_citation_guidelines(citation_style)
        }
        
        # Add any additional context from kwargs
        for key, value in kwargs.items():
            if key not in prompt_vars:
                prompt_vars[key] = value
        
        # Select the appropriate template based on section type
        if section_type == 'introduction':
            template = self.task_templates.get('full_paper')
            prompt_vars['section_type'] = 'introduction'
        elif section_type == 'literature_review':
            template = self.task_templates.get('literature_review')
        elif section_type == 'methodology':
            template = self.task_templates.get('methodology')
        elif section_type == 'results':
            template = self.task_templates.get('analysis')
            prompt_vars['section_type'] = 'results'
        elif section_type == 'discussion':
            template = self.task_templates.get('analysis')
            prompt_vars['section_type'] = 'discussion'
        elif section_type == 'conclusion':
            template = self.task_templates.get('full_paper')
            prompt_vars['section_type'] = 'conclusion'
        else:
            # Default to standard section template
            template = self.task_templates.get('full_paper')
            prompt_vars['section_type'] = 'standard'
        
        # Generate the section content using RAG-enhanced prompt
        rag_prompt = self.rag_enhancer.enhance_prompt_with_research(
            topic=f"{topic} {section_title}",
            base_prompt=template.format(**prompt_vars),
            num_sources=5
        )
        
        # Use the LLM to generate the section content
        response = self.llm.invoke(rag_prompt)
        
        # Extract the content and citations
        content = response
        citations = []
        
        # Extract citations from the content
        citation_pattern = r'\(([^)]+)\)'
        matches = re.findall(citation_pattern, content)
        for match in matches:
            if any(ref_id in match for ref_id in ref_id_map.keys()):
                citations.append(match)
        
        # Create the section object
        section = {
            'title': section_title,
            'content': content,
            'type': section_type,
            'citations': citations
        }
        
        logger.info(f"Generated {section_type} section with {len(citations)} citations")
        return section

    def _format_ref_list_for_prompt(self, references, ref_id_map):
        """Format reference list for prompt"""
        ref_list = []
        for ref in references:
            ref_id = ref.get("id")
            num = ref_id_map.get(ref_id, "?")
            authors = ref.get("authors", "")
            year = ref.get("year", "")
            title = ref.get("title", "")
            
            ref_list.append(f"[{num}] {authors} ({year}). {title}")
        
        return "\n".join(ref_list)

    def revise_content(self, previous_content: str, revision_instructions: str, 
                     original_task_type: str) -> Dict[str, Any]:
        """
        이전 콘텐츠를 수정 지시사항에 따라 재작성
        
        Args:
            previous_content: 이전에 생성된 콘텐츠
            revision_instructions: 수정 지시사항
            original_task_type: 원래 작업 유형
            
        Returns:
            Dict: 수정된 콘텐츠
        """
        prompt = f"""
        You are tasked with revising the following content according to specific instructions.
        
        ORIGINAL CONTENT:
        {previous_content}
        
        REVISION INSTRUCTIONS:
        {revision_instructions}
        
        Please revise the content to address the instructions. Maintain academic style and format.
        Focus specifically on addressing the issues mentioned in the revision instructions.
        
        Output the revised content only, without explaining your changes.
        """
        
        response = self.llm.invoke(prompt)
        
        return {
            "task_type": "revision",
            "original_task_type": original_task_type,
            "content": response.content,
            "status": "completed"
        }

    def update_template_config(self, config: Dict[str, Any]) -> None:
        """
        템플릿 설정을 업데이트합니다.
        
        Args:
            config: 새 설정 값
        """
        logger.info("작성 에이전트 템플릿 설정 업데이트")
        
        # 섹션 설정
        if "sections" in config:
            self.section_templates = config["sections"]
            logger.info(f"섹션 템플릿 업데이트: {len(self.section_templates)} 섹션")
        
        # 인용 스타일
        if "citation_style" in config:
            self.citation_style = config["citation_style"]
            logger.info(f"인용 스타일 설정: {self.citation_style}")
        
        # 포맷팅 가이드라인
        if "formatting" in config:
            self.formatting_guidelines = config["formatting"]
            logger.info("포맷팅 가이드라인 업데이트 완료")
        
        # 기타 설정들
        for key, value in config.items():
            if key not in ["sections", "citation_style", "formatting"]:
                setattr(self, f"_{key}", value)
                logger.info(f"추가 설정 업데이트: {key}")
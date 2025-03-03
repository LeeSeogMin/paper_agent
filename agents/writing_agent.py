"""
작가 에이전트 모듈
연구 자료를 바탕으로 논문을 작성하는 에이전트입니다.
"""

import os
import re
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from pydantic import BaseModel, Field

from langchain_core.runnables import RunnableConfig
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser

from config.settings import OUTPUT_DIR, MAX_SECTION_TOKENS
from config.templates import get_paper_template
from utils.logger import logger
from models.paper import Paper, PaperOutline, PaperSection, Reference
from models.research import ResearchMaterial
from prompts.paper_prompts import (
    PAPER_OUTLINE_PROMPT,
    PAPER_SECTION_PROMPT,
    PAPER_EDITING_PROMPT
)
from agents.base import BaseAgent


class OutlineTask(BaseModel):
    """논문 개요 작성 작업"""
    title: str = Field(description="논문 제목")
    main_points: List[str] = Field(description="논문에서 다룰 주요 요점 목록")
    sections: List[Dict[str, Any]] = Field(description="논문 섹션 목록 (제목과 설명 포함)")


class SectionTask(BaseModel):
    """논문 섹션 작성 작업"""
    section_title: str = Field(description="섹션 제목")
    section_content: str = Field(description="작성된 섹션 내용")
    citations: List[str] = Field(description="섹션에서 인용된 참고 문헌 목록")


class EditTask(BaseModel):
    """논문 편집 작업"""
    edited_content: str = Field(description="편집된 내용")
    changes_made: List[str] = Field(description="적용된 변경사항 목록")


class WriterAgent(BaseAgent[Paper]):
    """논문 작성 전문가 에이전트"""

    def __init__(
        self,
        name: str = "작성 에이전트",
        description: str = "논문 작성 전문가",
        verbose: bool = False
    ):
        """
        WriterAgent 초기화

        Args:
            name (str, optional): 에이전트 이름. 기본값은 "작성 에이전트"
            description (str, optional): 에이전트 설명. 기본값은 "논문 작성 전문가"
            verbose (bool, optional): 상세 로깅 활성화 여부. 기본값은 False
        """
        super().__init__(name, description, verbose=verbose)
        
        # 출력 디렉토리 생성
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # 프롬프트 초기화
        self._init_prompts()
        
        # 현재 작업 중인 논문
        self.current_paper = None
        
        logger.info(f"{self.name} 초기화 완료")

    def _init_prompts(self) -> None:
        """프롬프트와 체인 초기화"""
        # 개요 작업 파서 초기화
        self.outline_parser = PydanticOutputParser(pydantic_object=OutlineTask)
        
        # 섹션 작업 파서 초기화
        self.section_parser = PydanticOutputParser(pydantic_object=SectionTask)
        
        # 편집 작업 파서 초기화
        self.edit_parser = PydanticOutputParser(pydantic_object=EditTask)
        
        # 개요 작성 체인 초기화
        self.outline_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=PAPER_OUTLINE_PROMPT,
                input_variables=["topic", "template", "research_summary", "format_instructions"],
            ),
            verbose=self.verbose
        )
        
        # 섹션 작성 체인 초기화
        self.section_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=PAPER_SECTION_PROMPT,
                input_variables=["section_info", "research_materials", "previous_sections", "format_instructions"],
            ),
            verbose=self.verbose
        )
        
        # 편집 체인 초기화
        self.edit_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=PAPER_EDITING_PROMPT,
                input_variables=["content", "format_instructions"],
            ),
            verbose=self.verbose
        )
        
        logger.debug("작가 에이전트 프롬프트 및 체인 초기화 완료")

    def create_paper_outline(
        self, 
        topic: str, 
        research_materials: List[ResearchMaterial],
        template_name: str = "academic"
    ) -> PaperOutline:
        """
        논문 개요를 생성합니다.

        Args:
            topic (str): 논문 주제
            research_materials (List[ResearchMaterial]): 연구 자료 목록
            template_name (str, optional): 논문 템플릿 이름. 기본값은 "academic"

        Returns:
            PaperOutline: 생성된 논문 개요
        """
        logger.info(f"주제 '{topic}'에 대한 논문 개요 생성 중...")
        
        try:
            # 템플릿 가져오기
            template = get_paper_template(template_name)
            
            # 연구 자료 요약 생성
            research_summary = "\n\n".join([
                f"제목: {material.title}\n"
                f"저자: {', '.join(material.authors) if material.authors else '알 수 없음'}\n"
                f"요약: {material.summary}"
                for material in research_materials[:5]  # 상위 5개 자료만 사용
            ])
            
            # 개요 생성
            format_instructions = self.outline_parser.get_format_instructions()
            
            result = self.outline_chain.invoke({
                "topic": topic,
                "template": json.dumps(template, ensure_ascii=False),
                "research_summary": research_summary,
                "format_instructions": format_instructions
            })
            
            # 결과 파싱
            outline_task = self.outline_parser.parse(result["text"])
            
            # 참고 문헌 추출
            references = []
            for material in research_materials:
                ref = Reference(
                    title=material.title,
                    authors=material.authors,
                    year=material.year,
                    source=material.source,
                    url=material.url
                )
                references.append(ref)
            
            # PaperOutline 객체 생성
            paper_outline = PaperOutline(
                title=outline_task.title,
                topic=topic,
                main_points=outline_task.main_points,
                sections=outline_task.sections,
                references=references,
                template_name=template_name
            )
            
            logger.info(f"논문 개요 생성 완료: '{outline_task.title}' ({len(outline_task.sections)}개 섹션)")
            return paper_outline
            
        except Exception as e:
            logger.error(f"논문 개요 생성 중 오류 발생: {str(e)}")
            
            # 오류 발생 시 기본 개요 반환
            default_sections = [
                {"title": "서론", "description": "연구 배경 및 목적"},
                {"title": "관련 연구", "description": "기존 연구 검토"},
                {"title": "방법론", "description": "연구 방법 설명"},
                {"title": "결과", "description": "연구 결과 제시"},
                {"title": "논의", "description": "결과 해석 및 의미"},
                {"title": "결론", "description": "연구 요약 및 향후 방향"}
            ]
            
            return PaperOutline(
                title=f"{topic}에 관한 연구",
                topic=topic,
                main_points=["주제에 대한 기본 연구"],
                sections=default_sections,
                references=[],
                template_name=template_name
            )

    def write_section(
        self, 
        section: Dict[str, Any], 
        research_materials: List[ResearchMaterial],
        outline: PaperOutline,
        previous_sections: Optional[List[PaperSection]] = None
    ) -> PaperSection:
        """
        논문 섹션을 작성합니다.

        Args:
            section (Dict[str, Any]): 섹션 정보
            research_materials (List[ResearchMaterial]): 연구 자료 목록
            outline (PaperOutline): 논문 개요
            previous_sections (Optional[List[PaperSection]], optional): 이전에 작성된 섹션 목록

        Returns:
            PaperSection: 작성된 논문 섹션
        """
        section_title = section["title"]
        logger.info(f"섹션 '{section_title}' 작성 중...")
        
        try:
            # 연구 자료 요약 생성
            research_materials_text = "\n\n".join([
                f"제목: {material.title}\n"
                f"저자: {', '.join(material.authors) if material.authors else '알 수 없음'}\n"
                f"요약: {material.summary}\n"
                f"내용 샘플: {material.content[:500]}..." if material.content else ""
                for material in research_materials[:5]  # 상위 5개 자료만 사용
            ])
            
            # 이전 섹션 정보 생성
            previous_sections_text = ""
            if previous_sections and len(previous_sections) > 0:
                previous_sections_text = "\n\n".join([
                    f"섹션: {prev_section.title}\n내용: {prev_section.content[:300]}..."
                    for prev_section in previous_sections[-2:]  # 최근 2개 섹션만 사용
                ])
            
            # 섹션 정보 준비
            section_info = {
                "title": section_title,
                "description": section.get("description", ""),
                "paper_title": outline.title,
                "paper_topic": outline.topic,
                "main_points": outline.main_points
            }
            
            # 섹션 작성
            format_instructions = self.section_parser.get_format_instructions()
            
            result = self.section_chain.invoke({
                "section_info": json.dumps(section_info, ensure_ascii=False),
                "research_materials": research_materials_text,
                "previous_sections": previous_sections_text,
                "format_instructions": format_instructions
            })
            
            # 결과 파싱
            section_task = self.section_parser.parse(result["text"])
            
            # 인용 참고 문헌 처리
            citations = section_task.citations
            
            # PaperSection 객체 생성
            paper_section = PaperSection(
                title=section_title,
                content=section_task.section_content,
                citations=citations
            )
            
            logger.info(f"섹션 '{section_title}' 작성 완료 ({len(section_task.section_content)} 자)")
            return paper_section
            
        except Exception as e:
            logger.error(f"섹션 '{section_title}' 작성 중 오류 발생: {str(e)}")
            
            # 오류 발생 시 기본 섹션 반환
            return PaperSection(
                title=section_title,
                content=f"이 섹션은 '{section_title}'에 대한 내용을 다룹니다. 작성 중 오류가 발생했습니다.",
                citations=[]
            )

    def edit_paper(self, paper: Paper) -> Paper:
        """
        논문 전체를 편집합니다.

        Args:
            paper (Paper): 편집할 논문

        Returns:
            Paper: 편집된 논문
        """
        logger.info(f"논문 '{paper.title}' 편집 중...")
        
        try:
            # 논문 전체 내용 생성
            paper_content = f"# {paper.title}\n\n"
            
            for section in paper.sections:
                paper_content += f"## {section.title}\n\n{section.content}\n\n"
            
            # 참고 문헌 섹션 추가
            if paper.references:
                paper_content += "## 참고 문헌\n\n"
                for i, ref in enumerate(paper.references, 1):
                    authors = ", ".join(ref.authors) if ref.authors else "알 수 없음"
                    paper_content += f"{i}. {ref.title}. {authors}. {ref.year}. {ref.source}.\n"
            
            # 편집 수행
            format_instructions = self.edit_parser.get_format_instructions()
            
            result = self.edit_chain.invoke({
                "content": paper_content,
                "format_instructions": format_instructions
            })
            
            # 결과 파싱
            edit_task = self.edit_parser.parse(result["text"])
            
            # 편집된 논문 파싱
            edited_paper = self._parse_edited_paper(edit_task.edited_content, paper)
            
            # 변경 사항 기록
            edited_paper.edit_history.append({
                "timestamp": self._get_timestamp(),
                "changes": edit_task.changes_made
            })
            
            logger.info(f"논문 '{paper.title}' 편집 완료 ({len(edit_task.changes_made)}개 변경사항)")
            return edited_paper
            
        except Exception as e:
            logger.error(f"논문 편집 중 오류 발생: {str(e)}")
            
            # 오류 발생 시 원본 논문 반환
            paper.edit_history.append({
                "timestamp": self._get_timestamp(),
                "changes": ["편집 중 오류 발생"]
            })
            return paper

    def edit_section(self, section: PaperSection) -> PaperSection:
        """
        개별 섹션을 편집합니다.

        Args:
            section (PaperSection): 편집할 섹션

        Returns:
            PaperSection: 편집된 섹션
        """
        logger.info(f"섹션 '{section.title}' 편집 중...")
        
        try:
            # 섹션 내용 준비
            section_content = f"## {section.title}\n\n{section.content}"
            
            # 편집 수행
            format_instructions = self.edit_parser.get_format_instructions()
            
            result = self.edit_chain.invoke({
                "content": section_content,
                "format_instructions": format_instructions
            })
            
            # 결과 파싱
            edit_task = self.edit_parser.parse(result["text"])
            
            # 편집된 내용에서 섹션 제목과 내용 추출
            edited_content = edit_task.edited_content
            
            # 제목 추출
            title_match = re.search(r"##\s+(.+?)(?:\n|$)", edited_content)
            edited_title = title_match.group(1).strip() if title_match else section.title
            
            # 내용 추출 (제목 제외)
            content_match = re.search(r"##\s+.+?\n\n([\s\S]+)$", edited_content)
            edited_content = content_match.group(1).strip() if content_match else edited_content
            
            # 편집된 섹션 생성
            edited_section = PaperSection(
                title=edited_title,
                content=edited_content,
                citations=section.citations
            )
            
            logger.info(f"섹션 '{section.title}' 편집 완료")
            return edited_section
            
        except Exception as e:
            logger.error(f"섹션 '{section.title}' 편집 중 오류 발생: {str(e)}")
            
            # 오류 발생 시 원본 섹션 반환
            return section

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
            section_pattern = re.compile(r"##\s+(.+?)\n\n([\s\S]+?)(?=\n##|\n# |$)")
            
            for match in section_pattern.finditer(edited_content):
                section_title = match.group(1).strip()
                section_content = match.group(2).strip()
                
                # 참고 문헌 섹션은 건너뜀
                if "참고 문헌" in section_title or "References" in section_title:
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

    def save_paper(self, paper: Paper, filename: Optional[str] = None) -> str:
        """
        논문을 파일로 저장합니다.

        Args:
            paper (Paper): 저장할 논문
            filename (Optional[str], optional): 파일 이름. 기본값은 None (자동 생성)

        Returns:
            str: 저장된 파일 경로
        """
        # 파일 이름 생성
        if not filename:
            safe_title = re.sub(r'[^\w\s-]', '', paper.title).strip().lower()
            safe_title = re.sub(r'[-\s]+', '-', safe_title)
            filename = f"{safe_title}.md"
        
        # 파일 경로 생성
        file_path = os.path.join(OUTPUT_DIR, filename)
        
        try:
            # 논문 내용 생성
            content = f"# {paper.title}\n\n"
            
            for section in paper.sections:
                content += f"## {section.title}\n\n{section.content}\n\n"
            
            # 참고 문헌 섹션 추가
            if paper.references:
                content += "## 참고 문헌\n\n"
                for i, ref in enumerate(paper.references, 1):
                    authors = ", ".join(ref.authors) if ref.authors else "알 수 없음"
                    content += f"{i}. {ref.title}. {authors}. {ref.year}. {ref.source}.\n"
            
            # 파일 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"논문 '{paper.title}' 저장됨: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"논문 저장 중 오류 발생: {str(e)}")
            return ""

    def _get_timestamp(self) -> str:
        """현재 타임스탬프를 반환합니다."""
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
        작가 에이전트를 실행하고 논문을 생성합니다.

        Args:
            topic (str): 논문 주제
            research_materials (List[ResearchMaterial]): 연구 자료 목록
            template_name (str, optional): 논문 템플릿 이름. 기본값은 "academic"
            config (Optional[RunnableConfig], optional): 실행 구성

        Returns:
            Paper: 생성된 논문
        """
        logger.info(f"작가 에이전트 실행 중: 주제 '{topic}'")
        
        try:
            # 상태 초기화
            self.reset()
            
            # 1. 논문 개요 생성
            outline = self.create_paper_outline(topic, research_materials, template_name)
            
            # 2. 각 섹션 작성
            sections = []
            previous_sections = []
            
            for section_info in outline.sections:
                # 섹션 작성
                section = self.write_section(
                    section=section_info,
                    research_materials=research_materials,
                    outline=outline,
                    previous_sections=previous_sections
                )
                
                sections.append(section)
                previous_sections.append(section)
            
            # 3. 논문 객체 생성
            paper = Paper(
                title=outline.title,
                topic=topic,
                sections=sections,
                references=outline.references,
                template_name=template_name,
                edit_history=[]
            )
            
            # 4. 논문 편집
            edited_paper = self.edit_paper(paper)
            
            # 5. 논문 저장
            file_path = self.save_paper(edited_paper)
            
            # 상태 업데이트
            self.update_state({
                "topic": topic,
                "paper": edited_paper.dict(),
                "output_file": file_path
            })
            
            self.current_paper = edited_paper
            
            logger.info(f"작가 에이전트 실행 완료: 논문 '{edited_paper.title}' 생성됨")
            return edited_paper
            
        except Exception as e:
            logger.error(f"작가 에이전트 실행 중 오류 발생: {str(e)}")
            
            # 오류 발생 시 빈 논문 반환
            empty_paper = Paper(
                title=f"{topic}에 관한 연구",
                topic=topic,
                sections=[
                    PaperSection(
                        title="오류",
                        content="논문 생성 중 오류가 발생했습니다.",
                        citations=[]
                    )
                ],
                references=[],
                template_name=template_name,
                edit_history=[]
            )
            
            return empty_paper 
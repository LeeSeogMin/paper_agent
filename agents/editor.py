"""
편집 에이전트 모듈
논문을 검토하고 편집하는 에이전트입니다.
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
    """편집 작업 형식"""
    edited_content: str = Field(description="편집된 내용")
    changes_made: List[str] = Field(description="적용된 변경사항 목록")


class StyleGuide(BaseModel):
    """스타일 가이드 형식"""
    name: str = Field(description="스타일 가이드 이름")
    rules: List[str] = Field(description="스타일 규칙 목록")
    examples: Dict[str, str] = Field(description="예시 (잘못된 예와 올바른 예)")


class ReviewResult(BaseModel):
    """논문 리뷰 결과 형식"""
    overall_rating: int = Field(description="전체 평가 점수 (1-10)")
    strengths: List[str] = Field(description="논문의 강점")
    weaknesses: List[str] = Field(description="논문의 약점")
    suggestions: List[str] = Field(description="개선 제안")
    grammar_issues: List[str] = Field(description="문법 문제")
    structure_comments: str = Field(description="구조에 대한 의견")


class EditorAgent(BaseAgent[Paper]):
    """논문 편집 전문가 에이전트"""

    def __init__(
        self,
        name: str = "편집 에이전트",
        description: str = "논문 편집 및 검토 전문가",
        verbose: bool = False
    ):
        """
        EditorAgent 초기화

        Args:
            name (str, optional): 에이전트 이름. 기본값은 "편집 에이전트"
            description (str, optional): 에이전트 설명. 기본값은 "논문 편집 및 검토 전문가"
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
        # 편집 작업 파서 초기화
        self.editing_parser = PydanticOutputParser(pydantic_object=EditingTask)
        
        # 리뷰 결과 파서 초기화
        self.review_parser = PydanticOutputParser(pydantic_object=ReviewResult)
        
        # 편집 체인 초기화
        self.editing_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=PAPER_EDITING_PROMPT,
                input_variables=["content", "style_guide", "format_instructions"],
            ),
            verbose=self.verbose
        )
        
        # 리뷰 체인 초기화
        self.review_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=PAPER_REVIEW_PROMPT,
                input_variables=["paper_content", "format_instructions"],
            ),
            verbose=self.verbose
        )
        
        # 참고 문헌 형식 체인 초기화
        self.reference_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=REFERENCE_FORMATTING_PROMPT,
                input_variables=["references", "format_instructions"],
            ),
            verbose=self.verbose
        )
        
        logger.debug("편집 에이전트 프롬프트 및 체인 초기화 완료")

    def edit_section(
        self, 
        section: PaperSection, 
        style_guide: Optional[StyleGuide] = None
    ) -> PaperSection:
        """
        논문 섹션 편집
        
        Args:
            section: 편집할 섹션
            style_guide: 적용할 스타일 가이드
            
        Returns:
            PaperSection: 편집된 섹션
        """
        section_title = section.title
        section_content = section.content
        
        logger.info(f"섹션 편집 중: {section_title}")
        
        # 스타일 가이드 정보
        style_guide_text = ""
        if style_guide:
            style_guide_text = (
                f"스타일 가이드: {style_guide.name}\n"
                f"규칙:\n" + "\n".join([f"- {rule}" for rule in style_guide.rules]) + "\n"
            )
        
        # 프롬프트 준비
        messages = [
            {"role": "user", "content": (
                f"다음 논문 섹션을 편집해 주세요:\n\n"
                f"제목: {section_title}\n\n"
                f"내용:\n{section_content}\n\n"
                f"{style_guide_text}\n\n"
                "문법, 맞춤법, 문장 구조, 명확성을 개선하고, 학술적 문체를 유지하세요."
            )}
        ]
        
        # LLM으로 편집
        response = self.editing_chain.invoke({"content": section_content, "style_guide": style_guide_text})
        chain = response | self.llm | self.editing_parser
        
        try:
            edit_result = chain.invoke({})
            
            # 편집된 섹션 반환
            edited_section = PaperSection(
                title=section_title,
                content=edit_result.edited_content,
                citations=section.citations
            )
            
            logger.info(f"섹션 편집 완료: {section_title}, {len(edit_result.changes_made)}개 변경사항")
            return edited_section
            
        except Exception as e:
            logger.error(f"섹션 편집 중 오류: {str(e)}", error=str(e))
            return section  # 오류 발생 시 원본 섹션 반환
    
    def edit_paper(
        self, 
        paper: Paper, 
        style_guide: Optional[StyleGuide] = None
    ) -> Paper:
        """
        전체 논문 편집
        
        Args:
            paper: 편집할 논문
            style_guide: 적용할 스타일 가이드
            
        Returns:
            Paper: 편집된 논문
        """
        logger.info(f"논문 편집 중: {paper.title}")
        
        # 섹션별 편집
        edited_sections = []
        for section in paper.sections:
            edited_section = self.edit_section(section, style_guide)
            edited_sections.append(edited_section)
        
        # 초록 편집
        abstract_section = PaperSection(
            title="Abstract",
            content=paper.abstract,
            citations=[]
        )
        edited_abstract = self.edit_section(abstract_section, style_guide)
        
        # 편집된 논문 생성
        edited_paper = Paper(
            title=paper.title,
            abstract=edited_abstract.content,
            sections=edited_sections,
            references=paper.references,
            keywords=paper.keywords
        )
        
        logger.info(f"논문 편집 완료: {paper.title}")
        return edited_paper
    
    def review_paper(self, paper: Paper) -> ReviewResult:
        """
        논문 리뷰 및 피드백 생성
        
        Args:
            paper: 리뷰할 논문
            
        Returns:
            ReviewResult: 논문 리뷰 피드백
        """
        logger.info(f"논문 리뷰 중: {paper.title}")
        
        # 논문 전체 텍스트
        paper_text = paper.get_full_text()
        
        # 텍스트가 너무 길면 섹션 요약 사용
        if len(paper_text) > 8000:
            paper_text = (
                f"제목: {paper.title}\n\n"
                f"초록: {paper.abstract}\n\n"
            )
            
            # 섹션 요약 추가
            for section in paper.sections:
                # 각 섹션의 처음 부분만 포함
                content_preview = section.content[:500] + "..." if len(section.content) > 500 else section.content
                paper_text += f"## {section.title}\n\n{content_preview}\n\n"
            
            # 참고 문헌 정보 추가
            paper_text += f"참고 문헌 수: {len(paper.references)}\n"
        
        # 프롬프트 준비
        messages = [
            {"role": "user", "content": (
                f"다음 논문을 리뷰하고 피드백을 제공해 주세요:\n\n{paper_text}"
            )}
        ]
        
        # LLM으로 리뷰 생성
        response = self.review_chain.invoke({"paper_content": paper_text})
        chain = response | self.llm | self.review_parser
        
        try:
            review_result = chain.invoke({})
            logger.info(f"논문 리뷰 완료: {paper.title}, 평가: {review_result.overall_rating}/10")
            return review_result
            
        except Exception as e:
            logger.error(f"논문 리뷰 중 오류: {str(e)}", error=str(e))
            
            # 오류 시 기본 리뷰 반환
            return ReviewResult(
                strengths=["리뷰 생성 중 오류가 발생했습니다."],
                weaknesses=["리뷰 생성 중 오류가 발생했습니다."],
                suggestions=["논문을 다시 검토하거나 리뷰 프로세스를 재시도하세요."],
                overall_rating=5
            )
    
    def format_paper(
        self, 
        paper: Paper, 
        style_guide: StyleGuide,
        output_format: str = "markdown"
    ) -> str:
        """
        논문을 지정된 형식으로 포맷팅
        
        Args:
            paper: 포맷팅할 논문
            style_guide: 적용할 스타일 가이드
            output_format: 출력 형식 (markdown, latex, docx)
            
        Returns:
            str: 포맷팅된 논문 내용 또는 파일 경로
        """
        logger.info(f"논문 포맷팅 중: {paper.title}, 형식: {output_format}")
        
        if output_format.lower() == "markdown":
            return self._format_markdown(paper, style_guide)
        elif output_format.lower() == "latex":
            return self._format_latex(paper, style_guide)
        else:
            logger.warning(f"지원되지 않는 출력 형식: {output_format}, markdown으로 대체")
            return self._format_markdown(paper, style_guide)
    
    def _format_markdown(self, paper: Paper, style_guide: StyleGuide) -> str:
        """
        논문을 마크다운 형식으로 포맷팅
        
        Args:
            paper: 포맷팅할 논문
            style_guide: 적용할 스타일 가이드
            
        Returns:
            str: 마크다운 형식의 논문 내용
        """
        # 파일 이름 생성
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in paper.title)
        safe_title = safe_title.replace(" ", "_").lower()
        filename = f"{safe_title}_formatted.md"
        file_path = os.path.join(OUTPUT_DIR, filename)
        
        # 마크다운 내용 생성
        md_content = f"# {paper.title}\n\n"
        
        # 키워드
        if paper.keywords:
            md_content += "**키워드**: " + ", ".join(paper.keywords) + "\n\n"
        
        # 초록
        md_content += "## 초록\n\n"
        md_content += paper.abstract + "\n\n"
        
        # 섹션
        for section in paper.sections:
            md_content += f"## {section.title}\n\n"
            md_content += section.content + "\n\n"
        
        # 참고 문헌
        if paper.references:
            md_content += "## 참고 문헌\n\n"
            
            # 스타일 가이드에 따라 인용 형식 적용
            citation_style = style_guide.rules[0].lower()
            
            for i, ref in enumerate(paper.references, 1):
                authors = ref.authors if isinstance(ref.authors, str) else ", ".join(ref.authors)
                
                if citation_style == "apa":
                    # APA 스타일
                    md_content += f"{authors} ({ref.year}). {ref.title}. "
                    if ref.url:
                        md_content += f"Retrieved from {ref.url}"
                    md_content += "\n\n"
                    
                elif citation_style == "mla":
                    # MLA 스타일
                    md_content += f"{authors}. \"{ref.title}.\" {ref.year}. "
                    if ref.url:
                        md_content += f"Web. {ref.url}"
                    md_content += "\n\n"
                    
                elif citation_style == "chicago":
                    # Chicago 스타일
                    md_content += f"{authors}. {ref.title}. {ref.year}. "
                    if ref.url:
                        md_content += f"{ref.url}."
                    md_content += "\n\n"
                    
                else:
                    # 기본 형식
                    md_content += f"{i}. {authors} ({ref.year}). {ref.title}. "
                    if ref.url:
                        md_content += f"URL: {ref.url}"
                    md_content += "\n\n"
        
        # 파일로 저장
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        
        logger.info(f"마크다운 형식으로 포맷팅 완료: {file_path}")
        return file_path
    
    def _format_latex(self, paper: Paper, style_guide: StyleGuide) -> str:
        """
        논문을 LaTeX 형식으로 포맷팅
        
        Args:
            paper: 포맷팅할 논문
            style_guide: 적용할 스타일 가이드
            
        Returns:
            str: LaTeX 형식의 논문 내용
        """
        # 파일 이름 생성
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in paper.title)
        safe_title = safe_title.replace(" ", "_").lower()
        filename = f"{safe_title}_formatted.tex"
        file_path = os.path.join(OUTPUT_DIR, filename)
        
        # LaTeX 헤더
        latex_content = (
            "\\documentclass{article}\n"
            "\\usepackage[utf8]{inputenc}\n"
            "\\usepackage{hyperref}\n"
            "\\usepackage{natbib}\n"
            "\\usepackage{graphicx}\n"
            "\\usepackage{geometry}\n"
            "\\geometry{a4paper, margin=1in}\n\n"
            f"\\title{{{paper.title}}}\n"
            "\\author{}\n"
            "\\date{\\today}\n\n"
            "\\begin{document}\n\n"
            "\\maketitle\n\n"
        )
        
        # 키워드
        if paper.keywords:
            latex_content += "\\textbf{Keywords:} " + ", ".join(paper.keywords) + "\n\n"
        
        # 초록
        latex_content += "\\begin{abstract}\n"
        latex_content += paper.abstract + "\n"
        latex_content += "\\end{abstract}\n\n"
        
        # 섹션
        for section in paper.sections:
            latex_content += f"\\section{{{section.title}}}\n\n"
            # LaTeX 특수 문자 이스케이프
            content = section.content.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")
            latex_content += content + "\n\n"
        
        # 참고 문헌
        if paper.references:
            latex_content += "\\begin{thebibliography}{99}\n\n"
            
            for i, ref in enumerate(paper.references, 1):
                authors = ref.authors if isinstance(ref.authors, str) else " and ".join(ref.authors)
                label = f"ref{i}"
                
                latex_content += f"\\bibitem[{label}]{{{label}}}\n"
                latex_content += f"{authors} ({ref.year}). {ref.title}. "
                
                if ref.url:
                    latex_content += f"\\url{{{ref.url}}}"
                
                latex_content += "\n\n"
            
            latex_content += "\\end{thebibliography}\n\n"
        
        # LaTeX 푸터
        latex_content += "\\end{document}"
        
        # 파일로 저장
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(latex_content)
        
        logger.info(f"LaTeX 형식으로 포맷팅 완료: {file_path}")
        return file_path
    
    def run(
        self, 
        paper: Paper,
        style_guide: Optional[StyleGuide] = None,
        output_format: str = "markdown",
        config: Optional[RunnableConfig] = None
    ) -> Paper:
        """
        편집 에이전트 실행
        
        Args:
            paper: 편집할 논문
            style_guide: 적용할 스타일 가이드
            output_format: 출력 형식
            config: 실행 설정
            
        Returns:
            Paper: 편집된 논문
        """
        logger.info(f"편집 에이전트 실행 중: {paper.title}")
        
        # 기본 스타일 가이드 설정
        if not style_guide:
            style_guide = StyleGuide(
                name="Standard Academic",
                rules=[
                    "명확하고 간결한 문장 사용",
                    "학술적 어조 유지",
                    "능동태 우선 사용",
                    "일관된 용어 사용"
                ],
                examples={
                    "잘못된 예": "이 논문은 명확하고 간결한 문장을 사용하여 작성되었습니다.",
                    "올바른 예": "이 논문은 학술적 어조를 유지하면서 작성되었습니다."
                }
            )
        
        # 1. 논문 편집
        edited_paper = self.edit_paper(paper, style_guide)
        
        # 2. 논문 리뷰
        review = self.review_paper(edited_paper)
        
        # 리뷰 결과를 논문 상태에 저장
        edited_paper.metadata = edited_paper.metadata or {}
        edited_paper.metadata["review"] = review.dict()
        
        # 3. 논문 포맷팅
        formatted_file = self.format_paper(edited_paper, style_guide, output_format)
        edited_paper.metadata["formatted_file"] = formatted_file
        
        logger.info(f"편집 에이전트 실행 완료: {paper.title}, 평점: {review.overall_rating}/10")
        return edited_paper 
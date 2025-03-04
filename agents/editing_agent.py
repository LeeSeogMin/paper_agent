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
                input_variables=["references", "citation_style"],
            ),
            verbose=self.verbose
        )
        
        logger.debug("편집 에이전트 프롬프트 및 체인 초기화 완료")

    def create_style_guide(self, style_name: str = "Standard Academic") -> StyleGuide:
        """
        스타일 가이드를 생성합니다.

        Args:
            style_name (str, optional): 스타일 가이드 이름. 기본값은 "Standard Academic"

        Returns:
            StyleGuide: 생성된 스타일 가이드
        """
        logger.info(f"'{style_name}' 스타일 가이드 생성 중...")
        
        # 기본 학술 스타일 가이드
        if style_name == "Standard Academic":
            return StyleGuide(
                name="Standard Academic",
                rules=[
                    "명확하고 간결한 문장 사용",
                    "능동태 선호 (수동태 최소화)",
                    "일관된 시제 사용 (주로 현재 시제)",
                    "첫 사용 시 약어 정의",
                    "객관적인 어조 유지",
                    "주장에 대한 증거 제시",
                    "적절한 인용 사용",
                    "일인칭 사용 최소화",
                    "전문 용어 적절히 사용",
                    "단락 간 논리적 흐름 유지"
                ],
                examples={
                    "수동태": "연구가 수행되었다. → 연구자들이 연구를 수행했다.",
                    "모호한 표현": "이것은 중요하다. → 이 발견은 치료법 개발에 중요하다.",
                    "약어": "AI는 유용하다. → 인공지능(AI)은 유용하다.",
                    "주관적 표현": "놀라운 결과이다. → 결과는 기존 연구와 15% 차이를 보인다."
                }
            )
        
        # APA 스타일 가이드
        elif style_name == "APA":
            return StyleGuide(
                name="APA Style",
                rules=[
                    "명확하고 간결한 문장 사용",
                    "능동태 선호",
                    "과거 시제 사용 (연구 방법 및 결과 설명 시)",
                    "현재 시제 사용 (결론 및 확립된 지식 설명 시)",
                    "성별 중립적 언어 사용",
                    "정확한 용어 사용",
                    "약어 사용 시 첫 번째 언급에서 정의",
                    "숫자 10 이상은 숫자로 표기",
                    "문장 시작 시 숫자는 단어로 표기",
                    "직접 인용 시 페이지 번호 포함"
                ],
                examples={
                    "수동태": "실험이 수행되었다. → 연구자들이 실험을 수행했다.",
                    "성별 편향": "각 참가자는 그의 응답을 제출했다. → 각 참가자는 자신의 응답을 제출했다.",
                    "시제 불일치": "데이터는 수집되었고 분석된다. → 데이터는 수집되었고 분석되었다.",
                    "숫자 표기": "실험에 5명의 참가자가 참여했다. → 실험에 다섯 명의 참가자가 참여했다."
                }
            )
        
        # MLA 스타일 가이드
        elif style_name == "MLA":
            return StyleGuide(
                name="MLA Style",
                rules=[
                    "현재 시제 사용 (문학 작품 논의 시)",
                    "명확하고 간결한 문장 사용",
                    "능동태 선호",
                    "일인칭 사용 가능 (적절한 경우)",
                    "직접 인용 시 페이지 번호 포함",
                    "작품 제목은 이탤릭체 또는 따옴표로 표시",
                    "약어 사용 최소화",
                    "문학적 현재 시제 사용",
                    "정확한 인용 사용",
                    "논리적 단락 구성"
                ],
                examples={
                    "시제": "작가는 말했다. → 작가는 말한다.",
                    "제목 표기": "소설 '전쟁과 평화' → 소설 『전쟁과 평화』",
                    "인용": "이것은 중요하다(스미스). → 스미스는 \"이것이 중요하다\"고 주장한다(42).",
                    "약어": "등. → 기타 등등"
                }
            )
        
        # Chicago 스타일 가이드
        elif style_name == "Chicago":
            return StyleGuide(
                name="Chicago Style",
                rules=[
                    "명확하고 간결한 문장 사용",
                    "능동태 선호",
                    "일관된 시제 사용",
                    "약어 사용 시 첫 번째 언급에서 정의",
                    "숫자 100 이하는 단어로 표기 (특정 예외 있음)",
                    "직접 인용 시 페이지 번호 포함",
                    "각주 또는 미주 사용",
                    "작품 제목은 이탤릭체 또는 따옴표로 표시",
                    "정확한 인용 사용",
                    "논리적 단락 구성"
                ],
                examples={
                    "숫자 표기": "42개의 샘플 → 사십이 개의 샘플",
                    "인용": "스미스는 중요하다고 말했다. → 스미스는 \"이것이 중요하다\"고 말했다.¹",
                    "제목 표기": "논문 '인공지능의 미래' → 논문 \"인공지능의 미래\"",
                    "약어": "WHO는 → 세계보건기구(WHO)는"
                }
            )
        
        # 기본 스타일 가이드 반환
        else:
            logger.warning(f"알 수 없는 스타일 가이드 '{style_name}'. 기본 학술 스타일 가이드를 사용합니다.")
            return self.create_style_guide("Standard Academic")

    def edit_paper(self, paper: Paper, style_guide: StyleGuide) -> Paper:
        """
        논문을 편집합니다.

        Args:
            paper (Paper): 편집할 논문
            style_guide (StyleGuide): 적용할 스타일 가이드

        Returns:
            Paper: 편집된 논문
        """
        logger.info(f"논문 '{paper.title}' 편집 중...")
        
        try:
            # 논문 전체 내용 생성
            paper_content = f"# {paper.title}

"
            
            for section in paper.sections:
                paper_content += f"## {section.title}

{section.content}

"
            
            # 참고 문헌 섹션 추가
            if paper.references:
                paper_content += "## 참고 문헌

"
                for i, ref in enumerate(paper.references, 1):
                    authors = ", ".join(ref.authors) if ref.authors else "알 수 없음"
                    paper_content += f"{i}. {ref.title}. {authors}. {ref.year}. {ref.source}.
"
            
            # 스타일 가이드 정보 준비
            style_guide_info = json.dumps(style_guide.dict(), ensure_ascii=False)
            
            # 편집 수행
            format_instructions = self.editing_parser.get_format_instructions()
            
            result = self.editing_chain.invoke({
                "content": paper_content,
                "style_guide": style_guide_info,
                "format_instructions": format_instructions
            })
            
            # 결과 파싱
            edit_task = self.editing_parser.parse(result["text"])
            
            # 편집된 논문 파싱
            edited_paper = self._parse_edited_paper(edit_task.edited_content, paper)
            
            # 변경 사항 기록
            edited_paper.edit_history.append({
                "timestamp": self._get_timestamp(),
                "changes": edit_task.changes_made,
                "style_guide": style_guide.name
            })
            
            logger.info(f"논문 '{paper.title}' 편집 완료 ({len(edit_task.changes_made)}개 변경사항)")
            return edited_paper
            
        except Exception as e:
            logger.error(f"논문 편집 중 오류 발생: {str(e)}")
            
            # 오류 발생 시 원본 논문 반환
            paper.edit_history.append({
                "timestamp": self._get_timestamp(),
                "changes": ["편집 중 오류 발생"],
                "style_guide": style_guide.name
            })
            return paper

    def review_paper(self, paper: Paper) -> Dict[str, Any]:
        """
        논문을 검토하고 피드백을 제공합니다.

        Args:
            paper (Paper): 검토할 논문

        Returns:
            Dict[str, Any]: 검토 결과
        """
        logger.info(f"논문 '{paper.title}' 검토 중...")
        
        try:
            # 논문 전체 내용 생성
            paper_content = f"# {paper.title}

"
            
            for section in paper.sections:
                paper_content += f"## {section.title}

{section.content}

"
            
            # 참고 문헌 섹션 추가
            if paper.references:
                paper_content += "## 참고 문헌

"
                for i, ref in enumerate(paper.references, 1):
                    authors = ", ".join(ref.authors) if ref.authors else "알 수 없음"
                    paper_content += f"{i}. {ref.title}. {authors}. {ref.year}. {ref.source}.
"
            
            # 검토 수행
            format_instructions = self.review_parser.get_format_instructions()
            
            result = self.review_chain.invoke({
                "paper_content": paper_content,
                "format_instructions": format_instructions
            })
            
            # 결과 파싱
            review_result = self.review_parser.parse(result["text"])
            
            # 검토 결과를 딕셔너리로 변환
            review_dict = review_result.dict()
            
            logger.info(f"논문 '{paper.title}' 검토 완료 (평가: {review_result.overall_rating}/10)")
            return review_dict
            
        except Exception as e:
            logger.error(f"논문 검토 중 오류 발생: {str(e)}")
            
            # 오류 발생 시 기본 검토 결과 반환
            return {
                "overall_rating": 5,
                "strengths": ["검토 중 오류 발생"],
                "weaknesses": ["검토를 완료할 수 없음"],
                "suggestions": ["시스템 오류를 확인하세요"],
                "grammar_issues": [],
                "structure_comments": "검토 중 오류가 발생했습니다."
            }

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
            for line in result["text"].strip().split("
"):
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
                content = f"# {paper.title}

"
                
                for section in paper.sections:
                    content += f"## {section.title}

{section.content}

"
                
                # 참고 문헌 섹션 추가
                if paper.references:
                    content += "## 참고 문헌

"
                    for i, ref in enumerate(paper.references, 1):
                        authors = ", ".join(ref.authors) if ref.authors else "알 수 없음"
                        content += f"{i}. {ref.title}. {authors}. {ref.year}. {ref.source}.
"
                
                # 파일 저장
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logger.info(f"논문 '{paper.title}' 마크다운 형식으로 저장됨: {file_path}")
                return file_path
                
            except Exception as e:
                logger.error(f"논문 저장 중 오류 발생: {str(e)}")
                return ""
                
        elif output_format == "latex":
            filename = f"{safe_title}.tex"
            file_path = os.path.join(OUTPUT_DIR, filename)
            
            try:
                # LaTeX 형식으로 논문 내용 생성
                content = "\documentclass{article}
"
                content += "\usepackage[utf8]{inputenc}
"
                content += "\usepackage{natbib}
"
                content += "\usepackage{graphicx}
"
                content += "\usepackage{hyperref}

"
                
                content += f"\title{{{paper.title}}}
"
                
                # 저자 정보 (없으므로 기본값 사용)
                content += "\author{AI Paper Writer}
"
                content += "\date{\today}

"
                
                content += "\begin{document}

"
                content += "\maketitle

"
                
                # 초록 섹션 (있는 경우)
                abstract_section = next((s for s in paper.sections if s.title.lower() == "초록" or s.title.lower() == "abstract"), None)
                if abstract_section:
                    content += "\begin{abstract}
"
                    content += abstract_section.content + "
"
                    content += "\end{abstract}

"
                
                # 목차
                content += "\tableofcontents
\newpage

"
                
                # 각 섹션
                for section in paper.sections:
                    # 초록은 이미 처리했으므로 건너뜀
                    if section.title.lower() == "초록" or section.title.lower() == "abstract":
                        continue
                        
                    content += f"\section{{{section.title}}}

"
                    
                    # LaTeX 특수 문자 이스케이프
                    section_content = section.content
                    section_content = section_content.replace("_", "\_")
                    section_content = section_content.replace("%", "\%")
                    section_content = section_content.replace("&", "\&")
                    section_content = section_content.replace("#", "\#")
                    
                    content += section_content + "

"
                
                # 참고 문헌
                if paper.references:
                    content += "\begin{thebibliography}{99}

"
                    
                    for i, ref in enumerate(paper.references, 1):
                        authors = ", ".join(ref.authors) if ref.authors else "Unknown"
                        content += f"\bibitem{{{i}}}
"
                        content += f"{authors} ({ref.year}). {ref.title}. {ref.source}.

"
                    
                    content += "\end{thebibliography}

"
                
                content += "\end{document}"
                
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
            title_match = re.search(r"#\s+(.+?)(?:
|$)", edited_content)
            title = title_match.group(1).strip() if title_match else original_paper.title
            
            # 섹션 추출
            sections = []
            section_pattern = re.compile(r"##\s+(.+?)

([\s\S]+?)(?=
##|
# |$)")
            
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

    def _get_timestamp(self) -> str:
        """현재 타임스탬프를 반환합니다."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def run(
        self, 
        paper: Paper,
        style_guide_name: str = "Standard Academic",
        citation_style: str = "APA",
        output_format: str = "markdown",
        config: Optional[RunnableConfig] = None
    ) -> Paper:
        """
        편집 에이전트를 실행하고 편집된 논문을 반환합니다.

        Args:
            paper (Paper): 편집할 논문
            style_guide_name (str, optional): 스타일 가이드 이름. 기본값은 "Standard Academic"
            citation_style (str, optional): 인용 스타일. 기본값은 "APA"
            output_format (str, optional): 출력 형식. 기본값은 "markdown"
            config (Optional[RunnableConfig], optional): 실행 구성

        Returns:
            Paper: 편집된 논문
        """
        logger.info(f"편집 에이전트 실행 중: 논문 '{paper.title}'")
        
        try:
            # 상태 초기화
            self.reset()
            
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

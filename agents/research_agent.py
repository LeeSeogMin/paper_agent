"""
연구 에이전트 모듈
논문 주제에 대한 연구를 수행하고 관련 자료를 수집하는 에이전트입니다.
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

from config.settings import RESEARCH_DIR, MAX_RESEARCH_DEPTH
from config.api_keys import OPENAI_API_KEY, SERP_API_KEY
from utils.logger import logger
from utils.api_clients import GoogleSearchAPIClient
from utils.pdf_processor import PDFProcessor
from models.research import ResearchMaterial, Reference
from prompts.paper_prompts import (
    RESEARCH_PLAN_PROMPT,
    RESEARCH_QUESTION_PROMPT,
    PAPER_SUMMARY_PROMPT
)
from agents.base import BaseAgent


class SearchQuery(BaseModel):
    """검색 쿼리 형식"""
    query: str = Field(description="검색할 자세한 쿼리")
    rationale: str = Field(description="이 검색을 수행하는 이유")


class SearchResult(BaseModel):
    """검색 결과 형식"""
    query: str = Field(description="수행된 검색 쿼리")
    references: List[Reference] = Field(description="검색에서 찾은 참고 문헌 목록")
    summary: str = Field(description="검색 결과에 대한 간략한 요약")


class ResearchPlan(BaseModel):
    """연구 계획 형식"""
    topic: str = Field(description="연구 주제")
    research_questions: List[str] = Field(description="연구를 통해 답하려는 질문들")
    search_queries: List[SearchQuery] = Field(description="수행할 검색 쿼리 목록")
    rationale: str = Field(description="이 연구 계획을 수립한 이유")


class ResearchSummary(BaseModel):
    """연구 요약 형식"""
    topic: str = Field(description="연구 주제")
    key_findings: List[str] = Field(description="주요 연구 결과")
    collected_materials: List[ResearchMaterial] = Field(description="수집된 연구 자료")
    gaps: List[str] = Field(description="식별된 연구 격차")
    next_steps: List[str] = Field(description="권장되는 다음 단계")


class ResearchAgent(BaseAgent[ResearchSummary]):
    """논문 주제에 대한 연구를 수행하고 관련 자료를 수집하는 에이전트"""

    def __init__(
        self,
        name: str = "연구 에이전트",
        description: str = "논문 주제에 대한 연구 수행 및 관련 자료 수집",
        verbose: bool = False
    ):
        """
        ResearchAgent 초기화

        Args:
            name (str, optional): 에이전트 이름. 기본값은 "연구 에이전트"
            description (str, optional): 에이전트 설명. 기본값은 "논문 주제에 대한 연구 수행 및 관련 자료 수집"
            verbose (bool, optional): 상세 로깅 활성화 여부. 기본값은 False
        """
        super().__init__(name, description, verbose=verbose)
        
        # PDF 처리기 초기화
        self.pdf_processor = PDFProcessor(verbose=verbose)
        
        # 검색 API 클라이언트 초기화
        if SERP_API_KEY:
            self.search_client = GoogleSearchAPIClient(api_key=SERP_API_KEY)
        else:
            self.search_client = None
            logger.warning("SERP_API_KEY가 설정되지 않았습니다. 검색 기능이 제한됩니다.")
        
        # 연구 디렉토리 생성
        os.makedirs(RESEARCH_DIR, exist_ok=True)
        
        # 프롬프트 초기화
        self._init_prompts()
        
        logger.info(f"{self.name} 초기화 완료")

    def _init_prompts(self) -> None:
        """프롬프트와 체인 초기화"""
        # 연구 계획 파서 초기화
        self.research_plan_parser = PydanticOutputParser(pydantic_object=ResearchPlan)
        
        # 검색 결과 파서 초기화
        self.search_result_parser = PydanticOutputParser(pydantic_object=SearchResult)
        
        # 연구 요약 파서 초기화
        self.research_summary_parser = PydanticOutputParser(pydantic_object=ResearchSummary)
        
        # 연구 계획 체인 초기화
        self.research_plan_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=RESEARCH_PLAN_PROMPT,
                input_variables=["topic", "format_instructions"],
            ),
            verbose=self.verbose
        )
        
        # 연구 질문 체인 초기화
        self.research_question_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=RESEARCH_QUESTION_PROMPT,
                input_variables=["topic", "format_instructions"],
            ),
            verbose=self.verbose
        )
        
        # 요약 체인 초기화
        self.summary_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template=PAPER_SUMMARY_PROMPT,
                input_variables=["content"],
            ),
            verbose=self.verbose
        )
        
        logger.debug("연구 에이전트 프롬프트 및 체인 초기화 완료")

    def create_research_plan(self, topic: str) -> ResearchPlan:
        """
        주어진 주제에 대한 연구 계획을 생성합니다.

        Args:
            topic (str): 연구 주제

        Returns:
            ResearchPlan: 생성된 연구 계획
        """
        logger.info(f"주제 '{topic}'에 대한 연구 계획 생성 중...")
        
        try:
            format_instructions = self.research_plan_parser.get_format_instructions()
            
            result = self.research_plan_chain.invoke({
                "topic": topic,
                "format_instructions": format_instructions
            })
            
            plan = self.research_plan_parser.parse(result["text"])
            
            logger.info(f"연구 계획 생성 완료: {len(plan.search_queries)}개의 검색 쿼리 포함")
            return plan
            
        except Exception as e:
            logger.error(f"연구 계획 생성 중 오류 발생: {str(e)}")
            # 오류 발생 시 기본 계획 반환
            return ResearchPlan(
                topic=topic,
                research_questions=["이 주제에 대한 주요 개념은 무엇인가?"],
                search_queries=[
                    SearchQuery(
                        query=f"{topic} overview research papers",
                        rationale="주제에 대한 기본적인 이해를 위해"
                    )
                ],
                rationale="기본 연구 계획"
            )

    def execute_search(self, query: SearchQuery) -> SearchResult:
        """
        검색 쿼리를 실행하고 결과를 반환합니다.

        Args:
            query (SearchQuery): 검색 쿼리

        Returns:
            SearchResult: 검색 결과
        """
        logger.info(f"검색 실행 중: '{query.query}'")
        
        if not self.search_client:
            logger.warning("검색 클라이언트가 초기화되지 않았습니다. 가상의 결과를 반환합니다.")
            return SearchResult(
                query=query.query,
                references=[],
                summary="검색 API가 구성되지 않아 검색할 수 없습니다."
            )
        
        try:
            # 검색 실행
            search_results = self.search_client.search(
                query=query.query,
                num_results=5,
                include_domains=["scholar.google.com", "arxiv.org", "researchgate.net", "academia.edu"]
            )
            
            # 결과 변환
            references = []
            for result in search_results:
                ref = Reference(
                    title=result.get("title", "제목 없음"),
                    url=result.get("link", ""),
                    authors=result.get("authors", "").split(", ") if "authors" in result else [],
                    year=result.get("year", ""),
                    source=result.get("source", "웹 검색"),
                    summary=result.get("snippet", "요약 없음")
                )
                references.append(ref)
            
            # 결과 요약
            summary = f"'{query.query}'에 대한 검색에서 {len(references)}개의 결과를 찾았습니다."
            
            logger.info(f"검색 완료: {len(references)}개의 결과 발견")
            return SearchResult(
                query=query.query,
                references=references,
                summary=summary
            )
            
        except Exception as e:
            logger.error(f"검색 중 오류 발생: {str(e)}")
            return SearchResult(
                query=query.query,
                references=[],
                summary=f"검색 중 오류 발생: {str(e)}"
            )

    def download_and_process_papers(self, references: List[Reference]) -> List[ResearchMaterial]:
        """
        참고 문헌 목록에서 논문을 다운로드하고 처리합니다.

        Args:
            references (List[Reference]): 참고 문헌 목록

        Returns:
            List[ResearchMaterial]: 처리된 연구 자료 목록
        """
        logger.info(f"{len(references)}개의 참고 문헌 다운로드 및 처리 중...")
        materials = []
        
        for ref in references:
            try:
                # PDF URL인 경우에만 처리
                if ref.url.endswith('.pdf') or 'pdf' in ref.url:
                    logger.info(f"PDF 다운로드 중: {ref.title}")
                    
                    # PDF 다운로드 및 처리
                    output_path = os.path.join(RESEARCH_DIR, f"{ref.title[:50].replace(' ', '_')}.pdf")
                    
                    # PDF 처리
                    file_path, text, chunks, metadata = self.pdf_processor.process_url(
                        url=ref.url,
                        output_path=output_path
                    )
                    
                    # 텍스트 요약
                    summary = self._generate_summary(chunks[:3]) if chunks else "요약 없음"
                    
                    # 연구 자료 생성
                    material = ResearchMaterial(
                        title=ref.title,
                        authors=ref.authors,
                        year=ref.year,
                        source=ref.source,
                        url=ref.url,
                        local_path=file_path,
                        content=text[:10000] if text else "",  # 처음 10000자만 저장
                        summary=summary,
                        metadata=metadata
                    )
                    
                    materials.append(material)
                    logger.info(f"PDF 처리 완료: {ref.title}")
                else:
                    # PDF가 아닌 경우 참고 정보만 기록
                    logger.debug(f"PDF URL이 아닙니다: {ref.url}")
                    material = ResearchMaterial(
                        title=ref.title,
                        authors=ref.authors,
                        year=ref.year,
                        source=ref.source,
                        url=ref.url,
                        local_path="",
                        content="",
                        summary=ref.summary,
                        metadata={}
                    )
                    materials.append(material)
            except Exception as e:
                logger.error(f"참고 문헌 처리 중 오류 발생: {str(e)}")
        
        logger.info(f"{len(materials)}개의 연구 자료 처리 완료")
        return materials

    def _generate_summary(self, text_chunks: List[str]) -> str:
        """
        텍스트 청크 목록에서 요약을 생성합니다.

        Args:
            text_chunks (List[str]): 텍스트 청크 목록

        Returns:
            str: 생성된 요약
        """
        try:
            # 청크를 결합하고 최대 8000자로 제한
            combined_text = "\n\n".join(text_chunks)
            combined_text = combined_text[:8000] + "..." if len(combined_text) > 8000 else combined_text
            
            # 요약 생성
            result = self.summary_chain.invoke({"content": combined_text})
            summary = result["text"]
            
            return summary
        except Exception as e:
            logger.error(f"요약 생성 중 오류 발생: {str(e)}")
            return "요약을 생성할 수 없습니다."

    def search_related_materials(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        주어진 쿼리와 관련된 자료를 검색합니다.

        Args:
            query (str): 검색 쿼리
            limit (int, optional): 결과 제한. 기본값은 5

        Returns:
            List[Dict[str, Any]]: 검색 결과 목록
        """
        if not self.search_client:
            logger.warning("검색 클라이언트가 초기화되지 않았습니다.")
            return []
        
        try:
            logger.info(f"'{query}' 관련 자료 검색 중...")
            
            # 학술 검색 실행
            search_query = f"{query} research papers"
            results = self.search_client.search(
                query=search_query,
                num_results=limit,
                include_domains=["scholar.google.com", "arxiv.org", "researchgate.net", "academia.edu"]
            )
            
            processed_results = []
            for result in results:
                item = {
                    "title": result.get("title", "제목 없음"),
                    "url": result.get("link", ""),
                    "snippet": result.get("snippet", ""),
                    "source": result.get("source", "학술 검색")
                }
                processed_results.append(item)
            
            logger.info(f"{len(processed_results)}개의 관련 자료 찾음")
            return processed_results
            
        except Exception as e:
            logger.error(f"관련 자료 검색 중 오류 발생: {str(e)}")
            return []

    def generate_research_summary(self, topic: str) -> ResearchSummary:
        """
        주제에 대한 연구를 수행하고 요약을 생성합니다.

        Args:
            topic (str): 연구 주제

        Returns:
            ResearchSummary: 생성된 연구 요약
        """
        logger.info(f"주제 '{topic}'에 대한 연구 요약 생성 중...")
        
        try:
            # 1. 연구 계획 생성
            plan = self.create_research_plan(topic)
            
            # 2. 검색 쿼리 실행
            all_references = []
            for query in plan.search_queries[:3]:  # 최대 3개 쿼리만 처리
                search_result = self.execute_search(query)
                all_references.extend(search_result.references)
            
            # 3. 논문 다운로드 및 처리
            materials = self.download_and_process_papers(all_references[:5])  # 최대 5개 참고 문헌만 처리
            
            # 4. 추가 정보 검색
            additional_questions = []
            for question in plan.research_questions[:2]:  # 최대 2개 질문만 처리
                additional_materials = self.search_related_materials(question, limit=2)
                
                # 추가 자료에서 연구 자료 생성
                for material_info in additional_materials:
                    material = ResearchMaterial(
                        title=material_info.get("title", "제목 없음"),
                        authors=[],
                        year="",
                        source=material_info.get("source", "추가 검색"),
                        url=material_info.get("url", ""),
                        local_path="",
                        content="",
                        summary=material_info.get("snippet", "요약 없음"),
                        metadata={}
                    )
                    materials.append(material)
            
            # 5. 연구 요약 생성
            key_findings = [
                f"'{material.title}'에서의 발견: {material.summary[:150]}..."
                for material in materials[:5]
            ]
            
            gaps = [
                "추가 연구가 필요한 영역",
                "현재 자료에서 다루지 않은 주제 측면"
            ]
            
            next_steps = [
                "더 많은 학술 논문 검토",
                "주요 연구 질문에 대한 심층 분석 수행"
            ]
            
            # 연구 요약 객체 생성
            summary = ResearchSummary(
                topic=topic,
                key_findings=key_findings,
                collected_materials=materials,
                gaps=gaps,
                next_steps=next_steps
            )
            
            logger.info(f"연구 요약 생성 완료: {len(materials)}개의 자료 수집됨")
            return summary
            
        except Exception as e:
            logger.error(f"연구 요약 생성 중 오류 발생: {str(e)}")
            
            # 오류 발생 시 빈 요약 반환
            return ResearchSummary(
                topic=topic,
                key_findings=["연구 중 오류 발생"],
                collected_materials=[],
                gaps=["연구를 완료할 수 없음"],
                next_steps=["연구 에이전트 오류 해결"]
            )

    def run(self, topic: str, config: Optional[RunnableConfig] = None) -> ResearchSummary:
        """
        연구 에이전트를 실행하고 연구 요약을 반환합니다.

        Args:
            topic (str): 연구 주제
            config (Optional[RunnableConfig], optional): 실행 구성

        Returns:
            ResearchSummary: 생성된 연구 요약
        """
        logger.info(f"연구 에이전트 실행 중: 주제 '{topic}'")
        
        try:
            # 상태 초기화
            self.reset()
            
            # 연구 요약 생성
            summary = self.generate_research_summary(topic)
            
            # 상태 업데이트
            self.update_state({
                "topic": topic,
                "research_summary": summary
            })
            
            logger.info(f"연구 에이전트 실행 완료: 주제 '{topic}'에 대한 {len(summary.collected_materials)}개의 자료 수집됨")
            return summary
            
        except Exception as e:
            logger.error(f"연구 에이전트 실행 중 오류 발생: {str(e)}")
            
            # 오류 발생 시 빈 요약 반환
            return ResearchSummary(
                topic=topic,
                key_findings=["연구 실행 중 오류 발생"],
                collected_materials=[],
                gaps=["연구를 완료할 수 없음"],
                next_steps=["연구 에이전트 오류 해결"]
            ) 
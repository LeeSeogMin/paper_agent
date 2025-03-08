"""

Research Agent for academic paper writing.



This module contains the ResearchAgent class, which is responsible for gathering,

analyzing, and organizing research materials for academic paper writing.

"""



import os

import re

import json

import time

import uuid

import traceback

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from langchain.docstore.document import Document

from langchain.schema import AIMessage, HumanMessage, SystemMessage

from langchain.tools import tool

from langchain_core.runnables import RunnableSequence

from langchain_core.prompts import PromptTemplate

from langchain_openai import ChatOpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains.summarize import load_summarize_chain

from langchain.chains import LLMChain

from datetime import datetime

from typing import Dict, List, Tuple, Any, Optional, Union



from utils.logger import logger

from utils.vector_db import create_vector_db, search_vector_db, process_and_vectorize_paper

from utils.api_clients import search_google_scholar, search_academic_papers

from utils.pdf_processor import extract_text_from_pdf, process_local_pdfs

from utils.search_utils import academic_search

from utils.search import google_search, search_academic_resources, search_crossref, search_arxiv

from utils.pdf_downloader import get_local_pdf_path



from agents.base import BaseAgent

from models.research import ResearchMaterial, SearchQuery

from prompts.research_prompts import (

    RESEARCH_SYSTEM_PROMPT,

    QUERY_GENERATION_PROMPT,

    SOURCE_EVALUATION_PROMPT,

    SEARCH_RESULTS_ANALYSIS_PROMPT

)



# XAI 클라이언트 import 추가 (기존 import 섹션 아래)

from utils.xai_client import XAIClient  # utils 폴더에서 가져오기





class ResearchAgent(BaseAgent):

    """

    Agent responsible for research activities in the paper writing process.

    

    This agent searches for relevant papers, extracts information, and organizes

    findings for use in the paper writing phase.

    """

    

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.2, verbose: bool = False):

        """

        Initialize the research agent.

        

        Args:

            model_name (str): Name of the large language model to use

            temperature (float): Temperature parameter for LLM

            verbose (bool): Enable verbose logging

        """

        super().__init__(model_name, temperature)

        

        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)

        self.verbose = verbose

        

        # Initialize chains using RunnableSequence

        self.query_chain = RunnableSequence(

            first=QUERY_GENERATION_PROMPT,

            last=self.llm

        )

        

        self.evaluation_chain = RunnableSequence(

            first=SOURCE_EVALUATION_PROMPT,

            last=self.llm

        )

        

        self.analysis_chain = RunnableSequence(

            first=SEARCH_RESULTS_ANALYSIS_PROMPT,

            last=self.llm

        )

        

        # Initialize text splitter for handling large documents

        self.text_splitter = RecursiveCharacterTextSplitter(

            chunk_size=2000,

            chunk_overlap=200

        )

        

        # Initialize summarization chain

        self.summary_chain = load_summarize_chain(

            llm=self.llm,

            chain_type="map_reduce",

            verbose=False

        )

        

        logger.info("ResearchAgent initialized")

    

    def generate_search_queries(self, topic: str, n_queries: int = 3) -> List[SearchQuery]:
        """
        Generate search queries from a research topic.
        
        Args:
            topic: Research topic
            n_queries: Number of queries to generate
            
        Returns:
            List[SearchQuery]: List of search queries
        """
        logger.info(f"Generating {n_queries} search queries for topic: {topic}")
        
        try:
            # Create prompt for query generation
            prompt = f"""
You are a research assistant helping me find literature for a literature review on:
{topic}

Please generate {n_queries} specific search queries to find relevant academic papers. 
Each query should focus on a different aspect of the topic.
For each query, include:
1. The query text (optimized for academic search engines)
2. A brief explanation of what aspect this query is targeting

Format your response as follows (just the queries, no introduction or conclusion):

1. Query: "your first search query here"
   Explanation: Brief explanation here

2. Query: "your second search query here"
   Explanation: Brief explanation here

etc.
"""
            
            # Generate search queries
            result = self.llm.invoke(prompt)
            
            # Extract queries from the result
            result_text = result.content if hasattr(result, 'content') else str(result)
            
            # Parse result into queries
            queries = []
            pattern = r'(\d+)\.\s+Query:\s+["\']([^"\']+)["\'](?:\s+Explanation:\s+([^\n]+))?'
            matches = re.findall(pattern, result_text, re.DOTALL)
            
            for i, (_, query_text, explanation) in enumerate(matches):
                query = SearchQuery(
                    id=f"q{i+1}",
                    text=query_text.strip(),
                    explanation=explanation.strip() if explanation else ""
                )
                queries.append(query)
            
            # If no queries were extracted, use a fallback approach
            if not queries:
                logger.warning("Failed to parse search queries, using fallback approach")
                # Use topic as the first query
                queries.append(SearchQuery(
                    id="q1",
                    text=f"recent research on {topic}",
                    explanation="Direct search using the research topic"
                ))
            
            logger.info(f"Generated {len(queries)} search queries")
            return queries
            
        except Exception as e:
            logger.error(f"Error generating search queries: {str(e)}")
            
            # Fallback to a simple query
            return [SearchQuery(
                id="q1",
                text=f"recent research on {topic}",
                explanation="Direct search using the research topic"
            )]

    

    def search_for_papers(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:

        """

        Search for academic papers using available API clients.

        

        Args:

            query (str): Search query

            max_results (int): Maximum number of results to return

            

        Returns:

            List[Dict[str, Any]]: List of paper metadata

        """

        logger.info(f"검색 쿼리 '{query}'로 논문 검색 중...")

        

        try:

            # 통합 검색 함수 사용 (Semantic Scholar와 Google Scholar 모두 시도)

            results = search_academic_papers(query, max_results)

            logger.info(f"검색 완료: {len(results)}개 결과 발견")

            return results

            

        except Exception as e:

            logger.error(f"논문 검색 실패: {str(e)}")

            # 검색 실패 시 빈 결과 반환 대신 예외 전파

            raise

    

    def evaluate_source(self, source, topic, query=None, threshold=0.3, use_llm=True):
        """
        소스의 관련성을 평가합니다. LLM이나 키워드 일치를 통해 평가할 수 있습니다.
        
        Args:
            source (dict): 평가할 소스
            topic (str): 연구 주제
            query (str, optional): 검색 쿼리
            threshold (float, optional): 관련성 임계값
            use_llm (bool, optional): LLM을 사용하여 평가할지 여부
            
        Returns:
            tuple: (평가 결과, 관련성 점수)
        """
        try:
            # 소스에 필요한 필드가 없는 경우 최소 점수 반환
            for field in ['title', 'url']:
                if not source.get(field):
                    logger.warning(f"소스에 필수 필드 '{field}'가 없습니다: {source}")
                    return None, 0.0
                
            default_score = 0.5  # 기본 관련성 점수
            source_title = source.get('title', '')
            
            if not use_llm:
                # 키워드 일치에 따른 간단한 점수 계산
                relevance_score = default_score
                keywords = topic.lower().split()
                title_lower = source_title.lower()
                
                # 제목에 키워드가 있으면 점수 상승
                for keyword in keywords:
                    if keyword in title_lower:
                        relevance_score += 0.1
                
                # 쿼리가 있고 제목에 쿼리 키워드가 있으면 점수 상승
                if query:
                    query_keywords = query.lower().split()
                    for keyword in query_keywords:
                        if keyword in title_lower and keyword not in keywords:
                            relevance_score += 0.05
                
                # 최대 1.0으로 제한
                relevance_score = min(relevance_score, 1.0)
                
                # 결과 정보 구성
                result = {
                    "relevance_score": relevance_score,
                    "evaluation": "자동 키워드 평가",
                    "rationale": "키워드 일치 기반 평가"
                }
                
                return result, relevance_score
            
            # LLM을 사용한 평가
            content_preview = source.get('abstract', '') or source.get('description', '') or source.get('text_preview', '') or ''
            if not content_preview and 'content' in source:
                content_preview = source['content'][:500] if len(source['content']) > 500 else source['content']
            
            # 메타데이터 구성
            authors = source.get('authors', [])
            authors_str = ', '.join(authors) if authors else 'Unknown'
            year = source.get('year', 'Unknown')
            
            # 평가 프롬프트 작성
            prompt = (
                f"당신은 연구 논문과 자료의 관련성을 평가하는 전문가 연구원입니다. 다음 논문이 연구 주제와 얼마나 관련이 있는지 평가해주세요.\n\n"
                f"연구 주제: {topic}\n"
                f"검색 쿼리: {query if query else topic}\n\n"
                f"논문 정보:\n"
                f"제목: {source_title}\n"
                f"저자: {authors_str}\n"
                f"연도: {year}\n"
                f"내용 미리보기: {content_preview[:1000] if content_preview else 'No preview available'}\n\n"
                f"이 논문이 주제와 얼마나 관련이 있는지 0.0에서 1.0 사이의 점수로 평가하고, 그 이유를 설명해주세요.\n"
                f"평가 결과는 다음 JSON 형식으로 제공해주세요:\n"
                f"```json\n"
                f"{{\n"
                f"  \"relevance_score\": float,  // 0.0에서 1.0 사이의 관련성 점수\n"
                f"  \"evaluation\": string,  // 간단한 평가 요약\n"
                f"  \"rationale\": string  // 평가 이유 설명\n"
                f"}}\n"
                f"```"
            )
            
            # LLM 호출
            result_text = self.llm.invoke(prompt).content
            
            # JSON 추출
            pattern = r"```json\s*([\s\S]*?)\s*```"
            match = re.search(pattern, result_text)
            if match:
                json_str = match.group(1)
                result = json.loads(json_str)
            else:
                pattern = r"{[\s\S]*?}"
                match = re.search(pattern, result_text)
                if match:
                    json_str = match.group(0)
                    result = json.loads(json_str)
                else:
                    # JSON을 찾을 수 없는 경우 기본값 반환
                    logger.warning(f"LLM 응답에서 JSON을 추출할 수 없습니다: {result_text}")
                    result = {
                        "relevance_score": default_score,
                        "evaluation": "평가 실패",
                        "rationale": "결과 파싱 실패"
                    }
            
            relevance_score = float(result.get("relevance_score", default_score))
            
            # 최소 임계값과 비교
            if relevance_score < threshold:
                logger.info(f"관련성 점수({relevance_score})가 임계값({threshold}) 미만으로 소스를 제외합니다: {source_title}")
            
            return result, relevance_score
            
        except Exception as e:
            logger.error(f"소스 평가 중 오류 발생: {str(e)}")
            return None, 0.0

    

    def collect_research_materials(self, topic, research_plan=None, max_queries=3, results_per_source=10, final_result_count=20):

        """

        연구 주제 및 계획에 따른 자료 수집

        

        Args:

            topic: 연구 주제

            research_plan: 총괄 에이전트의 연구 계획

            max_queries: 생성할 최대 검색 쿼리 수

            results_per_source: 각 소스별 최대 결과 수

            final_result_count: 최종 선정할 결과 수

        

        Returns:

            List[ResearchMaterial]: 수집된 연구 자료 리스트

        """

        logger.info(f"Collecting research materials for topic: {topic}")

        

        # 수집된 모든 자료

        all_materials = []

        

        try:

            # 연구 계획 확인 (검색 전략 설정)

            search_scope = "all"  # 기본값

            if research_plan and "search_strategy" in research_plan:

                search_scope = research_plan["search_strategy"].get("search_scope", "all")

                if "min_papers" in research_plan["search_strategy"]:

                    final_result_count = research_plan["search_strategy"]["min_papers"]

                if "queries" in research_plan["search_strategy"] and research_plan["search_strategy"]["queries"]:

                    predefined_queries = research_plan["search_strategy"]["queries"]

                    max_queries = min(max_queries, len(predefined_queries))

            

            # 로컬 폴더와 pdfs 폴더의 PDF 파일 수 확인
            from utils.pdf_downloader import PDF_STORAGE_PATH
            
            local_dir = "data/local"
            pdfs_dir = PDF_STORAGE_PATH
            
            # 각 디렉토리의 PDF 파일 수 계산
            local_pdf_count = len([f for f in os.listdir(local_dir) if f.lower().endswith('.pdf')]) if os.path.exists(local_dir) else 0
            pdfs_pdf_count = len([f for f in os.listdir(pdfs_dir) if f.lower().endswith('.pdf')]) if os.path.exists(pdfs_dir) else 0
            
            total_pdf_count = local_pdf_count + pdfs_pdf_count
            logger.info(f"PDF 파일 수: 로컬 폴더 {local_pdf_count}개, pdfs 폴더 {pdfs_pdf_count}개, 총 {total_pdf_count}개")
            
            # PDF 파일이 30개 이상인 경우 외부 검색 건너뛰기
            skip_external_search = total_pdf_count >= 30
            if skip_external_search:
                logger.info(f"PDF 파일이 30개 이상 ({total_pdf_count}개) 존재하여 외부 검색을 건너뜁니다.")
                search_scope = "local_only"
            
            # 1. 로컬 PDF 파일 처리 및 벡터화 (로컬 검색 허용 시)
            if search_scope in ["local_only", "all"]:
                from utils.pdf_processor import process_local_pdfs
                local_papers = process_local_pdfs(local_dir="data/local", vector_db_path="data/vector_db")
                
                # 로컬 PDF는 관련성 점수와 상관없이 모두 포함
                for paper in local_papers:
                    relevance_score, explanation = self.evaluate_source(paper, topic)
                    material = self._create_research_material(paper, "local", relevance_score, explanation)
                    if material:
                        all_materials.append(material)
                
                # 로컬 PDF 처리 후 papers.json 파일 저장
                if local_papers and len(local_papers) > 0:
                    # 모든 자료를 papers.json에 저장
                    materials_to_save = []
                    for paper in local_papers:
                        # 연구 자료 생성 (관련성 점수는 기본값 사용)
                        material = {
                            "id": paper.get("id", f"local_{uuid.uuid4().hex[:8]}"),
                            "title": paper.get("title", ""),
                            "authors": paper.get("authors", []),
                            "year": paper.get("year", ""),
                            "abstract": paper.get("abstract", ""),
                            "content": "",
                            "url": "",
                            "pdf_url": "",
                            "local_path": paper.get("pdf_path", ""),
                            "relevance_score": 1.0,  # 로컬 파일은 모두 최대 관련성 점수 부여
                            "evaluation": "로컬 파일",
                            "query_id": "local",
                            "citation_count": 0,
                            "venue": "",
                            "source": "local"
                        }
                        
                        materials_to_save.append(material)
                    
                    self.save_research_materials_to_json(materials_to_save, file_path='data/papers.json')
                    logger.info(f"{len(local_papers)}개의 로컬 PDF 파일 정보를 data/papers.json에 저장했습니다.")
                
                # 2. 로컬 데이터베이스 검색
                local_results = self.search_local_papers(topic, max_results=results_per_source)
                local_results = [r for r in local_results if r.get('pdf_url') or 'pdf_path' in r]
                
                for result in local_results:
                    relevance_score, explanation = self.evaluate_source(result, topic)
                    
                    # 모든 로컬 파일 포함 (관련성 점수와 상관없이)
                    material = self._create_research_material(result, "local_db", relevance_score, explanation)
                    if material:
                        all_materials.append(material)
                
                # pdfs 폴더의 PDF 파일도 모두 포함
                if os.path.exists(pdfs_dir):
                    from utils.pdf_processor import PDFProcessor
                    pdf_processor = PDFProcessor(use_llm=True)
                    
                    for pdf_file in os.listdir(pdfs_dir):
                        if pdf_file.lower().endswith('.pdf'):
                            pdf_path = os.path.join(pdfs_dir, pdf_file)
                            try:
                                pdf_result = pdf_processor.process_pdf(pdf_path)
                                
                                if pdf_result["success"]:
                                    metadata = pdf_result["metadata"]
                                    
                                    paper_info = {
                                        "id": f"pdfs_{os.path.splitext(pdf_file)[0]}",
                                        "title": metadata.get("title", ""),
                                        "authors": metadata.get("authors", []),
                                        "abstract": metadata.get("abstract", ""),
                                        "year": metadata.get("year", ""),
                                        "pdf_path": pdf_path
                                    }
                                    
                                    # pdfs 폴더의 파일도 관련성 점수와 상관없이 포함
                                    relevance_score, explanation = self.evaluate_source(paper_info, topic)
                                    
                                    material = self._create_research_material(paper_info, "pdfs", relevance_score, explanation)
                                    if material:
                                        all_materials.append(material)
                            except Exception as e:
                                logger.error(f"pdfs 폴더 PDF 처리 중 오류: {pdf_file} - {str(e)}")
            
            # 외부 검색이 허용된 경우에만 실행
            if search_scope in ["web_only", "all"] and not skip_external_search:
                # 3. 검색 쿼리 생성 또는 계획에서 가져오기
                search_queries = []
                if research_plan and "search_strategy" in research_plan and "queries" in research_plan["search_strategy"]:
                    # 계획에서 제공된 쿼리 사용
                    predefined_queries = research_plan["search_strategy"]["queries"]
                    search_queries = [SearchQuery(id=f"q{i}", text=q) for i, q in enumerate(predefined_queries[:max_queries])]
                else:
                    # 자동 쿼리 생성
                    search_queries = self.generate_search_queries(topic, n_queries=max_queries)
                
                # 4. 외부 학술 검색 (영어 자료만)
                for query in search_queries:
                    query_text = query.text
                    logger.info(f"Processing query: {query_text}")
                    
                    # 학술 검색 및 웹 검색 실행 (영어만)
                    academic_results = academic_search(query_text, max_results=results_per_source, language='en')
                    web_results = academic_search(query_text, max_results=results_per_source, sources=["google", "arxiv", "crossref"], language='en')
                    
                    # PDF URL이 있는 결과만 유지
                    academic_results = [r for r in academic_results if r.get('pdf_url')]
                    web_results = [r for r in web_results if r.get('pdf_url')]
                    
                    # 결과 결합
                    combined_results = academic_results + web_results
                    
                    # 병렬 처리로 결과 평가
                    def evaluate_single_result(result):
                        try:
                            relevance_score, explanation = self.evaluate_source(result, topic)
                            result['relevance_score'] = relevance_score
                            result['evaluation'] = explanation
                            return result
                        except Exception as e:
                            logger.error(f"결과 평가 중 오류: {str(e)}")
                            # 오류 시 기본 점수 할당
                            result['relevance_score'] = 0.3
                            result['evaluation'] = f"평가 오류: {str(e)}"
                            return result
                    
                    # 병렬 처리로 결과 평가
                    logger.info(f"병렬 처리로 {len(combined_results)}개 결과 평가 중...")
                    with ThreadPoolExecutor(max_workers=5) as executor:
                        evaluated_results = list(executor.map(evaluate_single_result, combined_results))
                    
                    # 관련성 점수로 정렬하고 상위 결과만 선택
                    evaluated_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
                    top_results = evaluated_results[:results_per_source]
                    
                    # 관련성 기준 이상인 결과만 최종 자료로 변환
                    for result in top_results:
                        if result.get('relevance_score', 0) >= 0.3:  # 관련성 최소 기준을 0.3으로 낮춤
                            material = self._create_research_material(result, query.id, result.get('relevance_score', 0), result.get('evaluation', ''))
                            if material:
                                all_materials.append(material)
            
            # 자료가 없는 경우 처리
            if not all_materials:
                if search_scope == "local_only":
                    logger.error("로컬 자료가 없습니다. 외부 검색을 고려하세요.")
                else:
                    logger.error("검색 결과가 없습니다. 프로세스를 중단합니다.")
                return []

            

            # 중복 제거 및 최종 선택

            unique_materials = {}

            for material in all_materials:

                # 제목을 기준으로 중복 확인 (소문자 변환 및 공백 제거하여 비교)

                normalized_title = material.title.lower().strip()

                # 이미 있는 자료보다 관련성이 높은 경우 대체

                if normalized_title not in unique_materials or material.relevance_score > unique_materials[normalized_title].relevance_score:

                    unique_materials[normalized_title] = material

            

            # 5. 관련성 점수 기준 정렬 후 최종 선정

            final_materials = list(unique_materials.values())

            final_materials.sort(key=lambda x: x.relevance_score, reverse=True)

            final_materials = final_materials[:final_result_count]  # 최종 개수로 제한

            

            logger.info(f"Collected {len(final_materials)} research materials after deduplication and filtering")

            return final_materials

            

        except Exception as e:

            logger.error(f"Error collecting research materials: {str(e)}", exc_info=True)

            raise

    

    def _create_research_material(self, result, query_id, relevance_score, evaluation):

        """자료로부터 ResearchMaterial 객체 생성 (helper 메서드)"""

        # PDF URL 또는 PDF 경로 확인

        pdf_url = result.get('pdf_url')

        pdf_path = result.get('pdf_path')

        

        if not pdf_url and not pdf_path:

            return None

        

        # 저자 처리

        if isinstance(result.get('authors', ''), str):

            authors_list = [author.strip() for author in result.get('authors', '').split(',') if author.strip()]

        else:

            authors_list = result.get('authors', [])

        

        # 연도 처리

        year_value = result.get('year', result.get('published_date', ''))

        if isinstance(year_value, str):

            # 연도만 추출 시도

            year_match = re.search(r'(19|20)\d{2}', year_value)

            if year_match:

                year_int = int(year_match.group(0))

            else:

                year_int = None

        else:

            year_int = year_value

        

        # 연구 자료 생성

        return ResearchMaterial(

            id=result.get("id") or f"paper_{uuid.uuid4().hex[:8]}",

            title=result.get("title", ""),

            authors=authors_list,

            year=year_int,

            abstract=result.get("abstract", ""),

            url=result.get("url", ""),

            pdf_url=pdf_url,

            pdf_path=pdf_path,  # PDF 경로 추가

            relevance_score=relevance_score,

            evaluation=evaluation,

            query_id=query_id,

            content="",

            summary=""

        )

    

    def extract_content_from_pdf(self, pdf_url_or_path: str) -> str:

        """

        Extract text content from a PDF file.

        

        Args:

            pdf_url_or_path (str): URL or local path to PDF file

            

        Returns:

            str: Extracted text

        """

        logger.info(f"Extracting content from PDF: {pdf_url_or_path}")

        

        try:

            text = extract_text_from_pdf(pdf_url_or_path)

            return text

        except Exception as e:

            logger.error(f"Error extracting content from PDF: {str(e)}", exc_info=True)

            return ""

    

    def summarize_content(self, content: str, title: str) -> str:

        """

        Generate a summary of research material content.

        

        Args:

            content (str): Material content

            title (str): Material title

            

        Returns:

            str: Generated summary

        """

        try:

            # Split content into chunks

            text_chunks = self.text_splitter.split_text(content)

            

            # Convert text chunks to Document objects for the summarize chain

            documents = [Document(page_content=chunk) for chunk in text_chunks]

            

            # Generate summary

            result = self.summary_chain.invoke({"input_documents": documents})

            summary = result["output_text"] if "output_text" in result else result.get("text", "")

            

            return summary

            

        except Exception as e:

            logger.error(f"Error summarizing content: {str(e)}", exc_info=True)

            return f"Failed to summarize content for '{title}': {str(e)}"

    

    def enrich_research_materials(self, materials: List[ResearchMaterial]) -> List[ResearchMaterial]:
        """
        연구 자료 정보 강화
        PDF에서 내용 추출, 메타데이터 보강, 요약 생성 등을 수행합니다.
        
        Args:
            materials: 연구 자료 목록
            
        Returns:
            List[ResearchMaterial]: 강화된 연구 자료 목록
        """
        if not materials:
            logger.warning("강화할 연구 자료가 없습니다.")
            return []
            
        # 결과 리스트 초기화
        enriched_materials = []
        
        try:
            # 단일 자료 강화를 위한 내부 함수
            def enrich_single_material(material, index):
                try:
                    logger.info(f"자료 강화 중: {getattr(material, 'title', f'자료 {index+1}')}")
                    
                    # 필수 필드 확인
                    has_title = hasattr(material, 'title') and material.title
                    has_authors = hasattr(material, 'authors') and material.authors
                    
                    # 제목이나 저자가 없는 자료는 건너뜁니다
                    if not has_title or not has_authors:
                        logger.warning(f"필수 메타데이터(제목/저자)가 없어 자료를 건너뜁니다: {getattr(material, 'id', f'자료 {index+1}')}")
                        return None
                    
                    # PDF에서 내용 추출 (이미 있으면 건너뜀)
                    if not hasattr(material, 'content') or not material.content:
                        # PDF 경로 또는 URL이 있는 경우
                        if hasattr(material, 'pdf_path') and material.pdf_path:
                            logger.info(f"Extracting content from PDF: {material.pdf_path}")
                            try:
                                content = self.extract_content_from_pdf(material.pdf_path)
                                if content and len(content) > 100:  # 의미 있는 내용이 있는지 확인
                                    material.content = content
                                else:
                                    logger.warning(f"PDF에서 충분한 내용을 추출하지 못했습니다: {material.pdf_path}")
                                    # 내용이 없으면 건너뜁니다
                                    return None
                            except Exception as e:
                                logger.error(f"Error extracting text from PDF: {str(e)}")
                                # 내용 추출 실패 시 건너뜁니다
                                return None
                        # PDF 없이 초록만 있는 경우
                        elif hasattr(material, 'abstract') and material.abstract:
                            logger.info(f"PDF 없음, 초록을 내용으로 사용: {material.title}")
                            material.content = material.abstract
                        else:
                            logger.warning(f"내용이 없고 추출할 방법도 없어 자료를 건너뜁니다: {material.title}")
                            return None
                    
                    # 내용이 있는지 최종 확인
                    if not hasattr(material, 'content') or not material.content or len(material.content) < 100:
                        logger.warning(f"내용이 부족하여 자료를 건너뜁니다: {material.title}")
                        return None
                        
                    # 내용 요약 (이미 있으면 건너뜀)
                    if not hasattr(material, 'summary') or not material.summary:
                        if hasattr(material, 'content') and material.content:
                            logger.info(f"내용 요약 생성 중: {material.title}")
                            try:
                                material.summary = self.summarize_content(material.content, material.title)
                            except Exception as e:
                                logger.warning(f"요약 생성 실패: {str(e)}")
                                # 요약이 없어도 계속 진행 (요약은 필수가 아님)
                                material.summary = material.content[:200] + "..."
                    
                    # 발행 연도가 없는 경우 대체 설정
                    if not hasattr(material, 'year') or not material.year:
                        material.year = "미상"  # 연도 미상
                        
                    # 발표학술지/출처가 없는 경우 대체 설정
                    if not hasattr(material, 'source') or not material.source:
                        material.source = "미상"  # 출처 미상
                    
                    # 벡터 데이터베이스에서 유사 문서 검색 (선택적)
                    try:
                        if hasattr(material, 'abstract') and material.abstract:
                            similar_docs = search_vector_db(
                                db_name="research_papers",  # collection_name 대신 db_name
                                query=material.abstract, 
                                k=3
                            )
                            
                            # 유사 문서 정보 추가
                            if similar_docs:
                                related_info = []
                                
                                # 형식에 맞게 처리하도록 수정
                                for doc_item in similar_docs:
                                    try:
                                        # 이미 (doc, score) 형태인 경우
                                        if isinstance(doc_item, tuple) and len(doc_item) == 2:
                                            doc, score = doc_item
                                        else:
                                            # 단일 문서만 있는 경우
                                            doc = doc_item
                                            score = 1.0  # 기본 유사도 점수 (없는 경우)
                                        
                                        # 메타데이터 유효성 확인
                                        if not hasattr(doc, 'metadata') or not doc.metadata:
                                            logger.warning(f"메타데이터가 없는 문서를 건너뜁니다")
                                            continue
                                            
                                        paper_id = doc.metadata.get('paper_id')
                                        title = doc.metadata.get('title')
                                        
                                        # 필수 메타데이터가 없으면 건너뛰기
                                        if not paper_id or not title:
                                            logger.warning(f"필수 메타데이터가 없는 문서를 건너뜁니다")
                                            continue
                                        
                                        if paper_id != material.id:  # 자기 자신 제외
                                            related_info.append({
                                                'title': title,
                                                'similarity': f"{score:.2f}",
                                                'content': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content
                                            })
                                    except Exception as e:
                                        logger.warning(f"문서 메타데이터 처리 오류: {str(e)}")
                                        continue
                                
                                # 관련 정보가 있으면 저장
                                if related_info:
                                    material.related_papers = related_info
                    except Exception as e:
                        logger.warning(f"유사 문서 검색 중 오류: {str(e)}")
                    
                    return material
                    
                except Exception as e:
                    logger.error(f"자료 {getattr(material, 'title', f'자료 {index+1}')} 강화 중 오류: {str(e)}")
                    return None
            
            # 병렬 처리로 자료 강화
            with ThreadPoolExecutor(max_workers=5) as executor:
                material_indices = [(material, idx) for idx, material in enumerate(materials)]
                enriched_materials_list = list(executor.map(
                    lambda x: enrich_single_material(*x), material_indices
                ))
                
                # None 값 제거
                enriched_materials = [m for m in enriched_materials_list if m is not None]
            
            # 결과 수 로그
            if len(enriched_materials) < len(materials):
                logger.warning(f"{len(materials) - len(enriched_materials)}개 자료가 필수 데이터 부족으로 제외됨. 유효 자료 {len(enriched_materials)}개로 진행.")
            
            return enriched_materials
            
        except Exception as e:
            logger.error(f"연구 자료 강화 중 오류 발생: {str(e)}")
            traceback.print_exc()
            # 오류 발생 시 원본 반환 대신 지금까지 강화된 자료 반환
            return enriched_materials

    

    def analyze_research_materials(self, materials, topic):
        """
        Analyze research materials to extract key insights.
        
        Args:
            materials: List of research materials
            topic: Research topic
            
        Returns:
            dict: Analysis results
        """
        logger.info(f"Analyzing {len(materials)} research materials for topic: {topic}")
        
        try:
            # Prepare material summaries for analysis
            material_summaries = []
            for mat in materials:
                summary = f"Title: {mat.title}\n"
                if mat.authors:
                    summary += f"Authors: {', '.join(mat.authors)}\n"
                if mat.year:
                    summary += f"Year: {mat.year}\n"
                if mat.abstract:
                    summary += f"Abstract: {mat.abstract[:500]}...\n"
                elif mat.content:
                    content_preview = mat.content[:500] + "..." if len(mat.content) > 500 else mat.content
                    summary += f"Content: {content_preview}\n"
                material_summaries.append(summary)
            
            # Combine summaries
            all_summaries = "\n\n".join(material_summaries)
            
            # Create analysis prompt
            analysis_prompt = f"""
You are a research assistant analyzing materials for a literature review on: {topic}

Please analyze the following research materials and provide a comprehensive analysis including:
1. Main themes and patterns
2. Chronological development of ideas
3. Key methodologies used
4. Significant findings
5. Gaps or areas for further research
6. Relationships between different works

Materials:
{all_summaries}

Provide a thorough analysis that will help in writing a literature review.
"""
            
            # Generate analysis
            result = self.llm.invoke(analysis_prompt)
            
            # 수정: AIMessage 객체에서 텍스트 내용 추출
            analysis_text = result.content if hasattr(result, 'content') else str(result)
            
            # Structure the analysis
            analysis = {
                "topic": topic,
                "main_themes": [],
                "chronology": [],
                "methodologies": [],
                "key_findings": [],
                "research_gaps": [],
                "relationships": [],
                "analysis_text": analysis_text
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing research materials: {str(e)}")
            traceback.print_exc()
            
            # 오류 발생시 최소한의 분석 결과 반환
            return {
                "topic": topic,
                "main_themes": [],
                "chronology": [],
                "methodologies": [],
                "key_findings": [],
                "research_gaps": [],
                "relationships": [],
                "analysis_text": f"분석 중 오류 발생: {str(e)}"
            }

    

    def search_local_papers(self, query, data_path="data/papers.json", max_results=10):

        """

        로컬에 저장된 논문 데이터에서 검색

        

        Args:

            query: 검색어

            data_path: 논문 데이터 파일 경로

            max_results: 최대 결과 수

            

        Returns:

            List[Dict]: 검색 결과 목록

        """

        logger.info(f"로컬 데이터에서 검색: {query}")

        

        try:

            import os

            import json

            

            # 파일이 없으면 빈 결과 반환 (파일 생성 시도 안 함)

            if not os.path.exists(data_path):

                logger.info(f"로컬 데이터 파일이 없음: {data_path}, 빈 결과 반환")

                return []

            

            # 파일 크기 확인

            file_size = os.path.getsize(data_path)

            if file_size == 0:

                logger.warning(f"로컬 데이터 파일이 비어 있음: {data_path}")

                return []

            

            try:

                with open(data_path, 'r', encoding='utf-8') as f:

                    papers = json.load(f)

                

                # 빈 배열인 경우 처리

                if not papers or len(papers) == 0:

                    logger.warning(f"로컬 데이터 파일에 논문 정보가 없음: {data_path}")

                    return []

                

                logger.info(f"로컬 데이터베이스에서 {len(papers)} 논문 로드됨")

                

                # 모든 논문을 결과로 반환 (필터링 없이)

                for paper in papers:

                    paper['relevance'] = 1.0  # 모든 논문에 최대 관련성 점수 부여

                

                logger.info(f"로컬 데이터에서 {len(papers)}개 논문 찾음")

                return papers

                

            except json.JSONDecodeError as e:

                logger.error(f"로컬 데이터 파일 JSON 파싱 오류: {str(e)}")

                return []

            

        except Exception as e:

            logger.error(f"로컬 논문 검색 중 오류: {str(e)}", exc_info=True)

            return []

    

    def _extract_semantic_keywords(self, query):

        """

        LLM을 사용하여 검색 쿼리에서 의미 기반 키워드 추출

        

        Args:

            query: 검색 쿼리

            

        Returns:

            Dict[str, float]: 키워드와 가중치 딕셔너리

        """

        try:

            # LLM에게 키워드 추출 요청

            messages = [

                SystemMessage(content="You are a research assistant tasked with extracting and expanding search keywords."),

                HumanMessage(content=f"""

                    Extract and expand the most important keywords from this search query for academic paper search.

                    

                    The search query is: "{query}"

                    

                    Consider synonyms, related concepts, and specific terminology in the field.

                    Return a JSON dictionary with keywords as keys and relevance weights (0.0-1.0) as values.

                    Format: {{"keyword1": weight1, "keyword2": weight2, ...}}

                    

                    For example, if the query is "machine learning", you might return:

                    {{"machine learning": 1.0, "deep learning": 0.8, "neural networks": 0.7, "ai": 0.6, "artificial intelligence": 0.6, "ml": 0.9}}

                """)

            ]

            

            response = self.llm.invoke(messages)

            

            # 응답에서 JSON 추출

            import re

            import json

            

            # JSON 형식 추출 (중괄호 사이의 내용)

            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)

            if json_match:

                json_str = json_match.group(0)

                try:

                    keywords = json.loads(json_str)

                    return keywords

                except json.JSONDecodeError:

                    logger.warning(f"키워드 JSON 파싱 실패: {json_str}")

            

            # JSON 파싱 실패 시 기본 키워드 반환

            logger.warning(f"LLM 응답에서 키워드 추출 실패, 기본 키워드 사용")

            default_keywords = {query.lower(): 1.0}

            query_words = query.lower().split()

            for word in query_words:

                if word != query.lower():

                    default_keywords[word] = 0.8

            return default_keywords

            

        except Exception as e:

            logger.error(f"의미 기반 키워드 추출 중 오류: {str(e)}")

            # 오류 발생 시 기본 키워드 반환

            return {query.lower(): 1.0}



    def create_paper_outline(self, topic, analysis, materials):
        """
        연구 주제와 분석 결과를 바탕으로 논문 개요를 생성합니다.
        
        Args:
            topic (str): 연구 주제
            analysis (dict): 분석 결과
            materials (list): 연구 자료 목록
            
        Returns:
            dict: 논문 개요 (제목, 초록, 섹션 등)
        """
        logger.info("논문 개요 생성 중...")
        
        # 자료 요약 준비
        material_summaries = []
        for i, material in enumerate(materials):
            if i >= 15:  # 최대 15개 자료만 사용
                break
                
            try:
                title = getattr(material, 'title', f'자료 {i+1}')
                authors = getattr(material, 'authors', ['저자 미상'])
                if isinstance(authors, list):
                    authors = ', '.join(authors[:3])
                    if len(getattr(material, 'authors', [])) > 3:
                        authors += ' et al.'
                        
                year = getattr(material, 'year', '연도 미상')
                source = getattr(material, 'source', '출처 미상')
                
                # 초록 또는 내용 요약 사용
                if hasattr(material, 'summary') and material.summary:
                    preview = material.summary[:300] + '...' if len(material.summary) > 300 else material.summary
                elif hasattr(material, 'abstract') and material.abstract:
                    preview = material.abstract[:300] + '...' if len(material.abstract) > 300 else material.abstract
                elif hasattr(material, 'content') and material.content:
                    preview = material.content[:300] + '...' if len(material.content) > 300 else material.content
                else:
                    preview = "내용 없음"
                    
                material_summaries.append({
                    'title': title,
                    'authors': authors,
                    'year': year,
                    'source': source,
                    'preview': preview
                })
            except Exception as e:
                logger.error(f"자료 {i+1} 요약 생성 중 오류: {str(e)}")
                
        # 자료 요약이 없는 경우 처리
        if not material_summaries:
            logger.warning("요약할 자료가 없습니다.")
            material_summaries = [{'title': '자료 없음', 'authors': '저자 미상', 'year': '연도 미상', 'preview': '내용 없음'}]
            
        # 분석 요약 준비
        analysis_summary = ""
        if analysis:
            if 'main_themes' in analysis:
                analysis_summary += f"주요 주제: {analysis['main_themes']}\n"
            if 'methodologies' in analysis:
                analysis_summary += f"주요 방법론: {analysis['methodologies']}\n"
            if 'key_findings' in analysis:
                analysis_summary += f"주요 발견: {analysis['key_findings']}\n"
            if 'research_gaps' in analysis:
                analysis_summary += f"연구 공백: {analysis['research_gaps']}\n"
                
        # 분석 요약이 없는 경우 처리
        if not analysis_summary:
            analysis_summary = "분석 결과 없음"
            
        # 아웃라인 생성 프롬프트
        outline_prompt = f"""
        다음 정보를 바탕으로 "{topic}"에 관한 문헌 리뷰 논문의 아웃라인을 작성해주세요.
        
        ## 분석 요약
        {analysis_summary}
        
        ## 연구 자료 목록 (총 {len(material_summaries)}개)
        {json.dumps(material_summaries, indent=2, ensure_ascii=False)}
        
        ## 요구사항
        1. 논문의 제목을 작성하세요 (한글 또는 영어).
        2. 본문 섹션만 작성하세요 - 초록, 서론, 결론 섹션은 포함하지 마세요.
        3. 각 섹션에는 제목과 간략한 설명을 포함하세요.
        4. 섹션 수준은 최대 2단계까지만 사용하세요 (예: 3.1, 3.2).
        5. 참고문헌 목록은 제공된 연구 자료를 기반으로 작성하세요.
        
        다음 JSON 형식으로 응답해주세요:
        ```json
        {
          "title": "논문 제목",
          "sections": [
            {
              "title": "섹션 제목",
              "level": 1,
              "description": "섹션 설명"
            },
            {
              "title": "하위 섹션 제목",
              "level": 2,
              "description": "하위 섹션 설명"
            }
          ],
          "references": [
            "참고문헌 1",
            "참고문헌 2"
          ]
        }
        ```
        
        중요: 초록, 서론, 결론 섹션은 포함하지 마세요! 본문 내용(연구 주제의 주요 내용, 방법론, 역사적 발전, 분석 등)에 집중하세요.
        """
        
        try:
            # LLM을 사용하여 아웃라인 생성
            response = self.llm.invoke(outline_prompt)
            outline_text = response.content.strip()
            
            # JSON 추출
            match = re.search(r'```(?:json)?(.*?)```', outline_text, re.DOTALL)
            if match:
                outline_json = match.group(1).strip()
            else:
                outline_json = outline_text
                
            try:
                outline = json.loads(outline_json)
            except json.JSONDecodeError:
                logger.error("JSON 파싱 실패, 텍스트 분석 시도")
                # 텍스트에서 정보 추출 시도
                outline = self._extract_outline_from_text(outline_text)
                
            logger.info(f"논문 개요 생성 완료: {len(outline.get('sections', []))}개 섹션")
            return outline
            
        except Exception as e:
            logger.error(f"논문 개요 생성 중 오류: {str(e)}")
            traceback.print_exc()
            return None



    def generate_report(self, topic, analysis, materials):
        """
        분석 결과와 연구 자료를 기반으로 문헌 리뷰 보고서를 생성합니다.
        
        Args:
            topic (str): 연구 주제
            analysis (dict): 연구 자료 분석 결과
            materials (list): 연구 자료 목록
            
        Returns:
            dict: 생성된 보고서 정보 (성공 여부, 파일 경로 등)
        """
        try:
            logger.info("문헌 리뷰 보고서 생성 중...")
            
            # 1. 보고서 아웃라인 생성
            outline = self.create_paper_outline(topic, analysis, materials)
            if not outline:
                logger.error("보고서 아웃라인 생성에 실패했습니다.")
                return None
                
            logger.info(f"논문 아웃라인 생성됨: {len(outline.get('sections', []))}개 섹션")
            
            # 2. 보고서 내용 구성 - 초록, 서론, 결론 제외하고 본문만 포함
            # 제목만 포함하고 초록 제외
            title = outline.get('title', '토픽 모델링의 역사와 발전')
            report_content = f"# {title}\n\n"
            
            # 섹션 필터링 - 초록, 서론, 결론 제외
            sections = outline.get('sections', [])
            content_sections = []
            for section in sections:
                section_title = section.get('title', '').lower()
                # 초록, 서론, 결론 관련 섹션 제외
                if any(keyword in section_title for keyword in ['초록', 'abstract', '서론', 'introduction', '결론', 'conclusion']):
                    logger.info(f"섹션 제외: {section_title}")
                    continue
                content_sections.append(section)
            
            logger.info(f"본문 섹션 {len(content_sections)}개 선택됨")
            
            # 본문 섹션 내용 생성
            for section in content_sections:
                section_title = section.get('title', '제목 없음')
                section_desc = section.get('description', '')
                
                try:
                    # 각 섹션에 대한 상세 내용 생성
                    section_prompt = f"""
                    다음은 '{title}' 제목의 문헌 리뷰 논문의 섹션입니다:
                    
                    섹션: {section_title}
                    설명: {section_desc}
                    
                    이 섹션에 대해 약 500-700단어의 학술적이고 상세한 내용을 작성해주세요.
                    주제: {topic}
                    분석 내용: {analysis.get('main_themes', '')}
                    
                    학술 논문 스타일로 작성하고, 적절한 제목과 소제목을 포함하세요.
                    * 중요: 초록, 서론, 결론 스타일의 내용은 포함하지 마세요. 오직 본문 콘텐츠만 작성하세요.
                    """
                    
                    response = self.llm.invoke(section_prompt)
                    section_content = response.content.strip()
                    
                    report_content += f"## {section_title}\n\n"
                    report_content += f"{section_content}\n\n"
                    
                except Exception as e:
                    logger.error(f"섹션 '{section_title}' 내용 생성 중 오류 발생: {e}")
                    report_content += f"## {section_title}\n\n"
                    report_content += f"{section_desc}\n\n"
            
            # 참고문헌 추가
            report_content += "## References\n\n"
            
            # 참고문헌 추출 및 포맷팅
            references = []
            try:
                references = self.extract_references(materials)
                for i, ref in enumerate(references, 1):
                    author = ref.get('authors', '저자 미상')
                    title = ref.get('title', '제목 미상')
                    year = ref.get('year', '')
                    
                    if isinstance(author, list) and len(author) > 0:
                        author = ', '.join(author[:3])
                        if len(ref.get('authors', [])) > 3:
                            author += ' et al.'
                    
                    report_content += f"{i}. **{author} ({year}). {title}.**  \n"
                    if ref.get('abstract'):
                        report_content += f"   {ref.get('abstract')[:150]}...\n\n"
                    else:
                        report_content += "\n"
            except Exception as e:
                logger.error(f"참고문헌 생성 중 오류 발생: {e}")
            
            # 3. 보고서 저장
            os.makedirs('output', exist_ok=True)
            output_file = 'output/literature_review.md'
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
                
            logger.info(f"문헌 리뷰 보고서가 '{output_file}'에 저장되었습니다.")
            
            # 4. HTML 변환 시도 (선택적)
            if MARKDOWN_AVAILABLE:
                try:
                    html_file = output_file.replace('.md', '.html')
                    with open(output_file, 'r', encoding='utf-8') as md_file:
                        html_content = markdown.markdown(md_file.read(), extensions=['extra'])
                    
                    with open(html_file, 'w', encoding='utf-8') as f:
                        f.write(f"""<!DOCTYPE html>
                        <html>
                        <head>
                            <meta charset="utf-8">
                            <title>{title}</title>
                            <style>
                                body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
                                h1, h2, h3 {{ color: #333; }}
                                blockquote {{ border-left: 4px solid #ddd; padding-left: 15px; color: #777; }}
                                code {{ background-color: #f9f9f9; padding: 2px 4px; border-radius: 4px; }}
                            </style>
                        </head>
                        <body>
                        {html_content}
                        </body>
                        </html>""")
                    
                    logger.info(f"HTML 버전의 보고서가 '{html_file}'에 저장되었습니다.")
                except Exception as e:
                    logger.error(f"HTML 변환 중 오류 발생: {e}")
            else:
                logger.warning("markdown 모듈이 설치되지 않아 HTML 변환을 건너뜁니다.")
            
            return {
                "success": True,
                "file_path": output_file,
                "section_count": len(content_sections),
                "reference_count": len(references)
            }
            
        except Exception as e:
            logger.error(f"보고서 생성 중 오류 발생: {e}")
            return None



    def run(self, topic=None, **kwargs):

        """

        연구 에이전트 실행 메서드

        

        Args:

            topic (str): 연구 주제

            **kwargs: 추가 매개변수

            

        Returns:

            dict: 연구 결과

        """

        if not topic:

            logger.error("연구 주제가 제공되지 않았습니다.")

            return {"status": "error", "message": "연구 주제가 필요합니다."}

        

        try:

            # 1. 연구 자료 수집

            max_sources = kwargs.get("max_sources", 5)  # 기본값 5로 변경

            # 각 소스별 10개 결과, 최종 max_sources개 선정

            materials = self.collect_research_materials(

                topic, 

                max_queries=5,  # 쿼리 수 증가 

                results_per_source=10,  # 각 소스별 10개

                final_result_count=max_sources  # 최종 선정 개수

            )

            

            # 명시적으로 검색 결과가 없는지 확인

            if not materials or len(materials) == 0:

                logger.warning("검색 결과가 없습니다. 빈 JSON 파일을 생성합니다.")

                # 검색 결과가 없어도 빈 JSON 파일 생성

                self.save_research_materials_to_json([])

                return {

                    "status": "error",

                    "message": "검색 결과가 없어 연구를 진행할 수 없습니다. 다른 주제나 검색어를 시도해보세요."

                }

            

            # 2. 연구 자료 강화 (내용 및 요약 추가 + 벡터 DB 처리)

            enriched_materials = self.enrich_research_materials(materials)
            
            # 연구 자료 JSON 파일로 저장
            self.save_research_materials_to_json(enriched_materials)

            # 병렬로 연구 자료 벡터화
            logger.info("연구 자료 벡터화 시작...")
            self.vectorize_materials(enriched_materials)
            logger.info("연구 자료 벡터화 완료")

            # 3. 연구 자료 분석

            analysis = self.analyze_research_materials(enriched_materials, topic)

            

            # 4. 논문 개요 생성

            outline = self.create_paper_outline(topic, analysis, enriched_materials)

            

            # 5. 키워드 및 참고문헌 추출

            keywords = self.extract_keywords(topic, analysis)

            references = self.extract_references(enriched_materials)

            

            # 인용 스타일 설정 (기본값: APA)

            citation_style = kwargs.get("citation_style", "APA")

            

            # 결과 반환

            return {

                "status": "completed",

                "topic": topic,

                "materials": [material.dict() for material in enriched_materials],

                "analysis": analysis,

                "outline": outline,

                "keywords": keywords,

                "references": references,

                "citation_style": citation_style  # 인용 스타일 추가

            }

        except Exception as e:

            logger.error(f"연구 에이전트 실행 중 오류 발생: {str(e)}", exc_info=True)

            # 오류가 발생해도 빈 JSON 파일 생성

            self.save_research_materials_to_json([])

            return {

                "status": "error",

                "message": f"연구 중 오류가 발생했습니다: {str(e)}"

            }



    def search_academic_sources(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:

        """

        학술 소스 검색

        

        Args:

            query: 검색 쿼리

            max_results: 최대 결과 수

            

        Returns:

            List[Dict[str, Any]]: 검색 결과 목록

        """

        logger.info(f"학술 소스 검색: {query}")

        

        try:

            # 학술 검색 수행

            results = academic_search(query, num_results=max_results)

            

            if not results:

                logger.warning(f"학술 검색 결과 없음: {query}")

                # 폴백 검색 시도

                results = academic_search(query, num_results=max_results)

            

            return results

        except Exception as e:

            logger.error(f"학술 소스 검색 오류: {str(e)}")

            return []



    def search_papers(self, query, max_results=10):

        """학술 자료 검색 기능을 통합 사용"""

        from utils.search import search_academic_resources, search_crossref, search_arxiv

        

        # 새로운 검색 기능 사용

        results = []

        

        # Crossref 검색 시도

        crossref_results = search_crossref(query, rows=max_results)

        for result in crossref_results:

            results.append({

                'title': result.title,

                'authors': result.authors,

                'abstract': result.abstract,

                'year': result.published_date[:4] if result.published_date and result.published_date != '날짜 정보 없음' else None,

                'url': result.url,

                'venue': 'Unknown',  # Crossref에는 venue 정보가 명확하지 않을 수 있음

                'citation_count': result.citation_count or 0

            })

        

        # arXiv 검색 시도

        arxiv_results = search_arxiv(query, max_results=max_results)

        for result in arxiv_results:

            results.append({

                'title': result.title,

                'authors': result.authors,

                'abstract': result.abstract,

                'year': result.published_date[:4] if result.published_date and result.published_date != '날짜 정보 없음' else None,

                'url': result.url,

                'venue': 'arXiv',

                'citation_count': 0  # arXiv에는 인용 정보가 없음

            })

        

        return results



    def search_academic_papers(self, query, max_results=5):

        """

        학술 논문 검색 함수

        

        Args:

            query: 검색 쿼리

            max_results: 최대 검색 결과 수

            

        Returns:

            검색된 논문 목록

        """

        logger.info(f"검색 쿼리 '{query}'로 논문 검색 중...")

        

        # API 키 설정 (필요한 경우)

        api_keys = {}

        if hasattr(self, 'core_api_key') and self.core_api_key:

            api_keys['core'] = self.core_api_key

        

        # 통합 학술 검색 실행

        from utils.search import search_academic_resources

        

        logger.info(f"학술 논문 검색: '{query}', 최대 {max_results}개 결과")

        

        try:

            # 새로운 통합 검색 API 사용

            results = search_academic_resources(query, api_keys, max_results)

            

            # 결과 형식 변환

            papers = []

            for source, source_results in results.items():

                for result in source_results:

                    papers.append({

                        'title': result.title,

                        'abstract': result.abstract,

                        'url': result.url,

                        'authors': ', '.join(result.authors) if result.authors else '저자 정보 없음',

                        'year': result.published_date,

                        'source': result.source,

                        'citation_count': result.citation_count

                    })

            

            logger.info(f"총 {len(papers)}개 논문 찾음")

            return papers

            

        except Exception as e:

            logger.error(f"학술 논문 검색 중 오류 발생: {str(e)}", exc_info=True)

            return []



    def extract_keywords(self, topic, analysis):

        """

        주제와 분석 결과에서 중요 키워드 추출

        

        Args:

            topic: 연구 주제

            analysis: 분석 결과

            

        Returns:

            List[str]: 키워드 목록

        """

        logger.info(f"주제 '{topic}'에 대한 키워드 추출 중")

        

        try:

            # 슬라이싱 대신 문자열 길이 제한

            analysis_text = analysis

            if isinstance(analysis, str) and len(analysis) > 2000:

                analysis_text = analysis[:2000] + "..."

            

            # LLM을 사용하여 중요 키워드 추출

            messages = [

                SystemMessage(content="You are a research assistant tasked with extracting important keywords."),

                HumanMessage(content=f"""

                    Extract the most important keywords from this research topic and analysis.

                    

                    TOPIC: {topic}

                    

                    ANALYSIS: {analysis_text}

                    

                    List 10-15 important keywords as a comma-separated list.

                    Format: keyword1, keyword2, keyword3, ...

                """)

            ]

            

            response = self.llm.invoke(messages)

            

            # 응답에서 키워드 추출

            keywords_text = response.content.strip()

            keywords = [k.strip() for k in keywords_text.split(',')]

            

            # 중복 제거 및 빈 문자열 제외

            keywords = [k for k in keywords if k]

            keywords = list(dict.fromkeys(keywords))

            

            logger.info(f"{len(keywords)}개 키워드 추출됨")

            return keywords

        

        except Exception as e:

            logger.error(f"키워드 추출 중 오류 발생: {str(e)}", exc_info=True)

            # 오류 발생 시 기본 키워드 반환

            return ["인공지능", "머신러닝", "딥러닝", "AI", "연구동향"]



    def extract_references(self, materials):
        """
        연구 자료에서 참고문헌 정보를 추출합니다.
        
        Args:
            materials: 연구 자료 목록
            
        Returns:
            list: 참고문헌 목록
        """
        references = []
        
        for material in materials:
            try:
                ref_entry = {
                    "title": getattr(material, 'title', 'Unknown Title'),
                    "authors": getattr(material, 'authors', []),
                    "year": getattr(material, 'year', 'Unknown'),
                    "venue": getattr(material, 'source', 'Unknown Source')
                }
                references.append(ref_entry)
            except Exception as e:
                logger.error(f"참고문헌 추출 중 오류: {str(e)}")
                
        return references
    
    def vectorize_materials(self, materials):
        """
        연구 자료를 병렬로 벡터화합니다.
        
        Args:
            materials (List[ResearchMaterial]): 벡터화할 연구 자료 목록
            
        Returns:
            bool: 성공 여부
        """
        try:
            logger.info(f"총 {len(materials)}개 연구 자료 벡터화 시작")
            
            # 단일 자료 벡터화 함수
            def vectorize_single_material(material):
                try:
                    # PDF URL 또는 경로가 있는 경우 PDF 벡터화
                    if hasattr(material, 'pdf_url') and material.pdf_url:
                        logger.info(f"PDF URL 벡터화 중: {material.title}")
                        process_and_vectorize_paper(material.pdf_url)
                        return True
                    elif hasattr(material, 'pdf_path') and material.pdf_path:
                        logger.info(f"PDF 파일 벡터화 중: {material.title}")
                        process_and_vectorize_paper(material.pdf_path)
                        return True
                    # 내용이 있는 경우 텍스트 벡터화
                    elif hasattr(material, 'content') and material.content and len(material.content) > 100:
                        logger.info(f"내용 벡터화 중: {material.title}")
                        # 내용을 청크로 나누고 벡터화
                        from utils.vector_db import vectorize_content
                        vectorize_content(
                            content=material.content,
                            title=material.title,
                            material_id=material.id
                        )
                        return True
                    else:
                        logger.warning(f"벡터화 불가: {material.title} - 내용 또는 PDF URL/경로 필요")
                        return False
                except Exception as e:
                    logger.warning(f"자료 '{material.title}' 벡터화 중 오류 발생, 건너뜁니다: {str(e)}")
                    return False
            
            # 병렬 처리로 벡터화 실행
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for material in materials:
                    futures.append(executor.submit(vectorize_single_material, material))
                
                # 결과 수집
                results = []
                for future in as_completed(futures):
                    results.append(future.result())
            
            vectorized_count = sum(1 for r in results if r)
            logger.info(f"벡터화 완료: {vectorized_count}/{len(materials)} 자료")
            return True
            
        except Exception as e:
            logger.error(f"벡터화 과정 중 오류 발생: {str(e)}")
            traceback.print_exc()
            return False
        
    def save_research_materials_to_json(self, materials: List[Union[ResearchMaterial, Dict]], file_path: str = 'data/papers.json') -> None:
        """
        연구 자료를 JSON 파일로 저장합니다.
        
        Args:
            materials: 저장할 연구 자료 목록 (ResearchMaterial 객체 또는 딕셔너리)
            file_path: 저장할 파일 경로
        """
        try:
            # 디렉토리 생성
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 직렬화 가능한 형태로 변환
            materials_data = []
            for material in materials:
                try:
                    if hasattr(material, 'dict'):
                        # ResearchMaterial 객체인 경우
                        material_dict = material.dict()
                    elif isinstance(material, dict):
                        # 이미 딕셔너리인 경우
                        material_dict = material
                    else:
                        logger.warning(f"지원되지 않는 자료 유형: {type(material)}")
                        continue
                    
                    materials_data.append(material_dict)
                except Exception as e:
                    logger.warning(f"자료 JSON 처리 중 오류: {str(e)}")
                    continue
            
            # datetime 객체를 문자열로 변환하는 함수 정의
            def json_serial(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Type {type(obj)} not serializable")
            
            # JSON 파일로 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(materials_data, f, ensure_ascii=False, indent=2, default=json_serial)
            logger.info(f"연구 자료가 {file_path}에 저장됨")
            
        except Exception as e:
            logger.error(f"연구 자료를 JSON으로 저장하는 데 실패: {str(e)}")
            traceback.print_exc()

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

# Markdown 모듈 import 추가
try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

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
You are a research assistant helping me find literature for a literature review on the following topic:

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

    

    def collect_research_materials(self, topic, research_plan=None, max_queries=3, results_per_source=10, final_result_count=30):

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
                    try:
                        # 논문 평가 수행
                        evaluation_result, relevance_score = self.evaluate_source(paper, topic)
                        # 평가 결과와 점수로 연구 자료 생성 (파라미터 순서 변경: 평가 결과, 점수)
                        material = self._create_research_material(paper, "local", evaluation_result, relevance_score)
                        if material:
                            all_materials.append(material)
                    except Exception as e:
                        logger.warning(f"로컬 논문 처리 중 오류: {str(e)}")
                        # 오류 발생 시 기본값으로 자료 생성 시도
                        try:
                            material = self._create_research_material(
                                paper, 
                                "local", 
                                {"evaluation": "자동 평가", "rationale": "기본 평가"}, 
                                0.7  # 기본 점수
                            )
                            if material:
                                all_materials.append(material)
                        except Exception as inner_e:
                            logger.error(f"로컬 논문의 기본 처리도 실패: {str(inner_e)}")
                
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
                    try:
                        # 논문 평가 수행
                        evaluation_result, relevance_score = self.evaluate_source(result, topic)
                        # 평가 결과와 점수로 연구 자료 생성 (파라미터 순서 변경: 평가 결과, 점수)
                        material = self._create_research_material(result, "local_db", evaluation_result, relevance_score)
                        if material:
                            all_materials.append(material)
                    except Exception as e:
                        logger.warning(f"로컬 논문 처리 중 오류: {str(e)}")
                        # 오류 발생 시 기본값으로 자료 생성 시도
                        try:
                            material = self._create_research_material(
                                result, 
                                "local_db", 
                                {"evaluation": "자동 평가", "rationale": "기본 평가"}, 
                                0.7  # 기본 점수
                            )
                            if material:
                                all_materials.append(material)
                        except Exception as inner_e:
                            logger.error(f"로컬 논문의 기본 처리도 실패: {str(inner_e)}")
                
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
                                    evaluation_result, relevance_score = self.evaluate_source(paper_info, topic)
                                    
                                    material = self._create_research_material(paper_info, "pdfs", evaluation_result, relevance_score)
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
                    
                    # PDF URL이 있는 결과만 유지 (필터링 강화)
                    academic_results = [r for r in academic_results if r.get('pdf_url') and r.get('pdf_url').strip()]
                    web_results = [r for r in web_results if r.get('pdf_url') and r.get('pdf_url').strip()]
                    
                    # 검색 결과 로깅
                    logger.info(f"PDF URL이 있는 학술 검색 결과: {len(academic_results)}개")
                    logger.info(f"PDF URL이 있는 웹 검색 결과: {len(web_results)}개")
                    
                    # 결과 결합
                    combined_results = academic_results + web_results
                    
                    # 병렬 처리로 결과 평가
                    def evaluate_single_result(result):
                        try:
                            evaluation_result, relevance_score = self.evaluate_source(result, topic)
                            result['evaluation'] = evaluation_result
                            result['relevance_score'] = relevance_score
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
                            material = self._create_research_material(result, query.id, result.get('evaluation', ''), result.get('relevance_score', 0))
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

    

    def _create_research_material(self, result, query_id, evaluation_result, relevance_score):
        """
        자료로부터 ResearchMaterial 객체 생성 (helper 메서드)
        
        Args:
            result (dict): 자료 정보
            query_id (str): 쿼리 ID
            evaluation_result (dict/str): 평가 결과 객체 또는 문자열
            relevance_score (float): 관련성 점수
            
        Returns:
            ResearchMaterial: 생성된 연구 자료 객체
        """
        # PDF URL 또는 PDF 경로 확인 - PDF 자료가 없으면 건너뜀
        pdf_url = result.get('pdf_url')
        pdf_path = result.get('pdf_path')
        
        if not pdf_url and not pdf_path:
            logger.warning(f"PDF URL 또는 경로가 없어 자료를 생성하지 않습니다: {result.get('title', '제목 없음')}")
            return None
        
        # PDF 파일이 실제로 존재하는지 확인 (pdf_path가 있는 경우)
        if pdf_path and not os.path.exists(pdf_path):
            logger.warning(f"PDF 파일이 존재하지 않습니다: {pdf_path}")
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
                # 숫자로 변환할 수 없는 경우 None 대신 문자열로 설정
                # 이후 문자열이 int 필드에 들어갈 때 ResearchMaterial 생성 시 처리될 것임
                year_int = None
        else:
            year_int = year_value
        
        # 연구 자료 생성
        try:
            # 평가 결과 처리 - 문자열이나 딕셔너리로 들어올 수 있음
            if isinstance(evaluation_result, dict):
                evaluation_str = evaluation_result.get('evaluation', '') or evaluation_result.get('rationale', '')
            else:
                evaluation_str = str(evaluation_result) if evaluation_result else ''
            
            # 관련성 점수 확인
            if relevance_score is None:
                relevance_score = 0.5  # 기본값 설정
            
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
                evaluation=evaluation_str,  # 문자열로 변환된 평가
                query_id=query_id,
                content="",
                summary=""
            )
        except Exception as e:
            logger.error(f"ResearchMaterial 생성 중 오류: {str(e)}")
            return None

    

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
                        # PDF 없이 초록만 있는 경우 - 건너뜀
                        elif hasattr(material, 'abstract') and material.abstract:
                            logger.warning(f"PDF 없음, 자료를 건너뜁니다: {material.title}")
                            return None
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
                                    material.related_documents = related_info
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
        연구 자료를 분석하여 주요 인사이트를 추출합니다.
        
        Args:
            materials (List[ResearchMaterial]): 연구 자료 목록
            topic (str): 연구 주제
            
        Returns:
            dict: 분석 결과
        """
        if not materials:
            logger.warning("분석할 연구 자료가 없습니다. 기본 분석 결과를 반환합니다.")
            return {
                "topic": topic,
                "main_themes": ["정보 부족"],
                "chronology": [],
                "methodologies": [],
                "key_findings": ["분석할 자료가 충분하지 않습니다."],
                "research_gaps": ["자료 부족으로 연구 격차를 식별할 수 없습니다."],
                "relationships": [],
                "analysis_text": f"{topic}에 대한 연구 자료가 충분하지 않아 상세한 분석을 수행할 수 없습니다."
            }
            
        logger.info(f"{len(materials)}개 연구 자료 분석 중 (주제: {topic})")
        
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
        연구 주제, 분석 결과, 자료 기반으로 학술 논문 개요 생성.
        
        Args:
            topic (str): 연구 주제
            analysis (dict): 연구 자료 분석 결과
            materials (List[ResearchMaterial]): 연구 자료 목록
            
        Returns:
            dict: 논문 개요
        """
        logger.info(f"논문 개요 생성 시작: {topic[:100]}...")
        
        try:
            # 분석 결과와 자료 요약을 준비
            analysis_summary = ""
            if isinstance(analysis, dict):
                if 'analysis_text' in analysis:
                    analysis_summary = analysis['analysis_text'][:2000]  # 긴 분석 텍스트 처리
                else:
                    analysis_summary = json.dumps(analysis, ensure_ascii=False)[:2000]
            elif isinstance(analysis, str):
                analysis_summary = analysis[:2000]  # 긴 분석 텍스트 처리
            
            # 자료 요약 (최대 5개)
            material_summaries = []
            for mat in materials[:5]:
                if hasattr(mat, 'title') and hasattr(mat, 'authors'):
                    authors_str = ", ".join(mat.authors[:3]) if isinstance(mat.authors, list) else mat.authors
                    mat_summary = f"- {mat.title} by {authors_str}"
                    material_summaries.append(mat_summary)
            
            materials_text = "\n".join(material_summaries)
            
            # 프롬프트 생성 - 긴 연구 주제도 효과적으로 처리
            outline_prompt = f"""
당신은 학술 논문 작성을 돕는 전문 연구원입니다. 다음 연구 주제, 분석 결과, 자료를 바탕으로 학술 논문의 개요를 만들어주세요.

연구 주제:
{topic}

분석 결과 요약:
{analysis_summary}

주요 연구 자료 예시:
{materials_text}

학술 논문 개요에는 다음 항목이 포함되어야 합니다:
1. 적절한 논문 제목
2. 초록 (Abstract) - 연구의 목적, 방법, 결과, 시사점을 간략히 요약
3. 서론 (Introduction) - 연구 주제 소개 및 배경, 연구 목적, 연구 문제
4. 논문의 주요 섹션들 - 문헌 리뷰, 방법론, 결과, 토론 등을 포함하는 논리적 구조
5. 결론 및 향후 연구 방향

각 섹션에는 간략한 설명을 덧붙여주세요.

JSON 형태로 다음 형식에 맞게 응답해주세요:
{{
  "title": "논문 제목",
  "abstract": "초록 텍스트",
  "sections": [
    {{
      "title": "섹션 제목 (예: 서론)",
      "description": "섹션에 대한 간략한 설명"
    }},
    // 다른 섹션들도 같은 형식으로...
  ]
}}
"""
            
            # LLM을 사용하여 개요 생성
            response = self.llm.invoke(outline_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # JSON 형식 추출
            json_pattern = r"\{[\s\S]*\}"
            match = re.search(json_pattern, response_text)
            
            if match:
                json_text = match.group(0)
                outline = json.loads(json_text)
                logger.info(f"논문 개요 생성 성공: '{outline.get('title', '제목 정보 없음')}'")
                return outline
            else:
                logger.error("응답에서 JSON 형식을 찾을 수 없습니다. 텍스트 형식으로 처리합니다.")
                # JSON 변환 실패 시 기본 개요 반환
                return {
                    "title": f"{topic}에 관한 연구",
                    "abstract": "이 연구는 주제에 대한 문헌 리뷰와 분석을 제공합니다.",
                    "sections": [
                        {"title": "서론", "description": "연구 배경 및 목적"},
                        {"title": "관련 연구", "description": "선행 연구 분석"},
                        {"title": "방법론", "description": "연구 방법 설명"},
                        {"title": "결과 및 논의", "description": "주요 발견 및 논의"},
                        {"title": "결론", "description": "연구 결론 및 향후 방향"}
                    ]
                }
        
        except Exception as e:
            logger.error(f"논문 개요 생성 중 오류: {str(e)}")
            traceback.print_exc()
            return None



    def generate_report(self, topic, analysis, materials):
        """
        연구 자료를 바탕으로 문헌 리뷰 보고서를 생성합니다.
        기본적으로 영어로 보고서를 생성하고, 한글 번역본도 함께 저장합니다.
        
        Args:
            topic (str): 연구 주제
            analysis (dict): 연구 자료 분석 결과
            materials (List[ResearchMaterial]): 연구 자료 목록
            
        Returns:
            dict: 보고서 생성 결과
        """
        try:
            logger.info(f"'{topic}' 주제에 대한 문헌 리뷰 보고서 생성 시작")
            
            # 1. 참고문헌 추출
            references = self.extract_references(materials)
            
            # 참고문헌 수 제한 로직 추가 (materials에서 추출된 참고문헌이 많을 경우)
            # run 메서드에서 final_result_count의 기본값은 max_sources로 오버라이드되며 기본값은 30
            target_ref_count = getattr(self, 'final_result_count', 30)  # 기본값 30 설정
            
            # 참고문헌이 target_ref_count보다 많으면 제한
            if len(references) > target_ref_count:
                logger.info(f"참고문헌 {len(references)}개를 {target_ref_count}개로 제한합니다.")
                # 참고문헌을 정렬하여 가장 관련성이 높은 것만 유지
                # 정렬 기준: 1) 연도(최신순), 2) 제목(알파벳순)
                sorted_refs = sorted(references, 
                                     key=lambda ref: (
                                         -int(ref.get('year', '0') or '0'),  # 연도 내림차순(최신순), 숫자로 변환 불가능하면 0으로 처리
                                         ref.get('title', '')                # 제목 오름차순
                                     ))
                references = sorted_refs[:target_ref_count]
            
            # 2. 보고서 아웃라인 생성
            outline = self.create_paper_outline(topic, analysis, materials)
            if not outline:
                logger.error("보고서 아웃라인 생성에 실패했습니다.")
                return None
                
            logger.info(f"논문 아웃라인 생성됨: {len(outline.get('sections', []))}개 섹션")
            
            # 3. 보고서 내용 구성 (영어)
            # 제목만 포함하고 초록 제외
            # 영어 제목 사용
            title = outline.get('title', 'History and Development of Topic Modeling')
            report_content = f"# {title}\n\n"
            
            # 사용자 제공 목차 확인
            user_toc = getattr(self, 'user_toc', None)
            
            # 목차 결정 (사용자 제공 또는 자동 생성)
            if user_toc:
                logger.info(f"사용자가 제공한 목차를 사용합니다: {user_toc}")
                toc = user_toc
            else:
                logger.info("자동으로 목차를 생성합니다.")
                try:
                    # 영어 섹션 제목 사용을 위한 목차 생성
                    outline_text = outline.get('outline', '')
                    if not outline_text:
                        raise ValueError("아웃라인 텍스트가 비어있습니다.")
                    
                    toc_items = []
                    lines = outline_text.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line:
                            parts = line.split('.', 1)
                            if len(parts) < 2:
                                continue
                            section_info = parts[1].strip()
                            if section_info:
                                title_parts = section_info.split('-', 1) if '-' in section_info else [section_info, '']
                                section_title = title_parts[0].strip()
                                section_desc = title_parts[1].strip() if len(title_parts) > 1 else ''
                                toc_items.append({
                                    'title': section_title,
                                    'description': section_desc
                                })
                    
                    # 최소 2개 이상의 섹션 확보 (영어 제목 사용)
                    if len(toc_items) < 2:
                        toc_items = [
                            {'title': 'Introduction', 'description': 'Research background and purpose'},
                            {'title': 'Related Works', 'description': 'Analysis of related research trends'}
                        ]
                    
                    toc = toc_items
                    
                except Exception as e:
                    logger.error(f"목차 자동 생성 중 오류 발생: {e}")
                    # 기본 목차 설정 (영어)
                    toc = [
                        {'title': 'Introduction', 'description': 'Research background and purpose'},
                        {'title': 'Related Works', 'description': 'Analysis of related research trends'}
                    ]
            
            # 각 섹션에 대한 내용 생성
            section_count = 0
            for section in toc:
                section_title = section.get('title', 'Untitled')
                section_desc = section.get('description', '')
                
                report_content += f"## {section_title}\n\n"
                section_count += 1
                
                try:
                    # 각 섹션에 대한 상세 내용 생성 (영어로 프롬프트 작성)
                    section_prompt = f"""
                    The following is a section for a literature review paper titled '{title}':
                    
                    Section: {section_title}
                    Description: {section_desc}
                    
                    Please write approximately 500-700 words of academic and detailed content for this section.
                    Topic: {topic}
                    Analysis: {analysis.get('main_themes', '')}
                    
                    Write in an academic paper style and include appropriate subheadings.
                    """
                    
                    # 서론(Introduction)인 경우 특별한 프롬프트 사용
                    if section_title.lower() in ['introduction', 'intro', '서론', '소개', 'introduce']:
                        # 참고문헌 정보를 문자열로 변환하여 제공
                        references_info = ""
                        for i, ref in enumerate(references[:10], 1):  # 너무 많은 정보를 주지 않기 위해 10개로 제한
                            author = ref.get('authors', 'Unknown Author')
                            if isinstance(author, list) and len(author) > 0:
                                author_text = ', '.join(author[:3])
                                if len(ref.get('authors', [])) > 3:
                                    author_text += ' et al.'
                            else:
                                author_text = str(author)
                            
                            title = ref.get('title', 'Unknown Title')
                            year = ref.get('year', '')
                            references_info += f"{i}. {author_text} ({year}). {title}\n"
                        
                        # 사용자 지정 서론 단어 수 확인
                        intro_words_text = "approximately 400-600 words"
                        if hasattr(self, 'intro_words') and self.intro_words:
                            intro_words_text = f"approximately {self.intro_words} words"
                        
                        section_prompt = f"""
                        Please write an introduction for a research paper on the following topic:
                        
                        Topic: {topic}
                        
                        The introduction should include:
                        1. Introduction to the research topic and background
                        2. Purpose and significance of the research
                        3. Main research questions
                        4. Structure of the report
                        
                        Important: You must cite appropriate academic literature in the introduction. Choose at least 3-5 references from the list below to cite:
                        
                        Reference List:
                        {references_info}
                        
                        Citation format: Use either "(Author, Year)" or "Author(Year) states..." format for in-text citations.
                        Example: "Recent research (Smith et al., 2022) suggests..." or "Johnson(2020) argues..."
                        
                        Please write an academic and clear introduction in {intro_words_text}. 
                        Cite the selected literature to support your arguments in the introduction.
                        """
                    
                    # 관련 연구(Related Works)인 경우 특별한 프롬프트 사용
                    elif section_title.lower() in ['related works', 'related work', '관련 연구', '선행 연구']:
                        # 사용자 지정 관련 연구 단어 수 확인
                        related_works_words_text = "approximately 700-800 words"
                        if hasattr(self, 'related_works_words') and self.related_works_words:
                            related_works_words_text = f"approximately {self.related_works_words} words"
                        
                        # 영어로 관련 연구 프롬프트 작성
                        section_prompt = f"""
                        Please write a literature review for the following research topic:
                        
                        Topic: {topic}
                        
                        Write a comprehensive literature review in {related_works_words_text}, covering:
                        1. Key themes and trends in the research area
                        2. Major insights from existing literature
                        3. Research gaps and contradictions in current knowledge
                        4. How your research relates to existing knowledge
                        
                        Important: Properly cite at least 7-10 references from the literature. Use the reference list provided below.
                        
                        Reference List:
                        {references_info}
                        
                        Citation format: Use either "(Author, Year)" or "Author(Year) states..." format for in-text citations.
                        Example: "Recent research (Smith et al., 2022) suggests..." or "Johnson(2020) argues..."
                        
                        Organize the literature review by themes rather than simply listing studies chronologically.
                        """
                    
                    # 각 섹션에 대한 내용 생성
                    section_content = self.llm.chat_completion([
                        {"role": "system", "content": "You are an expert academic writer. You write in clear, concise, and scholarly English. Use academic language appropriate for a literature review."},
                        {"role": "user", "content": section_prompt}
                    ])
                    
                    if section_content:
                        report_content += f"{section_content}\n\n"
                    else:
                        report_content += "Content generation failed for this section.\n\n"
                    
                except Exception as e:
                    logger.error(f"섹션 '{section_title}' 내용 생성 중 오류 발생: {e}")
                    report_content += f"{section_desc}\n\n"
            
            # 참고문헌 추가
            report_content += "## References\n\n"
            
            # 참고문헌 포맷팅 (이미 상단에서 추출했음)
            try:
                for i, ref in enumerate(references, 1):
                    author = ref.get('authors', 'Unknown Author')
                    title = ref.get('title', 'Unknown Title')
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
            
            # 4. 한글 번역본 생성
            logger.info("영어 보고서 생성 완료, 한글 번역본 생성 시작")
            korean_report_content = self._translate_report_to_korean(report_content)
            
            # 5. 보고서 저장 (영어 및 한글 버전)
            os.makedirs('output', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 영어 버전 저장
            english_output_file = f'output/literature_review_en_{timestamp}.md'
            with open(english_output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"영어 문헌 리뷰 보고서가 '{english_output_file}'에 저장되었습니다.")
            
            # 한글 버전 저장
            korean_output_file = f'output/literature_review_ko_{timestamp}.md'
            with open(korean_output_file, 'w', encoding='utf-8') as f:
                f.write(korean_report_content)
            logger.info(f"한글 문헌 리뷰 보고서가 '{korean_output_file}'에 저장되었습니다.")
            
            # 6. HTML 변환 시도 (선택적, 영어 버전만)
            if MARKDOWN_AVAILABLE:
                try:
                    html_file = english_output_file.replace('.md', '.html')
                    with open(english_output_file, 'r', encoding='utf-8') as md_file:
                        # 'extra' 및 'tables' 확장을 사용하여 더 풍부한 마크다운 기능 지원
                        html_content = markdown.markdown(md_file.read(), extensions=['extra', 'tables'])
                    
                    # 향상된 CSS 스타일로 HTML 파일 생성
                    with open(html_file, 'w', encoding='utf-8') as f:
                        f.write(f"""<!DOCTYPE html>
                        <html>
                        <head>
                            <meta charset="utf-8">
                            <meta name="viewport" content="width=device-width, initial-scale=1.0">
                            <title>{title}</title>
                            <style>
                                body {{ 
                                    font-family: 'Segoe UI', Arial, sans-serif; 
                                    line-height: 1.6; 
                                    max-width: 900px; 
                                    margin: 0 auto; 
                                    padding: 30px; 
                                    color: #333;
                                    background-color: #fdfdfd;
                                }}
                                h1 {{ 
                                    color: #2c3e50; 
                                    border-bottom: 2px solid #eaecef;
                                    padding-bottom: 10px;
                                }}
                                h2 {{ 
                                    color: #3498db; 
                                    margin-top: 30px;
                                    border-bottom: 1px solid #eaecef;
                                    padding-bottom: 7px;
                                }}
                                h3 {{ 
                                    color: #2980b9; 
                                    margin-top: 25px;
                                }}
                                blockquote {{ 
                                    border-left: 4px solid #4a90e2; 
                                    padding: 10px 15px; 
                                    margin: 20px 0; 
                                    background-color: #f8f9fa;
                                    color: #555; 
                                }}
                                code {{ 
                                    background-color: #f6f8fa; 
                                    padding: 2px 4px; 
                                    border-radius: 3px;
                                    font-family: Consolas, monospace;
                                }}
                                pre {{ 
                                    background-color: #f6f8fa; 
                                    padding: 16px; 
                                    border-radius: 5px;
                                    overflow-x: auto;
                                }}
                                a {{ 
                                    color: #4183c4; 
                                    text-decoration: none;
                                }}
                                a:hover {{ 
                                    text-decoration: underline; 
                                }}
                                table {{
                                    border-collapse: collapse;
                                    width: 100%;
                                    margin: 25px 0;
                                }}
                                table, th, td {{
                                    border: 1px solid #ddd;
                                }}
                                th, td {{
                                    padding: 10px;
                                    text-align: left;
                                }}
                                th {{
                                    background-color: #f2f2f2;
                                }}
                                tr:nth-child(even) {{
                                    background-color: #f9f9f9;
                                }}
                                img {{
                                    max-width: 100%;
                                    height: auto;
                                }}
                                .references {{
                                    margin-top: 40px;
                                    border-top: 1px solid #eaecef;
                                    padding-top: 20px;
                                }}
                                .timestamp {{
                                    color: #888;
                                    font-size: 0.9em;
                                    margin-top: 40px;
                                    text-align: right;
                                }}
                            </style>
                        </head>
                        <body>
                        {html_content}
                        <div class="timestamp">Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
                        </body>
                        </html>""")
                    
                    logger.info(f"HTML 버전의 보고서가 '{html_file}'에 저장되었습니다.")
                    
                    # 한글 버전도 HTML로 변환
                    korean_html_file = korean_output_file.replace('.md', '.html')
                    with open(korean_output_file, 'r', encoding='utf-8') as md_file:
                        korean_html_content = markdown.markdown(md_file.read(), extensions=['extra', 'tables'])
                    
                    with open(korean_html_file, 'w', encoding='utf-8') as f:
                        f.write(f"""<!DOCTYPE html>
                        <html>
                        <head>
                            <meta charset="utf-8">
                            <meta name="viewport" content="width=device-width, initial-scale=1.0">
                            <title>{title} (한글)</title>
                            <style>
                                body {{ 
                                    font-family: 'Malgun Gothic', 'Segoe UI', Arial, sans-serif; 
                                    line-height: 1.6; 
                                    max-width: 900px; 
                                    margin: 0 auto; 
                                    padding: 30px; 
                                    color: #333;
                                    background-color: #fdfdfd;
                                }}
                                h1 {{ 
                                    color: #2c3e50; 
                                    border-bottom: 2px solid #eaecef;
                                    padding-bottom: 10px;
                                }}
                                h2 {{ 
                                    color: #3498db; 
                                    margin-top: 30px;
                                    border-bottom: 1px solid #eaecef;
                                    padding-bottom: 7px;
                                }}
                                h3 {{ 
                                    color: #2980b9; 
                                    margin-top: 25px;
                                }}
                                blockquote {{ 
                                    border-left: 4px solid #4a90e2; 
                                    padding: 10px 15px; 
                                    margin: 20px 0; 
                                    background-color: #f8f9fa;
                                    color: #555; 
                                }}
                                code {{ 
                                    background-color: #f6f8fa; 
                                    padding: 2px 4px; 
                                    border-radius: 3px;
                                    font-family: Consolas, monospace;
                                }}
                                pre {{ 
                                    background-color: #f6f8fa; 
                                    padding: 16px; 
                                    border-radius: 5px;
                                    overflow-x: auto;
                                }}
                                a {{ 
                                    color: #4183c4; 
                                    text-decoration: none;
                                }}
                                a:hover {{ 
                                    text-decoration: underline; 
                                }}
                                table {{
                                    border-collapse: collapse;
                                    width: 100%;
                                    margin: 25px 0;
                                }}
                                table, th, td {{
                                    border: 1px solid #ddd;
                                }}
                                th, td {{
                                    padding: 10px;
                                    text-align: left;
                                }}
                                th {{
                                    background-color: #f2f2f2;
                                }}
                                tr:nth-child(even) {{
                                    background-color: #f9f9f9;
                                }}
                                img {{
                                    max-width: 100%;
                                    height: auto;
                                }}
                                .references {{
                                    margin-top: 40px;
                                    border-top: 1px solid #eaecef;
                                    padding-top: 20px;
                                }}
                                .timestamp {{
                                    color: #888;
                                    font-size: 0.9em;
                                    margin-top: 40px;
                                    text-align: right;
                                }}
                            </style>
                        </head>
                        <body>
                        {korean_html_content}
                        <div class="timestamp">생성 시간: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
                        </body>
                        </html>""")
                    
                    logger.info(f"한글 HTML 버전의 보고서가 '{korean_html_file}'에 저장되었습니다.")
                    
                except Exception as e:
                    logger.error(f"HTML 변환 중 오류 발생: {e}")
            else:
                logger.warning("markdown 모듈이 설치되지 않아 HTML 변환을 건너뜁니다. 'pip install markdown'으로 설치할 수 있습니다.")
            
            return {
                "success": True,
                "english_file_path": english_output_file,
                "korean_file_path": korean_output_file,
                "section_count": section_count,
                "reference_count": len(references)
            }
            
        except Exception as e:
            logger.error(f"보고서 생성 중 오류 발생: {e}")
            return None
            
    def _translate_report_to_korean(self, english_report):
        """
        영어 보고서를 한글로 번역합니다.
        
        Args:
            english_report (str): 영어로 작성된 보고서 내용
            
        Returns:
            str: 한글로 번역된 보고서 내용
        """
        logger.info("영어 보고서를 한글로 번역하는 작업 시작")
        
        try:
            # 보고서 제목 추출
            title_match = re.match(r'# (.*?)\n', english_report)
            title = title_match.group(1) if title_match else "Literature Review"
            
            # 보고서를 섹션으로 분할
            sections = re.split(r'(## .*?\n)', english_report)
            
            translated_report = ""
            
            # 제목 번역
            translation_prompt = f"""
            Translate the following academic paper title from English to Korean:
            
            {title}
            
            Provide only the Korean translation with no additional text or explanation.
            """
            
            translated_title = self.llm.chat_completion([
                {"role": "system", "content": "You are a professional translator specializing in academic content translation from English to Korean."},
                {"role": "user", "content": translation_prompt}
            ])
            
            translated_report += f"# {translated_title}\n\n"
            
            # 각 섹션 번역
            for i in range(1, len(sections)):
                section = sections[i]
                
                # 섹션 제목 부분인 경우
                if section.startswith('## '):
                    # 섹션 제목 번역
                    section_title = section.strip().replace('## ', '')
                    
                    translation_prompt = f"""
                    Translate the following section title from English to Korean:
                    
                    {section_title}
                    
                    Provide only the Korean translation with no additional text or explanation.
                    """
                    
                    translated_section_title = self.llm.chat_completion([
                        {"role": "system", "content": "You are a professional translator specializing in academic content translation from English to Korean."},
                        {"role": "user", "content": translation_prompt}
                    ])
                    
                    translated_report += f"## {translated_section_title}\n\n"
                
                # 섹션 내용 부분인 경우
                else:
                    # References 섹션은 번역하지 않고 원본 그대로 사용
                    if i > 0 and sections[i-1].strip() == '## References':
                        translated_report += section
                        continue
                    
                    # 내용이 있는 경우만 번역
                    if section.strip():
                        # 섹션 내용 번역
                        translation_prompt = f"""
                        Translate the following academic content from English to Korean.
                        Preserve all formatting, including paragraphs, bullet points, and citations.
                        Keep all citations in their original format. For example, "(Smith et al., 2022)" should remain unchanged.
                        Keep all technical terms appropriately translated in the academic context.
                        
                        Content to translate:
                        
                        {section}
                        
                        Provide only the Korean translation with no additional text or explanation.
                        """
                        
                        translated_content = self.llm.chat_completion([
                            {"role": "system", "content": "You are a professional translator specializing in academic content translation from English to Korean."},
                            {"role": "user", "content": translation_prompt}
                        ])
                        
                        translated_report += f"{translated_content}\n\n"
            
            logger.info("영어 보고서의 한글 번역 완료")
            return translated_report
            
        except Exception as e:
            logger.error(f"보고서 번역 중 오류 발생: {e}")
            # 오류 발생 시 원본 영어 보고서 반환
            return english_report

    

    def run(self, topic=None, **kwargs):
        """
        연구 에이전트 실행 메서드
        
        Args:
            topic (str): 연구 주제 - 길이 제한 없이 상세한 주제를 입력할 수 있습니다.
            **kwargs: 추가 매개변수
                - max_sources (int): 최대 소스 수 (기본값: 30)
                - toc (List[Dict]): 사용자가 제공하는 목차 (선택 사항)
                    형식: [{'title': '제목1', 'description': '설명1'}, ...]
                - citation_style (str): 인용 스타일 (기본값: 'APA')
                - intro_words (int): 서론 섹션 단어 수 (기본값: 400-600)
                - related_works_words (int): 관련 연구 섹션 단어 수 (기본값: 700-800)
            
        Returns:
            dict: 연구 결과
        """
        if not topic:
            logger.error("연구 주제가 제공되지 않았습니다.")
            return {"status": "error", "message": "연구 주제가 필요합니다."}
        
        # 연구 주제 로깅 및 길이 체크
        topic_length = len(topic)
        token_estimate = topic_length // 4  # 대략적인 토큰 수 추정
        logger.info(f"연구 주제 입력됨: 길이 {topic_length}자 (약 {token_estimate} 토큰)")
        
        # 주제가 매우 긴 경우 (10,000자 이상) 경고 로그 추가
        if topic_length > 10000:
            logger.warning(f"연구 주제가 매우 깁니다 ({topic_length}자). 처리 시간이 오래 걸릴 수 있습니다.")
        
        try:
            # 1. 연구 자료 수집
            max_sources = kwargs.get("max_sources", 30)  # 기본값 30으로 변경
            # final_result_count 값을 인스턴스 변수로 저장
            self.final_result_count = max_sources
            
            # 사용자가 제공한 목차가 있으면 저장 (generate_report에서 사용)
            user_toc = kwargs.get("toc", None)
            if user_toc:
                if isinstance(user_toc, list) and len(user_toc) > 0:
                    logger.info(f"사용자가 {len(user_toc)}개의 목차 항목을 제공했습니다.")
                    self.user_toc = user_toc
                else:
                    logger.warning("사용자가 제공한 목차 형식이 올바르지 않아 무시합니다.")
                    self.user_toc = None
            else:
                self.user_toc = None
            
            # 서론과 관련 연구 섹션의 분량 설정 (사용자 지정 가능)
            self.intro_words = kwargs.get("intro_words", None)
            self.related_works_words = kwargs.get("related_works_words", None)
            
            if self.intro_words:
                logger.info(f"사용자가 지정한 서론 단어 수: {self.intro_words}")
            if self.related_works_words:
                logger.info(f"사용자가 지정한 관련 연구 단어 수: {self.related_works_words}")
            
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
            enriched_materials = []
            try:
                logger.info("연구 자료 강화 시작...")
                enriched_materials = self.enrich_research_materials(materials)
                logger.info(f"{len(enriched_materials)}개 자료 강화 완료")
            except Exception as e:
                logger.error(f"연구 자료 강화 중 오류 발생: {str(e)}")
                # 오류 발생 시 원본 자료 사용
                enriched_materials = materials
                logger.warning("오류로 인해 원본 자료를 사용합니다.")
            
            # 연구 자료 JSON 파일로 저장
            try:
                self.save_research_materials_to_json(enriched_materials)
                logger.info("연구 자료 JSON 파일 저장 완료")
            except Exception as e:
                logger.error(f"연구 자료 JSON 저장 중 오류: {str(e)}")
            
            # 병렬로 연구 자료 벡터화
            try:
                logger.info("연구 자료 병렬 벡터화 시작...")
                vectorize_result = self.vectorize_materials(enriched_materials)
                if vectorize_result:
                    logger.info("연구 자료 벡터화 완료")
                else:
                    logger.warning("연구 자료 벡터화가 일부 실패했습니다.")
            except Exception as e:
                logger.error(f"연구 자료 벡터화 중 오류 발생: {str(e)}")

            # 3. 연구 자료 분석
            analysis = self.analyze_research_materials(enriched_materials, topic)
            
            # 4. 논문 개요 생성 (비활성화)
            logger.info("논문 개요 생성 단계는 비활성화되었습니다")
            
            # 기본 개요 생성
            outline = {
                "title": f"{topic}에 관한 연구",
                "abstract": f"{topic}에 대한 문헌 검토입니다.",
                "sections": [
                    {
                        "title": "서론",
                        "content": "연구 배경 및 목적"
                    },
                    {
                        "title": "방법론",
                        "content": "연구 방법론 설명"
                    },
                    {
                        "title": "주요 결과",
                        "content": "주요 연구 결과"
                    },
                    {
                        "title": "논의 및 결론",
                        "content": "연구 결과에 대한 논의 및 결론"
                    }
                ],
                "references": self.extract_references(enriched_materials)
            }

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

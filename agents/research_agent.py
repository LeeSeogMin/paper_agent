"""
Research Agent for academic paper writing.

This module contains the ResearchAgent class, which is responsible for gathering,
analyzing, and organizing research materials for academic paper writing.
"""

import os
import re
import json
import time
from typing import List, Dict, Any, Optional, Tuple

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.schema import Document
from langchain.tools import tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

from utils.logger import logger
from utils.vector_db import create_vector_db, search_vector_db, process_and_vectorize_paper
from utils.api_clients import search_google_scholar, search_academic_papers
from utils.pdf_processor import extract_text_from_pdf
from utils.search_utils import academic_search, fallback_search
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


class ResearchAgent(BaseAgent):
    """
    Agent responsible for research activities in the paper writing process.
    
    This agent searches for relevant papers, extracts information, and organizes
    findings for use in the paper writing phase.
    """
    
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.2):
        """
        Initialize the research agent.
        
        Args:
            model_name (str): Name of the large language model to use
            temperature (float): Temperature parameter for LLM
        """
        super().__init__(model_name, temperature)
        
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        
        # Initialize chains
        self.query_chain = LLMChain(
            llm=self.llm,
            prompt=QUERY_GENERATION_PROMPT
        )
        
        self.evaluation_chain = LLMChain(
            llm=self.llm,
            prompt=SOURCE_EVALUATION_PROMPT
        )
        
        self.analysis_chain = LLMChain(
            llm=self.llm,
            prompt=SEARCH_RESULTS_ANALYSIS_PROMPT
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
        Generate search queries based on the paper topic.
        
        Args:
            topic (str): Paper topic
            n_queries (int): Number of queries to generate
            
        Returns:
            List[SearchQuery]: List of search query objects
        """
        logger.info(f"Generating {n_queries} search queries for topic: {topic}")
        
        try:
            result = self.query_chain.invoke({
                "topic": topic,
                "n_queries": n_queries
            })
            
            # Parse the result
            queries_text = result['text'].strip()
            
            queries = []
            for line in queries_text.split('\n'):
                if not line.strip():
                    continue
                    
                # Extract query and optional rationale
                parts = line.split(':', 1)
                if len(parts) == 2:
                    query = parts[1].strip()
                    try:
                        # Try to extract a query number from the first part
                        query_id = re.search(r'\d+', parts[0]).group(0)
                        rationale = parts[0].replace(query_id, "").strip()
                    except (AttributeError, IndexError):
                        query_id = str(len(queries) + 1)
                        rationale = parts[0].strip()
                else:
                    query = line.strip()
                    query_id = str(len(queries) + 1)
                    rationale = "General information search"
                
                queries.append(SearchQuery(
                    id=f"q{query_id}",
                    text=query,
                    rationale=rationale
                ))
            
            logger.info(f"Generated {len(queries)} search queries")
            return queries
            
        except Exception as e:
            logger.error(f"Error generating search queries: {str(e)}", exc_info=True)
            # Return a basic query as fallback
            return [SearchQuery(
                id="q1",
                text=f"recent research on {topic}",
                rationale="Fallback query due to error"
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
    
    def evaluate_source(self, source: Dict, topic: str) -> Tuple[float, str]:
        """
        Evaluate a research source for relevance to the topic.
        
        Args:
            source (Dict): Source data
            topic (str): Research topic
            
        Returns:
            Tuple[float, str]: (relevance score, explanation)
        """
        logger.info(f"Evaluating source: {source.get('title', 'Unknown source')}")
        
        try:
            # 소스 형식 표준화 - Google 검색과 학술 검색 통합 처리
            formatted_source = {
                "title": source.get('title', '제목 없음'),
                "abstract": source.get('abstract', source.get('snippet', '')),
                "authors": source.get('authors', ''),
                "year": source.get('year', source.get('published_date', '연도 미상')),
                "venue": source.get('venue', source.get('source', 'Web Source')),
                "citation_count": source.get('citation_count', 0)
            }
            
            # LLM에게 간단한 평가만 요청
            messages = [
                SystemMessage(content="You are a research assistant evaluating the relevance of sources."),
                HumanMessage(content=f"""
                    Evaluate the relevance of this source to the topic "{topic}".
                    
                    SOURCE:
                    Title: {formatted_source['title']}
                    Abstract: {formatted_source['abstract'][:500]}...
                    
                    Rate the relevance from 0.0 to 1.0 and explain why.
                    Format: {{
                        "relevance_score": [0.0-1.0],
                        "explanation": "Your explanation"
                    }}
                """)
            ]
            
            response = self.llm.invoke(messages)
            
            # 응답 파싱 (JSON 형식으로 반환되었다고 가정)
            response_text = response.content
            
            # JSON 부분 추출 시도
            if '{' in response_text and '}' in response_text:
                json_str = response_text[response_text.find('{'):response_text.rfind('}')+1]
                result = json.loads(json_str)
            else:
                # JSON 형식이 아니면 텍스트에서 점수 추출 시도
                score_match = re.search(r'relevance_score["\s:]+([0-9.]+)', response_text)
                score = float(score_match.group(1)) if score_match else 0.7
                
                result = {
                    "relevance_score": score,
                    "explanation": response_text
                }
            
            relevance_score = result.get('relevance_score', 0.7)
            explanation = result.get('explanation', 'No explanation provided')
            
            # 점수가 문자열이면 변환
            if isinstance(relevance_score, str):
                try:
                    relevance_score = float(relevance_score)
                except:
                    relevance_score = 0.7
            
            return relevance_score, explanation
            
        except Exception as e:
            logger.error(f"Error evaluating source: {str(e)}", exc_info=True)
            return 0.7, f"Error evaluating source: {str(e)}. Using default relevance score."
    
    def collect_research_materials(self, topic, max_queries=3, max_results_per_query=5):
        """
        연구 자료 수집 통합 함수
        """
        logger.info(f"Collecting research materials for topic: {topic}")
        
        # Generate search queries
        search_queries = self.generate_search_queries(topic, n_queries=max_queries)
        
        all_sources = []
        materials = []
        
        # 먼저 로컬 데이터에서 검색
        try:
            local_results = self.search_local_papers(topic, max_results=max_results_per_query)
            if local_results:
                logger.info(f"로컬 데이터에서 {len(local_results)}개 논문 찾음")
                
                # 각 로컬 결과를 처리
                for result in local_results:
                    # 소스 평가
                    relevance_score, explanation = self.evaluate_source(result, topic)
                    
                    if relevance_score >= 0.6:
                        # 저자 처리: 문자열이면 리스트로 변환
                        if isinstance(result.get('authors', ''), str):
                            authors_list = [author.strip() for author in result.get('authors', '').split(',') if author.strip()]
                        else:
                            authors_list = result.get('authors', [])
                        
                        # 연도 처리: 날짜 형식이면 연도만 추출
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
                        material = ResearchMaterial(
                            id=result.get("id") or f"local_{len(materials)}",
                            title=result.get("title", ""),
                            authors=authors_list,  # 리스트로 변환된 저자
                            year=year_int,  # 정수로 변환된 연도
                            abstract=result.get("abstract", ""),
                            url=result.get("url", ""),
                            pdf_url=result.get("pdf_url", ""),
                            relevance_score=relevance_score,
                            evaluation=explanation,
                            query_id="local",
                            content=result.get("content", ""),
                            summary=result.get("summary", "")
                        )
                        
                        materials.append(material)
                        all_sources.append(result)
        except Exception as e:
            logger.warning(f"로컬 데이터 검색 중 오류: {str(e)}")
        
        # 온라인 검색
        for query in search_queries:
            logger.info(f"Processing query: {query.text}")
            
            # 학술 논문 검색
            logger.info(f"검색 쿼리 '{query.text}'로 논문 검색 중...")
            papers = self.search_academic_papers(query.text, max_results_per_query)
            
            # 웹 검색 (Google)
            web_results = google_search(query.text, num_results=max_results_per_query)
            
            # 학술 및 웹 결과 결합
            combined_results = papers + web_results
            
            # 각 결과 처리
            for result in combined_results:
                # 이미 처리된 결과면 건너뛰기
                if any(s.get("url") == result.get("url") for s in all_sources):
                    continue
                
                # 소스 평가
                relevance_score, explanation = self.evaluate_source(result, topic)
                
                # 관련성 기준 이상인 경우만 포함
                if relevance_score >= 0.6:
                    # 저자 처리: 문자열이면 리스트로 변환
                    if isinstance(result.get('authors', ''), str):
                        authors_list = [author.strip() for author in result.get('authors', '').split(',') if author.strip()]
                    else:
                        authors_list = result.get('authors', [])
                    
                    # 연도 처리: 날짜 형식이면 연도만 추출
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
                    material = ResearchMaterial(
                        id=result.get("id") or f"paper_{len(materials)}",
                        title=result.get("title", ""),
                        authors=authors_list,  # 리스트로 변환된 저자
                        year=year_int,  # 정수로 변환된 연도
                        abstract=result.get("abstract", ""),
                        url=result.get("url", ""),
                        pdf_url=result.get("pdf_url", ""),
                        relevance_score=relevance_score,
                        evaluation=explanation,
                        query_id=query.id,
                        content="",
                        summary=""
                    )
                    
                    materials.append(material)
                    all_sources.append(result)
            
            # 충분한 자료를 수집했으면 중단
            if len(materials) >= max_results_per_query * 2:
                break
        
        logger.info(f"Collected {len(materials)} research materials")
        
        # 검색 결과가 없으면 오류 발생
        if len(materials) == 0:
            logger.error("검색 결과가 없습니다. 프로세스를 중단합니다.")
            raise ValueError("검색 결과가 없어 연구를 진행할 수 없습니다. 다른 주제나 검색어를 시도해보세요.")
        
        return materials[:max_results_per_query * 2]
    
    def extract_content_from_pdf(self, pdf_url: str) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_url (str): URL to PDF file
            
        Returns:
            str: Extracted text
        """
        logger.info(f"Extracting content from PDF: {pdf_url}")
        
        try:
            text = extract_text_from_pdf(pdf_url)
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
            str: Summary text
        """
        logger.info(f"Summarizing content for: {title}")
        
        try:
            # Handle empty or very short content
            if not content or len(content) < 100:
                return "Insufficient content available for summarization."
            
            # Split text into chunks for processing
            text_chunks = self.text_splitter.split_text(content)
            
            # Join chunks with newlines for processing
            combined_text = "\n\n".join(text_chunks)
            combined_text = combined_text[:8000] + "..." if len(combined_text) > 8000 else combined_text
            
            # Generate summary
            result = self.summary_chain.invoke({"content": combined_text})
            summary = result["text"]
            
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing content: {str(e)}", exc_info=True)
            return "Error generating summary due to technical issues."
    
    def enrich_research_materials(self, materials: List[ResearchMaterial]) -> List[ResearchMaterial]:
        """
        Enrich research materials with content and summaries.
        
        Args:
            materials (List[ResearchMaterial]): List of materials to enrich
            
        Returns:
            List[ResearchMaterial]: Enriched materials
        """
        logger.info(f"Enriching {len(materials)} research materials")
        
        enriched_materials = []
        
        for material in materials:
            logger.info(f"자료 강화 중: {material.title}")
            
            # 논문 정보 사전 생성
            paper_info = {
                "id": material.id,
                "title": material.title,
                "authors": material.authors,
                "year": material.year,
                "abstract": material.abstract,
                "url": material.url,
                "pdf_url": material.pdf_url,
                "source": getattr(material, 'source', 'unknown')
            }
            
            # 벡터 DB에 저장 (PDF 다운로드 및 처리 포함)
            process_and_vectorize_paper(paper_info, material.pdf_url)
            
            # PDF가 있는 경우 내용 추출
            if material.pdf_url and not material.content:
                try:
                    # PDF 다운로드 및 텍스트 추출
                    pdf_path = get_local_pdf_path(material.id, material.pdf_url)
                    
                    if pdf_path:
                        text = self.extract_content_from_pdf(pdf_path)
                        material.content = text
                        
                        # 내용이 있으면 요약 생성
                        if text:
                            material.summary = self.summarize_content(text, material.title)
                except Exception as e:
                    logger.error(f"PDF 처리 중 오류 발생: {str(e)}", exc_info=True)
            
            # 벡터 DB에서 관련 정보 검색 추가
            if material.abstract:
                try:
                    similar_docs = search_vector_db(material.abstract, top_k=3)
                    
                    # 유사 문서 정보 추가
                    if similar_docs:
                        related_info = []
                        for doc, score in similar_docs:
                            if doc.metadata.get('paper_id') != material.id:  # 자기 자신 제외
                                related_info.append({
                                    'title': doc.metadata.get('title', '관련 자료'),
                                    'similarity': f"{score:.2f}",
                                    'content': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content
                                })
                        
                        # 관련 정보 추가
                        if related_info:
                            material.related_info = related_info
                except Exception as e:
                    logger.error(f"벡터 DB 검색 중 오류 발생: {str(e)}", exc_info=True)
            
            enriched_materials.append(material)
        
        logger.info(f"Enriched {len(enriched_materials)} research materials")
        return enriched_materials
    
    def analyze_research_materials(
        self, 
        materials: List[ResearchMaterial],
        topic: str
    ) -> Dict[str, Any]:
        """
        Analyze collected research materials.
        
        Args:
            materials (List[ResearchMaterial]): List of research materials
            topic (str): Paper topic
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        logger.info(f"Analyzing {len(materials)} research materials for topic: {topic}")
        
        try:
            # Prepare materials for analysis
            materials_data = []
            
            for material in materials:
                material_info = {
                    "id": material.id,
                    "title": material.title,
                    "authors": [a.name for a in material.authors],
                    "year": material.year,
                    "relevance_score": material.relevance_score,
                    "summary": material.summary or material.abstract
                }
                materials_data.append(material_info)
            
            # Perform analysis
            result = self.analysis_chain.invoke({
                "topic": topic,
                "materials": json.dumps(materials_data, indent=2)
            })
            
            analysis_text = result['text']
            
            # Parse into structured format
            analysis = {
                "key_findings": [],
                "themes": [],
                "gaps": [],
                "methodologies": [],
                "recommendations": []
            }
            
            # Extract sections
            sections = {
                "key_findings": "Key Findings",
                "themes": "Themes",
                "gaps": "Research Gaps",
                "methodologies": "Methodologies",
                "recommendations": "Recommendations"
            }
            
            for key, section_title in sections.items():
                pattern = f"{section_title}:(.*?)(?:$|(?=\n\n[A-Z]))"
                match = re.search(pattern, analysis_text, re.DOTALL)
                
                if match:
                    section_text = match.group(1).strip()
                    # Extract bullet points
                    items = re.findall(r'\n\s*[-*]\s*(.*?)(?=$|\n\s*[-*])', "\n" + section_text)
                    
                    if items:
                        analysis[key] = [item.strip() for item in items]
                    else:
                        # If no bullet points, use the whole section
                        analysis[key] = [section_text]
            
            logger.info("Research materials analysis completed")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing research materials: {str(e)}", exc_info=True)
            return {
                "key_findings": ["Error occurred during analysis"],
                "themes": [],
                "gaps": [],
                "methodologies": [],
                "recommendations": []
            }
    
    def create_paper_outline(
        self, 
        topic: str,
        analysis: Dict[str, Any],
        materials: List[ResearchMaterial]
    ) -> Dict[str, Any]:
        """
        Create a paper outline based on research analysis.
        
        Args:
            topic (str): Paper topic
            analysis (Dict[str, Any]): Research analysis
            materials (List[ResearchMaterial]): Research materials
            
        Returns:
            Dict[str, Any]: Paper outline
        """
        logger.info(f"Creating paper outline for topic: {topic}")
        
        try:
            outline_prompt = """
            Create a comprehensive academic paper outline based on the following research:
            
            TOPIC: {topic}
            
            KEY FINDINGS:
            {key_findings}
            
            THEMES:
            {themes}
            
            GAPS:
            {gaps}
            
            METHODOLOGIES:
            {methodologies}
            
            Your outline should include:
            1. An introduction section
            2. Background/Literature Review
            3. Logical main sections based on the themes and key findings
            4. A methodology section if appropriate
            5. Results/Discussion sections as appropriate
            6. Conclusion section
            7. A list of key references to include
            
            Format the outline as a JSON object with the following structure:
            {{
                "title": "Suggested Paper Title",
                "sections": [
                    {{
                        "title": "Section Title",
                        "subsections": [
                            {{ "title": "Subsection Title" }}
                        ],
                        "key_points": ["Point 1", "Point 2"],
                        "references": ["reference_id_1", "reference_id_2"]
                    }}
                ]
            }}
            
            ONLY respond with the JSON object, no other text.
            """
            
            # Format research components
            key_findings_text = "\n".join([f"- {kf}" for kf in analysis.get("key_findings", [])])
            themes_text = "\n".join([f"- {t}" for t in analysis.get("themes", [])])
            gaps_text = "\n".join([f"- {g}" for g in analysis.get("gaps", [])])
            methodologies_text = "\n".join([f"- {m}" for m in analysis.get("methodologies", [])])
            
            # Create prompt with research data
            formatted_prompt = outline_prompt.format(
                topic=topic,
                key_findings=key_findings_text,
                themes=themes_text,
                gaps=gaps_text,
                methodologies=methodologies_text
            )
            
            # Create messages for the LLM
            messages = [
                SystemMessage(content="You are an academic outline generator."),
                HumanMessage(content=formatted_prompt)
            ]
            
            # Get response from LLM
            response = self.llm.invoke(messages)
            outline_text = response.content
            
            # Extract JSON
            outline_json = outline_text
            if "```json" in outline_text:
                outline_json = outline_text.split("```json")[1].split("```")[0].strip()
            elif "```" in outline_text:
                outline_json = outline_text.split("```")[1].strip()
            
            # Parse JSON
            outline = json.loads(outline_json)
            
            # Add references to outline
            reference_ids = [m.id for m in materials]
            outline["references"] = reference_ids
            
            logger.info("Paper outline created successfully")
            return outline
            
        except Exception as e:
            logger.error(f"Error creating paper outline: {str(e)}", exc_info=True)
            
            # Create a basic outline as fallback
            sections = [
                {"title": "Introduction", "subsections": [], "key_points": [], "references": []},
                {"title": "Background", "subsections": [], "key_points": [], "references": []},
                {"title": "Methodology", "subsections": [], "key_points": [], "references": []},
                {"title": "Results", "subsections": [], "key_points": [], "references": []},
                {"title": "Discussion", "subsections": [], "key_points": [], "references": []},
                {"title": "Conclusion", "subsections": [], "key_points": [], "references": []}
            ]
            
            return {
                "title": f"Research on {topic}",
                "sections": sections,
                "references": [m.id for m in materials]
            }

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
            max_sources = kwargs.get("max_sources", 10)
            # 새로운 파라미터 구조 사용
            materials = self.collect_research_materials(
                topic, 
                max_queries=3, 
                max_results_per_query=max_sources//3
            )
            
            # 명시적으로 검색 결과가 없는지 확인
            if not materials or len(materials) == 0:
                logger.error("검색 결과가 없습니다. 프로세스를 중단합니다.")
                return {
                    "status": "error",
                    "message": "검색 결과가 없어 연구를 진행할 수 없습니다. 다른 주제나 검색어를 시도해보세요."
                }
            
            # 2. 연구 자료 강화 (내용 및 요약 추가 + 벡터 DB 처리)
            enriched_materials = self.enrich_research_materials(materials)
            
            # 3. 연구 자료 분석
            analysis = self.analyze_research_materials(enriched_materials, topic)
            
            # 4. 논문 개요 생성
            outline = self.create_paper_outline(topic, analysis, enriched_materials)
            
            # 5. 키워드 및 참고문헌 추출
            keywords = self.extract_keywords(topic, analysis)
            references = self.extract_references(enriched_materials)
            
            # 결과 반환
            return {
                "status": "completed",
                "topic": topic,
                "materials": [material.to_dict() for material in enriched_materials],
                "analysis": analysis,
                "outline": outline,
                "keywords": keywords,
                "references": references
            }
        except Exception as e:
            logger.error(f"연구 에이전트 실행 중 오류 발생: {str(e)}", exc_info=True)
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
                results = fallback_search(query, num_results=max_results)
            
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

    def search_local_papers(self, query: str, data_path: str = 'data/papers.json', max_results: int = 10) -> List[Dict]:
        """
        로컬에 저장된 논문 데이터셋에서 검색
        
        Args:
            query: 검색 쿼리
            data_path: 논문 메타데이터 JSON 파일 경로
            max_results: 반환할 최대 결과 수
            
        Returns:
            List[Dict]: 검색된 논문 목록
        """
        logger.info(f"로컬 데이터에서 검색: {query}")
        
        try:
            # 데이터 파일이 존재하는지 확인
            if not os.path.exists(data_path):
                logger.warning(f"로컬 데이터 파일이 없음: {data_path}")
                return []
            
            # JSON 파일 로드
            with open(data_path, 'r', encoding='utf-8') as f:
                papers = json.load(f)
            
            logger.info(f"로컬 데이터베이스에서 {len(papers)} 논문 로드됨")
            
            # 간단한 키워드 매칭으로 검색 (실제 구현에서는 더 고급 검색 알고리즘 사용 가능)
            keywords = query.lower().split()
            results = []
            
            for paper in papers:
                # 제목과 초록에서 키워드 검색
                title = paper.get('title', '').lower()
                abstract = paper.get('abstract', '').lower()
                
                # 간단한 관련성 점수 계산 (키워드 일치 수)
                relevance = sum(1 for kw in keywords if kw in title or kw in abstract)
                
                if relevance > 0:
                    paper['relevance'] = relevance  # 관련성 점수 추가
                    results.append(paper)
            
            # 관련성 점수로 정렬하고 상위 결과만 반환
            results.sort(key=lambda x: x.get('relevance', 0), reverse=True)
            limited_results = results[:max_results]
            
            logger.info(f"로컬 데이터에서 {len(limited_results)}개 논문 찾음")
            return limited_results
            
        except Exception as e:
            logger.error(f"로컬 논문 검색 중 오류: {str(e)}", exc_info=True)
            return []
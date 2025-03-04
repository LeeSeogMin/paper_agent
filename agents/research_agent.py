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
from utils.vector_db import create_vector_db, search_vector_db
from utils.api_clients import search_semantic_scholar, search_google_scholar, search_academic_papers
from utils.pdf_processor import extract_text_from_pdf

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
    
    def evaluate_source(self, source: Dict[str, Any], topic: str) -> Tuple[float, str]:
        """
        Evaluate a source for relevance and quality.
        
        Args:
            source (Dict[str, Any]): Source metadata
            topic (str): Paper topic
            
        Returns:
            Tuple[float, str]: Relevance score (0-1) and explanation
        """
        try:
            # Prepare source info
            source_info = {
                "title": source.get("title", ""),
                "abstract": source.get("abstract", "No abstract available"),
                "authors": ", ".join([
                    author["name"] if isinstance(author, dict) else author
                    for author in source.get("authors", [])
                ]),
                "year": source.get("year", "Unknown"),
                "venue": source.get("venue", "Unknown"),
                "citation_count": source.get("citationCount", 0)
            }
            
            result = self.evaluation_chain.invoke({
                "source": source_info,
                "topic": topic
            })
            
            # Extract score
            score_match = re.search(r'Score:\s*(\d+\.?\d*)', result['text'])
            
            if score_match:
                score = float(score_match.group(1))
                # Normalize to 0-1 range if needed
                if score > 1:
                    score = score / 10
            else:
                score = 0.5  # Default middle score
            
            # Extract explanation (everything after the score)
            explanation = result['text'].split("Score:", 1)[1]
            explanation = re.sub(r'^\s*\d+\.?\d*\s*', '', explanation).strip()
            
            return score, explanation
            
        except Exception as e:
            logger.error(f"Error evaluating source: {str(e)}", exc_info=True)
            return 0.3, f"Error during evaluation: {str(e)}"
    
    def collect_research_materials(
        self, 
        topic: str, 
        max_sources: int = 10
    ) -> List[ResearchMaterial]:
        """
        Collect research materials based on the topic.
        
        Args:
            topic (str): Paper topic
            max_sources (int): Maximum number of sources to collect
            
        Returns:
            List[ResearchMaterial]: List of research materials
        """
        logger.info(f"Collecting research materials for topic: {topic}")
        
        # Generate search queries
        queries = self.generate_search_queries(topic)
        
        all_sources = []
        materials = []
        
        # Search for papers using each query
        for query in queries:
            logger.info(f"Processing query: {query.text}")
            
            # Get search results
            search_results = self.search_for_papers(
                query.text, 
                max_results=max_sources // len(queries)
            )
            
            # Process each result
            for result in search_results:
                # Skip if already processed
                if any(s.get("paperId") == result.get("paperId") for s in all_sources):
                    continue
                
                # Evaluate source
                relevance_score, explanation = self.evaluate_source(result, topic)
                
                # Only include if somewhat relevant
                if relevance_score >= 0.6:
                    # Create research material
                    material = ResearchMaterial(
                        id=result.get("paperId") or f"paper_{len(materials)}",
                        title=result.get("title", ""),
                        authors=result.get("authors", []),
                        year=result.get("year"),
                        abstract=result.get("abstract", ""),
                        url=result.get("url", ""),
                        pdf_url=result.get("pdfUrl", ""),
                        relevance_score=relevance_score,
                        evaluation=explanation,
                        query_id=query.id,
                        content="",  # Will be populated later if PDF is accessible
                        summary=""   # Will be populated later
                    )
                    
                    materials.append(material)
                    all_sources.append(result)
            
            # Break if we have enough materials
            if len(materials) >= max_sources:
                break
        
        logger.info(f"Collected {len(materials)} research materials")
        return materials[:max_sources]
    
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
            # Extract content from PDF if available
            if material.pdf_url:
                content = self.extract_content_from_pdf(material.pdf_url)
                material.content = content
                
                # Generate summary if content was extracted
                if content:
                    summary = self.summarize_content(content, material.title)
                    material.summary = summary
                else:
                    # Fallback to abstract if PDF extraction failed
                    material.summary = material.abstract
            else:
                # Use abstract as summary if PDF not available
                material.summary = material.abstract
            
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
            materials = self.collect_research_materials(topic, max_sources=max_sources)
            
            # 2. 연구 자료 강화 (내용 및 요약 추가)
            enriched_materials = self.enrich_research_materials(materials)
            
            # 3. 연구 자료 분석
            analysis = self.analyze_research_materials(enriched_materials, topic)
            
            # 4. 논문 개요 생성
            outline = self.create_paper_outline(topic, analysis, enriched_materials)
            
            # 결과 반환
            return {
                "status": "success",
                "materials": [material.dict() for material in enriched_materials],
                "analysis": analysis,
                "outline": outline
            }
        except Exception as e:
            logger.error(f"연구 에이전트 실행 중 오류 발생: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": f"연구 프로세스 중 오류: {str(e)}"
            }
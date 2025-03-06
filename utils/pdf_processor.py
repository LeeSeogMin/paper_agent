"""
PDF processing utilities for extracting and processing text from academic papers.

This module provides functions to extract text from PDF files and process it 
for use in research and writing workflows.
"""

import os
import io
import re
import tempfile
from typing import List, Dict, Any, Optional, Tuple

import requests
# Removed dependency on PyPDF2 for demonstration purposes
# import PyPDF2
from utils.logger import logger
from utils.api_clients import download_pdf
import fitz  # PyMuPDF
from datetime import datetime
import json
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils.vector_db import process_and_vectorize_paper
from utils.xai_client import XAIClient  # XAI 클라이언트 추가


def extract_text_from_pdf(pdf_path_or_url: str) -> str:
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_path_or_url (str): Path to a local PDF file or URL to a PDF file
        
    Returns:
        str: Extracted text content
    """
    logger.info(f"Extracting text from PDF: {pdf_path_or_url}")
    
    try:
        # Handle URL or local path
        if pdf_path_or_url.startswith(('http://', 'https://')):
            # Download PDF from URL to a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            temp_file.close()
            download_pdf(pdf_path_or_url, temp_file.name)
            pdf_path = temp_file.name
        else:
            pdf_path = pdf_path_or_url
        
        # Use PyMuPDF (fitz) to extract text
        doc = fitz.open(pdf_path)
        text = ""
        
        # Extract text from all pages
        for page in doc:
            text += page.get_text()
        
        doc.close()
        
        # Clean up temporary file if created
        if pdf_path_or_url.startswith(('http://', 'https://')):
            os.unlink(pdf_path)
            
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return f"Error extracting text from {pdf_path_or_url}: {str(e)}"


def clean_pdf_text(text: str) -> str:
    """
    Clean the extracted text from a PDF.
    
    Args:
        text (str): Raw text extracted from a PDF
        
    Returns:
        str: Cleaned text
    """
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Remove header/footer page numbers
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    # Fix broken words (words split across lines with a hyphen)
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    # Fix spacing around punctuation
    text = re.sub(r'\s+([.,;:?!])', r'\1', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def extract_sections_from_pdf(pdf_path_or_url: str) -> Dict[str, str]:
    """
    Extract sections from a PDF file.
    
    Args:
        pdf_path_or_url (str): Path to a local PDF file or URL to a PDF file
        
    Returns:
        Dict[str, str]: Dictionary mapping section titles to their content
    """
    logger.info(f"Extracting sections from PDF: {pdf_path_or_url}")
    
    # First extract all text
    full_text = extract_text_from_pdf(pdf_path_or_url)
    
    # Define common section titles in academic papers
    section_patterns = [
        r'(?i)abstract',
        r'(?i)introduction',
        r'(?i)related work',
        r'(?i)background',
        r'(?i)methodology|methods',
        r'(?i)experimental setup|experiment',
        r'(?i)results',
        r'(?i)discussion',
        r'(?i)conclusion',
        r'(?i)references|bibliography'
    ]
    
    # Create pattern to match sections
    combined_pattern = r'(?:\n|\A)(' + '|'.join(section_patterns) + r')(?:[:\.\s]|\Z)'
    
    # Find all section positions
    sections = {}
    matches = list(re.finditer(combined_pattern, full_text, re.IGNORECASE))
    
    for i, match in enumerate(matches):
        section_title = match.group(1).strip()
        start_pos = match.end()
        
        # If this is the last section, it goes to the end of the document
        if i == len(matches) - 1:
            section_content = full_text[start_pos:].strip()
        else:
            # Otherwise, it goes until the next section
            end_pos = matches[i + 1].start()
            section_content = full_text[start_pos:end_pos].strip()
        
        sections[section_title] = section_content
    
    logger.info(f"Extracted {len(sections)} sections from PDF")
    return sections


def extract_references_from_pdf(pdf_path_or_url: str) -> List[str]:
    """
    Extract references from a PDF file.
    
    Args:
        pdf_path_or_url (str): Path to a local PDF file or URL to a PDF file
        
    Returns:
        List[str]: List of reference strings
    """
    logger.info(f"Extracting references from PDF: {pdf_path_or_url}")
    
    # Extract sections and find the references section
    sections = extract_sections_from_pdf(pdf_path_or_url)
    
    # Look for references section using different possible titles
    reference_section = None
    for title in ['references', 'bibliography', 'works cited']:
        for section_title in sections:
            if title.lower() in section_title.lower():
                reference_section = sections[section_title]
                break
        if reference_section:
            break
    
    if not reference_section:
        logger.warning("No references section found in PDF")
        return []
    
    # Split references by common patterns
    # Look for numbered references: [1], 1., etc.
    references = re.split(r'\n\s*(?:\[\d+\]|\d+\.)\s+', reference_section)
    
    # If no numbered references found, try splitting by authors (capital letters followed by surname patterns)
    if len(references) <= 1:
        references = re.split(r'\n\s*(?:[A-Z][a-z]+,\s*[A-Z]\.|[A-Z][a-z]+\s+[A-Z]\.)', reference_section)
    
    # Remove empty entries and the first entry (which is usually just the heading)
    references = [ref.strip() for ref in references if ref.strip()]
    if references and not any(char.isalpha() for char in references[0]):
        references = references[1:]
    
    logger.info(f"Extracted {len(references)} references from PDF")
    return references


class PDFProcessor:
    """PDF 처리 및 메타데이터 추출 클래스"""
    
    def __init__(self, llm=None, use_llm=True):
        """
        PDF 프로세서 초기화
        
        Args:
            llm: 사용할 LLM 객체 (기본값: None, 내부에서 생성)
            use_llm: LLM 사용 여부 (기본값: True)
        """
        self.use_llm = use_llm
        self.llm = llm if llm else OpenAI(temperature=0)
        
        # XAI 클라이언트 초기화
        try:
            self.xai_client = XAIClient.get_instance()
            self.use_xai = True
        except Exception as e:
            logger.warning(f"XAI 클라이언트 초기화 실패: {str(e)}")
            self.use_xai = False
        
        # 메타데이터 추출 체인 초기화
        self.metadata_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template="""다음은 학술 논문의 텍스트입니다. 제목, 저자, 소속, 발행 연도, 주제어, 초록 등의 
                메타데이터를 추출해 JSON 형식으로 반환해주세요:
                
                {text}
                
                출력 형식:
                {
                  "title": "논문 제목",
                  "authors": ["저자1", "저자2"],
                  "affiliations": ["소속1", "소속2"],
                  "year": "출판 연도",
                  "keywords": ["키워드1", "키워드2"],
                  "abstract": "초록 내용"
                }
                """,
                input_variables=["text"]
            )
        )
        
        # 메타데이터 캐시
        self._metadata_cache = {}
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        PDF에서 텍스트 추출
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            str: 추출된 텍스트
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            # 모든 페이지의 텍스트 추출
            for page in doc:
                text += page.get_text()
            
            doc.close()
            return text
        except Exception as e:
            logger.error(f"PDF 텍스트 추출 오류: {str(e)}")
            return ""
    
    def extract_metadata_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        PDF에서 메타데이터 추출 (하이브리드 접근법)
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            Dict[str, Any]: 추출된 메타데이터
        """
        # 캐시 확인
        if pdf_path in self._metadata_cache:
            return self._metadata_cache[pdf_path]
        
        try:
            # 1. PDF 내장 메타데이터 추출
            doc = fitz.open(pdf_path)
            embedded_metadata = doc.metadata
            
            # 기본 메타데이터 구조 생성
            metadata = {
                "title": embedded_metadata.get("title", ""),
                "authors": [],
                "year": embedded_metadata.get("creationDate", "")[:4] if embedded_metadata.get("creationDate") else "",
                "abstract": "",
                "keywords": [],
                "source": "embedded"
            }
            
            # 저자 정보 처리 (내장 메타데이터에서)
            if embedded_metadata.get("author"):
                # 여러 구분자로 저자 분리 시도
                for separator in [";", ",", " and ", "&"]:
                    if separator in embedded_metadata["author"]:
                        metadata["authors"] = [a.strip() for a in embedded_metadata["author"].split(separator)]
                        break
                
                # 구분자가 없는 경우 단일 저자로 처리
                if not metadata["authors"]:
                    metadata["authors"] = [embedded_metadata["author"].strip()]
            
            # 메타데이터가 충분한지 확인
            if metadata["title"] and metadata["authors"]:
                doc.close()
                self._metadata_cache[pdf_path] = metadata
                return metadata
            
            # 2. 휴리스틱 접근법 (첫 페이지 텍스트 분석)
            first_page_text = doc[0].get_text()
            doc.close()
            
            # 제목 추출 시도 (일반적으로 첫 페이지 상단에 큰 텍스트)
            title_match = re.search(r'^(.+?)(?:\n|$)', first_page_text.strip())
            if title_match and not metadata["title"]:
                metadata["title"] = title_match.group(1).strip()
            
            # 저자 추출 시도 (일반적인 패턴)
            if not metadata["authors"]:
                # 여러 저자 패턴 시도
                author_patterns = [
                    r'(?:Authors?|By):\s*(.+?)(?:\n|$)',
                    r'^.*?\n(.+?)(?:\n|$)',  # 제목 다음 줄
                ]
                
                for pattern in author_patterns:
                    author_match = re.search(pattern, first_page_text, re.IGNORECASE)
                    if author_match:
                        author_text = author_match.group(1).strip()
                        # 저자 분리 시도
                        for separator in [";", ",", " and ", "&"]:
                            if separator in author_text:
                                metadata["authors"] = [a.strip() for a in author_text.split(separator)]
                                break
                        
                        if metadata["authors"]:
                            break
            
            # 초록 추출 시도
            abstract_match = re.search(r'(?:Abstract|ABSTRACT)[\s:]*(.+?)(?:\n\n|\n[A-Z]+\n|$)', 
                                      first_page_text, re.DOTALL | re.IGNORECASE)
            if abstract_match:
                metadata["abstract"] = abstract_match.group(1).strip()
            
            # 연도 추출 시도 (아직 없는 경우)
            if not metadata["year"]:
                year_match = re.search(r'(?:19|20)\d{2}', first_page_text)
                if year_match:
                    metadata["year"] = year_match.group(0)
            
            # 키워드 추출 시도
            keyword_match = re.search(r'(?:Keywords|Key\s*words|KEYWORDS)[\s:]*(.+?)(?:\n\n|\n[A-Z]+\n|$)', 
                                     first_page_text, re.DOTALL | re.IGNORECASE)
            if keyword_match:
                keyword_text = keyword_match.group(1).strip()
                # 키워드 분리 시도
                for separator in [";", ","]:
                    if separator in keyword_text:
                        metadata["keywords"] = [k.strip() for k in keyword_text.split(separator)]
                        break
            
            # 메타데이터 소스 업데이트
            metadata["source"] = "heuristic"
            
            # 3. LLM 접근법 (필요한 경우)
            if (self.use_llm and 
                (not metadata["title"] or not metadata["authors"] or not metadata["abstract"])):
                # 첫 페이지 또는 처음 2000자만 사용 (비용 절감)
                text_for_llm = first_page_text[:2000]
                
                try:
                    # XAI 클라이언트 사용 시도
                    if self.use_xai:
                        logger.info("XAI 클라이언트를 사용하여 메타데이터 추출 시도")
                        prompt = f"""다음은 학술 논문의 텍스트입니다. 제목, 저자, 소속, 발행 연도, 주제어, 초록 등의 
                        메타데이터를 추출해 JSON 형식으로 반환해주세요:
                        
                        {text_for_llm}
                        
                        출력 형식:
                        {{
                          "title": "논문 제목",
                          "authors": ["저자1", "저자2"],
                          "affiliations": ["소속1", "소속2"],
                          "year": "출판 연도",
                          "keywords": ["키워드1", "키워드2"],
                          "abstract": "초록 내용"
                        }}
                        """
                        
                        result = self.xai_client.generate_text(prompt)
                        # JSON 형식 추출
                        json_match = re.search(r'({.*})', result, re.DOTALL)
                        if json_match:
                            llm_metadata = json.loads(json_match.group(1))
                        else:
                            raise ValueError("XAI 응답에서 JSON 형식을 찾을 수 없습니다")
                    else:
                        # 기존 LLM 체인 사용
                        llm_result = self.metadata_chain.invoke({"text": text_for_llm})
                        llm_metadata = json.loads(llm_result["text"])
                    
                    # Log the LLM result for debugging
                    logger.debug(f"LLM result: {llm_metadata}")
                    
                    # LLM 결과로 빈 필드 채우기
                    if not metadata["title"] and llm_metadata.get("title"):
                        metadata["title"] = llm_metadata["title"]
                    
                    if not metadata["authors"] and llm_metadata.get("authors"):
                        metadata["authors"] = llm_metadata["authors"]
                    
                    if not metadata["abstract"] and llm_metadata.get("abstract"):
                        metadata["abstract"] = llm_metadata["abstract"]
                    
                    if not metadata["year"] and llm_metadata.get("year"):
                        metadata["year"] = llm_metadata["year"]
                    
                    if not metadata["keywords"] and llm_metadata.get("keywords"):
                        metadata["keywords"] = llm_metadata["keywords"]
                    
                    # 메타데이터 소스 업데이트
                    metadata["source"] = "llm"
                    
                except Exception as e:
                    logger.warning(f"LLM 메타데이터 추출 실패: {str(e)}")
                    logger.debug(f"Failed LLM input text: {text_for_llm}")
            
            # 캐시에 저장
            self._metadata_cache[pdf_path] = metadata
            return metadata
            
        except Exception as e:
            logger.error(f"PDF 메타데이터 추출 오류: {str(e)}")
            return {
                "title": os.path.basename(pdf_path),
                "authors": [],
                "year": "",
                "abstract": "",
                "keywords": [],
                "source": "fallback"
            }
    
    def extract_sections_from_pdf(self, pdf_path: str) -> List[Dict[str, str]]:
        """
        PDF에서 섹션 추출
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            List[Dict[str, str]]: 추출된 섹션 목록
        """
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            
            # 모든 페이지의 텍스트 추출
            for page in doc:
                full_text += page.get_text()
            
            doc.close()
            
            # 섹션 헤더 패턴 (일반적인 학술 논문 섹션)
            section_patterns = [
                r'\n\s*(\d+\.?\s*[A-Z][A-Za-z\s]+)\s*\n',  # 번호가 있는 섹션 (예: "1. Introduction")
                r'\n\s*([A-Z][A-Z\s]+)\s*\n',              # 대문자 섹션 (예: "INTRODUCTION")
                r'\n\s*([A-Z][a-z]+(?:\s+[A-Za-z]+)*)\s*\n'  # 일반 섹션 (예: "Introduction")
            ]
            
            sections = []
            last_pos = 0
            last_section = "Abstract"
            
            # 각 패턴으로 섹션 찾기
            for pattern in section_patterns:
                for match in re.finditer(pattern, full_text):
                    section_title = match.group(1).strip()
                    section_start = match.start()
                    
                    # 이전 섹션 내용 저장
                    if section_start > last_pos:
                        section_content = full_text[last_pos:section_start].strip()
                        if section_content:
                            sections.append({
                                "title": last_section,
                                "content": section_content
                            })
                    
                    last_section = section_title
                    last_pos = match.end()
            
            # 마지막 섹션 추가
            if last_pos < len(full_text):
                sections.append({
                    "title": last_section,
                    "content": full_text[last_pos:].strip()
                })
            
            return sections
            
        except Exception as e:
            logger.error(f"PDF 섹션 추출 오류: {str(e)}")
            return [{
                "title": "Full Text",
                "content": self.extract_text_from_pdf(pdf_path)
            }]
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        PDF 처리 통합 메서드
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            Dict[str, Any]: 처리 결과
        """
        try:
            metadata = self.extract_metadata_from_pdf(pdf_path)
            sections = self.extract_sections_from_pdf(pdf_path)
            
            return {
                "metadata": metadata,
                "sections": sections,
                "path": pdf_path,
                "success": True
            }
        except Exception as e:
            logger.error(f"PDF 처리 오류: {str(e)}")
            return {
                "metadata": {
                    "title": os.path.basename(pdf_path),
                    "authors": [],
                    "year": "",
                    "abstract": "",
                    "source": "error"
                },
                "sections": [],
                "path": pdf_path,
                "success": False,
                "error": str(e)
            }


def process_local_pdfs(local_dir="data/local", vector_db_path="data/vector_db"):
    """
    data/local 폴더의 PDF 파일을 처리하여 벡터화
    
    Args:
        local_dir: 로컬 PDF 파일 디렉토리
        vector_db_path: 벡터 데이터베이스 저장 경로
    
    Returns:
        List[Dict]: 처리된 PDF 정보 목록
    """
    # 로컬 디렉토리 확인
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)
        logger.info(f"로컬 PDF 디렉토리 생성됨: {local_dir}")
        return []
    
    # PDF 파일 스캔
    pdf_files = [f for f in os.listdir(local_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        logger.info(f"로컬 PDF 파일이 없음: {local_dir}")
        return []
    
    logger.info(f"{len(pdf_files)}개의 로컬 PDF 파일 처리 시작")
    
    # 각 PDF 파일 처리
    processed_papers = []
    
    # PDF 프로세서 초기화 (하이브리드 방식 사용)
    pdf_processor = PDFProcessor(use_llm=True)
    
    # 메타데이터 추출 실패 기록을 위한 딕셔너리
    extraction_failures = {}
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(local_dir, pdf_file)
        
        # 이전 실패 횟수 확인
        if pdf_file in extraction_failures and extraction_failures[pdf_file] >= 2:
            logger.warning(f"PDF 메타데이터 추출 2회 이상 실패, 건너뜀: {pdf_file}")
            continue
        
        try:
            # 하이브리드 방식으로 PDF 처리
            pdf_result = pdf_processor.process_pdf(pdf_path)
            
            if pdf_result["success"]:
                metadata = pdf_result["metadata"]
                
                # 메타데이터 유효성 검사
                if not metadata.get("title") or not metadata.get("authors"):
                    # 실패 횟수 증가
                    extraction_failures[pdf_file] = extraction_failures.get(pdf_file, 0) + 1
                    logger.warning(f"PDF 메타데이터 불완전, 실패 횟수: {extraction_failures[pdf_file]}: {pdf_file}")
                    
                    # 2회 이상 실패하면 건너뜀
                    if extraction_failures[pdf_file] >= 2:
                        logger.warning(f"PDF 메타데이터 추출 2회 이상 실패, 건너뜀: {pdf_file}")
                        continue
                
                # 텍스트 추출 및 벡터화
                text = pdf_processor.extract_text_from_pdf(pdf_path)
                if text:
                    # 벡터 DB에 저장
                    vector_result = process_and_vectorize_paper(
                        pdf_path=pdf_path
                    )
                
                paper_info = {
                    "id": f"local_{os.path.splitext(pdf_file)[0]}",
                    "title": metadata.get("title", ""),
                    "authors": metadata.get("authors", []),
                    "abstract": metadata.get("abstract", ""),
                    "year": metadata.get("year", ""),
                    "pdf_path": pdf_path,
                    "local_path": pdf_path,
                    "vectorized": True if text else False
                }
                
                processed_papers.append(paper_info)
                logger.info(f"로컬 PDF 처리 완료: {pdf_file}")
            else:
                # 실패 횟수 증가
                extraction_failures[pdf_file] = extraction_failures.get(pdf_file, 0) + 1
                logger.warning(f"PDF 처리 실패, 실패 횟수: {extraction_failures[pdf_file]}: {pdf_file}")
                
                # 2회 이상 실패하면 건너뜀
                if extraction_failures[pdf_file] >= 2:
                    logger.warning(f"PDF 처리 2회 이상 실패, 건너뜀: {pdf_file}")
                    continue
                
                # 하이브리드 방식 실패시 기존 방식 시도
                paper_info = process_and_vectorize_paper(pdf_path=pdf_path)
                # ...기존 처리 코드...
        except Exception as e:
            # 실패 횟수 증가
            extraction_failures[pdf_file] = extraction_failures.get(pdf_file, 0) + 1
            logger.error(f"로컬 PDF 처리 중 오류, 실패 횟수: {extraction_failures[pdf_file]}: {pdf_file} - {str(e)}")
            
            # 2회 이상 실패하면 건너뜀
            if extraction_failures[pdf_file] >= 2:
                logger.warning(f"PDF 처리 2회 이상 실패, 건너뜀: {pdf_file}")
                continue
    
    logger.info(f"{len(processed_papers)}개의 로컬 PDF 파일 처리 완료")
    
    # 실패 통계 로깅
    total_failures = sum(1 for count in extraction_failures.values() if count >= 2)
    if total_failures > 0:
        logger.warning(f"{total_failures}개의 PDF 파일이 메타데이터 추출 실패로 건너뜀")
    
    return processed_papers
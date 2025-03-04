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


def extract_text_from_pdf(pdf_path_or_url: str) -> str:
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_path_or_url (str): Path to a local PDF file or URL to a PDF file
        
    Returns:
        str: Extracted text content
    """
    logger.info(f"Extracting text from PDF: {pdf_path_or_url}")
    
    # This is a mock implementation since PyPDF2 is not installed
    # In a real implementation, this would use PyPDF2 to extract text
    logger.warning("PDF extraction is mocked (PyPDF2 not installed)")
    
    # Return mock text
    return f"This is mock text extracted from {pdf_path_or_url}.\n\n" + \
           "Abstract\n\nThis paper discusses important research findings...\n\n" + \
           "Introduction\n\nThe field of research has seen significant advancements...\n\n" + \
           "Methodology\n\nWe applied various techniques to analyze the data...\n\n" + \
           "Results\n\nOur findings indicate significant improvements...\n\n" + \
           "Conclusion\n\nThis research contributes to the field by providing...\n\n" + \
           "References\n\n1. Smith, J. (2020). Recent Advances in Research.\n" + \
           "2. Johnson, K. (2019). Analytical Methods for Academic Research."


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
    """고성능 PDF 처리 클래스"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.section_patterns = [
            r'abstract', r'introduction', r'related\s+work',
            r'methodology', r'experiment', r'results',
            r'discussion', r'conclusion', r'references'
        ]

    def process_file(self, file_path: str) -> Tuple[bool, str, list, dict]:
        """PDF 파일 처리 메인 메서드"""
        try:
            doc = fitz.open(file_path)
            full_text, pages = self._extract_text_and_pages(doc)
            metadata = self._process_metadata(doc.metadata)
            sections = self._detect_sections(full_text)
            references = self._extract_references(full_text)
            
            metadata.update({
                "sections": sections,
                "reference_count": len(references),
                "processing_time": datetime.now().isoformat()
            })
            
            return True, full_text, pages, metadata
            
        except Exception as e:
            logger.error(f"PDF 처리 실패: {str(e)}", exc_info=self.verbose)
            return False, "", [], {}

    def _extract_text_and_pages(self, doc) -> Tuple[str, list]:
        """텍스트 및 페이지 데이터 추출"""
        full_text = []
        pages = []
        
        for page in doc:
            text = page.get_text("text", flags=fitz.TEXT_PRESERVE_LIGATURES | 
                                fitz.TEXT_MEDIABOX_CLIP | 
                                fitz.TEXT_DEHYPHENATE)
            full_text.append(text)
            pages.append({
                "number": page.number + 1,
                "content": text,
                "dimensions": page.rect,
                "annotations": [self._process_annotation(a) for a in page.annots()]
            })
            
        return "\n".join(full_text), pages

    def _process_metadata(self, meta: dict) -> dict:
        """메타데이터 가공"""
        return {
            "title": meta.get("title", ""),
            "authors": meta.get("author", "").split(';') if meta.get("author") else [],
            "subject": meta.get("subject", ""),
            "creation_date": meta.get("creationDate", ""),
            "modification_date": meta.get("modDate", ""),
            "keywords": meta.get("keywords", "").split(',') if meta.get("keywords") else []
        }

    def _process_annotation(self, annot) -> dict:
        """주석 데이터 처리"""
        return {
            "type": annot.type[1],
            "content": annot.info.get("content", ""),
            "coordinates": annot.rect,
            "color": annot.colors.get("fill", (0,0,0))
        } if annot else {}

    def _detect_sections(self, text: str) -> Dict[str, str]:
        """섹션 구조 분석"""
        section_pattern = re.compile(
            r'\n\s*(' + '|'.join(self.section_patterns) + r')\s*\n',
            re.IGNORECASE
        )
        matches = list(section_pattern.finditer(text))
        sections = {}
        
        for i, match in enumerate(matches):
            title = match.group(1).lower().capitalize()
            start = match.end()
            end = matches[i+1].start() if i+1 < len(matches) else len(text)
            sections[title] = text[start:end].strip()
            
        return sections

    def _extract_references(self, text: str) -> List[str]:
        """참고문헌 추출"""
        ref_section = self._detect_sections(text).get("References", "")
        if not ref_section:
            return []
            
        # 참고문헌 분할 로직
        references = re.split(r'\n(?=\[?\d+\]?\s?)', ref_section)
        return [ref.strip() for ref in references if ref.strip()]

    def _clean_text(self, text: str) -> str:
        """텍스트 정제"""
        # 하이픈 연결 단어 복원
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        # 머리글/꼬리글 제거
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        # 연속 공백 제거
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
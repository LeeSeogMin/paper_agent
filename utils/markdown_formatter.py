"""
마크다운 형식의 논문 포매팅 유틸리티
"""

import re
from typing import List, Dict, Any


def format_paper_markdown(content: str, metadata: Dict[str, Any] = None) -> str:
    """
    논문 내용을 마크다운 형식으로 포매팅합니다.
    
    Args:
        content: 원본 논문 내용
        metadata: 논문 메타데이터 (선택)
        
    Returns:
        str: 포매팅된 마크다운 문서
    """
    # 기본 메타데이터 설정
    if metadata is None:
        metadata = {}
    
    # 제목 추출
    title_match = re.search(r'#\s+(.*?)[\r\n]', content)
    title = title_match.group(1) if title_match else "Untitled Paper"
    
    # YAML 헤더 생성
    yaml_header = f"""---
title: "{title}"
author: "{metadata.get('author', 'Research Agent')}"
date: "{metadata.get('date', '')}"
abstract: "{metadata.get('abstract', '')}"
keywords: "{', '.join(metadata.get('keywords', []))}"
---

"""
    
    # 인용 포맷 개선
    content = re.sub(
        r'\(([^)]+),\s*(\d{4})\)',
        r'\\cite{\1\2}',
        content
    )
    
    # 참고문헌 섹션 포맷팅
    ref_section_match = re.search(r'##\s+참고문헌[\r\n]+(.*?)($|\Z)', content, re.DOTALL)
    if ref_section_match:
        ref_text = ref_section_match.group(1)
        refs = re.findall(r'\d+\.\s+(.*?)[\r\n][\r\n]', ref_text + "\n\n")
        
        formatted_refs = "## 참고문헌\n\n"
        for i, ref in enumerate(refs):
            formatted_refs += f"{i+1}. {ref}\n\n"
        
        # 원본 참고문헌 섹션을 포맷팅된 버전으로 대체
        content = content.replace(ref_section_match.group(0), formatted_refs)
    
    # 최종 문서 생성
    return yaml_header + content 
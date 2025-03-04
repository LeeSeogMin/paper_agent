import os

def save_paper(paper_content, metadata, output_dir="output"):
    """
    생성된 논문 저장
    
    Args:
        paper_content: 논문 내용
        metadata: 메타데이터
        output_dir: 출력 디렉토리
    
    Returns:
        str: 저장된 파일 경로
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 파일명 생성
    title = metadata.get("topic", "paper").replace(" ", "_")
    filename = f"paper_{title[:30]}.md"
    filepath = os.path.join(output_dir, filename)
    
    # 마크다운 포맷팅 적용
    from utils.markdown_formatter import format_paper_markdown
    formatted_content = format_paper_markdown(paper_content, metadata)
    
    # 파일 저장
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(formatted_content)
    
    return filepath 
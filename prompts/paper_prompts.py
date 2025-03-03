"""
논문 관련 프롬프트 템플릿
논문 작성 및 편집을 위한 프롬프트 템플릿 모음입니다.
"""

from langchain_core.prompts import PromptTemplate


# 연구 주제 제안 프롬프트
RESEARCH_TOPIC_SUGGESTION_PROMPT = PromptTemplate(
    template="""당신은 학술 연구 전문가입니다. 주어진 관심 분야에 관련된 구체적이고 연구 가능한 주제를 제안해 주세요.
    
관심 분야: {field}

기대하는 결과:
1. 해당 분야에서 연구할 가치가 있는 5가지 구체적인 주제
2. 각 주제에 대한 간략한 설명과 연구의 중요성
3. 각 주제가 현재 학술 담론에서 어떤 격차를 해결하는지 설명

주제 제안 시 고려사항:
- 현재 연구 동향과 관련성
- 연구 가능성 및 실행 가능성
- 학문적 중요성과 잠재적 영향
- 혁신성 및 창의성

제안하는 주제:
""",
    input_variables=["field"],
)


# 연구 계획 작성 프롬프트
RESEARCH_PLAN_PROMPT = PromptTemplate(
    template="""당신은 학술 연구 계획 작성 전문가입니다. 다음 주제에 대한 체계적인 연구 계획을 개발해 주세요.

연구 주제: {topic}

연구 계획에는 다음이 포함되어야 합니다:
1. 연구 배경 및 목적
2. 핵심 연구 질문 (3-5개)
3. 방법론 개요
4. 필요한 자료 및 리소스
5. 예상 결과 및 기여도
6. 잠재적인 한계점과 대응 방안
7. 참고할 주요 문헌 (있을 경우)

최대한 구체적이고 실행 가능한 계획을 제시해 주세요.

연구 계획:
""",
    input_variables=["topic"],
)


# 논문 개요 작성 프롬프트
PAPER_OUTLINE_PROMPT = PromptTemplate(
    template="""당신은 학술 논문 개요 작성 전문가입니다. 다음 주제와 연구 자료를 바탕으로 체계적인 논문 개요를 작성해 주세요.

논문 주제: {topic}
논문 유형: {paper_type}

수집된 연구 자료:
{research_materials}

개요에는 다음이 포함되어야 합니다:
1. 논문 제목 (명확하고 설명적인 제목)
2. 초록 구조 (연구의 목적, 방법, 결과, 결론에 관한 간략한 설명)
3. 각 섹션의 제목과 내용 요약 (서론부터 결론까지)
4. 각 섹션에서 다룰 주요 논점

개요는 논리적 흐름을 가지고 있어야 하며, 연구 주제를 효과적으로 탐구할 수 있도록 구성되어야 합니다.

논문 개요:
""",
    input_variables=["topic", "paper_type", "research_materials"],
)


# 논문 섹션 작성 프롬프트
PAPER_SECTION_PROMPT = PromptTemplate(
    template="""당신은 학술 논문 섹션 작성 전문가입니다. 다음 정보를 바탕으로 고품질의 논문 섹션을 작성해 주세요.

논문 제목: {paper_title}
섹션 제목: {section_title}
섹션 목적: {section_purpose}

논문 개요:
{paper_outline}

관련 연구 자료:
{research_materials}

작성 지침:
1. 학술적 문체를 사용하여 명확하고 체계적으로 작성
2. 적절한 인용 포함 (관련 연구 자료 활용)
3. 논리적 흐름과 일관성 유지
4. 주요 개념과 아이디어에 대한 충분한 설명 제공
5. 비판적 분석과 논증 포함

섹션 내용:
""",
    input_variables=["paper_title", "section_title", "section_purpose", "paper_outline", "research_materials"],
)


# 논문 편집 및 개선 프롬프트
PAPER_EDITING_PROMPT = PromptTemplate(
    template="""당신은 학술 논문 편집 전문가입니다. 다음 논문 내용을 검토하고 개선해 주세요.

작업 유형: {editing_type}
스타일 가이드: {style_guide}

논문 내용:
{paper_content}

편집 및 개선 지침:
1. 문법, 맞춤법, 구두점 오류 수정
2. 문장 구조와 흐름 개선
3. 일관성과 명확성 향상
4. 중복 및 불필요한 내용 제거
5. 학술적 표현과 전문 용어 적절히 사용
6. 인용과 참고 문헌 형식 확인

원본 내용의 의미는 유지하면서 표현을 개선하세요.

편집된 내용:
""",
    input_variables=["editing_type", "style_guide", "paper_content"],
)


# 논문 리뷰 및 피드백 프롬프트
PAPER_REVIEW_PROMPT = PromptTemplate(
    template="""당신은 학술 논문 리뷰어입니다. 다음 논문을 철저히 검토하고 종합적인 피드백을 제공해 주세요.

논문 제목: {paper_title}
논문 유형: {paper_type}

논문 내용:
{paper_content}

리뷰 지침:
1. 연구 질문/목적의 명확성과 중요성 평가
2. 방법론의 적절성과 견고성 평가
3. 결과의 타당성과 해석 평가
4. 논의의 깊이와 통찰력 평가
5. 문헌 인용의 적절성과 포괄성 평가
6. 구성, 명확성, 독창성 평가

다음 형식으로 피드백을 제공해 주세요:
- 주요 강점
- 개선이 필요한 영역
- 구체적인 개선 제안
- 전반적인 평가

피드백:
""",
    input_variables=["paper_title", "paper_type", "paper_content"],
)


# 논문 요약 프롬프트
PAPER_SUMMARY_PROMPT = PromptTemplate(
    template="""당신은 학술 논문 요약 전문가입니다. 다음 논문을 간결하게 요약해 주세요.

논문 제목: {paper_title}
논문 내용:
{paper_content}

다음 구조로 요약을 작성해 주세요:
1. 연구 목적 (1-2문장)
2. 방법론 (2-3문장)
3. 주요 결과 (3-4문장)
4. 결론 및 의의 (1-2문장)

전체 요약은 300단어 이내로 작성하세요. 원문의 핵심 내용을 유지하면서 명확하고 간결하게 요약하세요.

요약:
""",
    input_variables=["paper_title", "paper_content"],
)


# 논문 번역 프롬프트
PAPER_TRANSLATION_PROMPT = PromptTemplate(
    template="""당신은 학술 논문 번역 전문가입니다. 다음 {source_language} 논문을 {target_language}로 번역해 주세요.

원문:
{source_text}

번역 지침:
1. 학술적 문체와 전문 용어를 유지하세요
2. 문장 구조와 논리적 흐름을 유지하세요
3. 전문 용어는 표준 용어를 사용하세요
4. 원문의 의미를 정확하게 전달하세요
5. 번역 후 자연스러운 {target_language} 문장이 되도록 하세요

번역:
""",
    input_variables=["source_language", "target_language", "source_text"],
)


# 참고 문헌 서식 변환 프롬프트
REFERENCE_FORMATTING_PROMPT = PromptTemplate(
    template="""당신은 학술 참고 문헌 서식 변환 전문가입니다. 다음 참고 문헌 목록을 {target_style} 형식으로 변환해 주세요.

원본 참고 문헌:
{references}

변환 지침:
1. {target_style} 스타일 가이드에 맞게 정확히 변환하세요
2. 모든 필수 요소(저자, 연도, 제목, 출처 등)를 포함하세요
3. 형식, 구두점, 기울임꼴 등을 정확히 적용하세요
4. 항목의 순서를 알파벳 순으로 정렬하세요 (특별한 지시가 없는 한)

{target_style} 형식 참고 문헌:
""",
    input_variables=["target_style", "references"],
)


# 논문 키워드 추출 프롬프트
KEYWORD_EXTRACTION_PROMPT = PromptTemplate(
    template="""당신은 학술 논문 키워드 추출 전문가입니다. 다음 논문 내용에서 가장 관련성 높은 키워드를 추출해 주세요.

논문 제목: {paper_title}
논문 내용:
{paper_content}

추출 지침:
1. 논문의 주제와 직접적으로 관련된 키워드 선택
2. 학술 분야에서 일반적으로 사용되는 용어 선택
3. 너무 광범위하거나 너무 좁은 용어는 피하기
4. 5-8개의 키워드/키워드 구문 추출
5. 키워드를 관련성 순서로 정렬

추출된 키워드:
""",
    input_variables=["paper_title", "paper_content"],
)


# 논문 인용문 생성 프롬프트
PAPER_CITATION_PROMPT = PromptTemplate(
    template="""당신은 학술 인용 전문가입니다. 다음 논문 정보를 바탕으로 {citation_style} 형식의 인용문을 생성해 주세요.

논문 정보:
- 제목: {title}
- 저자: {authors}
- 발행연도: {year}
- 저널/출판물: {journal}
- 권(Volume): {volume}
- 호(Issue): {issue}
- 페이지: {pages}
- DOI: {doi}
- URL: {url}

{citation_style} 형식 인용문:
""",
    input_variables=["citation_style", "title", "authors", "year", "journal", "volume", "issue", "pages", "doi", "url"],
)


# 연구 질문 개발 프롬프트
RESEARCH_QUESTION_PROMPT = PromptTemplate(
    template="""당신은 학술 연구 질문 개발 전문가입니다. 다음 주제에 대한 효과적인 연구 질문을 개발해 주세요.

연구 주제: {topic}
연구 분야: {field}
연구 유형: {research_type}

연구 질문 개발 지침:
1. 명확하고 구체적인 질문 작성
2. 연구 가능한 질문 설계 (측정/관찰 가능한 변수 포함)
3. 중요하고 의미 있는 질문 개발 (학문적/실용적 가치)
4. 주요 연구 질문과 부차적 질문 포함
5. 각 질문에 대한 간략한 근거 제시

연구 질문:
""",
    input_variables=["topic", "field", "research_type"],
)


# 논문 비판적 분석 프롬프트
CRITICAL_ANALYSIS_PROMPT = PromptTemplate(
    template="""당신은 학술 논문 비판적 분석 전문가입니다. 다음 논문에 대한 철저한 비판적 분석을 제공해 주세요.

논문 제목: {paper_title}
논문 내용:
{paper_content}

비판적 분석 지침:
1. 논문의 주요 주장과 증거 요약
2. 연구 방법론의 강점과 약점 평가
3. 데이터 분석과 해석의 타당성 검토
4. 논리적 일관성과 논증 평가
5. 연구의 한계와 잠재적 편향 식별
6. 연구의 학문적/실용적 기여도 평가
7. 후속 연구에 대한 제안

비판적 분석:
""",
    input_variables=["paper_title", "paper_content"],
)


# 논문 결론 작성 프롬프트
PAPER_CONCLUSION_PROMPT = PromptTemplate(
    template="""당신은 학술 논문 결론 작성 전문가입니다. 다음 논문 내용을 바탕으로 효과적인 결론을 작성해 주세요.

논문 제목: {paper_title}
논문 주요 내용:
{paper_content}

결론 작성 지침:
1. 연구 질문/목적 상기시키기
2. 주요 연구 결과 요약 (새로운 내용 도입 없이)
3. 결과의 중요성과 의의 설명
4. 연구의 한계 인정
5. 향후 연구 방향 제안
6. 최종 통찰이나 메시지로 마무리

결론은 논문의 "마지막 인상"을 만듭니다. 명확하고 인상적이며 논문의 가치를 강조하는 결론을 작성하세요.

결론:
""",
    input_variables=["paper_title", "paper_content"],
) 
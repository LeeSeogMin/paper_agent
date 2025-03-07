"""
Paper-related prompt templates
Collection of prompt templates for paper writing and editing.
"""

from langchain_core.prompts import PromptTemplate
from typing import Dict, List, Optional, Any

# 프롬프트 그룹화를 위한 클래스
class PaperPrompts:
    """Paper writing and editing prompts collection"""
    
    @staticmethod
    def get_prompt(prompt_name: str, **kwargs) -> str:
        """
        Get formatted prompt by name
        
        Args:
            prompt_name: Name of the prompt to retrieve
            **kwargs: Variables to format the prompt with
            
        Returns:
            Formatted prompt string
        """
        prompts = {
            "research_topic": RESEARCH_TOPIC_SUGGESTION_PROMPT,
            "research_plan": RESEARCH_PLAN_PROMPT,
            "paper_outline": PAPER_OUTLINE_PROMPT,
            "paper_section": PAPER_SECTION_PROMPT,
            "paper_editing": PAPER_EDITING_PROMPT,
            "paper_review": PAPER_REVIEW_PROMPT,
            "paper_summary": PAPER_SUMMARY_PROMPT,
            "paper_translation": PAPER_TRANSLATION_PROMPT,
            "reference_formatting": REFERENCE_FORMATTING_PROMPT,
            "keyword_extraction": KEYWORD_EXTRACTION_PROMPT,
            "citation_generation": PAPER_CITATION_PROMPT,
            "research_question": RESEARCH_QUESTION_PROMPT,
            "critical_analysis": CRITICAL_ANALYSIS_PROMPT,
            "paper_conclusion": PAPER_CONCLUSION_PROMPT,
            "literature_review": LITERATURE_REVIEW_PROMPT,
            "paper_introduction": PAPER_INTRODUCTION_PROMPT,
            "methodology": METHODOLOGY_PROMPT,
            "research_summary": RESEARCH_SUMMARY_PROMPT,
            "analysis": ANALYSIS_PROMPT,
            "custom_writing": CUSTOM_WRITING_PROMPT
        }
        
        if prompt_name not in prompts:
            raise ValueError(f"Prompt '{prompt_name}' not found. Available prompts: {list(prompts.keys())}")
        
        prompt_template = prompts[prompt_name]
        
        # 입력 변수 검증
        missing_vars = [var for var in prompt_template.input_variables if var not in kwargs]
        if missing_vars:
            raise ValueError(f"Missing required variables for prompt '{prompt_name}': {missing_vars}")
        
        return prompt_template.format(**kwargs)

    @staticmethod
    def validate_inputs(prompt_template: PromptTemplate, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate inputs for a prompt template
        
        Args:
            prompt_template: The prompt template to validate inputs for
            inputs: The input variables
            
        Returns:
            Validated inputs
        """
        # 필수 변수 확인
        missing_vars = [var for var in prompt_template.input_variables if var not in inputs]
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        # 불필요한 변수 제거
        return {k: v for k, v in inputs.items() if k in prompt_template.input_variables}


# Research topic suggestion prompt
RESEARCH_TOPIC_SUGGESTION_PROMPT = PromptTemplate(
    template="""You are an academic research expert. Please suggest specific and researchable topics related to the given field of interest.
    
Field of interest: {field}

Expected results:
1. Five specific topics worthy of research in the field
2. Brief description and importance of each topic
3. Explanation of how each topic addresses a gap in current academic discourse

Considerations when suggesting topics:
- Relevance to current research trends
- Feasibility and practicality
- Academic significance and potential impact
- Innovation and creativity

Suggested topics:
""",
    input_variables=["field"],
)


# Research plan writing prompt
RESEARCH_PLAN_PROMPT = PromptTemplate(
    template="""You are an expert in developing academic research plans. Please develop a systematic research plan for the following topic.

Research topic: {topic}

The research plan should include:
1. Research background and purpose
2. Key research questions (3-5)
3. Methodology overview
4. Required materials and resources
5. Expected results and contributions
6. Potential limitations and mitigation strategies
7. Key literature to reference (if applicable)

Please provide a specific and feasible plan.

Research plan:
""",
    input_variables=["topic"],
)


# Paper outline writing prompt
PAPER_OUTLINE_PROMPT = PromptTemplate(
    template="""You are an expert in creating academic paper outlines. Please create a systematic paper outline based on the following topic and research materials.

Paper topic: {topic}
Paper type: {paper_type}

Collected research materials:
{research_materials}

The outline should include:
1. Paper title (clear and descriptive)
2. Abstract structure (brief explanation of research purpose, methods, results, and conclusion)
3. Title and content summary for each section (from introduction to conclusion)
4. Main arguments to be addressed in each section

The outline should have a logical flow and be structured to effectively explore the research topic.

Paper outline:
""",
    input_variables=["topic", "paper_type", "research_materials"],
)


# Paper section writing prompt
PAPER_SECTION_PROMPT = PromptTemplate(
    template="""You are an expert in writing academic paper sections. Please write a high-quality paper section based on the following information.

Paper title: {paper_title}
Section title: {section_title}
Section purpose: {section_purpose}

Paper outline:
{paper_outline}

Related research materials:
{research_materials}

Writing guidelines:
1. Write clearly and systematically using academic style
2. Include appropriate citations (utilizing related research materials)
3. Maintain logical flow and consistency
4. Provide sufficient explanation of key concepts and ideas
5. Include critical analysis and argumentation

Section content:
""",
    input_variables=["paper_title", "section_title", "section_purpose", "paper_outline", "research_materials"],
)


# Paper editing and improvement prompt
PAPER_EDITING_PROMPT = PromptTemplate(
    template="""You are an expert in editing academic papers. Please review and improve the following paper content.

Task type: {editing_type}
Style guide: {style_guide}
Paper format: {paper_format}

Paper content:
{paper_content}

Editing and improvement guidelines:
1. Correct grammar, spelling, and punctuation errors
2. Improve sentence structure and flow
3. Enhance consistency and clarity
4. Remove redundancies and unnecessary content
5. Use appropriate academic expressions and terminology
6. Check citation and reference format

Special instructions for paper format:
- If paper_format is "standard": Format as a complete academic paper with all standard sections.
- If paper_format is "literature_review": Format as a literature review that summarizes and synthesizes existing research. Focus on organizing content by themes or findings rather than creating a full research paper. Ensure all information is properly cited with references at the end.

Improve the expression while maintaining the original meaning.

Edited content:
""",
    input_variables=["editing_type", "style_guide", "paper_content", "paper_format"],
)


# Paper review and feedback prompt
PAPER_REVIEW_PROMPT = PromptTemplate(
    template="""You are an academic paper reviewer. Please thoroughly review the following paper and provide comprehensive feedback.

Paper title: {paper_title}
Paper type: {paper_type}

Paper content:
{paper_content}

Review guidelines:
1. Evaluate the clarity and importance of the research question/purpose
2. Assess the appropriateness and robustness of the methodology
3. Evaluate the validity and interpretation of results
4. Assess the depth and insight of the discussion
5. Evaluate the appropriateness and comprehensiveness of literature citations
6. Assess organization, clarity, and originality

Please provide feedback in the following format:
- Key strengths
- Areas needing improvement
- Specific improvement suggestions
- Overall assessment

Feedback:
""",
    input_variables=["paper_title", "paper_type", "paper_content"],
)


# Paper summary prompt
PAPER_SUMMARY_PROMPT = PromptTemplate(
    template="""You are an expert in summarizing academic papers. Please concisely summarize the following paper.

Paper title: {paper_title}
Paper content:
{paper_content}

Please structure your summary as follows:
1. Research purpose (1-2 sentences)
2. Methodology (2-3 sentences)
3. Key results (3-4 sentences)
4. Conclusion and significance (1-2 sentences)

Keep the entire summary within 300 words. Summarize clearly and concisely while maintaining the key content of the original text.

Summary:
""",
    input_variables=["paper_title", "paper_content"],
)


# Paper translation prompt
PAPER_TRANSLATION_PROMPT = PromptTemplate(
    template="""You are an expert in translating academic papers. Please translate the following {source_language} paper into {target_language}.

Original text:
{source_text}

Translation guidelines:
1. Maintain academic style and terminology
2. Preserve sentence structure and logical flow
3. Use standard terminology for technical terms
4. Accurately convey the meaning of the original text
5. Ensure the translation reads naturally in {target_language}

Translation:
""",
    input_variables=["source_language", "target_language", "source_text"],
)


# Reference formatting prompt
REFERENCE_FORMATTING_PROMPT = PromptTemplate(
    template="""You are an expert in academic reference formatting. Please convert the following reference list to {target_style} format.

Original references:
{references}

Conversion guidelines:
1. Follow the {target_style} style guide precisely
2. Include all required elements (authors, year, title, source, etc.)
3. Apply formatting, punctuation, italics, etc. correctly
4. Sort entries alphabetically (unless otherwise specified)

{target_style} formatted references:
""",
    input_variables=["target_style", "references"],
)


# Paper keyword extraction prompt
KEYWORD_EXTRACTION_PROMPT = PromptTemplate(
    template="""You are an expert in extracting keywords from academic papers. Please extract the most relevant keywords from the following paper content.

Paper title: {paper_title}
Paper content:
{paper_content}

Extraction guidelines:
1. Select keywords directly related to the paper's topic
2. Choose terms commonly used in the academic field
3. Avoid terms that are too broad or too narrow
4. Extract 5-8 keywords/keyword phrases
5. Sort keywords by relevance

Extracted keywords:
""",
    input_variables=["paper_title", "paper_content"],
)


# Paper citation generation prompt
PAPER_CITATION_PROMPT = PromptTemplate(
    template="""You are an expert in academic citations. Please generate a citation in {citation_style} format based on the following paper information.

Paper information:
- Title: {title}
- Authors: {authors}
- Publication year: {year}
- Journal/publication: {journal}
- Volume: {volume}
- Issue: {issue}
- Pages: {pages}
- DOI: {doi}
- URL: {url}

{citation_style} format citation:
""",
    input_variables=["citation_style", "title", "authors", "year", "journal", "volume", "issue", "pages", "doi", "url"],
)


# Research question development prompt
RESEARCH_QUESTION_PROMPT = PromptTemplate(
    template="""You are an expert in developing academic research questions. Please develop effective research questions for the following topic.

Research topic: {topic}
Research field: {field}
Research type: {research_type}

Research question development guidelines:
1. Create clear and specific questions
2. Design researchable questions (including measurable/observable variables)
3. Develop important and meaningful questions (academic/practical value)
4. Include primary research questions and secondary questions
5. Provide brief rationale for each question

Research questions:
""",
    input_variables=["topic", "field", "research_type"],
)


# Paper critical analysis prompt
CRITICAL_ANALYSIS_PROMPT = PromptTemplate(
    template="""You are an expert in critical analysis of academic papers. Please provide a thorough critical analysis of the following paper.

Paper title: {paper_title}
Paper content:
{paper_content}

Critical analysis guidelines:
1. Summarize the main claims and evidence of the paper
2. Evaluate strengths and weaknesses of the research methodology
3. Review the validity of data analysis and interpretation
4. Assess logical consistency and argumentation
5. Identify limitations and potential biases of the research
6. Evaluate the academic/practical contribution of the research
7. Suggest directions for follow-up research

Critical analysis:
""",
    input_variables=["paper_title", "paper_content"],
)


# Paper conclusion writing prompt
PAPER_CONCLUSION_PROMPT = PromptTemplate(
    template="""You are an expert in writing academic paper conclusions. Please write an effective conclusion based on the following paper content.

Paper title: {paper_title}
Paper main content:
{paper_content}

Conclusion writing guidelines:
1. Remind readers of the research question/purpose
2. Summarize key research findings (without introducing new content)
3. Explain the importance and significance of the results
4. Acknowledge research limitations
5. Suggest directions for future research
6. End with a final insight or message

The conclusion creates the "last impression" of the paper. Write a clear, impressive conclusion that emphasizes the value of the paper.

Conclusion:
""",
    input_variables=["paper_title", "paper_content"],
)


# 문헌 리뷰 프롬프트 추가
LITERATURE_REVIEW_PROMPT = PromptTemplate(
    template="""당신은 학술 논문의 문헌 리뷰 섹션을 작성하는 전문가입니다.
다음 연구 자료를 바탕으로 주제 '{topic}'에 관한 체계적인 문헌 리뷰를 작성해 주세요.

## 형식
{format}

## 연구 자료
{materials}

## 지침
1. 각 연구의 핵심 주장, 방법론, 결과를 요약하세요.
2. 연구들 간의 관계와 발전 과정을 보여주세요.
3. 현재 연구 분야의 동향과 격차를 식별하세요.
4. 학술적이고 객관적인 어조를 유지하세요.
5. 적절한 인용 형식을 사용하세요.

문헌 리뷰:
""",
    input_variables=["topic", "format", "materials"],
)

# 별칭 추가
SECTION_WRITING_PROMPT = PAPER_SECTION_PROMPT 

# 논문 서론 작성 프롬프트 추가
PAPER_INTRODUCTION_PROMPT = PromptTemplate(
    template="""당신은 학술 논문의 서론을 작성하는 전문가입니다.
다음 정보를 바탕으로 논문 '{paper_title}'의 서론을 작성해 주세요.

## 연구 주제
{topic}

## 연구 배경
{background}

## 연구 목적
{purpose}

## 지침
1. 연구 주제의 중요성과 관련성을 설명하세요.
2. 현재까지의 연구 동향과 한계점을 간략히 설명하세요.
3. 본 연구의 목적과 의의를 명확히 제시하세요.
4. 연구 질문 또는 가설을 포함하세요.
5. 논문의 구성을 간략히 소개하세요.

서론:
""",
    input_variables=["paper_title", "topic", "background", "purpose"],
) 

# 연구 방법론 작성 프롬프트 추가
METHODOLOGY_PROMPT = PromptTemplate(
    template="""당신은 학술 논문의 연구 방법론 섹션을 작성하는 전문가입니다.
다음 정보를 바탕으로 논문 '{paper_title}'의 연구 방법론을 작성해 주세요.

## 연구 주제
{topic}

## 연구 질문/가설
{research_questions}

## 연구 설계
{research_design}

## 지침
1. 연구 접근법과 설계를 명확히 설명하세요.
2. 데이터 수집 방법과 도구를 상세히 기술하세요.
3. 분석 방법과 절차를 체계적으로 설명하세요.
4. 연구의 타당성과 신뢰성을 확보하기 위한 조치를 설명하세요.
5. 연구 윤리적 고려사항을 포함하세요.

연구 방법론:
""",
    input_variables=["paper_title", "topic", "research_questions", "research_design"],
) 

# 연구 요약 프롬프트 추가
RESEARCH_SUMMARY_PROMPT = PromptTemplate(
    template="""당신은 연구 자료를 요약하는 전문가입니다.
다음 연구 자료를 간결하고 명확하게 요약해 주세요.

## 연구 자료
{research_material}

## 요약 지침
1. 연구의 주요 목적과 연구 질문을 명시하세요.
2. 사용된 방법론을 간략히 설명하세요.
3. 핵심 결과와 발견을 요약하세요.
4. 연구의 의의와 한계점을 포함하세요.
5. 300단어 이내로 작성하세요.

연구 요약:
""",
    input_variables=["research_material"],
)

# 분석 프롬프트 추가
ANALYSIS_PROMPT = PromptTemplate(
    template="""당신은 연구 데이터 분석 전문가입니다.
다음 데이터와 정보를 바탕으로 체계적인 분석을 수행해 주세요.

## 분석 주제
{topic}

## 데이터 설명
{data_description}

## 분석 목표
{analysis_goals}

## 분석 지침
1. 데이터의 주요 패턴과 경향을 식별하세요.
2. 핵심 통계와 수치를 계산하고 해석하세요.
3. 데이터 간의 관계와 상관성을 분석하세요.
4. 발견한 내용의 의미와 시사점을 설명하세요.
5. 분석의 한계점을 명시하세요.

분석 결과:
""",
    input_variables=["topic", "data_description", "analysis_goals"],
)

# 맞춤형 작성 프롬프트 추가
CUSTOM_WRITING_PROMPT = PromptTemplate(
    template="""당신은 학술 논문 작성 전문가입니다.
다음 요구사항에 따라 맞춤형 학술 콘텐츠를 작성해 주세요.

## 작성 주제
{topic}

## 작성 유형
{content_type}

## 세부 요구사항
{requirements}

## 참고 자료
{reference_materials}

## 작성 지침
1. 요구사항에 정확히 부합하는 콘텐츠를 작성하세요.
2. 학술적이고 전문적인 어조를 유지하세요.
3. 논리적 구조와 명확한 논지를 개발하세요.
4. 적절한 인용과 참조를 포함하세요.
5. 요청된 형식과 길이를 준수하세요.

작성 결과:
""",
    input_variables=["topic", "content_type", "requirements", "reference_materials"],
) 
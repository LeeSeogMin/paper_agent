"""
Research prompts module
Defines prompt templates for conducting research and analyzing materials.
"""

from langchain_core.prompts import PromptTemplate
from typing import Dict, List, Optional, Any

# 프롬프트 그룹화를 위한 클래스
class ResearchPrompts:
    """Research-related prompts collection"""
    
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
            "research_system": RESEARCH_SYSTEM_PROMPT,
            "query_generation": QUERY_GENERATION_PROMPT,
            "source_evaluation": SOURCE_EVALUATION_PROMPT,
            "search_results_analysis": SEARCH_RESULTS_ANALYSIS_PROMPT,
            "literature_review": LITERATURE_REVIEW_PROMPT,
            "research_methodology": RESEARCH_METHODOLOGY_PROMPT,
            "data_analysis": DATA_ANALYSIS_PROMPT,
            "research_interpretation": RESEARCH_INTERPRETATION_PROMPT,
            "research_gap_identification": RESEARCH_GAP_IDENTIFICATION_PROMPT,
            "research_proposal": RESEARCH_PROPOSAL_PROMPT
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


# Research system prompt
RESEARCH_SYSTEM_PROMPT = PromptTemplate(
    input_variables=[],
    template="""You are a research assistant specialized in academic literature review and analysis.
Your role is to help researchers gather, analyze, and organize scholarly information effectively.
Provide thorough, accurate, and academically rigorous responses.
"""
)

# Query generation prompt
QUERY_GENERATION_PROMPT = PromptTemplate(
    input_variables=["n_queries", "topic"],
    template="""Generate {n_queries} specific search queries to find academic papers on the following topic:

TOPIC: {topic}

For each query:
1. Focus on different aspects of the topic
2. Use specific academic terminology
3. Include key concepts, methodologies, or frameworks relevant to the topic
4. Formulate queries that would yield high-quality academic results

Format your response as a numbered list of queries, with each query on a new line.
For each query, include a brief rationale explaining why this query would be valuable.
"""
)

# Source evaluation prompt
SOURCE_EVALUATION_PROMPT = PromptTemplate(
    input_variables=["source", "topic"],
    template="""Evaluate the following academic source for relevance and quality in relation to the research topic.

SOURCE INFORMATION:
- Title: {source.title}
- Authors: {source.authors}
- Year: {source.year}
- Venue: {source.venue}
- Abstract: {source.abstract}
- Citation Count: {source.citation_count}

RESEARCH TOPIC: {topic}

Provide an evaluation with:
1. Relevance score (0-10, with 10 being extremely relevant)
2. Explanation of relevance to the topic
3. Assessment of source quality based on venue, authors, citation count, and methodology
4. Potential value of this source for the research

Format: Begin your response with "Score: X/10" followed by your detailed evaluation.
"""
)

# Search results analysis prompt
SEARCH_RESULTS_ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["topic", "materials"],
    template="""Analyze the following research materials collected on the topic.

TOPIC: {topic}

MATERIALS:
{materials}

Provide a comprehensive analysis including:
1. Key Findings: Identify the main discoveries, arguments, and conclusions across the materials
2. Themes: Identify recurring themes and patterns across the sources
3. Research Gaps: Identify areas where more research is needed
4. Methodologies: Summarize the primary research methodologies used
5. Recommendations: Suggest specific directions for further research

Format your analysis with clear section headings for each of the five areas.
Be thorough but concise, highlighting the most significant points.
"""
)

# Literature review prompt
LITERATURE_REVIEW_PROMPT = PromptTemplate(
    input_variables=["literature_list"],
    template="""Review the following academic literature and provide a comprehensive synthesis:

Literature List:
{literature_list}

Your review should include:
1. Main findings and arguments of each work
2. Comparison of methodological approaches
3. Consistent themes and patterns
4. Discrepancies and points of contention
5. Identification of research gaps
6. Suggestions for future research directions

Provide an objective and critical review.
"""
)

# Research methodology design prompt
RESEARCH_METHODOLOGY_PROMPT = PromptTemplate(
    input_variables=["research_question", "research_objectives", "constraints"],
    template="""Design a methodology for the following research question:

Research Question: {research_question}
Research Objectives: {research_objectives}
Constraints: {constraints}

Your methodology should include:
1. Type of research design
2. Data collection methods
3. Sampling strategy
4. Data analysis techniques
5. Ethical considerations
6. Strategies for ensuring validity and reliability

Provide a systematic and scientifically rigorous methodology.
"""
)

# Data analysis prompt
DATA_ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["data_description", "analysis_objectives", "analysis_constraints"],
    template="""Analyze the following dataset:

Data Description: {data_description}
Analysis Objectives: {analysis_objectives}
Analysis Constraints: {analysis_constraints}

Your analysis should include:
1. Data preprocessing steps
2. Analysis techniques to apply
3. Exploration of relationships between key variables
4. Identification of patterns and trends
5. Interpretation of results
6. Limitations and caveats

Provide a systematic and rigorous data analysis.
"""
)

# Research findings interpretation prompt
RESEARCH_INTERPRETATION_PROMPT = PromptTemplate(
    input_variables=["research_findings", "research_question", "research_context"],
    template="""Interpret the following research findings:

Research Findings: {research_findings}
Research Question: {research_question}
Research Context: {research_context}

Your interpretation should include:
1. Significance of the main findings
2. Relationship to existing literature
3. Theoretical and practical implications
4. Explanation of unexpected results
5. Limitations and caveats
6. Directions for future research

Provide an objective and balanced interpretation.
"""
)

# Research gap identification prompt
RESEARCH_GAP_IDENTIFICATION_PROMPT = PromptTemplate(
    input_variables=["research_field", "current_knowledge"],
    template="""Identify the main research gaps in the following research field:

Research Field: {research_field}
Current State of Knowledge: {current_knowledge}

Your identification should include:
1. Unresolved research questions
2. Methodological limitations
3. Theoretical inconsistencies
4. Lack of empirical evidence
5. Lack of research in specific contexts or populations
6. Importance and priority of each gap

Provide a systematic and comprehensive analysis of research gaps.
"""
)

# Research proposal prompt
RESEARCH_PROPOSAL_PROMPT = PromptTemplate(
    input_variables=["topic", "research_context", "constraints"],
    template="""Write a research proposal on the following topic:

Topic: {topic}
Research Context: {research_context}
Constraints: {constraints}

Your proposal should include:
1. Background and problem definition
2. Summary of literature review
3. Research questions and objectives
4. Research methodology
5. Expected results and implications
6. Timeline and resource requirements
7. Potential limitations and mitigation strategies

Provide a well-structured and academically sound research proposal.
"""
)
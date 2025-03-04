"""
Writing prompts module
Defines prompt templates for various aspects of academic paper writing.
"""

from langchain_core.prompts import PromptTemplate
from typing import Dict, List, Optional, Any

# 프롬프트 그룹화를 위한 클래스
class WritingPrompts:
    """Writing-related prompts collection"""
    
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
            "paper_title": PAPER_TITLE_GENERATION_PROMPT,
            "abstract": ABSTRACT_WRITING_PROMPT,
            "introduction": INTRODUCTION_WRITING_PROMPT,
            "methodology": METHODOLOGY_WRITING_PROMPT,
            "results": RESULTS_WRITING_PROMPT,
            "discussion": DISCUSSION_WRITING_PROMPT,
            "conclusion": CONCLUSION_WRITING_PROMPT,
            "academic_comparison": ACADEMIC_COMPARISON_PROMPT,
            "citation_integration": CITATION_INTEGRATION_PROMPT
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


# Paper title generation prompt
PAPER_TITLE_GENERATION_PROMPT = PromptTemplate(
    template="""Generate effective academic paper titles for the following topic and research content:

Topic: {topic}
Research Content: {research_content}
Target Audience: {target_audience}

The title should meet the following criteria:
1. Clarity and conciseness
2. Accurate reflection of research content
3. Appeal to capture reader interest
4. Maintenance of academic tone
5. Inclusion of key keywords

Provide 3-5 possible title options, with a brief rationale for each option.
""",
    input_variables=["topic", "research_content", "target_audience"],
)

# Abstract writing prompt
ABSTRACT_WRITING_PROMPT = PromptTemplate(
    template="""Write an abstract for an academic paper based on the following information:

Paper Title: {title}
Research Purpose: {research_purpose}
Methodology: {methodology}
Key Findings: {key_findings}
Conclusion: {conclusion}

The abstract should meet the following criteria:
1. Limited to 250 words
2. Clear summary of purpose, methods, results, and conclusion
3. Self-contained and independently understandable
4. Academic tone
5. Inclusion of key keywords

Write a clear and informative abstract.
""",
    input_variables=["title", "research_purpose", "methodology", "key_findings", "conclusion"],
)

# Introduction writing prompt
INTRODUCTION_WRITING_PROMPT = PromptTemplate(
    template="""Write an introduction for an academic paper based on the following information:

Paper Title: {title}
Research Background: {research_background}
Research Problem: {research_problem}
Research Purpose: {research_purpose}
Research Significance: {research_significance}

The introduction should follow this structure:
1. Introduction to the research topic and background information
2. Identification of the research problem or gap
3. Statement of research purpose and objectives
4. Explanation of the significance and potential contributions of the research
5. Overview of the paper structure

Write an introduction that captures reader interest and clearly establishes the context of the research.
""",
    input_variables=["title", "research_background", "research_problem", "research_purpose", "research_significance"],
)

# Literature review writing prompt
LITERATURE_REVIEW_WRITING_PROMPT = PromptTemplate(
    template="""Write a literature review section for an academic paper based on the following information:

Research Topic: {research_topic}
Key Concepts: {key_concepts}
Related Studies: {related_studies}
Theoretical Framework: {theoretical_framework}

The literature review should meet the following criteria:
1. Systematic and critical analysis of relevant literature
2. Explanation of key theories and concepts related to the topic
3. Identification of strengths and limitations in existing research
4. Emphasis on research gaps
5. Explanation of the relationship between the current research and existing literature

Write a comprehensive and well-organized literature review.
""",
    input_variables=["research_topic", "key_concepts", "related_studies", "theoretical_framework"],
)

# Methodology writing prompt
METHODOLOGY_WRITING_PROMPT = PromptTemplate(
    template="""Write a methodology section for an academic paper based on the following information:

Research Design: {research_design}
Data Collection Methods: {data_collection}
Sampling Strategy: {sampling_strategy}
Analysis Techniques: {analysis_techniques}
Ethical Considerations: {ethical_considerations}

The methodology section should meet the following criteria:
1. Clear explanation of the research approach
2. Detailed description of data collection and analysis methods
3. Justification for methodological choices
4. Methods to ensure reliability and validity of the research
5. Potential limitations and how they were addressed

Write a reproducible and scientifically rigorous methodology section.
""",
    input_variables=["research_design", "data_collection", "sampling_strategy", "analysis_techniques", "ethical_considerations"],
)

# Results writing prompt
RESULTS_WRITING_PROMPT = PromptTemplate(
    template="""Write a results section for an academic paper based on the following information:

Research Questions: {research_questions}
Data Analysis Results: {data_analysis_results}
Key Findings: {key_findings}
Statistical Significance: {statistical_significance}

The results section should meet the following criteria:
1. Objective and factual presentation of results
2. Logical and systematic organization
3. Appropriate use of tables, graphs, or quotations
4. Reporting of results without interpretation
5. Clear connection to research questions

Write a clear and accurate results section.
""",
    input_variables=["research_questions", "data_analysis_results", "key_findings", "statistical_significance"],
)

# Discussion writing prompt
DISCUSSION_WRITING_PROMPT = PromptTemplate(
    template="""Write a discussion section for an academic paper based on the following information:

Results Summary: {results_summary}
Relationship to Literature: {relation_to_literature}
Implications: {implications}
Limitations: {limitations}
Future Research: {future_research}

The discussion section should meet the following criteria:
1. Interpretation of the meaning and significance of results
2. Explanation of agreement or disagreement with existing research
3. Discussion of theoretical and practical implications
4. Acknowledgment and explanation of research limitations
5. Suggestions for future research

Write an insightful and critical discussion section.
""",
    input_variables=["results_summary", "relation_to_literature", "implications", "limitations", "future_research"],
)

# Conclusion writing prompt
CONCLUSION_WRITING_PROMPT = PromptTemplate(
    template="""Write a conclusion section for an academic paper based on the following information:

Research Purpose: {research_purpose}
Key Findings: {key_findings}
Research Contributions: {research_contributions}
Final Thoughts: {final_thoughts}

The conclusion section should meet the following criteria:
1. Summary of research purpose and key findings
2. Emphasis on the significance and contributions of the research
3. Reinforcement of key implications
4. Brief mention of limitations
5. Strong and memorable concluding statement

Write a concise and effective conclusion section.
""",
    input_variables=["research_purpose", "key_findings", "research_contributions", "final_thoughts"],
)

# References writing prompt
REFERENCES_WRITING_PROMPT = PromptTemplate(
    template="""Format the following reference information according to the {citation_style} style:

Reference Information:
{reference_information}

The reference list should meet the following criteria:
1. Exact adherence to the {citation_style} style guide format
2. Alphabetical ordering (by author)
3. Inclusion of all required information
4. Consistent formatting

Write an accurate and complete reference list.
""",
    input_variables=["citation_style", "reference_information"],
)

# Academic paragraph writing prompt
ACADEMIC_PARAGRAPH_WRITING_PROMPT = PromptTemplate(
    template="""Write an academic paragraph on the following topic and point:

Topic: {topic}
Main Point: {main_point}
Supporting Evidence: {supporting_evidence}
Context: {context}

The paragraph should meet the following criteria:
1. Begin with a clear topic sentence
2. Logical flow and structure
3. Provision of appropriate evidence and examples
4. Maintenance of academic tone
5. Effective integration of main point and context
6. Concluding with a strong concluding sentence

Write a cohesive and persuasive academic paragraph.
""",
    input_variables=["topic", "main_point", "supporting_evidence", "context"],
)

# Academic critique writing prompt
ACADEMIC_CRITIQUE_WRITING_PROMPT = PromptTemplate(
    template="""Write an academic critique of the following work or theory:

Work/Theory: {work_or_theory}
Author/Creator: {author_or_creator}
Main Claims: {main_claims}
Critical Perspective: {critical_perspective}

The critique should meet the following criteria:
1. Fair and accurate summary of the work or theory
2. Clear presentation of critical perspective
3. Support of criticism with specific evidence and examples
4. Balanced approach (considering both strengths and weaknesses)
5. Maintenance of academic tone
6. Suggestion of alternative viewpoints or interpretations

Write a thorough and balanced academic critique.
""",
    input_variables=["work_or_theory", "author_or_creator", "main_claims", "critical_perspective"],
)

# Research question development prompt
RESEARCH_QUESTION_DEVELOPMENT_PROMPT = PromptTemplate(
    template="""Develop effective research questions for the following topic:

Research Field: {research_field}
Topic Area: {topic_area}
Research Gaps: {research_gaps}
Aspects of Interest: {aspects_of_interest}

The research questions should meet the following criteria:
1. Clarity and specificity
2. Researchability
3. Significance and relevance
4. Originality
5. Meeting SMART criteria (Specific, Measurable, Achievable, Relevant, Time-bound)

Develop 3-5 potential research questions that explore different aspects of the topic, providing a rationale for each question.
""",
    input_variables=["research_field", "topic_area", "research_gaps", "aspects_of_interest"],
)

# Hypothesis formulation prompt
HYPOTHESIS_FORMULATION_PROMPT = PromptTemplate(
    template="""Formulate testable hypotheses for the following research question:

Research Question: {research_question}
Theoretical Framework: {theoretical_framework}
Existing Evidence: {existing_evidence}
Variables: {variables}

The hypotheses should meet the following criteria:
1. Clear and concise statements
2. Testability
3. Specification of relationships between variables
4. Consistency with theoretical framework
5. Grounding in existing evidence

Provide both null and alternative hypotheses, explaining the rationale for each.
""",
    input_variables=["research_question", "theoretical_framework", "existing_evidence", "variables"],
)

# Academic definition writing prompt
ACADEMIC_DEFINITION_WRITING_PROMPT = PromptTemplate(
    template="""Write a comprehensive academic definition for the following concept or term:

Concept/Term: {concept_or_term}
Academic Field: {academic_field}
Context: {context}
Related Concepts: {related_concepts}

The definition should meet the following criteria:
1. Clarity and accuracy
2. Comprehensiveness (including all important aspects of the concept)
3. Explanation of position within academic context
4. Explanation of relationship to related concepts
5. Inclusion of different perspectives or interpretations when necessary

Write a definition that accurately reflects academic understanding in the field.
""",
    input_variables=["concept_or_term", "academic_field", "context", "related_concepts"],
)

# Academic comparison prompt
ACADEMIC_COMPARISON_PROMPT = PromptTemplate(
    template="""Provide a comparative analysis of the following concepts, theories, or approaches:

Item 1: {item_1}
Item 2: {item_2}
Comparison Criteria: {comparison_criteria}
Analysis Context: {analysis_context}

The comparative analysis should meet the following criteria:
1. Clear description of each item
2. Systematic comparison according to specified criteria
3. Balanced analysis of similarities and differences
4. Evaluation of strengths and limitations of each item
5. Explanation of relevance of comparison within context

Provide a thorough and balanced comparative analysis.
""",
    input_variables=["item_1", "item_2", "comparison_criteria", "analysis_context"],
)

# Citation integration prompt
CITATION_INTEGRATION_PROMPT = PromptTemplate(
    template="""Effectively integrate the following citation into academic text:

Quote: {quote}
Source: {source}
Context: {context}
Purpose of Integration: {integration_purpose}

The citation integration should meet the following criteria:
1. Natural flow and appropriate fit with context
2. Use of appropriate citation format ({citation_style} style)
3. Clear explanation of relevance and importance of citation
4. Provision of appropriate introduction and analysis before and after citation
5. Clear attribution to avoid plagiarism

Write an academic paragraph that effectively integrates the citation.
""",
    input_variables=["quote", "source", "context", "integration_purpose", "citation_style"],
)
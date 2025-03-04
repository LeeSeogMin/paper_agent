"""
Prompt templates for reviewing academic papers.

This module contains prompt templates that guide the review process for academic papers,
including comprehensive reviews, section-specific reviews, and specialized review types
such as methodology assessment, literature review evaluation, and statistical analysis review.
"""

from langchain_core.prompts import PromptTemplate
from typing import Dict, List, Optional, Any

# 프롬프트 그룹화를 위한 클래스
class ReviewPrompts:
    """Review-related prompts collection"""
    
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
            "comprehensive_review": COMPREHENSIVE_REVIEW_PROMPT,
            "introduction_review": INTRODUCTION_REVIEW_PROMPT,
            "ethical_considerations_review": ETHICAL_CONSIDERATIONS_REVIEW_PROMPT,
            # 다른 프롬프트들도 여기에 추가
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


# Comprehensive paper review prompt
COMPREHENSIVE_REVIEW_PROMPT = PromptTemplate(
    input_variables=["paper_title", "paper_content", "review_criteria", "field_of_study"],
    template="""
You are an expert academic reviewer in the field of {field_of_study}. Conduct a comprehensive review of the following paper:

TITLE: {paper_title}

PAPER CONTENT:
{paper_content}

REVIEW CRITERIA:
{review_criteria}

Provide a thorough academic review that addresses the following aspects:

1. OVERALL ASSESSMENT:
   - Evaluate the paper's contribution to the field
   - Assess the originality and significance of the work
   - Determine if the paper meets the standards for publication

2. CONTENT ANALYSIS:
   - Evaluate the clarity and coherence of the research question/hypothesis
   - Assess the theoretical framework and its application
   - Review the methodology and its appropriateness
   - Analyze the results and their interpretation
   - Evaluate the discussion and conclusions

3. STRUCTURE AND ORGANIZATION:
   - Assess the logical flow and organization of the paper
   - Evaluate the clarity and effectiveness of the abstract
   - Review the introduction and its ability to contextualize the research
   - Assess the coherence between sections

4. TECHNICAL ASPECTS:
   - Evaluate the quality of writing and clarity of expression
   - Assess the use of citations and adherence to referencing standards
   - Review figures, tables, and their integration with the text
   - Identify any technical errors or inconsistencies

5. SPECIFIC STRENGTHS:
   - Highlight the most significant contributions of the paper
   - Identify particularly well-executed aspects

6. AREAS FOR IMPROVEMENT:
   - Provide constructive criticism for enhancing the paper
   - Identify gaps, weaknesses, or limitations that should be addressed
   - Suggest specific revisions to strengthen the paper

7. RECOMMENDATION:
   - Provide a clear recommendation regarding publication (Accept, Minor Revision, Major Revision, Reject)
   - Justify your recommendation based on your assessment

Your review should be detailed, constructive, and objective, providing specific examples from the paper to support your evaluation.
"""
)

# Introduction review prompt
INTRODUCTION_REVIEW_PROMPT = PromptTemplate(
    input_variables=["paper_title", "introduction_content", "field_of_study"],
    template="""
You are an expert academic reviewer in the field of {field_of_study}. Review the introduction section of the following paper:

TITLE: {paper_title}

INTRODUCTION CONTENT:
{introduction_content}

Provide a detailed review of the introduction section by addressing the following aspects:

1. CONTEXT ESTABLISHMENT:
   - Evaluate how effectively the introduction establishes the research context
   - Assess the clarity of background information provided
   - Determine if the significance of the research area is adequately conveyed

2. PROBLEM STATEMENT:
   - Assess the clarity and specificity of the research problem
   - Evaluate how well the gap in existing knowledge is identified
   - Determine if the importance of addressing this problem is justified

3. RESEARCH OBJECTIVES:
   - Evaluate the clarity and specificity of research objectives/questions/hypotheses
   - Assess whether the objectives logically follow from the problem statement
   - Determine if the objectives are achievable and measurable

4. THEORETICAL FRAMEWORK:
   - Assess the presentation of the theoretical or conceptual framework
   - Evaluate how well the framework connects to the research problem
   - Determine if key concepts are clearly defined

5. SCOPE AND LIMITATIONS:
   - Evaluate the clarity of the research scope
   - Assess whether limitations are appropriately acknowledged
   - Determine if the boundaries of the study are well-defined

6. PAPER STRUCTURE:
   - Assess the overview of the paper's organization
   - Evaluate whether the introduction provides a clear roadmap for the reader

7. WRITING QUALITY:
   - Evaluate the clarity, conciseness, and flow of the writing
   - Assess the use of appropriate academic language
   - Determine if the introduction engages the reader effectively

8. SPECIFIC RECOMMENDATIONS:
   - Provide specific suggestions for improving the introduction
   - Identify any missing elements that should be included
   - Suggest revisions to enhance clarity, coherence, or impact

Your review should be detailed, constructive, and objective, providing specific examples from the introduction to support your evaluation.
"""
)

# Literature review evaluation prompt
LITERATURE_REVIEW_PROMPT = PromptTemplate(
    input_variables=["paper_title", "literature_review_content", "field_of_study"],
    template="""
You are an expert academic reviewer in the field of {field_of_study}. Review the literature review section of the following paper:

TITLE: {paper_title}

LITERATURE REVIEW CONTENT:
{literature_review_content}

Provide a detailed review of the literature review section by addressing the following aspects:

1. COMPREHENSIVENESS:
   - Evaluate the breadth and depth of the literature covered
   - Assess whether key seminal works in the field are included
   - Determine if the review includes recent and relevant publications
   - Identify any significant gaps in the literature coverage

2. ORGANIZATION AND STRUCTURE:
   - Assess the logical organization of the literature review
   - Evaluate whether the review is organized thematically, chronologically, or methodologically
   - Determine if the organization effectively supports the research objectives

3. CRITICAL ANALYSIS:
   - Evaluate the level of critical engagement with the literature
   - Assess whether the review merely summarizes or genuinely analyzes the literature
   - Determine if strengths and limitations of previous studies are identified
   - Evaluate how well conflicting findings or theories are addressed

4. SYNTHESIS:
   - Assess how effectively the literature is synthesized
   - Evaluate whether connections between different studies are established
   - Determine if patterns, trends, or themes in the literature are identified
   - Assess whether the review goes beyond description to create new insights

5. RELEVANCE TO RESEARCH QUESTION:
   - Evaluate how well the literature review connects to the research question/objectives
   - Assess whether the review clearly establishes the research gap
   - Determine if the review provides a foundation for the current study

6. CITATION PRACTICES:
   - Assess the appropriateness and accuracy of citations
   - Evaluate whether primary sources are used appropriately
   - Determine if citation practices meet disciplinary standards

7. SPECIFIC RECOMMENDATIONS:
   - Provide specific suggestions for improving the literature review
   - Identify additional sources that should be considered
   - Suggest revisions to enhance critical analysis or synthesis

Your review should be detailed, constructive, and objective, providing specific examples from the literature review to support your evaluation.
"""
)

# Methodology review prompt
METHODOLOGY_REVIEW_PROMPT = PromptTemplate(
    input_variables=["paper_title", "methodology_content", "field_of_study", "research_type"],
    template="""
You are an expert academic reviewer in the field of {field_of_study}. Review the methodology section of the following {research_type} paper:

TITLE: {paper_title}

METHODOLOGY CONTENT:
{methodology_content}

Provide a detailed review of the methodology section by addressing the following aspects:

1. RESEARCH DESIGN:
   - Evaluate the appropriateness of the research design for addressing the research questions
   - Assess the clarity of the research design description
   - Determine if the design choices are justified with reference to methodological literature

2. DATA COLLECTION:
   - Evaluate the data collection methods and their appropriateness
   - Assess the comprehensiveness of the data collection procedures
   - Determine if potential biases in data collection are addressed
   - Evaluate sample selection and size (if applicable)

3. DATA ANALYSIS:
   - Assess the appropriateness of analytical methods
   - Evaluate the clarity of the analytical procedures
   - Determine if the analysis approach aligns with the research questions
   - Assess whether limitations of the analytical methods are acknowledged

4. VALIDITY AND RELIABILITY:
   - Evaluate measures taken to ensure validity and reliability
   - Assess how potential threats to validity are addressed
   - Determine if triangulation or other validation strategies are employed

5. ETHICAL CONSIDERATIONS:
   - Assess the treatment of ethical issues
   - Evaluate whether appropriate ethical approvals were obtained
   - Determine if participant rights and welfare were adequately protected

6. REPLICABILITY:
   - Evaluate whether the methodology is described in sufficient detail for replication
   - Assess the clarity of procedural steps
   - Determine if materials, instruments, or protocols are adequately described

7. LIMITATIONS:
   - Assess whether methodological limitations are acknowledged
   - Evaluate the discussion of how limitations were mitigated
   - Determine if the impact of limitations on findings is addressed

8. SPECIFIC RECOMMENDATIONS:
   - Provide specific suggestions for improving the methodology section
   - Identify areas requiring more detail or clarification
   - Suggest alternative approaches where appropriate

Your review should be detailed, constructive, and objective, providing specific examples from the methodology section to support your evaluation.
"""
)

# Results review prompt
RESULTS_REVIEW_PROMPT = PromptTemplate(
    input_variables=["paper_title", "results_content", "field_of_study", "research_type"],
    template="""
You are an expert academic reviewer in the field of {field_of_study}. Review the results section of the following {research_type} paper:

TITLE: {paper_title}

RESULTS CONTENT:
{results_content}

Provide a detailed review of the results section by addressing the following aspects:

1. PRESENTATION OF FINDINGS:
   - Evaluate the clarity and organization of the results presentation
   - Assess whether results directly address the research questions/hypotheses
   - Determine if the results are presented in a logical sequence

2. DATA VISUALIZATION:
   - Evaluate the quality and appropriateness of tables, figures, and graphs
   - Assess whether visualizations effectively communicate the findings
   - Determine if visualizations are properly labeled and referenced in the text

3. STATISTICAL ANALYSIS (if applicable):
   - Assess the appropriateness of statistical tests used
   - Evaluate the reporting of statistical results (p-values, effect sizes, confidence intervals)
   - Determine if statistical assumptions are verified and addressed

4. QUALITATIVE FINDINGS (if applicable):
   - Evaluate the presentation of qualitative data
   - Assess the use of quotes, themes, or categories
   - Determine if the analysis process is transparent

5. OBJECTIVITY:
   - Assess whether results are presented objectively without interpretation
   - Evaluate if all relevant results are reported, including negative or unexpected findings
   - Determine if there is any evidence of selective reporting

6. TECHNICAL ACCURACY:
   - Evaluate the accuracy of calculations and data reporting
   - Assess the consistency between text, tables, and figures
   - Determine if units of measurement are clearly specified

7. CLARITY FOR READERS:
   - Evaluate whether results are accessible to the intended audience
   - Assess if technical terms are adequately explained
   - Determine if the significance of key findings is highlighted

8. SPECIFIC RECOMMENDATIONS:
   - Provide specific suggestions for improving the results section
   - Identify areas requiring clarification or additional analysis
   - Suggest alternative ways to present complex findings

Your review should be detailed, constructive, and objective, providing specific examples from the results section to support your evaluation.
"""
)

# Discussion and conclusion review prompt
DISCUSSION_CONCLUSION_REVIEW_PROMPT = PromptTemplate(
    input_variables=["paper_title", "discussion_conclusion_content", "field_of_study"],
    template="""
You are an expert academic reviewer in the field of {field_of_study}. Review the discussion and conclusion sections of the following paper:

TITLE: {paper_title}

DISCUSSION AND CONCLUSION CONTENT:
{discussion_conclusion_content}

Provide a detailed review of the discussion and conclusion sections by addressing the following aspects:

1. INTERPRETATION OF FINDINGS:
   - Evaluate how effectively the findings are interpreted in relation to the research questions
   - Assess whether interpretations are supported by the results
   - Determine if alternative explanations for findings are considered

2. CONTEXTUALIZATION:
   - Evaluate how well the findings are situated within the existing literature
   - Assess whether the discussion addresses how findings confirm, extend, or contradict previous research
   - Determine if theoretical implications are adequately explored

3. LIMITATIONS DISCUSSION:
   - Assess the thoroughness of the limitations discussion
   - Evaluate whether the impact of limitations on findings is addressed
   - Determine if strategies to address limitations in future research are proposed

4. IMPLICATIONS:
   - Evaluate the discussion of theoretical and practical implications
   - Assess whether implications are reasonable given the study's scope and findings
   - Determine if the significance of the findings is clearly articulated

5. FUTURE RESEARCH:
   - Assess the quality of suggestions for future research
   - Evaluate whether future research directions logically extend from the current study
   - Determine if specific research questions or approaches are proposed

6. CONCLUSION:
   - Evaluate whether the conclusion effectively summarizes the key findings
   - Assess if the conclusion addresses the original research questions/objectives
   - Determine if the conclusion avoids introducing new information
   - Evaluate whether the conclusion provides appropriate closure to the paper

7. OVERINTERPRETATION:
   - Assess whether there is any evidence of overreaching claims or unwarranted generalizations
   - Evaluate if conclusions are proportionate to the strength of the evidence
   - Determine if speculative statements are clearly identified as such

8. SPECIFIC RECOMMENDATIONS:
   - Provide specific suggestions for improving the discussion and conclusion
   - Identify areas requiring more depth or nuance
   - Suggest additional implications or connections that could be explored

Your review should be detailed, constructive, and objective, providing specific examples from the discussion and conclusion to support your evaluation.
"""
)

# Abstract review prompt
ABSTRACT_REVIEW_PROMPT = PromptTemplate(
    input_variables=["paper_title", "abstract_content", "field_of_study"],
    template="""
You are an expert academic reviewer in the field of {field_of_study}. Review the abstract of the following paper:

TITLE: {paper_title}

ABSTRACT:
{abstract_content}

Provide a detailed review of the abstract by addressing the following aspects:

1. COMPLETENESS:
   - Evaluate whether the abstract includes all essential elements (background, objectives, methods, results, conclusion)
   - Assess if key findings and implications are clearly stated
   - Determine if the abstract provides a complete yet concise summary of the paper

2. CLARITY AND CONCISENESS:
   - Evaluate the clarity and precision of language
   - Assess whether the abstract avoids unnecessary details and jargon
   - Determine if the word count is appropriate for the publication venue

3. ACCURACY:
   - Evaluate whether the abstract accurately represents the paper's content
   - Assess if there are any discrepancies between the abstract and the main text
   - Determine if the abstract avoids claims not supported in the paper

4. STRUCTURE AND FLOW:
   - Evaluate the logical flow and organization of the abstract
   - Assess whether the abstract follows a coherent structure
   - Determine if transitions between elements are smooth

5. IMPACT AND SIGNIFICANCE:
   - Evaluate how effectively the abstract communicates the significance of the research
   - Assess whether the abstract would engage the target audience
   - Determine if the abstract clearly states the paper's contribution to the field

6. KEYWORDS (if included):
   - Evaluate the appropriateness and relevance of keywords
   - Assess whether keywords effectively represent the paper's content
   - Determine if keywords would facilitate appropriate indexing and searchability

7. SPECIFIC RECOMMENDATIONS:
   - Provide specific suggestions for improving the abstract
   - Identify elements that should be added, removed, or modified
   - Suggest revisions to enhance clarity, completeness, or impact

Your review should be detailed, constructive, and objective, providing specific examples from the abstract to support your evaluation.
"""
)

# Statistical analysis review prompt
STATISTICAL_ANALYSIS_REVIEW_PROMPT = PromptTemplate(
    input_variables=["paper_title", "statistical_content", "field_of_study"],
    template="""
You are an expert statistical reviewer in the field of {field_of_study}. Review the statistical analysis in the following paper:

TITLE: {paper_title}

STATISTICAL CONTENT:
{statistical_content}

Provide a detailed review of the statistical analysis by addressing the following aspects:

1. APPROPRIATENESS OF STATISTICAL METHODS:
   - Evaluate whether the chosen statistical tests are appropriate for the research questions
   - Assess if the statistical methods align with the data type and distribution
   - Determine if more suitable alternative methods should have been considered

2. STATISTICAL ASSUMPTIONS:
   - Evaluate whether assumptions for statistical tests are verified and met
   - Assess how violations of assumptions are addressed
   - Determine if appropriate corrections or alternative approaches are used when assumptions are violated

3. SAMPLE SIZE AND POWER:
   - Evaluate the adequacy of the sample size for the analyses conducted
   - Assess whether power calculations are reported or if post-hoc power is discussed
   - Determine if sample size limitations are acknowledged and addressed

4. REPORTING OF STATISTICS:
   - Evaluate the completeness of statistical reporting (test statistics, degrees of freedom, p-values, effect sizes)
   - Assess the consistency of statistical reporting throughout the paper
   - Determine if confidence intervals or measures of variability are appropriately reported

5. INTERPRETATION OF RESULTS:
   - Evaluate whether statistical significance is appropriately interpreted
   - Assess if practical significance (effect sizes) is discussed
   - Determine if causal claims are proportionate to the study design and analysis

6. MULTIPLE COMPARISONS:
   - Evaluate how multiple testing issues are addressed
   - Assess whether appropriate corrections are applied
   - Determine if family-wise error rates or false discovery rates are controlled

7. MISSING DATA:
   - Evaluate how missing data are reported and handled
   - Assess the appropriateness of methods used to address missing data
   - Determine if potential biases from missing data are discussed

8. SPECIFIC RECOMMENDATIONS:
   - Provide specific suggestions for improving the statistical analysis
   - Identify additional analyses that should be conducted
   - Suggest alternative statistical approaches where appropriate

Your review should be detailed, constructive, and objective, providing specific examples from the paper to support your evaluation. Include technical details where necessary, but ensure explanations are clear enough for researchers who may not be statistical experts.
"""
)

# Peer review summary prompt
PEER_REVIEW_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["paper_title", "review_content", "editor_guidelines"],
    template="""
You are an academic editor summarizing a peer review of the following paper:

TITLE: {paper_title}

FULL REVIEW:
{review_content}

EDITOR GUIDELINES:
{editor_guidelines}

Create a concise yet comprehensive summary of the peer review that will be shared with the authors. Your summary should:

1. HIGHLIGHT KEY STRENGTHS:
   - Identify the major strengths of the paper noted by the reviewer
   - Emphasize positive aspects that should be maintained in any revision

2. PRIORITIZE CRITICAL ISSUES:
   - Identify the most significant concerns or weaknesses that must be addressed
   - Organize issues in order of importance rather than the order presented in the review

3. CLARIFY AMBIGUOUS FEEDBACK:
   - Clarify any vague or potentially confusing feedback from the reviewer
   - Ensure criticism is constructive and actionable

4. SYNTHESIZE RECOMMENDATIONS:
   - Consolidate specific recommendations for improvement
   - Present suggestions in a clear, actionable format

5. CONTEXTUALIZE TECHNICAL FEEDBACK:
   - Provide context for highly technical criticisms to ensure authors understand their importance
   - Explain the rationale behind methodological or analytical recommendations

6. BALANCE PERSPECTIVE:
   - Ensure the summary maintains a balanced perspective
   - Moderate overly harsh criticism while preserving important feedback
   - Highlight areas of agreement and disagreement with other reviewers (if applicable)

7. ALIGN WITH EDITORIAL STANDARDS:
   - Ensure the summary aligns with the journal's or publication's standards and expectations
   - Address any reviewer comments that may be outside the scope of the current submission

Your summary should be professional, constructive, and focused on helping the authors improve their manuscript. Maintain the substantive content of the review while presenting it in a way that facilitates effective revision.
"""
)

# Interdisciplinary research review prompt
INTERDISCIPLINARY_REVIEW_PROMPT = PromptTemplate(
    input_variables=["paper_title", "paper_content", "disciplines_involved"],
    template="""
You are an expert reviewer with knowledge spanning multiple disciplines. Review the following interdisciplinary paper that bridges these fields: {disciplines_involved}.

TITLE: {paper_title}

PAPER CONTENT:
{paper_content}

Provide a detailed review of this interdisciplinary research by addressing the following aspects:

1. INTEGRATION OF DISCIPLINES:
   - Evaluate how effectively the paper integrates concepts, methods, or theories from different disciplines
   - Assess whether the integration creates novel insights beyond what single disciplines could provide
   - Determine if the paper achieves true interdisciplinarity rather than merely juxtaposing disciplines

2. DISCIPLINARY RIGOR:
   - Evaluate whether the paper maintains appropriate rigor in its treatment of each discipline
   - Assess if concepts and methods from each discipline are applied correctly
   - Determine if the paper meets the standards of each discipline it engages with

3. ACCESSIBILITY:
   - Evaluate whether the paper is accessible to readers from different disciplinary backgrounds
   - Assess if discipline-specific terminology is adequately explained
   - Determine if the paper bridges communication gaps between disciplines effectively

4. CONTRIBUTION TO INVOLVED DISCIPLINES:
   - Evaluate the paper's contribution to each of the disciplines involved
   - Assess whether the interdisciplinary approach enhances understanding in each field
   - Determine if the paper identifies new research directions for individual disciplines

5. METHODOLOGICAL APPROACH:
   - Evaluate the appropriateness of the methodological approach for interdisciplinary research
   - Assess how well methodological challenges of combining disciplines are addressed
   - Determine if innovative methodological approaches are developed to bridge disciplines

6. THEORETICAL FRAMEWORK:
   - Evaluate the coherence of the theoretical framework spanning multiple disciplines
   - Assess how well theoretical tensions or contradictions between disciplines are resolved
   - Determine if a novel theoretical synthesis is achieved

7. LITERATURE ENGAGEMENT:
   - Evaluate engagement with relevant literature from all involved disciplines
   - Assess whether the literature review adequately represents different disciplinary perspectives
   - Determine if connections between disciplinary literatures are established

8. SPECIFIC RECOMMENDATIONS:
   - Provide specific suggestions for improving the interdisciplinary aspects of the paper
   - Identify areas where integration could be strengthened
   - Suggest ways to enhance the paper's contribution to each discipline

Your review should be detailed, constructive, and objective, providing specific examples from the paper to support your evaluation. Be mindful of the challenges inherent in interdisciplinary work while maintaining appropriate standards for scholarly rigor.
"""
)

# Ethical considerations review prompt
ETHICAL_CONSIDERATIONS_REVIEW_PROMPT = PromptTemplate(
    input_variables=["paper_title", "paper_content", "field_of_study", "research_type"],
    template="""
You are an expert reviewer specializing in research ethics in the field of {field_of_study}. Review the ethical considerations in the following {research_type} paper:

TITLE: {paper_title}

PAPER CONTENT:
{paper_content}

Provide a detailed review of the ethical aspects of this research by addressing the following:

1. ETHICAL APPROVAL AND CONSENT:
   - Evaluate whether appropriate ethical approval was obtained
   - Assess the adequacy of informed consent procedures (if applicable)
   - Determine if vulnerable populations are appropriately protected (if applicable)
   - Evaluate whether consent documentation meets current standards

2. RISK ASSESSMENT AND MANAGEMENT:
   - Evaluate the assessment of potential risks to participants
   - Assess whether appropriate measures were taken to minimize risks
   - Determine if the balance of risks and benefits is justified
   - Evaluate whether adverse events were monitored and addressed (if applicable)

3. PRIVACY AND CONFIDENTIALITY:
   - Evaluate measures to protect participant privacy and data confidentiality
   - Assess the handling of sensitive information
   - Determine if anonymization or de-identification procedures are adequate
   - Evaluate data security measures

4. RESEARCH INTEGRITY:
   - Assess transparency in reporting methods and results
   - Evaluate disclosure of conflicts of interest
   - Determine if there are any concerns regarding data fabrication or manipulation
   - Assess adherence to discipline-specific ethical guidelines

5. SOCIAL AND CULTURAL CONSIDERATIONS:
   - Evaluate sensitivity to cultural, social, or community contexts
   - Assess whether diverse perspectives are respected
   - Determine if research benefits are shared equitably with participants or communities
   - Evaluate potential societal implications of the research

6. ETHICAL REPORTING:
   - Assess whether limitations and uncertainties are honestly reported
   - Evaluate if conclusions are proportionate to the evidence
   - Determine if potential misuses of the research are acknowledged
   - Assess whether ethical challenges encountered during the research are discussed

7. SPECIFIC ETHICAL CONCERNS FOR THIS RESEARCH TYPE:
   - Identify and evaluate ethical issues specific to this type of research
   - Assess how well these specific concerns are addressed

8. RECOMMENDATIONS:
   - Provide specific suggestions for addressing ethical concerns
   - Identify additional ethical considerations that should be addressed
   - Suggest improvements to ethical procedures or reporting

Your review should be detailed, constructive, and objective, providing specific examples from the paper to support your evaluation. Focus on helping the authors strengthen the ethical dimensions of their research rather than simply identifying shortcomings.
"""
)
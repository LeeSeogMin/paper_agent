"""
Paper-related prompt templates
Collection of prompt templates for paper writing and editing.
"""

from langchain_core.prompts import PromptTemplate


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

Paper content:
{paper_content}

Editing and improvement guidelines:
1. Correct grammar, spelling, and punctuation errors
2. Improve sentence structure and flow
3. Enhance consistency and clarity
4. Remove redundancies and unnecessary content
5. Use appropriate academic expressions and terminology
6. Check citation and reference format

Improve the expression while maintaining the original meaning.

Edited content:
""",
    input_variables=["editing_type", "style_guide", "paper_content"],
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
"""
Paper template definition module
Defines templates for various paper formats.
"""

# Basic academic paper structure template
ACADEMIC_PAPER_TEMPLATE = {
    "sections": [
        {
            "name": "Title",
            "required": True,
            "description": "The title of the paper. Should clearly and concisely express the research content."
        },
        {
            "name": "Abstract",
            "required": True,
            "description": "A 200-300 word summary describing the purpose, methods, results, and conclusions of the research."
        },
        {
            "name": "Keywords",
            "required": True,
            "description": "4-6 keywords representing the main topics of the paper."
        },
        {
            "name": "Introduction",
            "required": True,
            "description": "A section explaining the research background, purpose, significance, and research questions."
        },
        {
            "name": "Literature Review",
            "required": True,
            "description": "A section analyzing related research and theoretical background."
        },
        {
            "name": "Methodology",
            "required": True,
            "description": "A section detailing research methods, data collection, and analysis methods."
        },
        {
            "name": "Results",
            "required": True,
            "description": "A section objectively presenting research results."
        },
        {
            "name": "Discussion",
            "required": True,
            "description": "A section discussing the interpretation, meaning, and implications of the research results."
        },
        {
            "name": "Conclusion",
            "required": True,
            "description": "A section summarizing the main findings, limitations, and future research directions."
        },
        {
            "name": "References",
            "required": True,
            "description": "A list of all cited references."
        },
        {
            "name": "Appendices",
            "required": False,
            "description": "Supplementary materials too detailed to include in the main text."
        }
    ],
    "format": {
        "font": "Times New Roman",
        "font_size": 12,
        "line_spacing": 1.5,
        "margins": "1 inch on all sides",
        "citation_style": "APA 7th Edition"
    }
}

# Computer science paper template (IEEE format)
CS_PAPER_TEMPLATE = {
    "sections": [
        {
            "name": "Title",
            "required": True,
            "description": "The title of the paper."
        },
        {
            "name": "Abstract",
            "required": True,
            "description": "A 150-250 word summary describing the purpose, methods, results, and conclusions of the research."
        },
        {
            "name": "Index Terms",
            "required": True,
            "description": "Keywords representing the main topics of the paper."
        },
        {
            "name": "Introduction",
            "required": True,
            "description": "A section explaining the research background, purpose, and significance."
        },
        {
            "name": "Related Work",
            "required": True,
            "description": "A section explaining related research and background technology."
        },
        {
            "name": "Proposed Method/System",
            "required": True,
            "description": "A description of the design and implementation of the proposed method or system."
        },
        {
            "name": "Experimental Setup",
            "required": True,
            "description": "A description of the experimental environment, datasets, evaluation metrics, etc."
        },
        {
            "name": "Results and Analysis",
            "required": True,
            "description": "Experimental results and analysis."
        },
        {
            "name": "Discussion",
            "required": True,
            "description": "Discussion of results, strengths and weaknesses, significance, etc."
        },
        {
            "name": "Conclusion and Future Work",
            "required": True,
            "description": "Conclusion and future research directions."
        },
        {
            "name": "References",
            "required": True,
            "description": "A list of all cited references."
        },
        {
            "name": "Appendices",
            "required": False,
            "description": "Supplementary materials."
        }
    ],
    "format": {
        "font": "Times New Roman",
        "font_size": 10,
        "line_spacing": 1.0,
        "margins": "1 inch on all sides",
        "citation_style": "IEEE"
    }
}

# IT paper template (ACM format)
IT_PAPER_TEMPLATE = {
    "sections": [
        {
            "name": "Title",
            "required": True,
            "description": "Clear and specific title describing the IT solution or research."
        },
        {
            "name": "Abstract",
            "required": True,
            "description": "150-200 word summary highlighting the problem, approach, and key contributions."
        },
        {
            "name": "Keywords",
            "required": True,
            "description": "5-7 technical keywords for indexing and search."
        },
        {
            "name": "Introduction",
            "required": True,
            "description": "Problem statement, motivation, and overview of the proposed solution."
        },
        {
            "name": "Background & Related Work",
            "required": True,
            "description": "Overview of existing technologies, solutions, and research gaps."
        },
        {
            "name": "System Architecture",
            "required": True,
            "description": "Detailed description of the system design, components, and interactions."
        },
        {
            "name": "Implementation Details",
            "required": True,
            "description": "Technical implementation specifics, technologies used, and development process."
        },
        {
            "name": "Evaluation",
            "required": True,
            "description": "Performance metrics, testing methodology, and benchmark results."
        },
        {
            "name": "Discussion",
            "required": True,
            "description": "Analysis of results, limitations, and practical implications."
        },
        {
            "name": "Future Work",
            "required": True,
            "description": "Potential improvements, extensions, and future research directions."
        },
        {
            "name": "Conclusion",
            "required": True,
            "description": "Summary of contributions and key takeaways."
        },
        {
            "name": "References",
            "required": True,
            "description": "List of cited sources following ACM format."
        },
        {
            "name": "Appendices",
            "required": False,
            "description": "Supplementary materials such as code snippets, additional diagrams, or detailed algorithms."
        }
    ],
    "format": {
        "font": "Arial",
        "font_size": 10,
        "line_spacing": 1.0,
        "margins": "0.75 inch on all sides",
        "citation_style": "ACM"
    }
}

# Social Science paper template (APA format)
SOCIAL_SCIENCE_PAPER_TEMPLATE = {
    "sections": [
        {
            "name": "Title",
            "required": True,
            "description": "Concise title reflecting the study's focus and variables."
        },
        {
            "name": "Abstract",
            "required": True,
            "description": "150-250 word structured summary including objective, method, results, and conclusions."
        },
        {
            "name": "Keywords",
            "required": True,
            "description": "4-6 keywords representing the main concepts and variables."
        },
        {
            "name": "Introduction",
            "required": True,
            "description": "Research problem, theoretical framework, literature review, and hypotheses."
        },
        {
            "name": "Theoretical Framework",
            "required": True,
            "description": "Detailed explanation of the theoretical perspective guiding the research."
        },
        {
            "name": "Literature Review",
            "required": True,
            "description": "Critical analysis of relevant previous research and identification of gaps."
        },
        {
            "name": "Methodology",
            "required": True,
            "description": "Research design, participants, sampling methods, data collection procedures, and measures."
        },
        {
            "name": "Results",
            "required": True,
            "description": "Statistical analyses, qualitative findings, and hypothesis testing outcomes."
        },
        {
            "name": "Discussion",
            "required": True,
            "description": "Interpretation of findings, theoretical implications, and relation to previous research."
        },
        {
            "name": "Limitations",
            "required": True,
            "description": "Methodological limitations and potential biases."
        },
        {
            "name": "Conclusion",
            "required": True,
            "description": "Summary of key findings, broader implications, and future research directions."
        },
        {
            "name": "References",
            "required": True,
            "description": "List of all cited sources in APA format."
        },
        {
            "name": "Appendices",
            "required": False,
            "description": "Supplementary materials such as survey instruments, interview protocols, or additional analyses."
        }
    ],
    "format": {
        "font": "Times New Roman",
        "font_size": 12,
        "line_spacing": 2.0,
        "margins": "1 inch on all sides",
        "citation_style": "APA 7th Edition"
    }
}

# Computational Social Science paper template (hybrid format)
COMPUTATIONAL_SOCIAL_SCIENCE_TEMPLATE = {
    "sections": [
        {
            "name": "Title",
            "required": True,
            "description": "Descriptive title integrating computational methods and social science concepts."
        },
        {
            "name": "Abstract",
            "required": True,
            "description": "200-250 word summary covering social science question, computational approach, data, methods, and key findings."
        },
        {
            "name": "Keywords",
            "required": True,
            "description": "6-8 keywords covering both computational methods and social science concepts."
        },
        {
            "name": "Introduction",
            "required": True,
            "description": "Research question, social relevance, and computational approach overview."
        },
        {
            "name": "Related Work",
            "required": True,
            "description": "Review of both social science literature and computational methods in the domain."
        },
        {
            "name": "Theoretical Framework",
            "required": True,
            "description": "Social science theories informing the research and computational operationalization."
        },
        {
            "name": "Data",
            "required": True,
            "description": "Data sources, collection methods, ethical considerations, and preprocessing steps."
        },
        {
            "name": "Computational Methods",
            "required": True,
            "description": "Detailed explanation of algorithms, models, and analytical techniques employed."
        },
        {
            "name": "Experimental Design",
            "required": True,
            "description": "Research design, variables, validation approaches, and evaluation metrics."
        },
        {
            "name": "Results",
            "required": True,
            "description": "Quantitative findings, visualizations, and statistical analyses."
        },
        {
            "name": "Social Interpretation",
            "required": True,
            "description": "Interpretation of computational results in social science context."
        },
        {
            "name": "Discussion",
            "required": True,
            "description": "Implications for both computational methods and social science theory."
        },
        {
            "name": "Ethical Considerations",
            "required": True,
            "description": "Discussion of ethical implications, privacy concerns, and potential societal impacts."
        },
        {
            "name": "Limitations and Future Work",
            "required": True,
            "description": "Methodological limitations and future research directions."
        },
        {
            "name": "Conclusion",
            "required": True,
            "description": "Summary of contributions to both computational methods and social science understanding."
        },
        {
            "name": "References",
            "required": True,
            "description": "Combined bibliography of social science and computer science sources."
        },
        {
            "name": "Appendices",
            "required": False,
            "description": "Supplementary materials including code repositories, additional analyses, and data documentation."
        }
    ],
    "format": {
        "font": "Arial",
        "font_size": 11,
        "line_spacing": 1.5,
        "margins": "1 inch on all sides",
        "citation_style": "Chicago Author-Date"
    }
}

# Paper format mapping
PAPER_TEMPLATES = {
    "academic": ACADEMIC_PAPER_TEMPLATE,
    "cs": CS_PAPER_TEMPLATE,
    "it": IT_PAPER_TEMPLATE,
    "social_science": SOCIAL_SCIENCE_PAPER_TEMPLATE,
    "computational_social": COMPUTATIONAL_SOCIAL_SCIENCE_TEMPLATE
}

def get_template(template_type="academic"):
    """
    Returns a paper template of the specified type.
    
    Args:
        template_type (str): Template type ('academic', 'cs', 'it', 'social_science', 'computational_social')
        
    Returns:
        dict: Paper template dictionary
    """
    return PAPER_TEMPLATES.get(template_type, ACADEMIC_PAPER_TEMPLATE)
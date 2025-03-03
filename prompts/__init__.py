"""
프롬프트 패키지
논문 작성 AI 에이전트 시스템에서 사용되는 다양한 프롬프트 템플릿을 제공합니다.
"""

# 각 모듈에서 필요한 프롬프트 템플릿 임포트
from prompts.paper_prompts import (
    PaperPrompts,
    RESEARCH_TOPIC_SUGGESTION_PROMPT,
    RESEARCH_PLAN_PROMPT,
    PAPER_OUTLINE_PROMPT,
    PAPER_SECTION_PROMPT,
    PAPER_EDITING_PROMPT,
    PAPER_REVIEW_PROMPT,
    PAPER_SUMMARY_PROMPT,
    PAPER_TRANSLATION_PROMPT,
    REFERENCE_FORMATTING_PROMPT,
    KEYWORD_EXTRACTION_PROMPT,
    PAPER_CITATION_PROMPT,
    RESEARCH_QUESTION_PROMPT,
    CRITICAL_ANALYSIS_PROMPT,
    PAPER_CONCLUSION_PROMPT
)

from prompts.agent_prompts import (
    AGENT_SYSTEM_PROMPT,
    AGENT_COMMUNICATION_PROMPT,
    AGENT_PLANNING_PROMPT,
    AGENT_SELF_EVALUATION_PROMPT,
    AGENT_ERROR_HANDLING_PROMPT,
    AGENT_KNOWLEDGE_RETRIEVAL_PROMPT,
    AGENT_DECISION_MAKING_PROMPT,
    AGENT_FEEDBACK_INTEGRATION_PROMPT,
    AGENT_COLLABORATION_PROMPT,
    AGENT_LEARNING_PROMPT
)

from prompts.research_prompts import (
    ResearchPrompts,
    RESEARCH_SYSTEM_PROMPT,
    QUERY_GENERATION_PROMPT,
    SOURCE_EVALUATION_PROMPT,
    SEARCH_RESULTS_ANALYSIS_PROMPT,
    LITERATURE_REVIEW_PROMPT,
    RESEARCH_METHODOLOGY_PROMPT,
    DATA_ANALYSIS_PROMPT,
    RESEARCH_INTERPRETATION_PROMPT,
    RESEARCH_GAP_IDENTIFICATION_PROMPT,
    RESEARCH_PROPOSAL_PROMPT
)

from prompts.workflow_prompts import (
    WorkflowPrompts,
    WORKFLOW_STATE_EVALUATION_PROMPT,
    WORKFLOW_ERROR_HANDLING_PROMPT,
    WORKFLOW_TRANSITION_PROMPT,
    WORKFLOW_OPTIMIZATION_PROMPT,
    WORKFLOW_MONITORING_PROMPT
)

from prompts.writing_prompts import (
    WritingPrompts,
    PAPER_TITLE_GENERATION_PROMPT,
    ABSTRACT_WRITING_PROMPT,
    INTRODUCTION_WRITING_PROMPT,
    METHODOLOGY_WRITING_PROMPT,
    RESULTS_WRITING_PROMPT,
    DISCUSSION_WRITING_PROMPT,
    CONCLUSION_WRITING_PROMPT,
    ACADEMIC_COMPARISON_PROMPT,
    CITATION_INTEGRATION_PROMPT
)

from prompts.review_prompts import (
    ReviewPrompts,
    COMPREHENSIVE_REVIEW_PROMPT,
    INTRODUCTION_REVIEW_PROMPT,
    ETHICAL_CONSIDERATIONS_REVIEW_PROMPT
)

from prompts.coordinator_prompts import (
    COORDINATION_INITIALIZATION_PROMPT,
    # 다른 coordinator 프롬프트들도 여기에 추가
)

# 편의를 위한 프롬프트 클래스 모음
prompts = {
    "paper": PaperPrompts,
    "research": ResearchPrompts,
    "workflow": WorkflowPrompts,
    "writing": WritingPrompts,
    "review": ReviewPrompts
}

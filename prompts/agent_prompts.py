"""
에이전트 프롬프트 모듈
에이전트 간 상호작용 및 내부 처리를 위한 프롬프트 템플릿을 정의합니다.
"""

from langchain_core.prompts import PromptTemplate

# 에이전트 시스템 프롬프트
AGENT_SYSTEM_PROMPT = PromptTemplate(
    input_variables=[],
    template="""당신은 학술 논문 작성을 돕는 AI 에이전트입니다.
주어진 작업을 정확하고 철저하게 수행하세요.
항상 객관적이고 학술적인 어조를 유지하며, 정확한 정보와 논리적 구조를 제공하세요.
"""
)

# 에이전트 간 통신 프롬프트
AGENT_COMMUNICATION_PROMPT = PromptTemplate(
    input_variables=["target_agent", "task", "context", "requirements"],
    template="""다음 정보를 바탕으로 {target_agent} 에이전트에게 전달할 메시지를 작성하세요:

작업: {task}
컨텍스트: {context}
요구사항: {requirements}

메시지는 명확하고 구체적이어야 하며, 필요한 모든 정보를 포함해야 합니다.
"""
)

# 에이전트 작업 계획 프롬프트
AGENT_PLANNING_PROMPT = PromptTemplate(
    input_variables=["task", "constraints", "resources"],
    template="""다음 작업을 완료하기 위한 단계별 계획을 수립하세요:

작업: {task}
제약 조건: {constraints}
가용 자원: {resources}

각 단계는 구체적이고 실행 가능해야 하며, 예상 결과를 포함해야 합니다.
"""
)

# 에이전트 자기 평가 프롬프트
AGENT_SELF_EVALUATION_PROMPT = PromptTemplate(
    input_variables=["task", "result", "criteria"],
    template="""다음 작업 결과를 평가하세요:

작업: {task}
결과: {result}
기준: {criteria}

강점과 약점을 식별하고, 개선을 위한 구체적인 제안을 제공하세요.
"""
)

# 에이전트 오류 처리 프롬프트
AGENT_ERROR_HANDLING_PROMPT = PromptTemplate(
    input_variables=["task", "error", "context"],
    template="""다음 오류를 분석하고 해결 방법을 제안하세요:

작업: {task}
오류: {error}
컨텍스트: {context}

오류의 가능한 원인을 식별하고, 해결을 위한 단계별 접근 방식을 제공하세요.
"""
)

# 에이전트 지식 검색 프롬프트
AGENT_KNOWLEDGE_RETRIEVAL_PROMPT = PromptTemplate(
    input_variables=["topic", "scope", "details"],
    template="""다음 주제에 대한 관련 정보를 검색하세요:

주제: {topic}
검색 범위: {scope}
필요한 세부 정보: {details}

가장 관련성이 높고 신뢰할 수 있는 정보를 우선적으로 제공하세요.
"""
)

# 에이전트 의사 결정 프롬프트
AGENT_DECISION_MAKING_PROMPT = PromptTemplate(
    input_variables=["situation", "options", "criteria"],
    template="""다음 옵션 중에서 최적의 선택을 결정하세요:

상황: {situation}
옵션:
{options}
결정 기준: {criteria}

각 옵션의 장단점을 분석하고, 선택한 옵션에 대한 근거를 제공하세요.
"""
)

# 에이전트 피드백 통합 프롬프트
AGENT_FEEDBACK_INTEGRATION_PROMPT = PromptTemplate(
    input_variables=["original_work", "feedback", "requirements"],
    template="""다음 피드백을 작업에 통합하세요:

원본 작업: {original_work}
피드백:
{feedback}
통합 요구사항: {requirements}

피드백을 어떻게 통합할지 설명하고, 수정된 작업을 제공하세요.
"""
)

# 에이전트 협업 프롬프트
AGENT_COLLABORATION_PROMPT = PromptTemplate(
    input_variables=["task", "agents", "roles", "requirements"],
    template="""다음 에이전트들과 협업하여 작업을 완료하세요:

작업: {task}
참여 에이전트:
{agents}
역할 분담: {roles}
협업 요구사항: {requirements}

효과적인 협업 전략을 수립하고, 각 에이전트의 기여를 조정하세요.
"""
)

# 에이전트 학습 프롬프트
AGENT_LEARNING_PROMPT = PromptTemplate(
    input_variables=["experience", "outcome", "learning_objectives"],
    template="""다음 경험에서 학습하여 향후 성능을 개선하세요:

경험: {experience}
결과: {outcome}
학습 목표: {learning_objectives}

경험에서 얻은 교훈을 식별하고, 향후 유사한 상황에서 적용할 수 있는 개선 사항을 제안하세요.
"""
) 

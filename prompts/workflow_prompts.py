"""
워크플로우 프롬프트 모듈
논문 작성 워크플로우 관리를 위한 프롬프트 템플릿을 정의합니다.
"""

from langchain_core.prompts import PromptTemplate
from typing import Dict, List, Optional, Any

# 프롬프트 그룹화를 위한 클래스
class WorkflowPrompts:
    """Workflow-related prompts collection"""
    
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
            "state_evaluation": WORKFLOW_STATE_EVALUATION_PROMPT,
            "error_handling": WORKFLOW_ERROR_HANDLING_PROMPT,
            "transition": WORKFLOW_TRANSITION_PROMPT,
            "optimization": WORKFLOW_OPTIMIZATION_PROMPT,
            "monitoring": WORKFLOW_MONITORING_PROMPT
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


# 워크플로우 상태 평가 프롬프트
WORKFLOW_STATE_EVALUATION_PROMPT = PromptTemplate(
    input_variables=["current_state", "completed_steps", "remaining_steps", "goal"],
    template="""현재 워크플로우 상태를 평가하고 다음 단계를 결정하세요:

현재 상태: {current_state}
완료된 단계: {completed_steps}
남은 단계: {remaining_steps}
목표: {goal}

평가에는 다음이 포함되어야 합니다:
1. 현재 진행 상황 분석
2. 발생한 문제 또는 장애물 식별
3. 다음 단계에 대한 권장 사항
4. 필요한 조정 또는 변경 사항

객관적이고 실행 가능한 평가를 제공하세요.
"""
)

# 워크플로우 오류 처리 프롬프트
WORKFLOW_ERROR_HANDLING_PROMPT = PromptTemplate(
    input_variables=["error_description", "error_context", "workflow_state"],
    template="""워크플로우에서 발생한 다음 오류를 처리하세요:

오류 설명: {error_description}
오류 컨텍스트: {error_context}
워크플로우 상태: {workflow_state}

오류 처리에는 다음이 포함되어야 합니다:
1. 오류의 근본 원인 분석
2. 즉각적인 해결 방법 제안
3. 워크플로우 복구 단계
4. 향후 유사한 오류 방지를 위한 권장 사항

효과적이고 실행 가능한 오류 처리 전략을 제공하세요.
"""
)

# 워크플로우 전환 프롬프트
WORKFLOW_TRANSITION_PROMPT = PromptTemplate(
    input_variables=["current_step", "next_step", "transition_requirements", "transition_constraints"],
    template="""다음 워크플로우 단계로의 전환을 관리하세요:

현재 단계: {current_step}
다음 단계: {next_step}
전환 요구사항: {transition_requirements}
전환 제약 조건: {transition_constraints}

전환 관리에는 다음이 포함되어야 합니다:
1. 현재 단계의 출력 검증
2. 다음 단계에 필요한 입력 준비
3. 데이터 변환 또는 형식 조정
4. 전환 중 발생할 수 있는 문제 예상 및 해결

원활하고 효과적인 워크플로우 전환을 보장하세요.
"""
)

# 워크플로우 최적화 프롬프트
WORKFLOW_OPTIMIZATION_PROMPT = PromptTemplate(
    input_variables=["workflow_description", "current_metrics", "optimization_goals", "constraints"],
    template="""다음 워크플로우 프로세스를 최적화하세요:

워크플로우 설명: {workflow_description}
현재 성능 지표: {current_metrics}
최적화 목표: {optimization_goals}
제약 조건: {constraints}

최적화에는 다음이 포함되어야 합니다:
1. 비효율성 및 병목 현상 식별
2. 구체적인 개선 제안
3. 제안된 변경의 예상 영향
4. 구현 우선순위 및 로드맵

효율성과 효과를 향상시키는 실행 가능한 최적화 전략을 제공하세요.
"""
)

# 워크플로우 모니터링 프롬프트
WORKFLOW_MONITORING_PROMPT = PromptTemplate(
    input_variables=["workflow_id", "monitoring_period", "key_metrics", "alert_thresholds"],
    template="""다음 워크플로우 실행을 모니터링하고 보고하세요:

워크플로우 ID: {workflow_id}
모니터링 기간: {monitoring_period}
주요 지표: {key_metrics}
알림 임계값: {alert_thresholds}

모니터링 보고서에는 다음이 포함되어야 합니다:
1. 주요 지표의 현재 값 및 추세
2. 임계값 위반 또는 이상 탐지
3. 성능 병목 현상 또는 오류 식별
4. 리소스 사용량 및 효율성 평가

실시간 모니터링 및 문제 해결을 위한 실행 가능한 권장 사항을 제공하세요.
"""
)

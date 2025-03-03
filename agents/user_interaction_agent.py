"""
사용자 상호작용 에이전트 모듈
사용자와의 상호작용을 관리하고 피드백을 수집하는 에이전트입니다.
"""

import os
import re
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from pydantic import BaseModel, Field
from datetime import datetime

from langchain_core.runnables import RunnableConfig
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser

from config.settings import OUTPUT_DIR
from utils.logger import logger
from models.paper import Paper
from models.state import PaperWorkflowState
from agents.base import BaseAgent


class UserFeedback(BaseModel):
    """사용자 피드백 형식"""
    feedback_text: str = Field(description="사용자 피드백 텍스트")
    timestamp: str = Field(description="피드백 시간")
    is_positive: Optional[bool] = Field(description="긍정적 피드백 여부", default=None)
    tags: List[str] = Field(description="피드백 태그", default_factory=list)


class FeedbackAnalysis(BaseModel):
    """피드백 분석 결과 형식"""
    is_positive: bool = Field(description="긍정적 피드백 여부")
    sentiment_score: float = Field(description="감정 점수 (-1.0 ~ 1.0)")
    main_points: List[str] = Field(description="주요 요점")
    action_items: List[str] = Field(description="필요한 조치 항목")
    priority: str = Field(description="우선순위 (높음, 중간, 낮음)")


class ProgressUpdate(BaseModel):
    """진행 상황 업데이트 형식"""
    current_stage: str = Field(description="현재 단계")
    progress_percentage: float = Field(description="진행률 (%)")
    completed_tasks: List[str] = Field(description="완료된 작업")
    pending_tasks: List[str] = Field(description="대기 중인 작업")
    estimated_completion: Optional[str] = Field(description="예상 완료 시간", default=None)
    issues: List[str] = Field(description="현재 이슈", default_factory=list)


class UserInteractionAgent(BaseAgent[Dict[str, Any]]):
    """사용자 상호작용 관리 에이전트"""

    def __init__(
        self,
        name: str = "사용자 상호작용 에이전트",
        description: str = "사용자 상호작용 관리 및 피드백 수집",
        verbose: bool = False
    ):
        """
        UserInteractionAgent 초기화

        Args:
            name (str, optional): 에이전트 이름. 기본값은 "사용자 상호작용 에이전트"
            description (str, optional): 에이전트 설명. 기본값은 "사용자 상호작용 관리 및 피드백 수집"
            verbose (bool, optional): 상세 로깅 활성화 여부. 기본값은 False
        """
        super().__init__(name, description, verbose=verbose)
        
        # 출력 디렉토리 생성
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # 프롬프트 초기화
        self._init_prompts()
        
        # 피드백 저장소 초기화
        self.feedback_history = []
        
        # 진행 상황 초기화
        self.current_progress = None
        
        logger.info(f"{self.name} 초기화 완료")

    def _init_prompts(self) -> None:
        """프롬프트와 체인 초기화"""
        # 피드백 분석 파서 초기화
        self.feedback_analysis_parser = PydanticOutputParser(pydantic_object=FeedbackAnalysis)
        
        # 피드백 분석 체인 초기화
        self.feedback_analysis_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template="""사용자가 제공한 피드백을 분석해 주세요.
                
                피드백:
                {feedback}
                
                다음 항목을 분석해주세요:
                1. 이 피드백이 전반적으로 긍정적인지 부정적인지 (is_positive)
                2. -1.0부터 1.0 사이의 감정 점수 (sentiment_score)
                3. 피드백에서 언급된 주요 요점 목록 (main_points)
                4. 필요한 조치 항목 목록 (action_items)
                5. 조치의 우선순위 (높음, 중간, 낮음) (priority)
                
                {format_instructions}
                """,
                input_variables=["feedback", "format_instructions"],
            ),
            verbose=self.verbose
        )
        
        # 진행 상황 생성 체인 초기화
        self.progress_update_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template="""현재 워크플로우 상태를 분석하고 사용자에게 적합한 진행 상황 업데이트를 생성해 주세요.
                
                워크플로우 상태:
                {workflow_state}
                
                다음 형식으로 진행 상황 업데이트를 생성해 주세요:
                1. 현재 진행 중인 단계
                2. 현재 진행률 (%)
                3. 완료된 작업 목록
                4. 대기 중인 작업 목록
                5. 예상 완료 시간 (있는 경우)
                6. 현재 이슈 목록 (있는 경우)
                
                업데이트는 명확하고 간결하게 작성해 주세요.
                """,
                input_variables=["workflow_state"],
            ),
            verbose=self.verbose
        )
        
        logger.debug("사용자 상호작용 에이전트 프롬프트 및 체인 초기화 완료")

    def collect_feedback(self, feedback_text: str) -> UserFeedback:
        """
        사용자 피드백을 수집합니다.

        Args:
            feedback_text (str): 사용자 피드백 텍스트

        Returns:
            UserFeedback: 수집된 피드백
        """
        logger.info("사용자 피드백 수집 중...")
        
        try:
            # 피드백 분석
            format_instructions = self.feedback_analysis_parser.get_format_instructions()
            
            result = self.feedback_analysis_chain.invoke({
                "feedback": feedback_text,
                "format_instructions": format_instructions
            })
            
            analysis = self.feedback_analysis_parser.parse(result["text"])
            
            # 피드백 태그 추출
            tags = []
            if "수정" in feedback_text or "변경" in feedback_text or "개선" in feedback_text:
                tags.append("수정_요청")
            if "질문" in feedback_text or "궁금" in feedback_text or "어떻게" in feedback_text:
                tags.append("질문")
            if "좋아" in feedback_text or "만족" in feedback_text or "훌륭" in feedback_text:
                tags.append("긍정_평가")
            if "나쁘" in feedback_text or "불만" in feedback_text or "실망" in feedback_text:
                tags.append("부정_평가")
            
            # 사용자 피드백 객체 생성
            user_feedback = UserFeedback(
                feedback_text=feedback_text,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                is_positive=analysis.is_positive,
                tags=tags
            )
            
            # 피드백 저장
            self.feedback_history.append(user_feedback)
            
            # 상태 업데이트
            current_feedback_count = len(self.feedback_history)
            self.update_state({
                "last_feedback": user_feedback.dict(),
                "feedback_count": current_feedback_count,
                "feedback_analysis": analysis.dict()
            })
            
            logger.info(f"피드백 수집 완료: {'긍정적' if analysis.is_positive else '부정적'} 피드백, {len(tags)}개 태그")
            return user_feedback
            
        except Exception as e:
            logger.error(f"피드백 수집 중 오류 발생: {str(e)}")
            
            # 오류 발생 시 기본 피드백 반환
            default_feedback = UserFeedback(
                feedback_text=feedback_text,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                tags=["처리_오류"]
            )
            
            # 피드백 저장
            self.feedback_history.append(default_feedback)
            
            return default_feedback

    def analyze_feedback(self, feedback: UserFeedback) -> FeedbackAnalysis:
        """
        수집된 피드백을 분석합니다.

        Args:
            feedback (UserFeedback): 분석할 피드백

        Returns:
            FeedbackAnalysis: 피드백 분석 결과
        """
        logger.info(f"피드백 분석 중: {feedback.feedback_text[:50]}...")
        
        try:
            # 피드백 분석
            format_instructions = self.feedback_analysis_parser.get_format_instructions()
            
            result = self.feedback_analysis_chain.invoke({
                "feedback": feedback.feedback_text,
                "format_instructions": format_instructions
            })
            
            analysis = self.feedback_analysis_parser.parse(result["text"])
            
            logger.info(f"피드백 분석 완료: 감정 점수 {analysis.sentiment_score}, 우선순위 {analysis.priority}")
            return analysis
            
        except Exception as e:
            logger.error(f"피드백 분석 중 오류 발생: {str(e)}")
            
            # 오류 발생 시 기본 분석 결과 반환
            return FeedbackAnalysis(
                is_positive=False,
                sentiment_score=0.0,
                main_points=["분석 중 오류 발생"],
                action_items=["피드백 다시 분석"],
                priority="중간"
            )

    def generate_progress_update(self, workflow_state: PaperWorkflowState) -> ProgressUpdate:
        """
        현재 진행 상황 업데이트를 생성합니다.

        Args:
            workflow_state (PaperWorkflowState): 현재 워크플로우 상태

        Returns:
            ProgressUpdate: 생성된 진행 상황 업데이트
        """
        logger.info("진행 상황 업데이트 생성 중...")
        
        try:
            # 현재 단계 결정
            current_stage = "초기화"
            progress_percentage = 0.0
            completed_tasks = []
            pending_tasks = ["연구", "작성", "편집"]
            issues = []
            
            if workflow_state.error:
                issues.append(workflow_state.error)
            
            status = workflow_state.status
            
            if status == "initialized":
                current_stage = "초기화"
                progress_percentage = 10.0
                completed_tasks = ["초기화"]
                pending_tasks = ["연구", "작성", "편집"]
            elif status == "research_completed":
                current_stage = "연구 완료"
                progress_percentage = 30.0
                completed_tasks = ["초기화", "연구"]
                pending_tasks = ["작성", "편집"]
            elif status == "writing_completed":
                current_stage = "작성 완료"
                progress_percentage = 60.0
                completed_tasks = ["초기화", "연구", "작성"]
                pending_tasks = ["편집"]
            elif status == "editing_completed":
                current_stage = "편집 완료"
                progress_percentage = 100.0
                completed_tasks = ["초기화", "연구", "작성", "편집"]
                pending_tasks = []
            
            # 예상 완료 시간 계산 (간단한 추정)
            estimated_completion = None
            if progress_percentage < 100.0:
                remaining_percentage = 100.0 - progress_percentage
                # 대략적으로 10%당 5분 소요된다고 가정
                remaining_minutes = int(remaining_percentage / 10.0 * 5)
                
                if remaining_minutes > 0:
                    current_time = datetime.now()
                    estimated_time = current_time.replace(
                        minute=current_time.minute + remaining_minutes,
                        second=0
                    )
                    estimated_completion = estimated_time.strftime("%H:%M")
            
            # 진행 상황 업데이트 객체 생성
            progress_update = ProgressUpdate(
                current_stage=current_stage,
                progress_percentage=progress_percentage,
                completed_tasks=completed_tasks,
                pending_tasks=pending_tasks,
                estimated_completion=estimated_completion,
                issues=issues
            )
            
            # 상태 저장
            self.current_progress = progress_update
            
            logger.info(f"진행 상황 업데이트 생성 완료: {current_stage}, {progress_percentage}% 완료")
            return progress_update
            
        except Exception as e:
            logger.error(f"진행 상황 업데이트 생성 중 오류 발생: {str(e)}")
            
            # 오류 발생 시 기본 업데이트 반환
            return ProgressUpdate(
                current_stage="알 수 없음",
                progress_percentage=0.0,
                completed_tasks=[],
                pending_tasks=["연구", "작성", "편집"],
                issues=["진행 상황 업데이트 생성 중 오류 발생"]
            )

    def format_progress_message(self, progress: ProgressUpdate) -> str:
        """
        진행 상황 업데이트를 사용자 친화적인 메시지로 포맷팅합니다.

        Args:
            progress (ProgressUpdate): 진행 상황 업데이트

        Returns:
            str: 포맷팅된 메시지
        """
        message = f"## 현재 진행 상황: {progress.current_stage}\n\n"
        message += f"전체 진행률: **{progress.progress_percentage:.1f}%**\n\n"
        
        message += "### 완료된 작업:\n"
        for task in progress.completed_tasks:
            message += f"- ✅ {task}\n"
        
        message += "\n### 대기 중인 작업:\n"
        for task in progress.pending_tasks:
            message += f"- ⏳ {task}\n"
        
        if progress.estimated_completion:
            message += f"\n예상 완료 시간: **{progress.estimated_completion}**\n"
        
        if progress.issues:
            message += "\n### 현재 이슈:\n"
            for issue in progress.issues:
                message += f"- ⚠️ {issue}\n"
        
        return message

    def get_feedback_history(self) -> List[Dict[str, Any]]:
        """
        피드백 히스토리를 가져옵니다.

        Returns:
            List[Dict[str, Any]]: 피드백 히스토리 목록
        """
        return [feedback.dict() for feedback in self.feedback_history]

    def run(
        self, 
        input_data: Dict[str, Any], 
        config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """
        사용자 상호작용 에이전트를 실행합니다.

        Args:
            input_data (Dict[str, Any]): 입력 데이터
                - action: 실행할 액션 (collect_feedback, generate_progress_update)
                - feedback_text: 사용자 피드백 텍스트 (collect_feedback 액션에 필요)
                - workflow_state: 워크플로우 상태 (generate_progress_update 액션에 필요)
            config (Optional[RunnableConfig], optional): 실행 구성

        Returns:
            Dict[str, Any]: 실행 결과
        """
        logger.info("사용자 상호작용 에이전트 실행 중...")
        
        try:
            action = input_data.get("action", "")
            
            if action == "collect_feedback":
                feedback_text = input_data.get("feedback_text", "")
                if not feedback_text:
                    raise ValueError("피드백 텍스트가 제공되지 않았습니다.")
                
                feedback = self.collect_feedback(feedback_text)
                analysis = self.analyze_feedback(feedback)
                
                return {
                    "success": True,
                    "action": "collect_feedback",
                    "feedback": feedback.dict(),
                    "analysis": analysis.dict()
                }
                
            elif action == "generate_progress_update":
                workflow_state = input_data.get("workflow_state")
                if not workflow_state:
                    raise ValueError("워크플로우 상태가 제공되지 않았습니다.")
                
                progress = self.generate_progress_update(workflow_state)
                message = self.format_progress_message(progress)
                
                return {
                    "success": True,
                    "action": "generate_progress_update",
                    "progress": progress.dict(),
                    "message": message
                }
                
            else:
                raise ValueError(f"지원되지 않는 액션: {action}")
                
        except Exception as e:
            logger.error(f"사용자 상호작용 에이전트 실행 중 오류 발생: {str(e)}")
            
            return {
                "success": False,
                "error": str(e),
                "action": input_data.get("action", "unknown")
            }
"""
논문 리뷰 및 사용자 상호작용 에이전트 모듈
논문을 검토하고 사용자와의 상호작용을 관리하는 통합 에이전트입니다.
"""

import os
import re
import json
from typing import Dict, Any, List, Optional, Tuple, Union, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import uuid
import time

from langchain_core.runnables import RunnableConfig
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser

from config.settings import OUTPUT_DIR
from utils.logger import logger
from models.paper import Paper
from models.state import PaperWorkflowState
from agents.base import BaseAgent
from prompts.review_prompts import (
    FEEDBACK_ANALYSIS_PROMPT,
    PAPER_REVIEW_PROMPT,
    REVIEW_ACTION_PROMPT,
    PROGRESS_UPDATE_PROMPT,
    FEEDBACK_PROMPT,
    COLLABORATION_PROMPT
)


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


class PaperReview(BaseModel):
    """논문 리뷰 결과 형식"""
    paper_title: str = Field(description="리뷰한 논문 제목")
    overall_score: int = Field(description="전체 평가 점수 (1-10)")
    strengths: List[str] = Field(description="장점")
    weaknesses: List[str] = Field(description="약점")
    improvement_suggestions: List[str] = Field(description="개선 제안")
    section_feedback: Dict[str, str] = Field(description="섹션별 피드백")
    structural_comments: str = Field(description="구조적 의견")
    language_comments: str = Field(description="언어적 의견")


class ReviewAction(BaseModel):
    """리뷰 조치 사항 형식"""
    action_type: Literal["REVISE", "ADD", "DELETE", "RESTRUCTURE"]
    section: Optional[str] = Field(description="관련 섹션", default=None)
    description: str = Field(description="조치 설명")
    priority: Literal["HIGH", "MEDIUM", "LOW"]
    rationale: str = Field(description="조치의 근거")
    impact_analysis: Dict[str, float] = Field(description="Predicted impact scores on different quality dimensions")


class EnhancedFeedbackAnalysis(FeedbackAnalysis):
    """Enhanced feedback analysis model"""
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    contextual_tags: List[str] = Field(..., description="Domain-specific tags")
    related_sections: List[str] = Field(..., description="Affected paper sections")


class ReviewAgent(BaseAgent):
    """Academic Paper Review and Collaboration Agent
    
    Features:
    - Multi-stage review workflow management
    - Sentiment-aware feedback analysis
    - Intelligent progress tracking
    - Actionable recommendation system
    - Collaborative editing support
    """

    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.3):
        super().__init__(
            name="Academic Review Agent",
            description="Handles paper review and collaborative editing processes",
            model_name=model_name,
            temperature=temperature
        )
        # 초기화 로직 추가
        os.makedirs(OUTPUT_DIR, exist_ok=True)  # 출력 디렉토리 생성
        self._init_prompts()  # 프롬프트 초기화
        self.feedback_history = []  # 피드백 저장소 초기화
        self.review_history = []  # 리뷰 저장소 초기화
        self.current_progress = None  # 진행 상황 초기화
        logger.info(f"{self.name} 초기화 완료")

    def _init_prompts(self) -> None:
        """프롬프트와 체인 초기화"""
        self.feedback_analysis_parser = PydanticOutputParser(pydantic_object=FeedbackAnalysis)
        self.paper_review_parser = PydanticOutputParser(pydantic_object=PaperReview)
        self.review_action_parser = PydanticOutputParser(pydantic_object=ReviewAction)

        # 피드백 분석 체인 초기화
        self.feedback_analysis_chain = LLMChain(
            llm=self.llm,
            prompt=FEEDBACK_ANALYSIS_PROMPT,
            verbose=self.verbose
        )

        # 논문 리뷰 체인 초기화
        self.paper_review_chain = LLMChain(
            llm=self.llm,
            prompt=PAPER_REVIEW_PROMPT,
            verbose=self.verbose
        )

        # 리뷰 조치 체인 초기화
        self.review_action_chain = LLMChain(
            llm=self.llm,
            prompt=REVIEW_ACTION_PROMPT,
            verbose=self.verbose
        )

        # 진행 상황 생성 체인 초기화
        self.progress_update_chain = LLMChain(
            llm=self.llm,
            prompt=PROGRESS_UPDATE_PROMPT,
            verbose=self.verbose
        )

        # 피드백 체인 초기화
        self.feedback_chain = LLMChain(
            llm=self.llm,
            prompt=FEEDBACK_PROMPT,
            verbose=self.verbose
        )

        # 협업 체인 초기화
        self.collaboration_chain = LLMChain(
            llm=self.llm,
            prompt=COLLABORATION_PROMPT,
            verbose=self.verbose
        )
        logger.debug("리뷰 및 상호작용 에이전트 프롬프트 및 체인 초기화 완료")

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
            default_feedback = UserFeedback(
                feedback_text=feedback_text,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                tags=["처리_오류"]
            )
            self.feedback_history.append(default_feedback)
            return default_feedback

    def analyze_feedback(self, feedback: UserFeedback) -> EnhancedFeedbackAnalysis:
        """Improved feedback analysis with contextual understanding"""
        logger.info(f"사용자 피드백 분석 중: '{feedback.feedback_text[:50]}...'")
        
        try:
            # 피드백 분석 프롬프트 생성
            prompt = f"""
            다음 사용자 피드백을 분석해주세요:
            
            피드백: "{feedback.feedback_text}"
            
            다음 항목을 분석해주세요:
            1. 감정 분석 (긍정/부정/중립)
            2. 주요 주제 및 관심사
            3. 구체적인 요청 또는 제안
            4. 우선순위가 높은 개선 사항
            5. 피드백의 맥락 및 배경
            
            JSON 형식으로 응답해주세요.
            """
            
            # LLM을 사용하여 피드백 분석
            messages = [
                SystemMessage(content="당신은 사용자 피드백 분석 전문가입니다. 제공된 피드백을 심층적으로 분석해주세요."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            analysis_text = response.content
            
            # JSON 추출
            import json
            import re
            
            json_match = re.search(r'```json\s*(.*?)\s*```', analysis_text, re.DOTALL)
            if json_match:
                analysis_json = json.loads(json_match.group(1))
            else:
                try:
                    analysis_json = json.loads(analysis_text)
                except:
                    # JSON 파싱 실패 시 기본 구조 생성
                    analysis_json = {
                        "sentiment": "neutral",
                        "topics": ["general feedback"],
                        "requests": [],
                        "priorities": [],
                        "context": "No specific context identified"
                    }
            
            # EnhancedFeedbackAnalysis 객체 생성
            return EnhancedFeedbackAnalysis(
                feedback_id=feedback.feedback_id if hasattr(feedback, 'feedback_id') else str(uuid.uuid4()),
                sentiment=analysis_json.get("sentiment", "neutral"),
                topics=analysis_json.get("topics", []),
                requests=analysis_json.get("requests", []),
                priorities=analysis_json.get("priorities", []),
                context=analysis_json.get("context", ""),
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"피드백 분석 중 오류 발생: {str(e)}")
            return EnhancedFeedbackAnalysis(
                feedback_id=str(uuid.uuid4()),
                sentiment="neutral",
                topics=["error"],
                requests=[],
                priorities=[],
                context=f"Error during analysis: {str(e)}",
                timestamp=datetime.now().isoformat()
            )

    def generate_progress_update(self, workflow_state: PaperWorkflowState) -> ProgressUpdate:
        """Intelligent progress tracking with predictive analytics"""
        logger.info("작업 진행 상황 업데이트 생성 중")
        
        try:
            # 워크플로우 상태에서 정보 추출
            current_stage = workflow_state.current_stage
            completed_stages = workflow_state.completed_stages
            total_stages = len(workflow_state.stage_definitions)
            completion_percentage = (len(completed_stages) / total_stages) * 100 if total_stages > 0 else 0
            
            # 남은 작업 예상
            remaining_stages = [stage for stage in workflow_state.stage_definitions if stage not in completed_stages]
            
            # 예상 완료 시간 계산 (간단한 추정)
            current_time = time.time()
            start_time = workflow_state.start_time if hasattr(workflow_state, 'start_time') else current_time
            
            elapsed_time = current_time - start_time
            if len(completed_stages) > 0:
                avg_time_per_stage = elapsed_time / len(completed_stages)
                estimated_remaining_time = avg_time_per_stage * len(remaining_stages)
            else:
                # 기본 예상 (각 단계당 10분)
                estimated_remaining_time = len(remaining_stages) * 600
            
            # 진행 상황 요약 생성
            summary_prompt = f"""
            다음 논문 작성 워크플로우의 진행 상황을 요약해주세요:
            
            현재 단계: {current_stage}
            완료된 단계: {', '.join(completed_stages)}
            남은 단계: {', '.join(remaining_stages)}
            전체 진행률: {completion_percentage:.1f}%
            
            간결하고 정보가 풍부한 진행 상황 요약을 제공해주세요.
            """
            
            messages = [
                SystemMessage(content="당신은 프로젝트 진행 상황 보고 전문가입니다."),
                HumanMessage(content=summary_prompt)
            ]
            
            response = self.llm.invoke(messages)
            progress_summary = response.content
            
            # ProgressUpdate 객체 생성
            return ProgressUpdate(
                current_stage=current_stage,
                completion_percentage=completion_percentage,
                completed_stages=completed_stages,
                remaining_stages=remaining_stages,
                estimated_completion_time=datetime.fromtimestamp(current_time + estimated_remaining_time).isoformat(),
                progress_summary=progress_summary,
                bottlenecks=[],  # 실제 구현에서는 병목 현상 분석 추가
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"진행 상황 업데이트 생성 중 오류 발생: {str(e)}")
            return ProgressUpdate(
                current_stage="unknown",
                completion_percentage=0.0,
                completed_stages=[],
                remaining_stages=[],
                estimated_completion_time=datetime.now().isoformat(),
                progress_summary=f"진행 상황 업데이트 생성 중 오류가 발생했습니다: {str(e)}",
                bottlenecks=[],
                timestamp=datetime.now().isoformat()
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

    def review_paper(self, paper: Paper) -> PaperReview:
        """
        논문을 검토하고 종합적인 피드백을 제공합니다.

        Args:
            paper (Paper): 검토할 논문

        Returns:
            PaperReview: 논문 리뷰 결과
        """
        logger.info(f"논문 '{paper.title}' 검토 중...")
        try:
            paper_content = f"# {paper.title}\n\n"
            for section in paper.sections:
                paper_content += f"## {section.title}\n\n{section.content}\n\n"
            if paper.references:
                paper_content += "## 참고 문헌\n\n"
                for i, ref in enumerate(paper.references, 1):
                    authors = ", ".join(ref.authors) if ref.authors else "알 수 없음"
                    paper_content += f"{i}. {ref.title}. {authors}. {ref.year}. {ref.source}.\n"

            format_instructions = self.paper_review_parser.get_format_instructions()
            result = self.paper_review_chain.invoke({
                "paper_title": paper.title,
                "paper_content": paper_content,
                "format_instructions": format_instructions
            })
            review = self.paper_review_parser.parse(result["text"])
            self.review_history.append(review)

            self.update_state({
                "last_review": review.dict(),
                "review_count": len(self.review_history)
            })
            logger.info(f"논문 '{paper.title}' 검토 완료: 평가 점수 {review.overall_score}/10")
            return review

        except Exception as e:
            logger.error(f"논문 검토 중 오류 발생: {str(e)}")
            default_review = PaperReview(
                paper_title=paper.title,
                overall_score=5,
                strengths=["리뷰 중 오류 발생"],
                weaknesses=["검토를 완료할 수 없음"],
                improvement_suggestions=["시스템 오류를 확인하세요"],
                section_feedback={},
                structural_comments="검토 중 오류가 발생했습니다.",
                language_comments="검토 중 오류가 발생했습니다."
            )
            return default_review

    def suggest_review_actions(self, paper: Paper, review: PaperReview) -> List[ReviewAction]:
        """
        리뷰 결과를 바탕으로 구체적인 조치 사항을 제안합니다.

        Args:
            paper (Paper): 검토된 논문
            review (PaperReview): 논문 리뷰 결과

        Returns:
            List[ReviewAction]: 제안된 조치 사항 목록
        """
        logger.info(f"논문 '{paper.title}'에 대한 조치 사항 제안 중...")
        try:
            review_result = json.dumps(review.dict(), ensure_ascii=False)
            format_instructions = self.review_action_parser.get_format_instructions()
            actions = []

            for weakness in review.weaknesses[:3]:  # 상위 3개 약점만 처리
                result = self.review_action_chain.invoke({
                    "paper_title": paper.title,
                    "review_result": f"약점: {weakness}\n{review_result}",
                    "format_instructions": format_instructions
                })
                action = self.review_action_parser.parse(result["text"])
                actions.append(action)

            logger.info(f"논문 '{paper.title}'에 대한 {len(actions)}개 조치 사항 제안 완료")
            return actions

        except Exception as e:
            logger.error(f"조치 사항 제안 중 오류 발생: {str(e)}")
            return [
                ReviewAction(
                    action_type="REVISE",  # Literal에 맞게 수정
                    section=None,
                    description="조치 사항 제안 중 오류가 발생했습니다.",
                    priority="HIGH",  # Literal에 맞게 수정
                    rationale="시스템 오류를 해결하세요.",
                    impact_analysis={"quality": 0.5}  # 필수 필드 추가
                )
            ]

    def format_review_report(self, review: PaperReview, actions: List[ReviewAction]) -> str:
        """Dynamic report generation with template support"""
        report_template = """# Academic Paper Review Report

## Executive Summary
{overview}

## Detailed Assessment
{sections}

## Action Plan
{actions}

## Review Metadata
{metadata}"""
        # 주석으로 남겨진 미완성 메서드
        return report_template

    def run(self, input_data: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """
        리뷰 및 상호작용 에이전트를 실행합니다.

        Args:
            input_data (Dict[str, Any]): 입력 데이터
            config (Optional[RunnableConfig], optional): 실행 구성

        Returns:
            Dict[str, Any]: 실행 결과
        """
        logger.info("리뷰 및 상호작용 에이전트 실행 중...")
        try:
            action = input_data.get("action", "")
            if action == "collect_feedback":
                feedback_text = input_data.get("feedback_text", "")
                if not feedback_text:
                    raise ValueError("피드백 텍스트가 제공되지 않았습니다.")
                feedback = self.collect_feedback(feedback_text)
                # analyze_feedback 미완성으로 주석 처리
                return {
                    "success": True,
                    "action": "collect_feedback",
                    "feedback": feedback.dict(),
                    # "analysis": analysis.dict()  # 미완성
                }
            else:
                raise ValueError(f"지원되지 않는 액션: {action}")
        except Exception as e:
            logger.error(f"리뷰 및 상호작용 에이전트 실행 중 오류 발생: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "action": input_data.get("action", "unknown")
            }

    def enable_collaboration(self, paper: Paper) -> Dict[str, Any]:
        """Real-time collaboration setup"""
        # 주석으로 남겨진 미완성 메서드
        return {}

    def _monitor_performance(self):
        """Real-time performance monitoring"""
        # 주석으로 남겨진 미완성 메서드
        return {}


class ReviewError(Exception):
    """Custom exception for review operations"""
    def __init__(self, message: str, error_code: int):
        super().__init__(message)
        self.error_code = error_code
        logger.error(f"ReviewError {error_code}: {message}")
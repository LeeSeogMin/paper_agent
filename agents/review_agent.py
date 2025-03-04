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

        # 논문 리뷰 체인 초기화
        self.paper_review_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template="""다음 논문을 철저히 검토하고 종합적인 학술 리뷰를 제공해 주세요.
                
                논문 제목: {paper_title}
                
                논문 내용:
                {paper_content}
                
                다음 항목을 포함한 철저한 학술 리뷰를 작성해 주세요:
                1. 전체 평가 점수 (1-10)
                2. 주요 장점 목록
                3. 주요 약점 목록
                4. 구체적인 개선 제안 목록
                5. 주요 섹션별 피드백
                6. 구조적 의견 (논문 구성, 흐름, 논리적 일관성)
                7. 언어적 의견 (명확성, 간결성, 학술적 표현)
                
                리뷰는 객관적이고 건설적이어야 하며, 논문의 학술적 가치를 향상시키는 데 도움이 되어야 합니다.
                
                {format_instructions}
                """,
                input_variables=["paper_title", "paper_content", "format_instructions"],
            ),
            verbose=self.verbose
        )

        # 리뷰 조치 체인 초기화
        self.review_action_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                template="""논문 리뷰 결과를 바탕으로 구체적인 조치 사항을 제안해 주세요.
                
                논문 제목: {paper_title}
                리뷰 결과:
                {review_result}
                
                다음 형식으로 가장 중요한 조치 사항을 제안해 주세요:
                1. 조치 유형 (수정, 추가, 삭제, 재구성 등)
                2. 관련 섹션 (해당되는 경우)
                3. 상세한 조치 설명
                4. 우선순위 (높음, 중간, 낮음)
                5. 이 조치가 필요한 근거
                
                {format_instructions}
                """,
                input_variables=["paper_title", "review_result", "format_instructions"],
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
        analysis_prompt = """Analyze academic feedback:
        {feedback}
        
        Analysis Requirements:
        1. Determine sentiment polarity (-1.0 to 1.0)
        2. Identify key discussion points
        3. Tag relevant paper sections
        4. Generate actionable items
        5. Estimate confidence level
        
        {format_instructions}"""
        # 분석 체인 업그레이드 (미완성 주석으로 남겨진 부분)
        pass

    def generate_progress_update(self, workflow_state: PaperWorkflowState) -> ProgressUpdate:
        """Intelligent progress tracking with predictive analytics"""
        # 주석으로 남겨진 미완성 메서드
        pass

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
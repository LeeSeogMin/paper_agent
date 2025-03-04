"""
Prompt templates for the coordinator agent in the paper writing workflow.

This module contains prompt templates that guide the coordinator agent in managing
the overall paper writing process, including task delegation, progress tracking,
and workflow coordination between different specialized agents.
"""

from langchain_core.prompts import PromptTemplate

# Prompt for initializing the coordination process
COORDINATION_INITIALIZATION_PROMPT = PromptTemplate(
    input_variables=["topic", "paper_type", "requirements"],
    template="""
You are the Coordinator Agent responsible for managing the entire paper writing process.

TOPIC: {topic}
PAPER TYPE: {paper_type}
REQUIREMENTS: {requirements}

Your task is to initialize the paper writing workflow by:
1. Analyzing the topic and requirements
2. Breaking down the paper writing process into manageable tasks
3. Determining the sequence of tasks
4. Identifying which specialized agents should handle each task
5. Creating a comprehensive coordination plan

Provide a detailed coordination plan that includes:
- Task breakdown with clear objectives for each task
- Task dependencies and sequence
- Agent assignments for each task
- Key milestones and checkpoints
- Initial timeline estimates
- Potential challenges and contingency plans

The coordination plan should be comprehensive yet adaptable to changes that may arise during the paper writing process.
"""
)

# Prompt for task delegation to specialized agents
TASK_DELEGATION_PROMPT = PromptTemplate(
    input_variables=["agent_type", "task_description", "context", "requirements", "deadline"],
    template="""
You are the Coordinator Agent delegating a task to a specialized agent.

AGENT TYPE: {agent_type}
TASK: {task_description}
CONTEXT: {context}
REQUIREMENTS: {requirements}
DEADLINE: {deadline}

Create a detailed task specification for the {agent_type} that includes:
1. Clear objectives and expected outcomes
2. Relevant background information from the context
3. Specific requirements and constraints
4. Quality criteria for the deliverable
5. Format and structure expectations
6. Integration points with other components of the paper
7. Timeline and milestone expectations

The task specification should be comprehensive enough for the specialized agent to work independently while ensuring their output aligns with the overall paper requirements and integrates seamlessly with other components.
"""
)

# Prompt for progress monitoring and assessment
PROGRESS_MONITORING_PROMPT = PromptTemplate(
    input_variables=["current_state", "completed_tasks", "ongoing_tasks", "pending_tasks", "issues", "timeline"],
    template="""
You are the Coordinator Agent monitoring the progress of the paper writing workflow.

CURRENT STATE SUMMARY:
{current_state}

COMPLETED TASKS:
{completed_tasks}

ONGOING TASKS:
{ongoing_tasks}

PENDING TASKS:
{pending_tasks}

ISSUES OR BLOCKERS:
{issues}

TIMELINE:
{timeline}

Assess the current state of the paper writing process by:
1. Evaluating the quality and completeness of completed tasks
2. Checking if ongoing tasks are on schedule and meeting quality expectations
3. Identifying any bottlenecks or dependencies affecting pending tasks
4. Analyzing reported issues and determining their impact on the workflow
5. Comparing current progress against the planned timeline

Provide a comprehensive progress assessment that includes:
- Overall status evaluation (on track, at risk, delayed)
- Quality assessment of completed work
- Identification of critical path items
- Recommendations for addressing issues or optimizing the workflow
- Updated timeline projections if necessary
- Specific actions needed to maintain or improve progress
"""
)

# Prompt for workflow adjustment and optimization
WORKFLOW_ADJUSTMENT_PROMPT = PromptTemplate(
    input_variables=["current_state", "issues", "new_requirements", "progress_assessment"],
    template="""
You are the Coordinator Agent adjusting the paper writing workflow based on current progress and emerging needs.

CURRENT STATE:
{current_state}

ISSUES OR CHALLENGES:
{issues}

NEW OR CHANGED REQUIREMENTS:
{new_requirements}

PROGRESS ASSESSMENT:
{progress_assessment}

Develop a workflow adjustment plan that addresses current challenges and optimizes the process by:
1. Analyzing the root causes of identified issues
2. Evaluating the impact of new requirements on the existing workflow
3. Identifying opportunities to optimize task sequences or agent assignments
4. Determining if resource reallocation is necessary
5. Considering alternative approaches to problematic workflow components

Provide a detailed workflow adjustment plan that includes:
- Specific changes to task definitions, sequences, or assignments
- Rationale for each proposed change
- Expected impact on quality, timeline, and resource utilization
- Implementation approach for the adjustments
- Communication plan for affected agents
- Updated success criteria if applicable

The adjustment plan should maintain focus on the overall paper quality and requirements while addressing immediate challenges and improving workflow efficiency.
"""
)

# Prompt for integrating components from different agents
COMPONENT_INTEGRATION_PROMPT = PromptTemplate(
    input_variables=["components", "paper_structure", "style_guide", "integration_issues"],
    template="""
You are the Coordinator Agent responsible for integrating components created by different specialized agents into a cohesive paper.

COMPONENTS TO INTEGRATE:
{components}

PAPER STRUCTURE:
{paper_structure}

STYLE GUIDE:
{style_guide}

KNOWN INTEGRATION ISSUES:
{integration_issues}

Create an integration plan that will combine the various components into a unified paper by:
1. Analyzing each component for content quality, completeness, and adherence to requirements
2. Identifying gaps, overlaps, or inconsistencies between components
3. Ensuring logical flow and transitions between sections
4. Maintaining consistent terminology, style, and formatting throughout
5. Addressing any known integration issues

Provide a detailed integration strategy that includes:
- Assessment of each component's readiness for integration
- Specific integration actions needed for each component
- Approach to resolving inconsistencies or conflicts
- Methods for creating seamless transitions between sections
- Quality control measures for the integrated document
- Feedback to be provided to specialized agents if revisions are needed

The integration strategy should prioritize creating a cohesive, high-quality paper that reads as if written by a single author while preserving the specialized expertise contributed by each agent.
"""
)

# Prompt for quality assessment and feedback
QUALITY_ASSESSMENT_PROMPT = PromptTemplate(
    input_variables=["paper_content", "requirements", "quality_criteria", "target_audience"],
    template="""
You are the Coordinator Agent assessing the quality of the paper and providing feedback for improvements.

PAPER CONTENT:
{paper_content}

REQUIREMENTS:
{requirements}

QUALITY CRITERIA:
{quality_criteria}

TARGET AUDIENCE:
{target_audience}

Conduct a comprehensive quality assessment of the paper by evaluating:
1. Alignment with the original requirements and objectives
2. Adherence to academic standards and conventions
3. Logical structure and flow of arguments
4. Clarity and precision of language
5. Appropriate use of evidence and citations
6. Depth and originality of analysis
7. Suitability for the target audience
8. Overall impact and contribution to the field

Provide a detailed quality assessment report that includes:
- Overall evaluation of the paper's quality and readiness
- Specific strengths to maintain or emphasize
- Areas requiring improvement or revision
- Actionable feedback for each identified issue
- Prioritization of recommended changes
- Specific guidance for addressing major concerns
- Assessment of alignment with target audience expectations

The quality assessment should be constructive and focused on enhancing the paper's impact and scholarly contribution while maintaining academic rigor.
"""
)

# Prompt for final review and submission preparation
FINAL_REVIEW_PROMPT = PromptTemplate(
    input_variables=["paper_content", "submission_guidelines", "previous_feedback", "final_checklist"],
    template="""
You are the Coordinator Agent conducting a final review of the paper before submission.

PAPER CONTENT:
{paper_content}

SUBMISSION GUIDELINES:
{submission_guidelines}

PREVIOUS FEEDBACK ADDRESSED:
{previous_feedback}

FINAL CHECKLIST:
{final_checklist}

Conduct a thorough final review of the paper by:
1. Verifying that all previous feedback has been adequately addressed
2. Ensuring perfect alignment with submission guidelines and requirements
3. Checking for any remaining issues with content, structure, or formatting
4. Verifying citations, references, and academic integrity
5. Conducting a final quality assessment against the standard of excellence expected in the field

Provide a final review report that includes:
- Confirmation that the paper is ready for submission or specific issues preventing submission
- Verification that all items on the final checklist have been completed
- Any last-minute adjustments needed before submission
- Assessment of the paper's strengths and potential impact
- Confirmation that all submission requirements have been met
- Recommendations for the submission process

The final review should ensure that the paper represents the highest quality work possible and is fully prepared for the submission process.
"""
)

RESEARCH_PLAN_PROMPT = """
너는 학술 연구 계획을 수립하는 전문가입니다. 사용자의 요구사항에 따라 상세한 연구 계획을 작성해주세요.

주제: {topic}
논문 유형: {paper_type}
제약사항: {constraints}
참고자료: {references}

다음 요소를 포함한 연구 계획을 JSON 형식으로 작성해주세요:

1. 연구 목표 (3-5개)
2. 검색 전략
   - 검색 범위 (local_only: 로컬 데이터만, web_only: 웹 검색만, all: 모두)
   - 필요한 논문 수
   - 권장 검색 쿼리 (5-10개)
3. 논문 구조 (섹션 및 서브섹션)
4. 일정 계획

JSON 형식:
{
  "topic": "연구 주제",
  "paper_type": "논문 유형",
  "objectives": ["목표1", "목표2", "목표3"],
  "search_strategy": {
    "search_scope": "local_only|web_only|all",
    "min_papers": 10,
    "queries": ["쿼리1", "쿼리2", "쿼리3"]
  },
  "outline": ["섹션1", "섹션2", "섹션3"],
  "timeline": {
    "days": 7,
    "milestones": ["자료 수집 (1-2일)", "초안 작성 (3-5일)", "검토 및 수정 (6-7일)"]
  }
}

제약사항이나 참고자료가 있다면 이를 고려해서 계획을 세우세요.
"""
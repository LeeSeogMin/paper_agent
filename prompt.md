# LangGraph 기반 논문 작성 AI 에이전트 시스템 개발 프로젝트

## 프로젝트 요구사항

LangGraph를 활용한 논문 작성 AI 에이전트 시스템을 개발해야 합니다. 이 시스템은 중앙집중형 멀티 에이전트 아키텍처로, 총괄 에이전트를 중심으로 다양한 특화된 에이전트들이 협업하여 학술 논문을 작성하는 시스템입니다.

## 기본 구조

1. **총괄 에이전트(Coordinator Agent)**: 사용자 요청 분석, 논문 계획 수립, 전체 워크플로우 조정
2. **자료조사 에이전트(Research Agent)**: 학술 DB에서 관련 자료 검색 및 PDF 다운로드
3. **자료정리 에이전트(Data Processing Agent)**: PDF 파싱, 벡터화, 벡터 DB 저장
4. **연구 에이전트(Writing Agent)**: 논문 작성 및 수정
5. **논문심사 에이전트(Review Agent)**: 작성된 논문 검토 및 피드백 제공
6. **사용자 상호작용 에이전트(User Interaction Agent)**: 사용자 피드백 수집 및 진행상황 보고
7. **편집 에이전트(Editing Agent)**: 문법, 스타일, 형식 교정

## 워크플로우

1. 사용자가 논문 주제와 요구사항 제출
2. 총괄 에이전트가 요구사항 분석 후 논문 계획서 작성
3. 자료조사 에이전트가 관련 논문 수집 (최대 22개 PDF)
4. 자료정리 에이전트가 수집된 PDF를 파싱하여 벡터 DB에 저장
5. 총괄 에이전트가 연구 에이전트에게 작업 지시
6. 연구 에이전트가 자료정리 에이전트에게 필요 자료 요청하며 논문 작성
7. 작성된 논문 초안이 총괄 에이전트를 통해 논문심사 에이전트에게 전달
8. 심사 결과에 따라 수정 작업 반복 (최대 5회 반복)
9. 최종 논문 사용자에게 제출

## 기술 스택

- LangGraph: 에이전트 간 워크플로우 구성
- OpenAI API: 총괄, 자료조사, 연구, 논문심사 등 대부분의 에이전트
- XAI API: 자료정리 에이전트
- 벡터 DB: 논문 데이터 저장 (Chroma 또는 Pinecone)
- PDF 처리 라이브러리: PyPDF2, pdfplumber 등
- 학술 검색 API: Semantic Scholar, Crossref 등

## 폴더 구조 및 파일 요구사항

프로젝트는 다음 폴더 구조와 파일들로 구성되어야 합니다:

```
paper_writing_system/
│
├── agents/                      # 각 에이전트 구현 폴더
│   ├── __init__.py
│   ├── coordinator_agent.py     # 총괄 에이전트
│   ├── research_agent.py        # 자료조사 에이전트
│   ├── data_processing_agent.py # 자료정리 에이전트
│   ├── writing_agent.py         # 연구 에이전트
│   ├── review_agent.py          # 논문심사 에이전트
│   ├── user_interaction_agent.py # 사용자 상호작용 에이전트
│   └── editing_agent.py         # 편집 에이전트
│
├── graph/                      # LangGraph 워크플로우 구현
│   ├── __init__.py
│   ├── nodes.py                # 그래프 노드 정의
│   ├── edges.py                # 그래프 엣지 정의
│   └── workflow.py             # 워크플로우 로직
│
├── models/                     # 데이터 모델
│   ├── __init__.py
│   ├── paper.py                # 논문 구조 모델
│   ├── research.py             # 연구 자료 모델
│   └── state.py                # 상태 관리 모델
│
├── utils/                      # 유틸리티 함수들
│   ├── __init__.py
│   ├── api_clients.py          # API 클라이언트 
│   ├── pdf_processor.py        # PDF 처리 유틸리티
│   ├── vector_db.py            # 벡터 DB 통합
│   └── logger.py               # 로깅 시스템
│
├── prompts/                    # 각 에이전트용 프롬프트 템플릿
│   ├── coordinator_prompts.py
│   ├── research_prompts.py
│   ├── writing_prompts.py
│   └── review_prompts.py
│
├── config/                     # 설정 파일
│   ├── __init__.py
│   ├── settings.py             # 기본 설정
│   ├── api_keys.py             # API 키 관리 (gitignore에 추가할 것)
│   └── templates.py            # 논문 템플릿 정의
│
├── data/                       # 데이터 저장 폴더
│   ├── papers/                 # 다운로드된 PDF 저장
│   ├── vector_db/              # 벡터 DB 데이터
│   └── output/                 # 생성된 논문 저장
│
├── app.py                      # 애플리케이션 진입점
├── requirements.txt            # 의존성 정의
└── README.md                   # 프로젝트 문서
```

## 핵심 기능 구현 요구사항

1. **상태 관리 시스템**:
   - 전체 워크플로우 상태 추적
   - 에이전트 간 데이터 전달
   - 오류 복구 및 재개 지원

2. **에이전트 통신 프로토콜**:
   - 표준화된 메시지 형식
   - 비동기 통신 지원
   - 명확한 인터페이스 정의

3. **벡터 DB 통합**:
   - PDF 파싱 및 텍스트 추출
   - 효율적인 청크 설계
   - 메타데이터 관리
   - 관련성 기반 검색

4. **API 사용 최적화**:
   - 토큰 사용량 모니터링
   - 비용 효율적인 모델 선택
   - 병렬 처리 구현

5. **사용자 인터페이스**:
   - 진행 상황 보고
   - 중간 결과물 확인
   - 피드백 제공 인터페이스

## 세부 파일별 요구사항

1. **app.py**:
   - 메인 애플리케이션 진입점
   - 워크플로우 초기화 및 실행
   - 사용자 입력 처리
   - 결과 출력 및 저장

2. **agents/coordinator_agent.py**:
   - 논문 계획 수립 로직
   - 전체 워크플로우 조정
   - 에이전트 간 작업 할당
   - 최종 결과물 품질 관리

3. **agents/research_agent.py**:
   - 학술 검색 API 활용
   - 관련 논문 필터링
   - PDF 다운로드 관리
   - 검색 전략 최적화

4. **agents/data_processing_agent.py**:
   - PDF 텍스트 추출
   - 텍스트 전처리 및 청크 분할
   - 벡터 임베딩 생성
   - 벡터 DB 저장 및 관리

5. **agents/writing_agent.py**:
   - 논문 섹션별 작성 로직
   - 자료 인용 및 참고
   - RAG 기반 콘텐츠 생성
   - 논문 구조 관리

6. **graph/workflow.py**:
   - LangGraph 기반 워크플로우 정의
   - 노드 및 엣지 구성
   - 상태 전이 로직
   - 조건부 분기 처리

7. **utils/vector_db.py**:
   - 벡터 DB 초기화 및 연결
   - 문서 저장 및 검색
   - 임베딩 생성 및 관리
   - 관련성 점수 계산

8. **models/state.py**:
   - 워크플로우 상태 정의
   - 상태 업데이트 및 조회
   - 직렬화 및 역직렬화
   - 체크포인트 관리

## 구현 시 고려사항

1. 모듈화 및 확장성을 고려한 설계
2. 명확한 에러 처리 및 로깅
3. API 키 보안 관리
4. 비용 효율적인 API 사용
5. 프로세스 중단 및 재개 기능
6. 단위 테스트 및 통합 테스트

이 프로젝트는 LangGraph를 기반으로 하는 복잡한 AI 에이전트 시스템이므로, 각 컴포넌트 간의 인터페이스를 명확히 정의하고, 상태 관리에 특별한 주의를 기울여야 합니다. 또한 API 호출 비용과 효율성을 고려한 설계가 중요합니다.
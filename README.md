# AI 논문 작성 지원 시스템

인공지능을 활용한 학술 논문 작성 지원 시스템입니다. 연구 주제에 대한 자료 수집, 분석 및 논문 작성을 자동화합니다.

## 기능

- 연구 주제에 대한 학술 자료 자동 수집
- 관련 논문 분석 및 요약
- 문헌 리뷰 자동 생성
- 연구 계획 수립 지원
- 논문 초안 작성

## 설치 방법

1. 저장소 복제
```bash
git clone https://github.com/username/paper_agent.git
cd paper_agent
```

2. 필요한 패키지 설치
```bash
pip install -r requirements.txt

# 또는 필요한 패키지를 직접 설치
pip install openai langchain faiss-cpu python-dotenv markdown
```

3. 환경 변수 설정
`.env` 파일을 생성하고 필요한 API 키를 설정하세요:
```
OPENAI_API_KEY=your_openai_api_key
SERPAPI_API_KEY=your_serpapi_api_key
```

## 사용 방법

시스템은 웹 인터페이스 또는 콘솔 모드로 실행할 수 있습니다.

### 통합 실행 스크립트 (run.py)

`run.py` 스크립트는 시스템의 주요 진입점으로, 웹 모드와 콘솔 모드를 모두 지원합니다.

#### 웹 인터페이스 모드

웹 브라우저를 통해 시스템과 상호작용할 수 있습니다:

```bash
python run.py --mode=web --port=8080 --host=0.0.0.0
```

옵션:
- `--port`: 웹 서버 포트 (기본값: 8080)
- `--host`: 웹 서버 호스트 (기본값: 0.0.0.0)
- `--debug`: Flask 디버그 모드 활성화

웹 브라우저에서 `http://localhost:8080`에 접속하여 시스템을 사용할 수 있습니다.

#### 콘솔 모드

명령줄에서 직접 시스템을 실행할 수 있습니다:

```bash
python run.py --mode=console --topic="인공지능 윤리" --paper-type=literature_review
```

옵션:
- `--topic`: 연구 주제
- `--paper-type`: 논문 유형 (literature_review, experimental_research, case_study, theoretical_analysis)
- `--output-format`: 출력 형식 (markdown, docx, pdf, html)
- `--requirements`: 사용자 요구사항 파일 경로

### 디버깅 스크립트 (debug_agent.py)

논문 작성 과정의 각 단계를 디버깅할 수 있는 스크립트입니다:

```bash
python debug_agent.py
```

이 스크립트는 기본적으로 "토픽 모델링의 역사와 발전"이라는 주제로 문헌 리뷰를 생성합니다.

## 프로젝트 구조

- `agents/`: 시스템에서 사용되는 AI 에이전트 구현
  - `research_agent.py`: 연구 에이전트 (자료 수집 및 분석)
  - `writing_agent.py`: 작성 에이전트 (논문 초안 작성)
  - `coordinator_agent.py`: 조정 에이전트 (전체 프로세스 관리)

- `utils/`: 유틸리티 함수 모음
  - `vector_db.py`: 벡터 데이터베이스 관련 기능
  - `pdf_processor.py`: PDF 파일 처리 기능
  - `search_tools.py`: 학술 검색 도구
  - `requirements_utils.py`: 사용자 요구사항 처리 도구

- `web/`: 웹 인터페이스 구현
  - `app.py`: Flask 웹 애플리케이션
  - `templates/`: HTML 템플릿
  - `static/`: 정적 파일 (CSS, JS 등)

- `config/`: 설정 파일
  - `settings.py`: 시스템 설정
  - `api_keys.py`: API 키 관리

- `data/`: 데이터 저장소
  - `papers.json`: 수집된 논문 정보

- `output/`: 생성된 문서 저장소

- `run.py`: 통합 실행 스크립트 (웹/콘솔 모드)
- `debug_agent.py`: 디버깅용 스크립트

## 라이선스

MIT License 
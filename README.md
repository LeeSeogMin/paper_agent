# API 설정 가이드

## API 선택 설정

이 프로젝트는 OpenAI API, xAI API, Anthropic API를 모두 지원합니다. 사용자는 환경 변수를 통해 사용할 API를 선택할 수 있습니다.

### 환경 변수 설정

`.env` 파일에 다음 환경 변수를 추가하세요:

```
# API 선택 (true 또는 false)
USE_OPENAI_API=false
USE_XAI_API=false
USE_ANTHROPIC_API=true

# OpenAI API 설정
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.7

# xAI API 설정
XAI_API_KEY=your_xai_api_key
XAI_MODEL=grok-2
XAI_TEMPERATURE=0.7

# Anthropic API 설정
ANTHROPIC_API_KEY=your_anthropic_api_key
ANTHROPIC_MODEL=claude-3-7-sonnet-20240229
ANTHROPIC_TEMPERATURE=0.7
```

### API 선택 방법

- OpenAI API만 사용하려면: `USE_OPENAI_API=true`, `USE_XAI_API=false`, `USE_ANTHROPIC_API=false`
- xAI API만 사용하려면: `USE_OPENAI_API=false`, `USE_XAI_API=true`, `USE_ANTHROPIC_API=false`
- Anthropic API만 사용하려면: `USE_OPENAI_API=false`, `USE_XAI_API=false`, `USE_ANTHROPIC_API=true`
- 여러 API를 활성화한 경우 우선순위는 OpenAI > xAI > Anthropic 순입니다.

### 지원되는 모델

- OpenAI: gpt-4, gpt-4o, gpt-4o-mini, gpt-3.5-turbo 등
- xAI: grok-2
- Anthropic: claude-3-7-sonnet-20240229, claude-3-opus-20240229, claude-3-haiku-20240307 등

## API 테스트

API 설정이 올바르게 되었는지 확인하려면 다음 명령어를 실행하세요:

```
python test_api_selection.py
```

이 스크립트는 현재 API 설정을 출력하고, 선택된 API로 에이전트를 초기화하여 테스트합니다. 
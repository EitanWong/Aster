<div align="center">
  <img src="assets/logo.svg" alt="Aster Logo" width="200" height="200">

  # Aster

  **프로덕션 지향 Apple Silicon 로컬 LLM 추론 런타임**

  [English](README.md) | [中文](README.zh.md) | [日本語](README.ja.md) | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | [한국어](README.ko.md)
</div>

Aster는 장문맥, OpenClaw 스타일의 에이전트 워크로드에 최적화된 프로덕션 지향 Apple Silicon 로컬 LLM 추론 런타임입니다.

## Aster를 선택하는 이유

Aster는 다음에 최적화되어 있습니다:

- 거대한 프롬프트 및 반복되는 긴 접두사
- 도구 집약적 에이전트 프롬프트
- 긴 대화
- 지속적인 로컬 백그라운드 서빙
- 벤치마크 검증된 런타임 정책 선택
- Apple Silicon + MLX 배포

OpenAI 호환 API를 노출하고 고급 최적화를 교리가 아닌 후보 전략으로 취급합니다. 추측 디코딩, 접두사 캐싱, 배치 처리, 스케줄링 및 스트리밍 케이던스는 모두 벤치마크되고 측정된 로컬 성능 및 안정성을 기반으로 선택됩니다.

## 핵심 아이디어

- 스트리밍 및 비스트리밍 엔드포인트가 있는 OpenAI 호환 API
- 명시적 프리필/디코드 분할
- 큐 인식 배치 처리가 있는 적응형 스케줄러
- 페이징된 KV 관리자 추상화
- 결정론적 해싱이 있는 자동 접두사 캐시
- 자동 비활성화 폴백이 있는 추측 디코딩 컨트롤러
- 가장 빠르고 안정적인 프로필을 유지하는 벤치마크/자동 조정 하위 시스템
- 구조화된 로그, 메트릭, 감독 및 준비/상태 보고

## 빠른 시작

```bash
cd /Users/eitan/Documents/Projects/Python/Aster

# 가상 환경 생성
/opt/homebrew/bin/python3.13 -m venv .venv
source .venv/bin/activate

# 의존성 설치 (ASR/TTS용 mlx-audio 포함)
python -m pip install -r requirements.txt

# 모델 다운로드 (ASR, LLM, TTS)
bash scripts/setup/download_models.sh

# 서버 시작
python -m aster --config configs/config.yaml
```

API는 `http://127.0.0.1:8080`에서 사용 가능합니다

### 설치 확인

```bash
# 상태 확인
curl http://127.0.0.1:8080/health

# LLM 추론 테스트
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B",
    "messages": [{"role": "user", "content": "안녕하세요"}],
    "max_tokens": 100
  }'

# ASR (음성 텍스트 변환) 테스트
python scripts/test_audio_cli.py --tts "안녕하세요 세계" --output test.wav
python scripts/test_audio_cli.py --asr test.wav

# 엔드투엔드 파이프라인 테스트
python scripts/test_audio_cli.py --pipeline "이것은 테스트입니다"
```

## Python 버전

Aster는 최신 Python을 대상으로 하며 가능한 경우 Python 3.13.x에서 실행해야 합니다(최소 3.12+). macOS 시스템 Python은 이 프로젝트에서 지원되지 않는 것으로 간주됩니다.

## API

- `GET /health`
- `GET /ready`
- `GET /metrics`
- `GET /v1/models`
- `POST /v1/chat/completions` — LLM 채팅 추론
- `POST /v1/completions` — LLM 텍스트 완성
- `POST /v1/audio/transcriptions` — ASR (음성 텍스트 변환)
- `POST /v1/audio/speech` — TTS (텍스트 음성 변환)

호환성 참고:
- Aster의 기본 호환성 계약 및 선택적 디버그 확장은 `docs/api/OPENAI_COMPAT.md`를 참조하세요.

## 오디오 서비스 (ASR & TTS)

Aster는 Qwen3 모델로 구동되는 통합 음성 인식 및 합성을 포함합니다:

### ASR (음성 텍스트 변환)
- 모델: Qwen3-ASR-0.6B (0.66GB)
- 여러 언어 지원
- 빠른 로컬 전사

### TTS (텍스트 음성 변환)
- 기본 모델: Qwen3-TTS-0.6B (1.59GB)
- CustomVoice 모델: Qwen3-TTS-CustomVoice-0.6B (선택사항, 음성 복제용)
- 조정 가능한 음성 속도
- 참조 오디오를 사용한 음성 복제

### 오디오 API 예제

**TTS (텍스트 음성 변환):**
```bash
curl -X POST http://127.0.0.1:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-TTS-0.6B",
    "input": "안녕하세요, 이것은 테스트입니다",
    "voice": "default",
    "speed": 1.0
  }' \
  --output output.wav
```

**ASR (음성 텍스트 변환):**
```bash
curl -X POST http://127.0.0.1:8080/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=Qwen3-ASR-0.6B"
```

### 오디오 테스트

제공된 CLI 테스트 도구를 사용합니다:
```bash
# TTS 테스트
python scripts/test_audio_cli.py --tts "안녕하세요 세계" --output output.wav

# ASR 테스트
python scripts/test_audio_cli.py --asr output.wav

# 엔드투엔드 파이프라인 테스트 (TTS -> ASR)
python scripts/test_audio_cli.py --pipeline "테스트 메시지"

# 전체 테스트 스위트 실행
pytest tests/test_audio_services.py -v -s
```

자세한 오디오 서비스 문서는 `docs/guides/DEPLOYMENT.md`를 참조하세요.

## 벤치마크 철학

시작 자동 조정은 짧은 워밍업 벤치마크를 실행하여 가장 빠르고 안정적인 정책을 선택할 수 있습니다. 벤치마크 하위 시스템은 다음을 비교합니다:

- 추측 디코딩 켜기/끄기
- 드래프트 토큰 수
- 접두사 캐싱 켜기/끄기
- 배치 윈도우
- 배치 상한
- 페이지 크기
- 스케줄링 모드
- 스트리밍 플러시 케이던스

프로필은 유지되고 후속 시작 시 사용됩니다.

## Apple Silicon 튜닝 참고

- 반복되는 동적 할당보다 사전 할당 및 페이지 풀 선호
- 통합 메모리 스래싱을 피하기 위해 MLX 모델 상주를 신중하게 사용
- 머신별 접두사 캐싱 및 추측 디코딩 벤치마크
- Python 핫 경로를 작게 유지; 조정을 안정적인 루프로 이동
- 긴 프롬프트 아래에서 일관된 첫 토큰 지연 시간 우선 순위

## 동적 최적화 철학

Aster는 로컬 머신에서 유리한 것으로 입증된 최적화만 활성화합니다:

- 추측 디코딩은 전역적으로 또는 요청 클래스별로 비활성화할 수 있습니다
- 히트율이 낮거나 메모리 압력이 증가할 때 접두사 캐시를 줄이거나 비활성화할 수 있습니다
- 지연 시간이 증가하면 배치 윈도우가 자동으로 축소됩니다
- 불안정성 또는 회귀가 감지되면 폴백 프로필이 선택됩니다

## 모델 설정

hfd + aria2 가속을 사용한 원클릭 모델 다운로드:

```bash
# 모든 필수 모델 다운로드 (ASR, LLM, TTS)
bash scripts/setup/download_models.sh

# 또는 더 많은 제어를 위해 Python을 직접 사용
python scripts/download_models.py --all
python scripts/download_models.py --group llm
python scripts/download_models.py --list
```

자세한 지침은 `scripts/setup/README-model-download.md`를 참조하세요.

## 모델 경로

`model.path` 및 `model.draft_path`는 다음 중 하나일 수 있습니다:
- MLX 변환 모델 디렉토리에 대한 절대 로컬 경로
- `mlx-lm`으로 로드 가능한 호환 Hugging Face 리포지토리 ID

프로덕션의 경우 로컬 MLX 변환 디렉토리를 선호합니다. `configs/config.yaml` 업데이트:

```yaml
model:
  path: models/qwen3.5-9b-mlx
  draft_path: models/qwen3.5-0.8b-mlx

audio:
  asr_model_path: models/qwen3-asr-0.6b
  tts_model_path: models/qwen3-tts-0.6b-base
```

## OpenClaw 통합

OpenClaw를 Aster의 OpenAI 호환 기본 URL 및 모델 ID로 지정합니다. Aster는 반복되는 시스템/도구 접두사 및 장기 에이전트 세션을 위해 구축되었으므로 안정적인 스캐폴딩 및 장문맥 재사용이 있는 워크로드에서 특히 이점을 얻을 수 있습니다.

## 프로젝트 지침 문서

- `docs/guides/QUICK_START_MODELS.md` — 모델 다운로드 빠른 시작 가이드
- `docs/reference/MODEL_SETUP.md` — 자세한 설정 및 문제 해결
- `docs/development/MODEL_DOWNLOAD_ARCHITECTURE.md` — 시스템 설계
- `docs/reference/ROADMAP.md` — 장기 아키텍처 진화 계획
- `docs/api/OPENAI_COMPAT.md` — 호환성 경계 및 디버그 확장
- `docs/development/DEBUGGING.md` — 운영자 디버깅 가이드
- `docs/operations/OPERATIONS.md` — 일일 서비스 운영
- `docs/guides/BENCHMARK_GUIDE.md` — 성능 벤치마크 가이드
- `docs/guides/BACKGROUND_SERVICE_SETUP.md` — 백그라운드 서비스 설정
- `DOCS.md` — 완전한 문서 네비게이션

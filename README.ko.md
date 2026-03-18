<div align="center">
  <img src="assets/logo.svg" alt="Aster Logo" width="200" height="200">
  
  # Aster
  
  **Apple Silicon용 프로덕션 지향 로컬 LLM 추론 런타임**
  
  [English](README.md) | [中文](README.zh.md) | [日本語](README.ja.md) | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md) | [한국어](README.ko.md)
</div>

Aster는 긴 컨텍스트와 OpenClaw 스타일의 에이전트 워크로드를 위해 최적화된 Apple Silicon용 프로덕션 지향 로컬 LLM 추론 런타임입니다.

## Aster를 선택하는 이유

Aster는 다음을 위해 최적화되었습니다:

- 거대한 프롬프트와 반복되는 긴 접두사
- 도구 집약적인 에이전트 프롬프트
- 긴 대화
- 지속적인 로컬 백그라운드 서빙
- 벤치마크 검증된 런타임 정책 선택
- Apple Silicon + MLX 배포

OpenAI 호환 API를 제공하며 고급 최적화를 교리가 아닌 후보 전략으로 취급합니다. 추측 디코딩, 접두사 캐싱, 배치 처리, 스케줄링 및 스트리밍 속도는 모두 벤치마크되고 측정된 로컬 성능 및 안정성을 기반으로 선택됩니다.

## 핵심 기능

- 스트리밍 및 비스트리밍 엔드포인트가 있는 OpenAI 호환 API
- 명시적 prefill/decode 분리
- 큐 인식 적응형 스케줄러
- 페이징된 KV 관리자 추상화
- 결정론적 해싱을 사용한 자동 접두사 캐싱
- 자동 비활성화 폴백이 있는 추측 디코딩 컨트롤러
- 가장 빠르고 안정적인 프로필을 유지하는 벤치마크/자동 튜닝 서브시스템
- 구조화된 로그, 메트릭, 감시 및 준비/상태 보고

## 빠른 시작

```bash
cd /Users/eitan/Documents/Projects/Python/Aster
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp configs/config.yaml.example configs/config.yaml
python -m aster --config configs/config.yaml
```

## Python 버전

Aster는 최신 Python을 대상으로 하며 Python 3.13.x(사용 가능한 경우) 또는 3.12 이상에서 실행해야 합니다. macOS 시스템 Python은 이 프로젝트에서 지원되지 않습니다.

## API 엔드포인트

- `GET /health` - 상태 확인
- `GET /ready` - 준비 상태 확인
- `GET /metrics` - Prometheus 메트릭
- `GET /v1/models` - 모델 목록
- `POST /v1/chat/completions` - 채팅 완성
- `POST /v1/completions` - 텍스트 완성

호환성 참고:
- Aster의 기본 호환성 계약 및 선택적 디버그 확장에 대해서는 `docs/OPENAI_COMPAT.md`를 참조하세요.

## 벤치마크 철학

시작 자동 튜닝은 짧은 워밍업 벤치마크를 실행하여 가장 빠르고 안정적인 정책을 선택할 수 있습니다. 벤치마크 서브시스템은 다음을 비교합니다:

- 추측 디코딩 켜기/끄기
- 드래프트 토큰 수
- 접두사 캐싱 켜기/끄기
- 배치 윈도우
- 배치 상한
- 페이지 크기
- 스케줄링 모드
- 스트리밍 플러시 속도

프로필은 유지되며 후속 시작 시 사용됩니다.

## Apple Silicon 튜닝 참고

- 반복된 동적 할당보다 사전 할당 및 페이지 풀 선호
- 통합 메모리 스래싱을 피하기 위해 MLX 모델 상주를 신중하게 사용
- 머신별 접두사 캐싱 및 추측 디코딩 벤치마크
- Python 핫 경로를 작게 유지; 조정을 안정적인 루프로 이동
- 긴 프롬프트 아래에서 일관된 첫 토큰 지연 시간 우선 순위 지정

## 동적 최적화 철학

Aster는 로컬 머신에서 유리한 것으로 입증된 최적화만 활성화합니다:

- 추측 디코딩은 전역적으로 또는 요청 클래스별로 비활성화할 수 있습니다
- 히트율이 낮거나 메모리 압력이 증가할 때 접두사 캐싱을 줄이거나 비활성화할 수 있습니다
- 지연 시간이 증가하면 배치 윈도우가 자동으로 축소됩니다
- 불안정성 또는 회귀가 감지되면 폴백 프로필이 선택됩니다

## 모델 경로

`model.path` 및 `model.draft_path`는 다음 중 하나일 수 있습니다:
- MLX 변환 모델 디렉토리에 대한 절대 로컬 경로
- `mlx-lm`으로 로드 가능한 호환 Hugging Face 리포지토리 ID

의도된 프로덕션 설정의 경우 9B 대상 모델과 0.8B 드래프트 모델 모두에 대해 로컬 MLX 변환 디렉토리를 선호합니다.

유용한 설정 및 검증 명령:

```bash
bash scripts/setup/download_models.sh
# 또는 더 다운로드 복원력 있는 경로:
USE_HFD=1 bash scripts/setup/download_models.sh
source .venv/bin/activate
python scripts/dev/model_smoke.py --config configs/config.yaml
python scripts/dev/benchmark_live.py --config configs/config.yaml
```

## OpenClaw 통합

OpenClaw를 Aster의 OpenAI 호환 기본 URL 및 모델 ID로 지정합니다. Aster는 반복되는 시스템/도구 접두사 및 장기 에이전트 세션을 위해 구축되었으므로 안정적인 스캐폴딩 및 긴 컨텍스트 재사용이 있는 워크로드에서 특히 이점을 얻을 수 있습니다.

## 프로젝트 문서

- `docs/ROADMAP.md` — 장기 아키텍처 진화 계획
- `docs/OPENAI_COMPAT.md` — 호환성 경계 및 디버그 확장 규칙
- `docs/DEBUGGING.md` — 운영자 디버깅 가이드
- `docs/OPERATIONS.md` — 일일 서비스 운영
- `docs/DEVELOPMENT.md` — 개발 가이드

## 라이선스

MIT License - [LICENSE](LICENSE) 참조

## 기여

기여를 환영합니다! [CONTRIBUTING.ko.md](CONTRIBUTING.ko.md)에서 기여 지침을 참조하세요.

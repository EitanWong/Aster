# Aster Operations

Minimal ops skeleton for running Aster like a local service.

## Paths

- Project root: `/Users/eitan/Documents/Projects/Python/Aster`
- Default config: `configs/config.yaml`
- PID file: `run/aster.pid`
- Log file: `logs/aster.log`

## First-time setup

```bash
cd /Users/eitan/Documents/Projects/Python/Aster
/opt/homebrew/bin/python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp configs/config.yaml.example configs/config.yaml
```

## Core commands

### Start in background

```bash
bash scripts/start.sh
```

### Start in foreground

```bash
bash scripts/start.sh configs/config.yaml --foreground
```

### Stop

```bash
bash scripts/stop.sh
```

### Restart

```bash
bash scripts/restart.sh
```

### Status

```bash
bash scripts/status.sh
```

### Health + metrics

```bash
bash scripts/health.sh
```

### Smoke test

```bash
bash scripts/smoke_test.sh
```

### Tail logs

```bash
bash scripts/tail_logs.sh
```

### Interactive local chat CLI

```bash
bash scripts/chat.sh --repl --stream
```

Note: `chat.sh` opts into Aster debug streaming metadata with `X-Aster-Debug: 1`, so the CLI can display real service-side generation stats. Default API behavior remains OpenAI-compatible unless that header is explicitly sent.

Useful options:

```bash
bash scripts/chat.sh --repl --stream --save-transcript run/session.json
bash scripts/chat.sh --load-transcript run/session.json --repl --stream
bash scripts/chat.sh --stream --format json "Reply with exactly: Aster is online"
bash scripts/chat.sh --stream --stats-only "Reply with exactly: Aster is online"
```

REPL commands:
- `/stats`
- `/save <path>`
- `/load <path>`
- `/messages`
- `/metrics`
- `/reset`
- `/quit`

### One-shot local prompt

```bash
bash scripts/chat.sh "Reply with exactly: Aster is online"
```

## Alternate config

All scripts accept an optional config path as the first argument:

```bash
bash scripts/start.sh configs/config.yaml
bash scripts/status.sh configs/config.yaml
```

You can also export:

```bash
export ASTER_CONFIG_PATH=configs/config.yaml
```

## What status checks

`status.sh` reports:

- runtime paths
- whether the PID is alive
- `/health`
- `/ready`
- `/v1/models`
- last 20 lines of the log file

`metrics_summary.sh` reports:

- average request latency
- average first-token latency
- prefill / decode average latency
- average batch size
- queue depth
- KV pages used
- prefix cache hits / misses / hit rate
- speculative acceptance average
- worker restarts
- total errors

## What smoke test checks

`smoke_test.sh` verifies:

1. `/health`
2. `/v1/models`
3. one non-streaming `/v1/chat/completions` call

This is the quickest "is the service actually usable?" validation.

## Current operational notes

- Logs currently go to `logs/aster.log`
- Background mode uses `nohup`
- If Aster crashes and leaves a stale PID file behind, `stop.sh` and `status.sh` handle cleanup safely
- Current persisted tuning profile lives at `configs/autotune_profile.json`

## Recommended operator flow

```bash
bash scripts/start.sh
bash scripts/status.sh
bash scripts/smoke_test.sh
bash scripts/tail_logs.sh
```

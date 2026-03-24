#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# Aster model download helper for macOS / Apple Silicon.
#
# Features:
# - downloads LLM, ASR, TTS, and embedding models
# - uses hfd + aria2 for fast accelerated parallel downloads
# - supports HF mirror for mainland China
# - one-click environment setup (Homebrew, Python, venv)
#
# Usage:
#   bash scripts/setup/download_models.sh                        # interactive menu
#   bash scripts/setup/download_models.sh --list                 # list all available models
#   bash scripts/setup/download_models.sh --model llm:qwen35_9b  # download a specific model
#   bash scripts/setup/download_models.sh --model llm:qwen35_9b --show-config
#
# Environment variables:
#   HF_ENDPOINT    HuggingFace endpoint (default: https://hf-mirror.com for CN mirror)
#   USE_HFD        Use hfd+aria2 accelerated downloads, 1=yes 0=no (default: 1)
#   INSTALL_ARIA2  Auto-install aria2 via Homebrew if missing (default: 1)
#   HF_TOKEN       HuggingFace access token for gated models

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"
DOWNLOADER="$SCRIPTS_DIR/setup/download_model_interactive.py"
MODELS_DIR="$PROJECT_ROOT/models"
VENV_PATH="$PROJECT_ROOT/.venv"
TOOLS_DIR="$PROJECT_ROOT/tools"
HFD_PATH="$TOOLS_DIR/hfd.sh"

# ── Configuration ───────────────────────────────────────────────────────────
HF_ENDPOINT_VALUE="${HF_ENDPOINT:-https://hf-mirror.com}"
USE_HFD="${USE_HFD:-1}"
INSTALL_ARIA2="${INSTALL_ARIA2:-1}"
HF_TOKEN="${HF_TOKEN:-}"

BREW_BIN=""
PYTHON_BIN=""

# ── Logging ─────────────────────────────────────────────────────────────────
log()  { printf '\n[Aster] %s\n' "$*"; }
warn() { printf '[Aster][warn] %s\n' "$*" >&2; }
err()  { printf '[Aster][error] %s\n' "$*" >&2; }

on_error() {
  local exit_code=$?
  err "Setup failed on line ${BASH_LINENO[0]} with exit code ${exit_code}."
  err "Fix the issue above and re-run the script."
  exit "$exit_code"
}
trap on_error ERR

# ── Prerequisite checks ──────────────────────────────────────────────────────
require_macos() {
  if [[ "$(uname -s)" != "Darwin" ]]; then
    err "This script is for macOS only."
    exit 1
  fi
}

require_project_root() {
  if [[ ! -d "$PROJECT_ROOT" ]]; then
    err "Project root not found: $PROJECT_ROOT"
    exit 1
  fi
  if [[ ! -f "$DOWNLOADER" ]]; then
    err "Python downloader not found: $DOWNLOADER"
    err "Expected: scripts/setup/download_model_interactive.py"
    exit 1
  fi
}

# ── Homebrew ─────────────────────────────────────────────────────────────────
ensure_brew() {
  if command -v brew >/dev/null 2>&1; then
    BREW_BIN="$(command -v brew)"
    log "Homebrew found: $BREW_BIN"
    return
  fi

  warn "Homebrew not found. Installing automatically."
  warn "This may prompt for your macOS password and can take a few minutes."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

  if   [[ -x /opt/homebrew/bin/brew ]]; then BREW_BIN="/opt/homebrew/bin/brew"
  elif [[ -x /usr/local/bin/brew    ]]; then BREW_BIN="/usr/local/bin/brew"
  else
    err "Homebrew installed but 'brew' not found on PATH."
    err "Open a new shell and re-run this script."
    exit 1
  fi

  eval "$($BREW_BIN shellenv)"
  log "Homebrew installed: $BREW_BIN"
}

# ── Python ───────────────────────────────────────────────────────────────────
python_version_ok() {
  "$1" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 12) else 1)
PY
}

ensure_python() {
  local candidates=(
    python3.13
    python3.12
    python3
    /opt/homebrew/bin/python3.13
    /opt/homebrew/bin/python3.12
  )

  for py in "${candidates[@]}"; do
    local resolved
    if [[ "$py" == /* ]]; then
      resolved="$py"
    else
      resolved="$(command -v "$py" 2>/dev/null || true)"
    fi
    if [[ -n "$resolved" && -x "$resolved" ]] && python_version_ok "$resolved"; then
      PYTHON_BIN="$resolved"
      log "Using Python: $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"
      return
    fi
  done

  warn "No suitable Python 3.12+ found. Installing python@3.13 via Homebrew."
  ensure_brew
  "$BREW_BIN" install python@3.13

  for p in /opt/homebrew/bin/python3.13 "$(command -v python3.13 2>/dev/null || true)"; do
    if [[ -n "$p" && -x "$p" ]] && python_version_ok "$p"; then
      PYTHON_BIN="$p"
      break
    fi
  done

  if [[ -z "$PYTHON_BIN" ]]; then
    err "Python 3.13 installed but no usable executable found."
    exit 1
  fi

  log "Installed Python: $PYTHON_BIN"
}

# ── Virtualenv ───────────────────────────────────────────────────────────────
ensure_venv() {
  if [[ ! -d "$VENV_PATH" ]]; then
    log "Creating virtual environment: $VENV_PATH"
    "$PYTHON_BIN" -m venv "$VENV_PATH"
  fi

  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"
  log "Active Python: $(python --version 2>&1)"

  python -m pip install --quiet --upgrade pip setuptools wheel
}

# ── Python dependencies ───────────────────────────────────────────────────────
ensure_dependencies() {
  log "Installing Python dependencies (PyYAML, huggingface-hub)..."
  python -m pip install --quiet PyYAML huggingface-hub
}

# ── HF endpoint ───────────────────────────────────────────────────────────────
configure_hf_endpoint() {
  export HF_ENDPOINT="$HF_ENDPOINT_VALUE"
  log "HF_ENDPOINT: $HF_ENDPOINT"
}

# ── aria2 ────────────────────────────────────────────────────────────────────
ensure_aria2() {
  if command -v aria2c >/dev/null 2>&1; then
    log "aria2 already available: $(command -v aria2c)"
    return
  fi
  ensure_brew
  log "Installing aria2 via Homebrew for accelerated parallel downloads..."
  "$BREW_BIN" install aria2 || warn "aria2 installation failed; downloads will still work, just slower."
}

# ── hfd helper ───────────────────────────────────────────────────────────────
ensure_hfd() {
  mkdir -p "$TOOLS_DIR"

  if [[ -f "$HFD_PATH" ]]; then
    log "hfd helper already present: $HFD_PATH"
    return
  fi

  log "Downloading hfd helper to $HFD_PATH"
  if ! curl -fsSL "$HF_ENDPOINT_VALUE/hfd/hfd.sh" -o "$HFD_PATH" 2>/dev/null; then
    # Fallback: fetch directly from GitHub
    warn "Mirror download failed; falling back to GitHub."
    curl -fsSL "https://raw.githubusercontent.com/LetheanVPN/hfd/main/hfd.sh" -o "$HFD_PATH"
  fi
  chmod +x "$HFD_PATH"
  log "hfd downloaded: $HFD_PATH"
}

# ── Preflight summary ─────────────────────────────────────────────────────────
show_preflight() {
  log "Configuration:"
  printf '  %-16s %s\n' "Project root:"  "$PROJECT_ROOT"
  printf '  %-16s %s\n' "Models dir:"    "$MODELS_DIR"
  printf '  %-16s %s\n' "HF endpoint:"   "$HF_ENDPOINT_VALUE"
  printf '  %-16s %s\n' "Accelerated:"   "$([ "$USE_HFD" = "1" ] && echo "hfd+aria2" || echo "huggingface-hub")"
  printf '  %-16s %s\n' "Python:"        "$PYTHON_BIN"
  printf '  %-16s %s\n' "HF token:"      "$([ -n "$HF_TOKEN" ] && echo "set" || echo "not set (gated models may fail)")"
}

# ── Run Python downloader ─────────────────────────────────────────────────────
run_downloader() {
  log "Launching model downloader..."
  cd "$PROJECT_ROOT"

  [[ -n "$HF_TOKEN" ]] && export HF_TOKEN && log "HF_TOKEN exported."
  [[ "$USE_HFD" == "1" ]] && export HFD_PATH

  python "$DOWNLOADER" "$@"
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
  require_macos
  require_project_root

  ensure_brew
  ensure_python
  ensure_venv
  ensure_dependencies
  configure_hf_endpoint

  if [[ "$INSTALL_ARIA2" == "1" ]]; then
    ensure_aria2
  fi
  if [[ "$USE_HFD" == "1" ]]; then
    ensure_hfd
  fi

  show_preflight

  # No args → interactive menu
  run_downloader "$@"
}

main "$@"

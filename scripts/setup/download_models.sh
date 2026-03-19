#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# Aster model download helper for macOS / Apple Silicon.
#
# Features:
# - downloads all required models (ASR, LLM, TTS)
# - uses hfd + aria2 for fast accelerated downloads
# - supports HF mirror for mainland China
# - verifies downloads
# - one-click setup
#
# Usage:
#   bash scripts/download_models.sh                    # Download all required
#   bash scripts/download_models.sh --all              # Download all (required + optional)
#   bash scripts/download_models.sh --group llm        # Download LLM only
#   bash scripts/download_models.sh --group asr        # Download ASR only
#   bash scripts/download_models.sh --group tts        # Download TTS only
#   bash scripts/download_models.sh --list             # List available models
#   bash scripts/download_models.sh --verify-only      # Verify existing models

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"
MODELS_DIR="$PROJECT_ROOT/models"
VENV_PATH="$PROJECT_ROOT/.venv"
TOOLS_DIR="$PROJECT_ROOT/tools"
HFD_PATH="$TOOLS_DIR/hfd.sh"

# Configuration
HF_ENDPOINT_VALUE="${HF_ENDPOINT:-https://hf-mirror.com}"
USE_HFD="${USE_HFD:-1}"
INSTALL_ARIA2="${INSTALL_ARIA2:-1}"
HF_TOKEN="${HF_TOKEN:-}"

BREW_BIN=""
PYTHON_BIN=""

log() {
  printf '\n[Aster] %s\n' "$*"
}

warn() {
  printf '[Aster][warn] %s\n' "$*" >&2
}

err() {
  printf '[Aster][error] %s\n' "$*" >&2
}

on_error() {
  local exit_code=$?
  err "Setup failed on line ${BASH_LINENO[0]} with exit code ${exit_code}."
  err "You can re-run the script after fixing the issue."
  exit "$exit_code"
}
trap on_error ERR

require_macos() {
  if [[ "$(uname -s)" != "Darwin" ]]; then
    err "This setup script is intended for macOS only."
    exit 1
  fi
}

require_project_root() {
  if [[ ! -d "$PROJECT_ROOT" ]]; then
    err "Project root not found: $PROJECT_ROOT"
    exit 1
  fi
  if [[ ! -f "$SCRIPTS_DIR/download_models.py" ]]; then
    err "Python downloader not found: $SCRIPTS_DIR/download_models.py"
    exit 1
  fi
}

ensure_brew() {
  if command -v brew >/dev/null 2>&1; then
    BREW_BIN="$(command -v brew)"
    log "Homebrew found at $BREW_BIN"
    return
  fi

  warn "Homebrew not found. Installing Homebrew automatically."
  warn "This may prompt for your macOS password and can take a few minutes."

  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

  if [[ -x /opt/homebrew/bin/brew ]]; then
    BREW_BIN="/opt/homebrew/bin/brew"
  elif [[ -x /usr/local/bin/brew ]]; then
    BREW_BIN="/usr/local/bin/brew"
  else
    err "Homebrew installation completed but brew was not found on PATH."
    err "Try opening a new shell and re-running this script."
    exit 1
  fi

  eval "$($BREW_BIN shellenv)"
  log "Homebrew installed at $BREW_BIN"
}

python_version_ok() {
  local py="$1"
  "$py" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 12) else 1)
PY
}

ensure_python() {
  local candidates=()

  if command -v python3.13 >/dev/null 2>&1; then
    candidates+=("$(command -v python3.13)")
  fi
  if command -v python3.12 >/dev/null 2>&1; then
    candidates+=("$(command -v python3.12)")
  fi
  if command -v python3 >/dev/null 2>&1; then
    candidates+=("$(command -v python3)")
  fi
  if [[ -x /opt/homebrew/bin/python3.13 ]]; then
    candidates+=("/opt/homebrew/bin/python3.13")
  fi
  if [[ -x /opt/homebrew/bin/python3.12 ]]; then
    candidates+=("/opt/homebrew/bin/python3.12")
  fi

  for py in "${candidates[@]:-}"; do
    if [[ -n "$py" ]] && python_version_ok "$py"; then
      PYTHON_BIN="$py"
      log "Using Python: $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"
      return
    fi
  done

  warn "No suitable Python 3.12+ found. Installing python@3.13 via Homebrew."
  ensure_brew
  "$BREW_BIN" install python@3.13

  if [[ -x /opt/homebrew/bin/python3.13 ]]; then
    PYTHON_BIN="/opt/homebrew/bin/python3.13"
  elif command -v python3.13 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3.13)"
  else
    err "Python 3.13 installation did not produce a usable executable."
    exit 1
  fi

  if ! python_version_ok "$PYTHON_BIN"; then
    err "Resolved Python is still below 3.12: $($PYTHON_BIN --version 2>&1)"
    exit 1
  fi

  log "Installed and selected Python: $PYTHON_BIN"
}

ensure_venv() {
  if [[ ! -d "$VENV_PATH" ]]; then
    log "Creating virtual environment at $VENV_PATH"
    "$PYTHON_BIN" -m venv "$VENV_PATH"
  fi

  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"

  log "Active Python: $(python --version 2>&1)"
  python -m pip install --upgrade pip setuptools wheel >/dev/null 2>&1
}

ensure_dependencies() {
  log "Installing dependencies..."
  python -m pip install -q PyYAML huggingface-hub
}

configure_hf_endpoint() {
  export HF_ENDPOINT="$HF_ENDPOINT_VALUE"
  log "HF_ENDPOINT set to: $HF_ENDPOINT"
}

ensure_aria2_if_requested() {
  if [[ "$INSTALL_ARIA2" != "1" ]]; then
    return
  fi
  if command -v aria2c >/dev/null 2>&1; then
    log "aria2 already available: $(command -v aria2c)"
    return
  fi
  ensure_brew
  log "Installing aria2 via Homebrew for accelerated downloads..."
  "$BREW_BIN" install aria2 || warn "aria2 installation failed; continuing without it"
}

ensure_hfd_if_requested() {
  if [[ "$USE_HFD" != "1" ]]; then
    return
  fi

  ensure_aria2_if_requested
  mkdir -p "$TOOLS_DIR"

  if [[ ! -f "$HFD_PATH" ]]; then
    log "Downloading hfd helper to $HFD_PATH"
    curl -fsSL "$HF_ENDPOINT_VALUE/hfd/hfd.sh" -o "$HFD_PATH"
    chmod +x "$HFD_PATH"
  else
    log "hfd helper already present: $HFD_PATH"
  fi
}

show_preflight() {
  log "Configuration:"
  log "  Project root:  $PROJECT_ROOT"
  log "  Models dir:    $MODELS_DIR"
  log "  HF mirror:     $HF_ENDPOINT_VALUE"
  log "  Use hfd:       $USE_HFD"
  log "  Python:        $PYTHON_BIN"
}

run_downloader() {
  local args=("$@")
  
  log "Running model downloader..."
  cd "$PROJECT_ROOT"
  
  # Set environment for hfd if requested
  if [[ "$USE_HFD" == "1" ]]; then
    export HFD_PATH="$HFD_PATH"
  fi
  
  python "$SCRIPTS_DIR/download_models.py" "${args[@]}"
}

main() {
  require_macos
  require_project_root
  
  # Parse arguments
  local args=()
  if [[ $# -eq 0 ]]; then
    # Default: download required models only
    args=("--required-only")
  else
    args=("$@")
  fi
  
  show_preflight
  ensure_brew
  ensure_python
  ensure_venv
  configure_hf_endpoint
  ensure_dependencies
  ensure_aria2_if_requested
  ensure_hfd_if_requested
  
  if [[ "$HF_TOKEN" != "" ]]; then
    export HF_TOKEN
    log "HF_TOKEN set from environment"
  fi
  
  run_downloader "${args[@]}"
}

main "$@"

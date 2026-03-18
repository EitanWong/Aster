#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# Aster model download + environment preparation helper for macOS / Apple Silicon.
#
# Features:
# - verifies macOS + project structure
# - checks for / installs Homebrew if missing
# - checks for modern Python (prefers 3.13, minimum 3.12)
# - checks for / creates the project venv if needed
# - installs huggingface_hub CLI into the venv if missing
# - supports both `hf` and `huggingface-cli`
# - uses HF mirror by default for faster downloads in mainland China
# - optional hfd + aria2 accelerated path
# - prepares model directories and config paths
# - downloads MLX-compatible Qwen models into local directories

PROJECT_ROOT="/Users/eitan/Documents/Projects/Python/Aster"
MODELS_DIR="$PROJECT_ROOT/models"
MAIN_DIR="$MODELS_DIR/qwen3.5-9b-mlx"
DRAFT_DIR="$MODELS_DIR/qwen3.5-0.8b-mlx"
CONFIG_PATH="$PROJECT_ROOT/configs/config.yaml"
EXAMPLE_CONFIG_PATH="$PROJECT_ROOT/configs/config.yaml.example"
VENV_PATH="$PROJECT_ROOT/.venv"
TOOLS_DIR="$PROJECT_ROOT/tools"
HFD_PATH="$TOOLS_DIR/hfd.sh"

MAIN_REPO="${MAIN_REPO:-mlx-community/Qwen3.5-9B-4bit}"
DRAFT_REPO="${DRAFT_REPO:-mlx-community/Qwen3.5-0.8B-4bit}"
HF_ENDPOINT_VALUE="${HF_ENDPOINT_VALUE:-https://hf-mirror.com}"
USE_HFD="${USE_HFD:-0}"
INSTALL_ARIA2="${INSTALL_ARIA2:-1}"
HF_TOKEN="${HF_TOKEN:-}"

BREW_BIN=""
PYTHON_BIN=""
HF_CMD=""
HF_MODE=""

log() {
  printf '\n[%s] %s\n' "Aster" "$*"
}

warn() {
  printf '\n[%s][warn] %s\n' "Aster" "$*" >&2
}

err() {
  printf '\n[%s][error] %s\n' "Aster" "$*" >&2
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
  if [[ ! -f "$EXAMPLE_CONFIG_PATH" ]]; then
    err "Example config missing: $EXAMPLE_CONFIG_PATH"
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
  python -m pip install --upgrade pip setuptools wheel
}

configure_hf_endpoint() {
  export HF_ENDPOINT="$HF_ENDPOINT_VALUE"
  log "HF_ENDPOINT set to: $HF_ENDPOINT"
}

resolve_hf_cli() {
  if command -v hf >/dev/null 2>&1; then
    HF_CMD="$(command -v hf)"
    HF_MODE="hf"
    return 0
  fi
  if command -v huggingface-cli >/dev/null 2>&1; then
    HF_CMD="$(command -v huggingface-cli)"
    HF_MODE="huggingface-cli"
    return 0
  fi
  if [[ -x "$VENV_PATH/bin/hf" ]]; then
    HF_CMD="$VENV_PATH/bin/hf"
    HF_MODE="hf"
    return 0
  fi
  if [[ -x "$VENV_PATH/bin/huggingface-cli" ]]; then
    HF_CMD="$VENV_PATH/bin/huggingface-cli"
    HF_MODE="huggingface-cli"
    return 0
  fi
  return 1
}

ensure_hf_cli() {
  if resolve_hf_cli; then
    log "Hugging Face CLI found: $HF_CMD ($HF_MODE)"
    return
  fi

  warn "No Hugging Face CLI found. Installing huggingface_hub[cli] into the venv."
  python -m pip install -U "huggingface_hub[cli]"

  if resolve_hf_cli; then
    log "Hugging Face CLI installed: $HF_CMD ($HF_MODE)"
    return
  fi

  err "No usable Hugging Face CLI was found after installation."
  err "Expected one of: hf or huggingface-cli"
  exit 1
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
  log "Installing aria2 via Homebrew for optional accelerated downloads"
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

hf_login_hint() {
  if [[ "$HF_MODE" == "hf" ]]; then
    echo "hf auth login"
  else
    echo "huggingface-cli login"
  fi
}

ensure_config() {
  mkdir -p "$MODELS_DIR" "$MAIN_DIR" "$DRAFT_DIR"

  if [[ ! -f "$CONFIG_PATH" ]]; then
    cp "$EXAMPLE_CONFIG_PATH" "$CONFIG_PATH"
    log "Created config from example: $CONFIG_PATH"
  fi

  python - <<PY
from pathlib import Path
config_path = Path(r"$CONFIG_PATH")
text = config_path.read_text()
text = text.replace('/ABSOLUTE/PATH/TO/Qwen3.5-9B-MLX   # or a compatible HF repo id', r'$MAIN_DIR')
text = text.replace('/ABSOLUTE/PATH/TO/Qwen3.5-0.8B-MLX   # or a compatible HF repo id', r'$DRAFT_DIR')
config_path.write_text(text)
print(f"[Aster] Updated config paths in {config_path}")
PY
}

show_preflight() {
  log "Project root: $PROJECT_ROOT"
  log "Models dir:   $MODELS_DIR"
  log "Main repo:    $MAIN_REPO"
  log "Draft repo:   $DRAFT_REPO"
  log "Mirror:       $HF_ENDPOINT_VALUE"
  log "USE_HFD:      $USE_HFD"
}

hf_download() {
  local repo="$1"
  local target_dir="$2"

  if [[ "$HF_MODE" == "hf" ]]; then
    if [[ -n "$HF_TOKEN" ]]; then
      "$HF_CMD" download "$repo" --token "$HF_TOKEN" --local-dir "$target_dir"
    else
      "$HF_CMD" download "$repo" --local-dir "$target_dir"
    fi
  else
    if [[ -n "$HF_TOKEN" ]]; then
      "$HF_CMD" download "$repo" --token "$HF_TOKEN" --resume-download --local-dir "$target_dir" --local-dir-use-symlinks False
    else
      "$HF_CMD" download "$repo" --resume-download --local-dir "$target_dir" --local-dir-use-symlinks False
    fi
  fi
}

hfd_download() {
  local repo="$1"
  local target_dir="$2"

  if [[ ! -x "$HFD_PATH" ]]; then
    err "Requested USE_HFD=1 but hfd helper is unavailable at $HFD_PATH"
    exit 1
  fi

  mkdir -p "$target_dir"
  (
    cd "$target_dir"
    if [[ -n "$HF_TOKEN" ]]; then
      "$HFD_PATH" "$repo" --hf_token "$HF_TOKEN"
    else
      "$HFD_PATH" "$repo"
    fi
  )
}

download_model() {
  local repo="$1"
  local target_dir="$2"
  local label="$3"

  log "Downloading $label model to: $target_dir"
  if [[ "$USE_HFD" == "1" ]]; then
    hfd_download "$repo" "$target_dir"
  else
    hf_download "$repo" "$target_dir"
  fi
}

postflight() {
  log "Download complete."
  log "Main model path:  $MAIN_DIR"
  log "Draft model path: $DRAFT_DIR"
  printf '\n[Aster] Suggested next commands:\n'
  printf '  cd %s\n' "$PROJECT_ROOT"
  printf '  source .venv/bin/activate\n'
  printf '  python scripts/model_smoke.py --config configs/config.yaml\n'
  printf '  python scripts/benchmark_live.py --config configs/config.yaml\n\n'
}

main() {
  require_macos
  require_project_root
  show_preflight
  ensure_brew
  ensure_python
  ensure_venv
  configure_hf_endpoint
  ensure_hf_cli
  ensure_hfd_if_requested
  ensure_config

  warn "If the repositories require authentication, run: $(hf_login_hint)"
  warn "If a gated repo still fails, set HF_TOKEN=hf_xxx when invoking the script."

  download_model "$MAIN_REPO" "$MAIN_DIR" "main"
  download_model "$DRAFT_REPO" "$DRAFT_DIR" "draft"

  postflight
}

main "$@"

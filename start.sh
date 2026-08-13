#!/usr/bin/env bash
#
# Launch guru — local Ollama chat agent with on-demand tool directory.
#
# Prereqs (one-time):
#   1. Ollama app running.
#   2. uv sync  (creates .venv and installs dependencies)
#
# Usage:
#   ./start.sh
#   ./start.sh --model qwen3:8b   # override the model
#
set -euo pipefail

MODEL="qwen3-abliterated-32k:latest"
OLLAMA_URL="http://localhost:11434"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/.venv/bin/python"

# Allow model override via --model flag
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# --- 1. Check the venv exists -------------------------------------------------
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: .venv not found. Run: uv sync" >&2
  exit 1
fi

# --- 2. Ensure Ollama is reachable --------------------------------------------
if ! curl -sf "${OLLAMA_URL}/api/version" >/dev/null 2>&1; then
  echo "ERROR: Ollama server not reachable at ${OLLAMA_URL}." >&2
  echo "       Launch the Ollama app (menu-bar) and try again." >&2
  exit 1
fi

# --- 3. Pull the model if not already present ---------------------------------
if ! curl -sf "${OLLAMA_URL}/api/tags" | grep -q "\"${MODEL}\""; then
  echo "Pulling ${MODEL} (one-time download) ..."
  ollama pull "${MODEL}"
fi

# --- 4. Launch ----------------------------------------------------------------
echo "Starting guru -> ${MODEL}"
exec "${PYTHON}" "${SCRIPT_DIR}/guru.py" --model "${MODEL}"

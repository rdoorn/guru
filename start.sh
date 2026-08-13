#!/usr/bin/env bash
#
# Launch guru — local LLM agent with pluggable provider adapters.
#
# Prereqs (one-time):
#   uv sync   (creates .venv and installs dependencies)
#
# The Ollama daemon check and model pull now happen inside the Ollama adapter,
# so this launcher just runs the app. Pass through any arguments (e.g. --model).
#
# Usage:
#   ./start.sh
#   ./start.sh --model qwen3:8b
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: .venv not found. Run: uv sync" >&2
  exit 1
fi

exec "${PYTHON}" -m guru "$@"

#!/usr/bin/env bash
set -euo pipefail

if command -v uv >/dev/null 2>&1; then
  echo "[INFO] Using uv"
  exec uv run state "$@"
else
  echo "[INFO] uv not found; using pip + venv fallback"
  if [ ! -d ".venv" ]; then
    python -m venv .venv
  fi
  # shellcheck disable=SC1091
#   source .venv/bin/activate
  exec python -m state "$@"
fi
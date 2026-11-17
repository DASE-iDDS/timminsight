#!/usr/bin/env bash
# Isolated venv for the Palimpzest baseline (official PyPI package palimpzest).
# Installed version: palimpzest==1.5.3 (latest on PyPI, 2026-11-14).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$ROOT/.baselines/venvs/palimpzest"
PY="${PYTHON:-python3.12}"
command -v "$PY" >/dev/null || { echo "python3.12 not found; set PYTHON=/path/to/python3.12" >&2; exit 1; }
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
    || { echo "$PY is not Python 3.12" >&2; exit 1; }
"$PY" -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install "palimpzest==1.5.3"
"$VENV/bin/python" -c "import importlib.metadata as m; print('palimpzest', m.version('palimpzest'))"

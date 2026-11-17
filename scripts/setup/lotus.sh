#!/usr/bin/env bash
# Isolated venv for the LOTUS baseline (official PyPI package lotus-ai).
# Installed version: lotus-ai==1.2.4 (litellm 1.109.1), installed 2026-11-14.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$ROOT/.baselines/venvs/lotus"
PY="${PYTHON:-python3.12}"
command -v "$PY" >/dev/null || { echo "python3.12 not found; set PYTHON=/path/to/python3.12" >&2; exit 1; }
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
    || { echo "$PY is not Python 3.12" >&2; exit 1; }
"$PY" -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install "lotus-ai==1.2.4"
"$VENV/bin/python" -c "import importlib.metadata as m; print('lotus-ai', m.version('lotus-ai'))"

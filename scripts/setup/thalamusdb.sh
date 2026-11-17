#!/usr/bin/env bash
# Isolated venv for the ThalamusDB baseline (official PyPI package `thalamusdb`).
# Installed version: thalamusdb==0.1.15 (2026-11-14; deps: litellm 1.75.9, duckdb 1.3.2, sqlglot 27.0.0).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$ROOT/.baselines/venvs/thalamusdb"
PY="${PYTHON:-python3.12}"
command -v "$PY" >/dev/null || { echo "python3.12 not found; set PYTHON=/path/to/python3.12" >&2; exit 1; }
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
    || { echo "$PY is not Python 3.12" >&2; exit 1; }
"$PY" -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q "thalamusdb==0.1.15"
"$VENV/bin/pip" show thalamusdb | grep -i '^version'

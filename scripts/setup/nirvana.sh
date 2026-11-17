#!/usr/bin/env bash
# Isolated venv for the Nirvana baseline (Zhu et al., PACMMOD 2026).
# Installed from the local clone .baselines/src/nirvana at commit
# 7ef95e614a97836c37899a23db7f94e4fa836a35 (package nirvana-ai 1.3.2).
# nirvana-core is not a separate package: it is a Rust extension
# (nirvana.nirvana_core) built by maturin from nirvana-core/Cargo.toml, so a
# Rust toolchain (cargo) must be on PATH.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$ROOT/.baselines/venvs/nirvana"
export PATH="$HOME/.cargo/bin:$PATH"
PY="${PYTHON:-python3.12}"
command -v "$PY" >/dev/null || { echo "python3.12 not found; set PYTHON=/path/to/python3.12" >&2; exit 1; }
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
    || { echo "$PY is not Python 3.12" >&2; exit 1; }
SRC_DIR="$ROOT/.baselines/src/nirvana"
[ -d "$SRC_DIR" ] || git clone https://github.com/JunHao-Zhu/nirvana.git "$SRC_DIR"
git -C "$SRC_DIR" checkout -q 7ef95e6   # the evaluated commit
"$PY" -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install "$ROOT/.baselines/src/nirvana"
"$VENV/bin/python" -I -c "import importlib.metadata as m, nirvana, nirvana.nirvana_core; print('nirvana-ai', m.version('nirvana-ai'))"

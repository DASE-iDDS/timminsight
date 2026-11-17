#!/usr/bin/env bash
# Isolated venv for the CAESURA baseline (Urban & Binnig, CIDR 2024), installed
# from the cloned official source .baselines/src/caesura (README: python 3.11,
# torch 2.0; we use python 3.12 with the pinned requirements where they build).
# Pins kept from requirements.txt: openai==0.28, langchain==0.0.197, chromadb==0.3.20.
# Added (not in requirements.txt): torch/torchvision (README step 2, latest wheels
# for 3.12), pillow, pydantic<2 (langchain 0.0.197 needs pydantic v1),
# transformers<4.46 (old BLIP/BART APIs), numpy<2 (chromadb 0.3.x / pandasql).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$ROOT/.baselines/src/caesura"
VENV="$ROOT/.baselines/venvs/caesura"
PY="${PYTHON:-python3.12}"
command -v "$PY" >/dev/null || { echo "python3.12 not found; set PYTHON=/path/to/python3.12" >&2; exit 1; }
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' \
    || { echo "$PY is not Python 3.12" >&2; exit 1; }
SRC_DIR="$ROOT/.baselines/src/caesura"
[ -d "$SRC_DIR" ] || git clone https://github.com/DataManagementLab/caesura.git "$SRC_DIR"
git -C "$SRC_DIR" checkout -q 00ebb18   # the evaluated commit
"$PY" -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip setuptools wheel
"$VENV/bin/pip" install torch torchvision pillow "numpy<2" "pandas<2.2" "pydantic<2" "transformers<4.46" \
    -r "$SRC/requirements.txt"
"$VENV/bin/pip" install --no-deps "$SRC"
COMMIT="$(git -C "$SRC" rev-parse HEAD)"
echo "$COMMIT" > "$VENV/CAESURA_COMMIT"
echo "CAESURA commit $COMMIT"
"$VENV/bin/python" -c "import importlib.metadata as m; [print(p, m.version(p)) for p in ('langchain','openai','chromadb','transformers','torch','pydantic','numpy')]"

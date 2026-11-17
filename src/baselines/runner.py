"""Run a baseline system end to end on a natural-language question.

Each baseline runs as a subprocess in its own virtual environment
(``.baselines/venvs/<system>``, created by ``scripts/setup/<system>.sh``) with
the worker ``src/baselines/<system>_worker.py``; see ``common.py`` for the
protocol (NL-to-program translation with the same backbone LLM, error-feedback
retries, translation cost charged to the baseline).
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
SYSTEMS = ("caesura", "lotus", "thalamusdb", "palimpzest", "nirvana")
KEY_ENV = "TIMM_BASELINE_API_KEY"
MARKER = "__BASELINE_RESPONSE__"


@dataclass
class BaselineResponse:
    system: str
    rows: List[Dict[str, Any]] = field(default_factory=list)
    program: str = ""
    attempts: int = 0
    errors: List[str] = field(default_factory=list)
    translation_tokens: Dict[str, int] = field(default_factory=dict)
    execution_tokens: Dict[str, int] = field(default_factory=dict)
    translation_seconds: float = 0.0
    execution_seconds: float = 0.0
    wall_seconds: float = 0.0
    ok: bool = False


def worker_python(system: str) -> Path:
    return REPO / ".baselines" / "venvs" / system / "bin" / "python"


def _image_bytes(value: Any) -> Optional[bytes]:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, dict) and isinstance(value.get("bytes"), (bytes, bytearray)):
        return bytes(value["bytes"])
    return None


def write_tables(tables: Dict[str, List[Dict[str, Any]]], directory: str) -> Dict[str, str]:
    """Write in-memory tables as CSV files and return their paths.  Image cells
    (raw bytes or {"bytes": ...}) are written as image files and replaced by
    their absolute paths, which is how the baselines consume images."""
    paths = {}
    for name, rows in tables.items():
        path = os.path.join(directory, f"{name}.csv")
        cols: List[str] = []
        for r in rows:
            cols += [c for c in r if c not in cols]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for i, r in enumerate(rows):
                out = {}
                for c in cols:
                    img = _image_bytes(r.get(c))
                    if img is not None:
                        img_path = os.path.join(directory, f"{name}_{c}_{i}.jpg")
                        with open(img_path, "wb") as fi:
                            fi.write(img)
                        out[c] = img_path
                    else:
                        out[c] = r.get(c)
                w.writerow(out)
        paths[name] = path
    return paths


def image_columns(rows: List[Dict[str, Any]]) -> List[str]:
    """Columns holding image cells."""
    return [c for c in (rows[0] if rows else {}) if _image_bytes(rows[0].get(c)) is not None]


def run_baseline(system: str, question: str, tables: Dict[str, List[Dict[str, Any]]],
                 model: str, base_url: str, api_key: str,
                 table_info: Optional[Dict[str, Dict[str, Any]]] = None,
                 max_attempts: int = 3, timeout: int = 1800) -> BaselineResponse:
    """Answer ``question`` over ``tables`` with ``system``'s official release."""
    if system not in SYSTEMS:
        raise ValueError(f"unknown baseline {system!r}; expected one of {SYSTEMS}")
    python = worker_python(system)
    if not python.exists():
        raise FileNotFoundError(f"{python} not found; run scripts/setup/{system}.sh first")
    table_info = table_info or {}
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        paths = write_tables(tables, tmp)
        request = {
            "question": question,
            "tables": {name: {"path": path, "image_columns": image_columns(tables[name]),
                              **table_info.get(name, {})} for name, path in paths.items()},
            "llm": {"model": model, "base_url": base_url, "api_key_env": KEY_ENV},
            "max_attempts": max_attempts,
        }
        env = {**os.environ, KEY_ENV: api_key, "PYTHONPATH": str(REPO)}
        try:
            proc = subprocess.run(
                [str(python), str(REPO / "src" / "baselines" / f"{system}_worker.py")],
                input=json.dumps(request), capture_output=True, text=True,
                timeout=timeout, cwd=tmp, env=env)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"baseline {system} exceeded {timeout}s on: {question[:200]}") from exc
    wall = time.time() - t0
    out = proc.stdout
    if MARKER not in out:
        return BaselineResponse(system, errors=[
            f"worker exited with code {proc.returncode}: " + (proc.stderr or out)[-4000:]],
            wall_seconds=wall)
    payload = json.loads(out.rsplit(MARKER, 1)[1].strip().splitlines()[0])
    known = {k: payload[k] for k in BaselineResponse.__dataclass_fields__ if k in payload}
    resp = BaselineResponse(**{**known, "system": system, "wall_seconds": wall})
    if proc.returncode != 0:
        resp.ok = False
        resp.errors.append(f"worker exited with code {proc.returncode}: {proc.stderr[-2000:]}")
    return resp

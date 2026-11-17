"""User datasets: import, column roles, loading.

An uploaded dataset is a directory under the data root that the benchmark's
``dir:`` loader reads, so tables, text columns, and image columns are found
by exactly the same logic as for the built-in datasets:

    <root>/<name>/tables/*.csv        uploaded CSV/Parquet tables; XLSX sheets
                                      become one CSV each
    <root>/<name>/<image files>       images, referenced by file name from a
                                      table, or listed in tables/images.csv
    <root>/<name>/tables/documents.csv  one row per uploaded .txt file
    <root>/<name>/roles.json          column roles the user corrected

Nothing is changed in the data; images and texts that no uploaded table
references are given a table so that TiMetaGraph and the planner see them.
"""
from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

from demo import core

TABLE_EXT = {".csv", ".parquet", ".xlsx"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
TEXT_EXT = {".txt"}
NAME = re.compile(r"[A-Za-z0-9_-]{1,40}")


def _safe(name: str) -> str:
    base = PurePosixPath(name.replace("\\", "/")).name
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    if not base:
        raise ValueError(f"unusable file name {name!r}")
    return base


def _expand(files: List[Tuple[str, bytes]]) -> List[Tuple[str, bytes]]:
    """Files, with zip archives replaced by their members (directories and
    hidden files skipped, paths flattened)."""
    out = []
    for name, data in files:
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for info in z.infolist():
                    member = PurePosixPath(info.filename)
                    if info.is_dir() or any(p.startswith((".", "__MACOSX")) for p in member.parts):
                        continue
                    out.append((member.name, z.read(info)))
        else:
            out.append((name, data))
    return out


def import_files(root: Path, name: str, files: List[Tuple[str, bytes]]) -> Dict[str, Any]:
    """Store uploaded files as a dataset directory; return its manifest."""
    if not NAME.fullmatch(name):
        raise ValueError("dataset name: 1-40 letters, digits, '-' or '_'")
    target = root / name
    (target / "tables").mkdir(parents=True, exist_ok=True)
    kept = {"tables": [], "images": [], "texts": [], "skipped": []}
    texts: List[Tuple[str, str]] = []
    for raw, data in _expand(files):
        base = _safe(raw)
        ext = Path(base).suffix.lower()
        if ext == ".xlsx":
            import pandas as pd
            for sheet, frame in pd.read_excel(io.BytesIO(data), sheet_name=None).items():
                stem = f"{Path(base).stem}_{_safe(str(sheet))}"
                frame.to_csv(target / "tables" / f"{stem}.csv", index=False)
                kept["tables"].append(f"{stem}.csv")
        elif ext in (".csv", ".parquet"):
            (target / "tables" / base).write_bytes(data)
            kept["tables"].append(base)
        elif ext in IMAGE_EXT:
            (target / base).write_bytes(data)
            kept["images"].append(base)
        elif ext in TEXT_EXT:
            texts.append((base, data.decode("utf-8", errors="replace")))
            kept["texts"].append(base)
        else:
            kept["skipped"].append(base)
    referenced = _referenced_files(target)
    loose = sorted(i for i in kept["images"] if i not in referenced)
    if loose:
        _write_csv(target / "tables" / "images.csv", ["name", "image"],
                   [[Path(i).stem, i] for i in loose])
        kept["tables"].append("images.csv (images not referenced by a table)")
    if texts:
        _write_csv(target / "tables" / "documents.csv", ["file", "text"],
                   [[n, t] for n, t in sorted(texts)])
        kept["tables"].append("documents.csv (uploaded text files)")
    if not any((target / "tables").iterdir()):
        raise ValueError("no table, image, or text file in the upload")
    return kept


def _write_csv(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _referenced_files(target: Path) -> set:
    """Image file names that cells of the uploaded tables mention."""
    names = {p.name for p in target.iterdir() if p.suffix.lower() in IMAGE_EXT}
    hits = set()
    for t in (target / "tables").glob("*.csv"):
        with open(t, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.reader(f):
                hits.update(Path(c.strip()).name for c in row if Path(c.strip()).name in names)
    return hits


def dataset_name(name: str) -> str:
    return f"upload/{name}"


def load(root: Path, name: str, limit: Optional[int] = None):
    """The uploaded dataset through the benchmark's dir: loader, with the
    column roles the user set (if any)."""
    data = core.load(f"dir:{root / name}", limit)
    roles = roles_of(root, name)
    for table, r in roles.items():
        if table not in data.tables:
            continue
        cols = set(data.tables[table][0]) if data.tables[table] else set()
        data.text_columns[table] = [c for c in r.get("text_columns", []) if c in cols]
        data.image_columns[table] = [c for c in r.get("image_columns", []) if c in cols]
    return data


def roles_of(root: Path, name: str) -> Dict[str, Dict[str, List[str]]]:
    path = root / name / "roles.json"
    return json.loads(path.read_text()) if path.exists() else {}


def set_roles(root: Path, name: str, roles: Dict[str, Dict[str, List[str]]]) -> None:
    """Store user-corrected roles; an image column must hold images."""
    data = core.load(f"dir:{root / name}", None)
    for table, r in roles.items():
        if table not in data.tables:
            raise ValueError(f"no table {table!r}")
        cols = set(data.tables[table][0]) if data.tables[table] else set()
        for kind in ("text_columns", "image_columns"):
            bad = [c for c in r.get(kind, []) if c not in cols]
            if bad:
                raise ValueError(f"{table}: no columns {bad}")
        detected_images = set(data.image_columns.get(table, []))
        for c in r.get("image_columns", []):
            if c not in detected_images:
                raise ValueError(f"{table}.{c} does not hold images (files, bytes, or URLs)")
    (root / name / "roles.json").write_text(json.dumps(roles, indent=1))


def uploads(root: Path) -> List[str]:
    return sorted(p.name for p in root.iterdir() if (p / "tables").is_dir()) if root.exists() else []

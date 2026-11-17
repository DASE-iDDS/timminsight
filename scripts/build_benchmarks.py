"""Build the benchmarks of the paper's experiments (Sec. VI).

    python -m scripts.build_benchmarks [--stages data,base,variants]

Base queries (71):
* Nirvana-Bench, 36 queries: the workloads of the Nirvana release
  (experiments/workloads/{estate,imdb,steam}/q*.py; imdb/run_exp.py is a
  runner, not a query).  The question is the program's ``Qn: ...`` log line.
* SemBench, 35 queries without audio: Movie Q1-Q10, E-Commerce q1-q14, and
  MMQA q1, q2a, q2b, q3a, q3g, q4, q5, q6a, q6b, q6c, q7.  MMQA's q3a-q3g are
  one template ("Which movies are <genre>?") over seven genres; q3a (a single
  genre) and q3g (a compound genre) represent it.

Data: Nirvana-Bench reads the release's testdata in full (steam: all 18,891
games; the crawl lists some games twice, and gold answers count each game
once).
SemBench scenarios are staged by SemBench's own data generators, in its
layout, at the scale factors SemBench reports: movie sf_2000, ecomm sf_500
(seed 12345600), and mmqa sf_200 (data/sembench/files/<scenario>/data/sf_N).

Gold answers:
* SemBench: its official ground truth (gold SQL over the staged data, the
  ground-truth SQL of E-Commerce, the ground_truth field of MMQA).
* Nirvana-Bench has no released ground truth.  Every query has a gold
  function below that follows its question: answers determined by structured
  fields are computed exactly (``structured_columns``); perceptual or
  free-text judgements are labelled per row by a reference model at
  temperature 0 (``reference_llm``: qwen3-vl-plus for images, qwen-max for
  text); a currency conversion uses a fixed rate (``reference_rule``).

Ambiguity variants: one per base query and type of the TiQC taxonomy (426
items).  Given the base question and how it is answered (the Nirvana
program, SemBench's ground-truth query), the LLM rewrites the question so
that it is ambiguous in that way while the base reading stays valid with the
same answer, and states how the base question resolves it; an independent
audit pass checks each variant and rejected ones are rewritten.  The hidden
intent is the exact base question plus that resolution, so the gold answer
is unchanged.  Every item is listed in
benchmarks/review_checklist.csv for human review.

Outputs: benchmarks/base_queries.jsonl, nirvana_bench.jsonl, sembench.jsonl,
review_checklist.csv.  LLM outputs are cached under runs/benchmark_build/,
so the build is resumable.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import statistics
import sys
import threading
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

NIRVANA = REPO / "data" / "nirvana_repo" / "nirvana-main"
NIRVANA_DATA = NIRVANA / "testdata"
WORKLOADS = NIRVANA / "experiments" / "workloads"
SEMBENCH_SRC = REPO / ".baselines" / "src" / "sembench"          # read only
SEMBENCH_OUT = REPO / "data" / "sembench"                        # staged data
ECOMM_SOURCE = Path(os.environ.get(
    "ECOMM_SOURCE", "/Volumes/future_zhu/panghu2/ecomm/1/fashion-dataset"))
MMQA_SOURCE = Path(os.environ.get(
    "MMQA_SOURCE", "/Volumes/future_zhu/panghu2/mmqa/extracted/mmqa_data"))
BENCH = REPO / "benchmarks"
CACHE = REPO / "runs" / "benchmark_build"

STEAM_ROWS: Optional[int] = None    # None: the full steam table
SCALE = {"movie": 2000, "ecomm": 500, "mmqa": 200}
ECOMM_SEED = 12345600
MMQA_SELECTED = ["q1", "q2a", "q2b", "q3a", "q3g", "q4", "q5", "q6a", "q6b", "q6c", "q7"]
# Movie questions that ask for "five"/"ten" items: any k of the gold set
MOVIE_SUBSET_K = {1: 5, 2: 5, 5: 10, 6: 10}
IDR_PER_USD = 14850          # 2022 annual average; the Steam crawl lists IDR prices
POSITIVE_REVIEWS = {"Overwhelmingly Positive", "Very Positive", "Positive", "Mostly Positive"}

AMBIGUITY_TYPES = {
    "cross_modal_referential": "a mention may resolve to different objects across modalities "
                               "(e.g. 'positive reviews' = a rating column or the sentiment of "
                               "review texts)",
    "intra_modal_attribute": "an attribute admits multiple bindings within one modality "
                             "(e.g. 'best-received' = rating or number of votes)",
    "multimodal_intent": "the analytical goal itself admits multiple readings",
    "schema_structure": "a query term matches no or several tables and columns",
    "value_content": "a literal matches multiple cell values or entities (e.g. 'sci-fi' = the "
                     "Sci-Fi or the Science Fiction label)",
    "temporal_spatial": "an unspecified time range or spatial scope (e.g. 'recent')",
}


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# =============================================================== data
def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stage_data() -> None:
    """Stage the SemBench scenarios with SemBench's own generators."""
    import duckdb
    # movie: the sf_2000 tables shipped with SemBench
    dst = SEMBENCH_OUT / "files" / "movie" / "data" / f"sf_{SCALE['movie']}"
    if not (dst / "Reviews.csv").exists():
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("Movies.csv", "Reviews.csv"):
            shutil.copy(SEMBENCH_SRC / "files" / "movie" / "data" / f"sf_{SCALE['movie']}" / f, dst / f)
    log(f"movie staged: {dst}")

    # mmqa: MMQADataGenerator over the extracted source data
    out = SEMBENCH_OUT / "files" / "mmqa" / "data" / f"sf_{SCALE['mmqa']}"
    if not (out / "thalamusdb_images.csv").exists():
        if not MMQA_SOURCE.is_dir():
            raise FileNotFoundError(f"MMQA source data not found at {MMQA_SOURCE} (set MMQA_SOURCE)")
        src = SEMBENCH_OUT / "files" / "mmqa" / "source_data"
        src.parent.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            src.symlink_to(MMQA_SOURCE, target_is_directory=True)
        gen = _load_module("sembench_mmqa_gen", SEMBENCH_SRC / "src" / "scenario" / "mmqa"
                           / "preparation" / "generate_data.py")
        gen.MMQADataGenerator(str(SEMBENCH_OUT), SCALE["mmqa"], skip_download=True).generate_data()
    log(f"mmqa staged: {out}")

    # ecomm: SemBench's Parquet conversion of the Fashion Product Images data,
    # then its deterministic sample (_create_sample)
    target = SEMBENCH_OUT / "files" / "ecomm" / "data"
    sample = target / f"sf_{SCALE['ecomm']}"
    if not (sample / "styles_details.parquet").exists():
        if not (ECOMM_SOURCE / "styles.csv").exists():
            raise FileNotFoundError(f"Fashion Product Images data not found at {ECOMM_SOURCE} "
                                    "(set ECOMM_SOURCE)")
        full = target / "fashion_product_images"
        full.mkdir(parents=True, exist_ok=True)
        if not (full / "styles.parquet").exists():
            duckdb.sql(f"""COPY (SELECT * FROM read_csv('{ECOMM_SOURCE / 'styles.csv'}', header=true,
                       delim=',', quote='"', ignore_errors=true))
                       TO '{full / 'styles.parquet'}' (FORMAT PARQUET)""")
        if not (full / "image_mapping.parquet").exists():
            duckdb.sql(f"""COPY (SELECT split_part(filename, '.', 1) AS id, filename, link
                       FROM read_csv('{ECOMM_SOURCE / 'images.csv'}', header=true))
                       TO '{full / 'image_mapping.parquet'}' (FORMAT PARQUET)""")
        if not (full / "styles_details.parquet").exists():
            log("ecomm: converting 44k style documents (one-off, a few minutes)")
            with duckdb.connect() as con:
                con.execute("SET threads = 2;")
                con.execute("SET preserve_insertion_order=false;")
                con.execute(f"""COPY (SELECT unnest(data) FROM read_json(
                            '{ECOMM_SOURCE / 'styles' / '*.json'}', union_by_name=true))
                            TO '{full / 'styles_details.parquet'}' (FORMAT PARQUET)""")
        if not (full / "images").exists():
            (full / "images").symlink_to(ECOMM_SOURCE / "images", target_is_directory=True)
        gen = _load_module("sembench_ecomm_gen", SEMBENCH_SRC / "src" / "scenario" / "ecomm"
                           / "preparation" / "generate_data.py")
        gen._create_sample(str(target), SCALE["ecomm"], ECOMM_SEED)
        if not (sample / "images").exists():
            (sample / "images").symlink_to(full / "images", target_is_directory=True)
    log(f"ecomm staged: {sample}")


# =============================================================== base queries
def _call_kwargs(node: ast.Call) -> Dict[str, Any]:
    return {k.arg: ast.literal_eval(k.value) for k in node.keywords if k.arg}


def parse_nirvana_queries() -> List[Dict[str, Any]]:
    """The 36 Nirvana-Bench queries: question and semantic-operator program."""
    out = []
    for domain in ("estate", "imdb", "steam"):
        files = sorted(WORKLOADS.glob(f"{domain}/q*.py"), key=lambda p: int(p.stem[1:]))
        for f in files:
            src = f.read_text(encoding="utf-8")
            m = re.search(r'logger\.info\(f"Q(\d+): ([^"]+)"\)', src)
            if not m:
                raise ValueError(f"{f} has no 'Qn: <question>' line")
            ops = []
            for node in ast.walk(ast.parse(src)):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr.startswith("semantic_")):
                    ops.append((node.lineno, {"op": node.func.attr, **_call_kwargs(node)}))
            out.append({"base_id": f"nirvana/{domain}/{f.stem}", "domain": domain,
                        "question": m.group(2).strip(),
                        "original_program": [o for _, o in sorted(ops, key=lambda x: x[0])]})
    return out


def sembench_queries() -> List[Dict[str, Any]]:
    """The 35 SemBench queries (question, official ground-truth source)."""
    files = SEMBENCH_SRC / "files"
    out = []
    for i in range(1, 11):
        out.append({"base_id": f"sembench/movie/Q{i}", "scenario": "movie", "qid": i,
                    "question": (files / "movie" / "query" / "natural_language" / f"Q{i}.txt")
                    .read_text(encoding="utf-8").strip()})
    for p in sorted((files / "ecomm" / "queries").glob("q*.toml"), key=lambda p: int(p.stem[1:])):
        q = tomllib.loads(p.read_text(encoding="utf-8"))
        out.append({"base_id": f"sembench/ecomm/{p.stem}", "scenario": "ecomm",
                    "qid": q["metadata"]["query_id"],
                    "question": " ".join(q["definition"]["natural_language"].split()),
                    "ground_truth_sql": q["definition"]["ground_truth"],
                    "accuracy_metric": q["definition"].get("accuracy_metric")})
    for name in MMQA_SELECTED:
        q = json.loads((files / "mmqa" / "query" / "natural_language" / f"{name}.json")
                       .read_text(encoding="utf-8"))
        out.append({"base_id": f"sembench/mmqa/{name}", "scenario": "mmqa", "qid": name,
                    "question": q["nl_question"].strip(), "ground_truth": q["ground_truth"],
                    "modalities": q.get("modalities")})
    return out


def dataset_of(base: Dict[str, Any]) -> str:
    if base["base_id"].startswith("nirvana/"):
        d = base["domain"]
        return f"nirvana/steam:{STEAM_ROWS}" if d == "steam" and STEAM_ROWS else f"nirvana/{d}"
    return f"sembench/{base['scenario']}/sf_{SCALE[base['scenario']]}"


# =============================================================== reference labels
def _normalized_image(data: bytes, max_side: int = 2048) -> Optional[bytes]:
    """The image as RGB JPEG with its longer side at most ``max_side`` pixels
    (web images come in formats and sizes vision APIs reject); None if the
    bytes are not an image."""
    import io
    from PIL import Image, UnidentifiedImageError
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, ValueError):
        return None
    img = img.convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=90)
    return out.getvalue()


class Labeler:
    """Per-row reference labels at temperature 0, cached on disk."""

    def __init__(self, text_model: str, vision_model: str, workers: int):
        from core.query_planner.llm_query_analyzer import create_provider
        self.llm = create_provider("dashscope", text_model)
        self.vlm = create_provider("dashscope", vision_model)
        self.workers = workers
        self._lock = threading.Lock()

    def _cache(self, task: str) -> Tuple[Path, Dict[str, Any]]:
        path = CACHE / "labels" / (hashlib.sha1(task.encode()).hexdigest()[:16] + ".json")
        data = json.loads(path.read_text()) if path.exists() else {"task": task, "labels": {}}
        return path, data

    def _save(self, path: Path, data: Dict[str, Any]) -> None:
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False))
            tmp.replace(path)

    def text(self, task: str, kind: str, items: Dict[str, str], batch: int = 20) -> Dict[str, Any]:
        """One value per item; kind is bool | number | text | list."""
        from core.query_planner.llm_query_analyzer import complete_json
        path, data = self._cache(f"text|{kind}|{task}")
        todo = [k for k in items if k not in data["labels"]]
        fmt = {"bool": "true or false", "number": "a number, or null if not stated",
               "text": "a short string", "list": "a JSON list of short strings"}[kind]

        def run(keys: List[str]) -> None:
            def complete(v: dict) -> None:
                missing = [k for k in keys if k not in v]
                if missing:
                    raise ValueError(f"omitted ids {missing[:5]}")
                for k in keys:
                    x = v[k]
                    ok = {"bool": isinstance(x, bool), "number": x is None or isinstance(x, (int, float)),
                          "text": isinstance(x, str), "list": isinstance(x, list)}[kind]
                    if not ok:
                        raise ValueError(f"id {k}: expected {fmt}, got {x!r}"[:200])
            got = complete_json(self.llm, [
                {"role": "system", "content": "You label data records for a benchmark. Answer "
                                              "only from each record's content. Output JSON only."},
                {"role": "user", "content": f"Task: {task}\nFor every record give {fmt}.\n"
                 f"Records (id -> content):\n{json.dumps({k: items[k] for k in keys}, ensure_ascii=False)}\n"
                 "Return a JSON object mapping every id to its value."}],
                dict, complete, "reference label")
            with self._lock:
                data["labels"].update({k: got[k] for k in keys})
            self._save(path, data)

        chunks = [todo[i:i + batch] for i in range(0, len(todo), batch)]
        if chunks:
            log(f"  labelling {len(todo)} records: {task[:70]}")
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                list(pool.map(run, chunks))
        return {k: data["labels"][k] for k in items}

    @staticmethod
    def _fetch(url: str) -> Optional[bytes]:
        """Image bytes of a URL (cached); None if it cannot be fetched."""
        import urllib.request
        path = CACHE / "images" / hashlib.sha1(url.encode()).hexdigest()
        if path.exists():
            data = path.read_bytes()
            return data or None
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    data = r.read()
                break
            except Exception:
                time.sleep(2 * (attempt + 1))
        else:
            data = b""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return data or None

    def image(self, task: str, kind: str, cells: Dict[str, Any]) -> Dict[str, Any]:
        """One answer per image; kind is bool | text.  An image that cannot be
        fetched gets None (no label) and is left out of the gold answer."""
        from core.query_planner.plan_executor import _vision_answer, _yes
        path, data = self._cache(f"image|{kind}|{task}")
        todo = [k for k in cells if k not in data["labels"]]
        prompt = task + (" Answer yes or no." if kind == "bool" else " Answer in a few words.")
        done = [0]

        def run(k: str) -> None:
            cell = cells[k]
            if isinstance(cell, str) and cell.startswith("http"):
                cell = self._fetch(cell)
            if isinstance(cell, (bytes, bytearray)):
                cell = _normalized_image(cell)
            try:
                ans = None if cell is None else _vision_answer(self.vlm, prompt, cell)
            except RuntimeError as exc:
                if "HTTP 400" not in str(exc):
                    raise
                log(f"    image {k}: rejected by the vision model as invalid input; no label")
                ans = None
            with self._lock:
                data["labels"][k] = None if ans is None else (_yes(ans) if kind == "bool" else ans.strip())
                done[0] += 1
                if done[0] % 50 == 0:
                    log(f"    {done[0]}/{len(todo)} images")
            if done[0] % 25 == 0:
                self._save(path, data)

        if todo:
            log(f"  labelling {len(todo)} images: {task[:70]}")
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                list(pool.map(run, todo))
            self._save(path, data)
        return {k: data["labels"][k] for k in cells}


# =============================================================== Nirvana gold
_CORP_SUFFIX = re.compile(r"(co|corp|corporation|inc|incorporated|ltd|limited|llc|l\.l\.c|"
                          r"gmbh|s\.a|s\.r\.l|s\.l|b\.v|pty|plc|ab|oy|as)\.?", re.I)


def developers(field: str) -> List[str]:
    """The developers of a Steam game: the field separates developers by
    commas, but a company name may itself contain a comma before its legal
    suffix ("CAPCOM Co., Ltd.", "CarX Technologies, LLC"), which stays with it."""
    out: List[str] = []
    for part in (p.strip() for p in str(field or "").split(",")):
        if not part:
            continue
        if out and _CORP_SUFFIX.fullmatch(part):
            out[-1] = f"{out[-1]}, {part}"
        else:
            out.append(part)
    return out


def _num(s: Any) -> Optional[float]:
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", str(s or "").replace(" ", ""))
    return float(m.group(0).replace(",", "")) if m else None


def _idr(s: str) -> Optional[float]:
    s = (s or "").strip()
    if not s:
        return None
    if s.lower().startswith("free"):
        return 0.0
    return _num(s) if s.startswith("Rp") else None


def _date(s: str) -> Optional[datetime]:
    for fmt in ("%d %b, %Y", "%b %d, %Y", "%b %Y", "%d %B, %Y"):
        try:
            return datetime.strptime((s or "").strip(), fmt)
        except ValueError:
            pass
    return None


class Nirvana:
    """Gold answers of the Nirvana-Bench queries.  ``source`` records whether
    a query needed reference labels."""

    def __init__(self, labeler: Optional[Labeler]):
        import pyarrow.parquet as pq
        self.labeler = labeler
        self.estate = pq.read_table(NIRVANA_DATA / "multimodal_real_estate.parquet").to_pylist()
        with open(NIRVANA_DATA / "movie_data.csv", newline="", encoding="utf-8") as f:
            self.imdb = list(csv.DictReader(f))
        with open(NIRVANA_DATA / "steam_games.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))[:STEAM_ROWS]
        # the crawl lists some games twice (same app id): one game each
        seen, self.steam = set(), []
        for r in rows:
            app = re.search(r"/app/(\d+)", r["url"])
            key = app.group(1) if app else r["title"]
            if key not in seen:
                seen.add(key)
                self.steam.append(r)
        self.source = "structured_columns"

    def _llm(self) -> Labeler:
        if self.labeler is None:
            raise RuntimeError("this gold answer needs reference labels (an LLM)")
        self.source = "reference_llm"
        return self.labeler

    # ---- estate: key = the release's row id ("Unnamed: 0")
    @staticmethod
    def _eid(r):
        return str(r["Unnamed: 0"])

    @staticmethod
    def _bedrooms(r) -> Optional[int]:
        m = re.match(r"\s*(\d+)\s*bedroom", r["Title"], re.I)
        return int(m.group(1)) if m else None

    def _estate_text(self, task, kind, rows):
        return self._llm().text(task, kind, {self._eid(r): r["Details"] for r in rows})

    def price(self, rows):
        return self._estate_text("Extract the asking price of the house in Naira from the estate "
                                 "details, as a plain number (e.g. N250Million -> 250000000).",
                                 "number", rows)

    def amenities(self, rows):
        return self._estate_text("Extract the amenities of the estate from its details.", "list", rows)

    def has(self, rows, thing):
        got = self._estate_text(f"According to the estate details, does the estate have a {thing}?",
                                "bool", rows)
        return [r for r in rows if got[self._eid(r)]]

    def yard(self, rows):
        got = self._llm().image("Observed from the house picture, does the house have a yard?",
                                "bool", {self._eid(r): r["image"] for r in rows})
        return [r for r in rows if got[self._eid(r)]]

    def bedrooms_between(self, rows, lo, hi):
        return [r for r in rows if self._bedrooms(r) is not None and lo < self._bedrooms(r) < hi]

    @staticmethod
    def _agg(values, fn):
        vals = [v for v in values if isinstance(v, (int, float))]
        if not vals:
            return None
        return round({"min": min, "max": max, "sum": sum, "avg": statistics.mean}[fn](vals), 2)

    def estate_gold(self, q: str):
        E = self.estate
        if q == "q1":
            return sorted(self._eid(r) for r in self.yard(E))
        if q == "q2":
            return self.price(E)
        if q == "q3":
            return sorted(self._eid(r) for r in E if "Ajah" in r["Location"] and "Lagos" in r["Location"])
        if q == "q4":
            return self.amenities(E)
        if q == "q5":
            return self.amenities(self.bedrooms_between(E, 3, 6))
        if q == "q6":
            ys = self.yard(E)
            return self._agg(self.price(ys).values(), "avg")
        if q in ("q7", "q8"):
            two3 = [r for r in E if self._bedrooms(r) in (2, 3)]
            if q == "q8":
                return self.amenities(two3)
            return self._estate_text("Extract the features of the estate from its details.", "list", two3)
        if q == "q9":
            return self._agg(self.price(self.bedrooms_between(E, 3, 6)).values(), "avg")
        if q == "q10":
            dd = [r for r in self.bedrooms_between(E, 3, 6)
                  if "detached duplex" in r["Title"].lower() and "semi-detached" not in r["Title"].lower()]
            return self._agg(self.price(dd).values(), "min")
        if q == "q11":
            return self._agg(self.price(self.has(E, "swimming pool")).values(), "min")
        if q == "q12":
            lekki = [r for r in E if "Lekki" in r["Location"] and "Lagos" in r["Location"]]
            both = self.has(self.has(lekki, "swimming pool"), "gym")
            return self._agg(self.price(both).values(), "avg")
        raise KeyError(q)

    # ---- imdb: key = title
    @staticmethod
    def _genres(r):
        return [g for g in (r["Genre1"], r["Genre2"], r["Genre3"]) if g]

    @staticmethod
    def _rating(r):
        return _num(r["IMDB_rating"]) if r["IMDB_rating"] not in ("", "N/A") else None

    @staticmethod
    def _oscars(r):
        m = re.search(r"Won (\d+) Oscars?", r["Awards"])
        return int(m.group(1)) if m else 0

    def imdb_gold(self, q: str):
        M = self.imdb
        crime = [r for r in M if "Crime" in self._genres(r)]
        above9 = [r for r in M if (self._rating(r) or 0) > 9]
        if q == "q1":
            return {r["Title"]: self._genres(r) for r in M}
        if q == "q2":
            return sorted(r["Title"] for r in M if "Christopher Nolan" in r["Director"])
        if q == "q3":
            got = self._llm().image("Is this movie poster in a dark style?", "bool",
                                    {r["Title"]: r["Poster"] for r in M if r["Poster"].startswith("http")})
            return sorted(t for t, v in got.items() if v)
        if q == "q4":
            return sorted(r["Title"] for r in M if self._oscars(r) > 3)
        if q == "q5":
            return self._agg([_num(r["BoxOffice"]) for r in above9 if r["BoxOffice"] not in ("", "N/A")], "sum")
        if q == "q6":
            return sum("Quentin Tarantino" in r["Director"] for r in M)
        if q == "q7":
            sp = [r for r in M if "Steven Spielberg" in r["Director"] and self._rating(r) is not None]
            best = max(sp, key=self._rating)
            return {"movie": best["Title"], "rating": self._rating(best), "genres": self._genres(best)}
        if q == "q8":
            return sum(1 for r in above9 if self._oscars(r) == 2)
        mid = [r for r in crime if self._rating(r) is not None and 8.5 < self._rating(r) < 9]
        if q == "q9":
            return max(self._rating(r) for r in mid) if mid else None
        if q == "q10":
            return len(mid)
        crime9 = [r for r in crime if (self._rating(r) or 0) > 9]
        if q == "q11":
            return self._agg([_num(r["Runtime"]) for r in crime9], "avg")
        if q == "q12":
            return self._llm().text("Who is the main character of this movie, according to its plot? "
                                    "Give the character's name.", "text",
                                    {r["Title"]: r["Plot"] for r in crime9})
        raise KeyError(q)

    # ---- steam (all games): key = title
    def steam_gold(self, q: str):
        S = self.steam
        vr = [r for r in S if "VR Supported" in r["platforms"]]
        shooter = [r for r in S if "Shooter" in r["tags"].split(",")]
        if q == "q1":
            return sorted(r["title"] for r in S if r["rating"].split("?")[0].endswith("/18.png"))
        if q == "q2":
            return {r["title"]: ("positive" if r["overall_reviews"] in POSITIVE_REVIEWS else "negative")
                    for r in S if r["overall_reviews"]}
        if q == "q3":
            return sorted(r["title"] for r in vr)
        if q == "q4":
            return {t: v for t, v in self._llm().image("Summarize the graphic style of this video game, judging from "
                                     "its cover image.", "text",
                                     {r["title"]: r["image"] for r in S if r["image"]}).items() if v}
        if q == "q5":
            pubs = [r["publisher"] for r in S if "Sports" in r["genre"].split(",") and r["publisher"]]
            top = max(set(pubs), key=pubs.count)
            return {"publisher": top, "sports_games": pubs.count(top)}
        if q == "q6":
            mac = [r for r in S if "Mac OS" in r["platforms"]]
            return self._agg([_idr(r["discounted_price"]) for r in mac], "min")
        if q == "q7":
            return sorted(r["title"] for r in shooter if "Chinese" in r["language"])
        if q == "q8":
            return sum(1 for r in S if len(developers(r["developer"])) == 1
                       and (_num(r["metacriticts"]) or 0) > 90)
        if q == "q9":
            self.source = "reference_rule"
            usd = [_idr(r["original_price"]) for r in vr if r in shooter]
            return self._agg([p / IDR_PER_USD for p in usd if p is not None], "avg")
        if q == "q10":
            wm = [r for r in S if "Windows" in r["platforms"] and "Mac OS" in r["platforms"]
                  and r["overall_reviews"] in POSITIVE_REVIEWS]
            return self._agg([_idr(r["original_price"]) for r in wm], "avg")
        if q == "q11":
            dates = [d for d in (_date(r["release_date"]) for r in vr) if d]
            return min(dates).strftime("%d %b, %Y") if dates else None
        if q == "q12":
            adv = [r for r in S if "Adventure" in r["genre"].split(",") and r["image"]]
            got = self._llm().image("Is the graphic style of this video game cover cartoon?", "bool",
                                    {r["title"]: r["image"] for r in adv})
            dates = [d for d in (_date(r["release_date"]) for r in adv if got[r["title"]]) if d]
            return min(dates).strftime("%d %b, %Y") if dates else None
        raise KeyError(q)

    def gold(self, domain: str, q: str) -> Tuple[Any, str]:
        self.source = "structured_columns"
        value = getattr(self, f"{domain}_gold")(q)
        return value, self.source


# Questions whose answer is the gold of a program step the release words
# differently: documented corrections so the gold answers the question.
NIRVANA_NOTES = {
    "nirvana/steam/q10": "the release's program also filters PEGI-18 games, which the question "
                         "does not ask; gold follows the question",
    "nirvana/imdb/q7": "the release's program returns the highest rating; the question asks for "
                       "that movie's genre",
    "nirvana/steam/q9": f"USD at a fixed {IDR_PER_USD} IDR per USD",
    "nirvana/steam/q5": "the publisher with the most sports games",
}


# =============================================================== SemBench gold
def sembench_gold(base: Dict[str, Any]) -> Tuple[Any, str]:
    import duckdb
    sc = base["scenario"]
    data = SEMBENCH_OUT / "files" / sc / "data" / f"sf_{SCALE[sc]}"
    if sc == "mmqa":
        return base["ground_truth"], "set"
    con = duckdb.connect()
    try:
        if sc == "movie":
            import pandas as pd
            con.register("Movies", pd.read_csv(data / "Movies.csv"))
            con.register("Reviews", pd.read_csv(data / "Reviews.csv"))
            sql = (SEMBENCH_SRC / "files" / "movie" / "query" / "gold_sql" / f"Q{base['qid']}.sql").read_text()
        else:
            con.execute(f"set file_search_path = '{data}'")
            sql = base["ground_truth_sql"]
        df = con.execute(sql).fetchdf()
    finally:
        con.close()
    if sc == "movie" and base["qid"] in (5, 6, 7):
        # the pairs of the gold SQL are determined by the sentiment labels of
        # the movie's reviews; state them by those groups, not as ~32k pairs
        import pandas as pd
        rv = pd.read_csv(data / "Reviews.csv")
        rv = rv[rv["id"] == "ant_man_and_the_wasp_quantumania"]
        groups = {k: sorted(set(map(int, g["reviewId"]))) for k, g in rv.groupby("scoreSentiment")}
        same = base["qid"] == 5
        rule = ("two distinct reviews in the SAME group" if same
                else "one review from each group")
        return {"rule": f"a pair is correct iff it holds {rule}", "groups": groups}, \
            ("any_10_of" if base["qid"] in (5, 6) else "all_pairs")
    rows = json.loads(df.to_json(orient="records", force_ascii=False))
    if df.shape == (1, 1):
        value, kind = rows[0][df.columns[0]], "scalar"
    elif df.shape[1] == 1:
        value, kind = [r[df.columns[0]] for r in rows], "set"
    else:
        value, kind = rows, "table"
    if sc == "movie":
        if base["qid"] in MOVIE_SUBSET_K:
            kind = f"any_{MOVIE_SUBSET_K[base['qid']]}_of"
        elif base["qid"] == 9:
            kind = "scores"
        elif base["qid"] == 10:
            kind = "ranking"
    return value, kind


def render_gold(value: Any, kind: str) -> str:
    """Judge-readable gold answer.  Unordered collections are written in a
    canonical order, so a rebuild reproduces the file byte for byte (queries
    without ORDER BY return rows in an arbitrary order)."""
    if isinstance(value, list) and kind not in ("ranking",):
        value = sorted(value, key=lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True))
    body = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if kind.startswith("any_"):
        k = kind.split("_")[1]
        return f"Any {k} distinct items from this set are correct: {body}"
    if kind == "scores":
        return f"Per-item scores (an answer is correct if its scores agree closely): {body}"
    if kind == "all_pairs":
        return f"All pairs satisfying the rule (as ordered pairs of distinct reviews): {body}"
    if kind == "ranking":
        return f"Reference scores that define the ranking (higher ranks first): {body}"
    return body


def gold_kind(value: Any) -> str:
    if isinstance(value, dict):
        return "map"
    if isinstance(value, list):
        return "set"
    return "scalar"


def build_base(labeler: Optional[Labeler]) -> List[Dict[str, Any]]:
    bases: List[Dict[str, Any]] = []
    nv = Nirvana(labeler)
    for b in parse_nirvana_queries():
        log(f"gold {b['base_id']}: {b['question']}")
        value, source = nv.gold(b["domain"], b["base_id"].rsplit("/", 1)[1])
        kind = gold_kind(value)
        bases.append({"base_id": b["base_id"], "dataset": dataset_of(b), "question": b["question"],
                      "gold_answer": render_gold(value, kind), "gold_kind": kind,
                      "gold_source": source, "needs_review": source != "structured_columns",
                      "note": NIRVANA_NOTES.get(b["base_id"], ""),
                      "original_program": b["original_program"]})
    for b in sembench_queries():
        log(f"gold {b['base_id']}")
        value, kind = sembench_gold(b)
        bases.append({"base_id": b["base_id"], "dataset": dataset_of(b), "question": b["question"],
                      "gold_answer": render_gold(value, kind), "gold_kind": kind,
                      "gold_source": "sembench_official", "needs_review": False,
                      "accuracy_metric": b.get("accuracy_metric"), "note": ""})
    return bases


# =============================================================== variants
def schema_summary(dataset: str) -> str:
    """Tables and columns of a dataset, with a few short sample values."""
    from experiments.benchmark import SEMBENCH, parse_dataset_name
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    name, _ = parse_dataset_name(dataset)
    if name.startswith("nirvana/"):
        files = {"estate": [NIRVANA_DATA / "multimodal_real_estate.parquet"],
                 "imdb": [NIRVANA_DATA / "movie_data.csv"],
                 "steam": [NIRVANA_DATA / "steam_games.csv"]}[name.split("/")[1]]
    else:
        scenario, _, sf = name.split("/", 1)[1].partition("/")
        root = SEMBENCH / scenario / "data" / sf
        files = sorted(p for p in root.rglob("*") if p.suffix in (".csv", ".parquet")
                       and "images" not in p.parts)
    lines = []
    for p in files:
        if p.suffix == ".csv":
            df = pd.read_csv(p, nrows=3)
        else:
            t = pq.read_table(p).slice(0, 3)
            while any(pa.types.is_struct(f.type) for f in t.schema):
                t = t.flatten()
            df = t.to_pandas()
        cols = []
        for c in df.columns:
            sample = [str(v) for v in df[c].tolist() if isinstance(v, (str, int, float)) and len(str(v)) <= 40]
            cols.append(f"{c}" + (f" (e.g. {', '.join(sample[:2])})" if sample else ""))
        lines.append(f"table {p.stem}: " + "; ".join(cols))
    return "\n".join(lines)


def validate_variant(base_question: str, out: Dict[str, Any]) -> None:
    q, res = str(out.get("question") or "").strip(), str(out.get("resolution") or "").strip()
    if not q or not res:
        raise ValueError("needs non-empty 'question' and 'resolution'")
    norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
    if norm(q) == norm(base_question):
        raise ValueError("the variant must differ from the base question")


def answer_procedure(base: Dict[str, Any]) -> str:
    """How the base question is answered: the Nirvana program's operators, or
    SemBench's ground-truth query and modalities."""
    if base.get("original_program"):
        steps = []
        for op in base["original_program"]:
            cols = op.get("input_columns") or [op.get("input_column")]
            steps.append(f"{op['op']}({op.get('user_instruction', '')!r} on {cols})")
        note = f" Note: {base['note']}." if base.get("note") else ""
        return " -> ".join(steps) + note
    sq = {q["base_id"]: q for q in sembench_queries()}[base["base_id"]]
    if sq["scenario"] == "movie":
        return "SQL: " + (SEMBENCH_SRC / "files" / "movie" / "query" / "gold_sql"
                          / f"Q{sq['qid']}.sql").read_text().strip()
    if sq["scenario"] == "ecomm":
        return "SQL over the product tables: " + " ".join(sq["ground_truth_sql"].split())
    return f"answered over the modalities {sq.get('modalities')}"


def make_variant(llm: Any, base: Dict[str, Any], atype: str, schema: str,
                 writer: Any = None, rounds: int = 3, revise_failed: bool = False) -> Dict[str, Any]:
    """One ambiguity variant, written from the base question and how it is
    answered, then checked by an independent pass (``llm``); rejected variants
    are rewritten with the reason (up to ``rounds`` rounds) by ``writer``
    (``llm`` by default).  With ``revise_failed``, a cached variant that failed
    the check is rewritten starting from its rejection."""
    from core.query_planner.llm_query_analyzer import LLMOutputError, complete_json
    writer = writer or llm
    path = CACHE / "variants_v3" / (base["base_id"].replace("/", "_") + f"__{atype}.json")
    cached = json.loads(path.read_text()) if path.exists() else None
    if cached is not None and not (revise_failed and cached["auto_check"] != "passed"):
        return cached
    procedure = answer_procedure(base)
    system = (
        "You build items of a benchmark for query clarification in multimodal data analysis. "
        "Rewrite the analyst's question into a natural question that is ambiguous in exactly the "
        f"given way, '{atype}': {AMBIGUITY_TYPES[atype]}. Requirements: (1) the new question is "
        "MORE ambiguous than the original, never less (do not name the column or modality the "
        "original relies on); (2) the original question stays one valid reading with exactly the "
        "same answer: keep its answer form (a list stays a list, a count stays a count, no new "
        "superlatives, filters, or scopes); (3) ground the ambiguity in the dataset's actual "
        "tables, columns, and modalities; (4) word it differently from the original. The "
        "resolution states, consistently with how the original is answered, which reading the "
        "analyst means. "
        'Return JSON {"question": <ambiguous question>, "resolution": <one sentence>}.')
    user = (f"Dataset schema:\n{schema}\n\nOriginal question: {base['question']}\n"
            f"How the original is answered: {procedure}")
    checker = (
        "You audit an item of a query-clarification benchmark. A rewritten question is SUPPOSED to "
        "admit several readings; the analyst's true reading is the original question, as stated "
        "in the resolution. Given the original question, how it is answered, the ambiguity type "
        f"('{atype}': {AMBIGUITY_TYPES[atype]}), the rewritten question, and the resolution, "
        "decide three things. ambiguous: the rewritten question admits at least two reasonable "
        "readings of this type, one of which is the original. same_answer: read as the "
        "resolution states, the rewritten question asks for exactly the original's answer (same "
        "form and constraints); the OTHER readings are expected and do not count against this. "
        "consistent: the resolution matches how the original is answered. Report false only "
        "for a concrete defect, and name it in reason. "
        'Return JSON {"ambiguous": bool, "same_answer": bool, "consistent": bool, "reason": str}.')

    def check_ok(v: Dict[str, Any]) -> None:
        if not all(isinstance(v.get(k), bool) for k in ("ambiguous", "same_answer", "consistent")):
            raise ValueError("needs boolean ambiguous, same_answer, consistent")

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if cached is not None:      # revise: start from the rejected variant and its reason
        messages += [{"role": "assistant", "content": json.dumps(
                          {"question": cached["question"], "resolution": cached["resolution"]},
                          ensure_ascii=False)},
                     {"role": "user", "content": f"Rejected by the audit: "
                      f"{cached['auto_check'].removeprefix('failed: ')} "
                      "Rewrite it to satisfy all requirements."}]
    verdict: Dict[str, Any] = {}
    out: Dict[str, Any] = {}
    for _ in range(rounds):
        try:
            out = complete_json(writer, messages, dict, lambda v: validate_variant(base["question"], v),
                                "ambiguity variant")
        except LLMOutputError as exc:
            messages.append({"role": "user", "content": f"Rejected: {str(exc)[:300]}. Write a "
                             "differently worded ambiguous question."})
            continue
        verdict = complete_json(llm, [
            {"role": "system", "content": checker},
            {"role": "user", "content": json.dumps({
                "original": base["question"], "answered_by": procedure,
                "rewritten": out["question"], "resolution": out["resolution"]}, ensure_ascii=False)}],
            dict, check_ok, "variant check")
        if verdict["ambiguous"] and verdict["same_answer"] and verdict["consistent"]:
            break
        messages += [{"role": "assistant", "content": json.dumps(out, ensure_ascii=False)},
                     {"role": "user", "content": f"Rejected by the audit: {verdict.get('reason', '')} "
                      "Rewrite it to satisfy all requirements."}]
    if not out:
        raise LLMOutputError(f"no valid variant for {base['base_id']} / {atype}")
    item = {"question": out["question"].strip(), "resolution": out["resolution"].strip(),
            "auto_check": "passed" if verdict.get("ambiguous") and verdict.get("same_answer")
            and verdict.get("consistent") else f"failed: {verdict.get('reason', '')}"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(item, ensure_ascii=False))
    return item


def build_variants(bases: List[Dict[str, Any]], model: str, workers: int,
                   writer_model: Optional[str] = None, rounds: int = 3,
                   revise_failed: bool = False) -> List[Dict[str, Any]]:
    from core.query_planner.llm_query_analyzer import create_provider
    llm = create_provider("dashscope", model)
    writer = create_provider("dashscope", writer_model) if writer_model else llm
    schemas = {d: schema_summary(d) for d in {b["dataset"] for b in bases}}
    jobs = [(b, t) for b in bases for t in AMBIGUITY_TYPES]
    done = [0]
    lock = threading.Lock()

    def run(job):
        b, t = job
        v = make_variant(llm, b, t, schemas[b["dataset"]], writer, rounds, revise_failed)
        with lock:
            done[0] += 1
            if done[0] % 25 == 0:
                log(f"variants {done[0]}/{len(jobs)}")
        return {"id": f"{b['base_id']}#{t}", "dataset": b["dataset"], "base_id": b["base_id"],
                "ambiguity_type": t, "question": v["question"],
                "hidden_intent": f"{b['question']} ({v['resolution']})",
                "gold_answer": b["gold_answer"], "gold_kind": b["gold_kind"],
                "gold_source": b["gold_source"], "auto_check": v["auto_check"],
                "needs_review": True}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, jobs))


# =============================================================== main
def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default="data,base,variants")
    ap.add_argument("--model", default="qwen-max", help="reference text model and variant writer")
    ap.add_argument("--vision-model", default="qwen3-vl-plus", help="reference vision model")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--writer-model", default=None,
                    help="variant writer (default: --model); the audit always uses --model")
    ap.add_argument("--rounds", type=int, default=3, help="write/audit rounds per variant")
    ap.add_argument("--revise-failed", action="store_true",
                    help="rewrite cached variants that failed the audit")
    args = ap.parse_args()
    stages = set(args.stages.split(","))
    t0 = time.time()
    if "data" in stages:
        stage_data()
    base_path = BENCH / "base_queries.jsonl"
    if "base" in stages:
        labeler = Labeler(args.model, args.vision_model, args.workers)
        bases = build_base(labeler)
        if (n := sum(b["base_id"].startswith("nirvana/") for b in bases)) != 36 or len(bases) != 71:
            raise RuntimeError(f"expected 36 + 35 base queries, got {n} + {len(bases) - n}")
        write_jsonl(base_path, bases)
        log(f"wrote {base_path} ({len(bases)} base queries)")
    if "variants" in stages:
        bases = [json.loads(l) for l in open(base_path, encoding="utf-8")]
        items = build_variants(bases, args.model, args.workers, args.writer_model,
                               args.rounds, args.revise_failed)
        write_jsonl(BENCH / "nirvana_bench.jsonl", [i for i in items if i["base_id"].startswith("nirvana/")])
        write_jsonl(BENCH / "sembench.jsonl", [i for i in items if i["base_id"].startswith("sembench/")])
        by_base = {b["base_id"]: b for b in bases}
        with open(BENCH / "review_checklist.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "ambiguity_type", "base_question", "variant_question", "hidden_intent",
                        "gold_answer", "gold_source", "auto_check", "note", "ok (y/n)", "comment"])
            for i in items:
                b = by_base[i["base_id"]]
                w.writerow([i["id"], i["ambiguity_type"], b["question"], i["question"],
                            i["hidden_intent"], i["gold_answer"][:500], i["gold_source"],
                            i["auto_check"], b.get("note", ""), "", ""])
        log(f"wrote {len(items)} items (nirvana_bench.jsonl, sembench.jsonl, review_checklist.csv)")
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

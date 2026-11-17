"""Offline micro-benchmarks of the registered physical implementations.

Measures the profile (cost, conf, gpu, cpu, mem) of every implementation in
I(o) (paper Sec. V-A/V-D: "measured by offline micro-benchmarks when an
implementation is registered") and writes them, with their provenance, to
src/core/query_planner/impl_profiles.json, which default_impl_registry.py loads.

Every implementation runs through its executor (plan_executor.py) on real,
labeled data:

  * SemBench Movie sf_2000 (Movies 116 rows, Reviews 2000 rows): relational
    operators, sentiment (Reviews.scoreSentiment), review-to-movie matching
    (Reviews.id = Movies.id), same-movie review joins;
  * SemBench E-Commerce sf_500 (500 products with images): image extraction
    and encoders (masterCategory), semantic search (articleType);
  * the TiMetaGraph of SemBench Movie (precomputed sentiment, entity links).

Profile fields
  cost  wall-clock seconds per 1,000 input rows (sum over the operator's
        inputs), the median of REPEATS runs for deterministic implementations
        and of one run for model calls; encoders and indexes are loaded/built
        before timing (their one-off time is in the provenance).
  conf  accuracy on clean labeled inputs (paper Sec. V-C: reliability given
        correct inputs).  A relational implementation scores 1 per query
        whose output equals the reference result computed independently with
        pandas, else 0; a model-based one scores its accuracy against the
        labels.  cost and conf are means over the implementation's task
        families, each family weighing equally (PREDICATE_FILTER serves
        structured filters and semantic ones; CONTENT_EXTRACT implementations
        text and images).
  cpu   CPU time / wall time of the process during the run (cores in use).
  gpu   MB of accelerator memory: the parameters of the encoders it runs on
        the device (MPS/CUDA) plus the allocator growth during the run;
        remote model calls count 0.
  mem   peak MB of Python/numpy allocations during the run (tracemalloc).

LLM: DashScope deepseek-v4.1-flash-1017 (text) and qwen3-vl-plus (images),
temperature 0.  Samples are drawn with random.Random(SEED).  Measurements of
model-based implementations are cached under runs/microbench/ (--refresh
re-measures them).

Usage: python -m scripts.microbenchmark_impls [--graph PATH] [--refresh]
"""
import argparse
import csv
import hashlib
import io
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from core.query_planner import plan_executor as pe  # noqa: E402
from core.query_planner.mcts_plan_search import LogicalPlanNode  # noqa: E402
from core.query_planner.metadata_graph import (  # noqa: E402
    doc_key, document_entities, entity_ids_by_surface)

MOVIE = REPO / "data/sembench/files/movie/data/sf_2000"
ECOMM = REPO / "data/sembench/files/ecomm/data/sf_500"
DEFAULT_GRAPH = REPO / "runs/metadata_graphs/sembench_movie_sf_2000.pkl"
OUT = REPO / "src/core/query_planner/impl_profiles.json"
CACHE = REPO / "runs/microbench"
LLM_MODEL, VISION_MODEL = "deepseek-v4.1-flash-1017", "qwen3-vl-plus"
SEED, REPEATS = 0, 5
N_SENTIMENT, N_FILTER, N_MATCH, N_SIMJOIN_PROBES, N_IMAGES = 200, 200, 60, 40, 100
EXTERNAL_SORT_RUN_ROWS = 256

# the registry, in the order planners list the implementations of a type
REGISTRY = {
    "SCAN": ["TABLE_SCAN", "INDEX_SCAN"],
    "FILTER": ["PREDICATE_FILTER", "THRESHOLD_FILTER"],
    "PROJECT": ["COLUMN_PROJECT"],
    "AGGREGATE": ["HASH_AGGREGATE", "STREAM_AGGREGATE"],
    "SORT": ["MEMORY_SORT", "EXTERNAL_SORT"],
    "LIMIT": ["STREAMING_LIMIT"],
    "DISTINCT": ["HASH_DISTINCT", "STREAM_DISTINCT"],
    "JOIN": ["HASH_JOIN", "NESTED_LOOP_JOIN"],
    "UNION": ["UNION_ALL"],
    "SEMANTIC_SEARCH": ["VECTOR_SEARCH", "VECTOR_SCAN"],
    "CROSS_MODAL_MATCH": ["ENTITY_LINK_MATCH", "LLM_MATCH"],
    "CONTENT_EXTRACT": ["METADATA_LOOKUP", "TOOL_EXTRACT", "LLM_EXTRACT"],
    "SIMILARITY_JOIN": ["INTRA_MODAL_JOIN", "SEMANTIC_JOIN"],
    "FEATURE_TRANSFORM": ["CLIP_IMAGE_ENCODER", "CLIP_TEXT_ENCODER"],
    "VISUALIZE": ["MATERIALIZE"],
    "EXPORT": ["CSV_EXPORT", "JSON_EXPORT"],
}
# encoders each implementation runs on the device
DEVICE_MODELS = {
    "VECTOR_SEARCH": [pe.TEXT_MODEL], "VECTOR_SCAN": [pe.TEXT_MODEL],
    "INTRA_MODAL_JOIN": [pe.TEXT_MODEL], "CLIP_IMAGE_ENCODER": [pe.CLIP_MODEL],
    "CLIP_TEXT_ENCODER": [pe.CLIP_MODEL], "TOOL_EXTRACT": [pe.CLIP_MODEL],
}


# ---------------------------------------------------------------- data
def read_csv(path: Path) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def load_data() -> Dict[str, Any]:
    movies, reviews = read_csv(MOVIE / "Movies.csv"), read_csv(MOVIE / "Reviews.csv")
    styles = pd.read_parquet(ECOMM / "styles.parquet")
    images = pd.read_parquet(ECOMM / "image_mapping.parquet")
    styles["id"] = styles["id"].astype(str)
    products = styles.merge(images, on="id").sort_values("id", kind="stable")
    products = [{"id": str(r.id), "productDisplayName": r.productDisplayName,
                 "masterCategory": r.masterCategory, "articleType": r.articleType,
                 "image": {"bytes": (ECOMM / "images" / r.filename).read_bytes()}}
                for r in products.itertuples()]
    return {"movies": movies, "reviews": reviews, "products": products}


def sample(rows: List[Any], n: int, salt: str) -> List[Any]:
    rng = random.Random(f"{SEED}:{salt}")
    return rows if n >= len(rows) else rng.sample(rows, n)


# ---------------------------------------------------------------- measurement
def _device_mb(names: List[str]) -> float:
    total = 0
    for name in names:
        model = pe._encoder(name)
        if str(model.device) != "cpu":
            total += sum(p.numel() * p.element_size() for p in model.parameters())
    return total / 2 ** 20


def _accel_allocated() -> int:
    try:
        import torch
        if torch.backends.mps.is_available():
            return torch.mps.driver_allocated_memory()
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated()
    except Exception:
        pass
    return 0


def run_once(fn: Callable[[], Any], trace_memory: bool) -> Dict[str, Any]:
    if trace_memory:
        tracemalloc.start()
    accel0, cpu0, t0 = _accel_allocated(), time.process_time(), time.perf_counter()
    out = fn()
    wall, cpu = time.perf_counter() - t0, time.process_time() - cpu0
    accel = max(0, _accel_allocated() - accel0) / 2 ** 20
    peak = tracemalloc.get_traced_memory()[1] / 2 ** 20 if trace_memory else None
    if trace_memory:
        tracemalloc.stop()
    return {"out": out, "wall": wall, "cpu": cpu, "accel_mb": accel, "peak_mb": peak}


def measure(fn: Callable[[], Any], deterministic: bool) -> Dict[str, Any]:
    """Timing without tracemalloc (median of REPEATS runs if deterministic),
    then the peak memory of one traced run (the timed run itself for model
    calls, which are not repeated)."""
    if not deterministic:
        r = run_once(fn, trace_memory=True)
        return {**r, "runs": 1}
    runs = [run_once(fn, trace_memory=False) for _ in range(REPEATS)]
    traced = run_once(fn, trace_memory=True)
    walls = [r["wall"] for r in runs]
    mid = runs[walls.index(sorted(walls)[len(walls) // 2])]
    return {"out": mid["out"], "wall": statistics.median(walls), "cpu": mid["cpu"],
            "accel_mb": max(r["accel_mb"] for r in runs), "peak_mb": traced["peak_mb"],
            "runs": REPEATS}


class Bench:
    def __init__(self, data, graph, llm, vision, refresh: bool):
        self.data, self.graph, self.llm, self.vision = data, graph, llm, vision
        self.refresh = refresh
        self.results: Dict[str, List[dict]] = {i: [] for impls in REGISTRY.values() for i in impls}

    def ctx(self, bindings: Dict[str, Any], sources: Optional[Dict[str, list]] = None,
            **metadata) -> pe.ExecutionContext:
        return pe.ExecutionContext(llm_provider=self.llm, vision_provider=self.vision,
                                   metadata={"bindings": bindings, "sources": sources or {},
                                             **metadata})

    def run(self, op: str, impl: str, task: str, inputs: List[list], ctx: pe.ExecutionContext,
            score: Callable[[list], float], node: Optional[LogicalPlanNode] = None,
            model: bool = False, input_rows: Optional[int] = None,
            family: Optional[str] = None) -> None:
        """Execute ``impl`` on ``inputs``, time it, and score its output
        (``input_rows``: rows a leaf reads, i.e. its source size)."""
        executor = pe.EXECUTOR_REGISTRY[op]
        node = node or LogicalPlanNode(node_id=f"bench-{op}", operator_type=op)
        key = hashlib.sha1(json.dumps(
            [impl, task, LLM_MODEL, VISION_MODEL, SEED, [len(x) for x in inputs],
             hashlib.sha1(json.dumps(inputs, default=str, sort_keys=True).encode()).hexdigest()],
            default=str).encode()).hexdigest()[:16]
        cache = CACHE / f"{impl}.{task}.{key}.json"
        if model and cache.exists() and not self.refresh:
            rec = json.loads(cache.read_text())
            rec["family"] = family or task
        else:
            m = measure(lambda: executor.execute(inputs, ctx, node, impl), deterministic=not model)
            n_in = input_rows if input_rows is not None else sum(len(x) for x in inputs)
            rec = {"op": op, "impl": impl, "task": task, "family": family or task,
                   "input_rows": n_in,
                   "output_rows": len(m["out"]), "seconds": m["wall"], "runs": m["runs"],
                   "cpu_cores": m["cpu"] / max(m["wall"], 1e-9),
                   "accel_mb": m["accel_mb"], "peak_mb": m["peak_mb"],
                   "accuracy": float(score(m["out"])),
                   "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            if model:
                CACHE.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(rec, indent=1))
        self.results[impl].append(rec)
        print(f"  {impl:20s} {task:28s} n={rec['input_rows']:5d} "
              f"acc={rec['accuracy']:.3f} {rec['seconds']:.4f}s", flush=True)


# ---------------------------------------------------------------- references
def rows_df(rows: List[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).astype(str)


def same_rows(out: List[dict], ref: List[dict], ordered: bool = False) -> bool:
    enc = [json.dumps(r, sort_keys=True, default=str) for r in out]
    want = [json.dumps(r, sort_keys=True, default=str) for r in ref]
    return enc == want if ordered else sorted(enc) == sorted(want)


def ref_filter(rows: List[dict], conds: List[dict]) -> List[dict]:
    """Equality/IN conditions with pandas (case-insensitive, as the planner's
    conditions are)."""
    df = rows_df(rows)
    mask = pd.Series(True, index=df.index)
    for c in conds:
        col = df[c["column"]].str.lower()
        vals = c["value"] if c["op"] == "in" else [c["value"]]
        mask &= col.isin([str(v).lower() for v in vals])
    return [rows[i] for i in df.index[mask]]


# ---------------------------------------------------------------- relational tasks
def bench_relational(b: Bench) -> None:
    movies, reviews = b.data["movies"], b.data["reviews"]
    critics = sorted({r["criticName"] for r in reviews})[:40:8]
    sources = {"Reviews": reviews, "Movies": movies}

    print("SCAN")
    scan_queries = [
        ("Reviews", [{"column": "scoreSentiment", "op": "=", "value": "POSITIVE"}]),
        ("Reviews", [{"column": "reviewState", "op": "=", "value": "rotten"},
                     {"column": "isTopCritic", "op": "=", "value": "True"}]),
        ("Reviews", [{"column": "criticName", "op": "in", "value": critics}]),
        ("Movies", [{"column": "originalLanguage", "op": "=", "value": "English"}]),
        ("Movies", [{"column": "rating", "op": "in", "value": ["PG-13", "R"]}]),
    ]
    build = {}
    for i, (src, conds) in enumerate(scan_queries):
        fb = {"source": src, "conditions": conds}
        ref = ref_filter(sources[src], conds)
        for impl in REGISTRY["SCAN"]:
            filt = LogicalPlanNode(node_id="f", operator_type="FILTER")
            scan = LogicalPlanNode(node_id="s", operator_type="SCAN", parent=filt, source_name=src)
            filt.children = [scan]
            ctx = b.ctx({"FILTER": fb}, sources)
            if impl == "INDEX_SCAN":   # the index exists before the query (built once)
                t0 = time.perf_counter()
                for c in conds:
                    pe.ScanExecutor.index(src, sources[src], c["column"])
                build.setdefault(f"{src}.{'+'.join(c['column'] for c in conds)}",
                                 time.perf_counter() - t0)

            def score(out, ctx=ctx, filt=filt, ref=ref):
                kept = pe.FilterExecutor().execute([out], ctx, filt, "PREDICATE_FILTER")
                return same_rows(kept, ref)
            b.run("SCAN", impl, f"scan_q{i}", [], ctx, score, node=scan,
                  input_rows=len(sources[src]))
    b.index_build_seconds = build

    print("FILTER (structured)")
    for i, (src, conds) in enumerate(scan_queries + [
            ("Movies", [{"column": "genre", "op": "contains", "value": "Sci-fi"},
                        {"column": "releaseDateTheaters", "op": ">=", "value": "2020-01-01"}])]):
        rows = sources[src]
        if all(c["op"] in ("=", "in") for c in conds):
            ref = ref_filter(rows, conds)
        else:   # the running example's filter, with pandas
            df = rows_df(rows)
            mask = df["genre"].str.lower().str.contains("sci-fi", regex=False) & \
                (df["releaseDateTheaters"].str.lower() >= "2020-01-01") & (df["releaseDateTheaters"] != "")
            ref = [rows[j] for j in df.index[mask]]
        b.run("FILTER", "PREDICATE_FILTER", f"filter_q{i}", [rows],
              b.ctx({"FILTER": {"source": src, "conditions": conds}}),
              lambda out, ref=ref: same_rows(out, ref, True), family="structured")

    print("PROJECT")
    for i, cols in enumerate([["id", "reviewText"], ["criticName", "scoreSentiment", "creationDate"]]):
        ref = rows_df(reviews)[cols].to_dict("records")
        b.run("PROJECT", "COLUMN_PROJECT", f"project_q{i}", [reviews],
              b.ctx({"PROJECT": {"columns": cols}}), lambda out, ref=ref: same_rows(out, ref, True))

    print("AGGREGATE")
    agg_queries = [
        (reviews, {"group_by": ["id"], "function": "count"}, "count"),
        (reviews, {"group_by": ["id"], "function": "avg", "attribute": "scoreSentiment",
                   "condition": {"op": "=", "value": "POSITIVE"}, "as": "share"}, "share"),
        (movies, {"group_by": ["rating"], "function": "avg", "attribute": "runtimeMinutes",
                  "as": "avg_runtime"}, "mean"),
        (movies, {"group_by": ["originalLanguage"], "function": "max",
                  "attribute": "audienceScore", "as": "best"}, "max"),
        (reviews, {"function": "count"}, "global"),
    ]
    for i, (rows, spec, kind) in enumerate(agg_queries):
        ref = ref_aggregate(rows, spec, kind)
        for impl in REGISTRY["AGGREGATE"]:
            b.run("AGGREGATE", impl, f"aggregate_q{i}", [rows], b.ctx({"AGGREGATE": spec}),
                  lambda out, ref=ref: agg_equal(out, ref))

    print("SORT")
    sort_queries = [(reviews, "creationDate", True), (reviews, "reviewId", False),
                    (movies, "title", False), (movies, "audienceScore", True)]
    for i, (rows, key, desc) in enumerate(sort_queries):
        ref = ref_sort(rows, key, desc)
        for impl in REGISTRY["SORT"]:
            spec = {"key": key, "descending": desc, "run_rows": EXTERNAL_SORT_RUN_ROWS}
            b.run("SORT", impl, f"sort_q{i}", [rows], b.ctx({"SORT": spec}),
                  lambda out, ref=ref: same_rows(out, ref, True))

    print("LIMIT")
    for i, k in enumerate([1, 10, 500]):
        ref = rows_df(reviews).head(k).to_dict("records")
        b.run("LIMIT", "STREAMING_LIMIT", f"limit_q{i}", [reviews],
              b.ctx({"LIMIT": {"count": k}}), lambda out, ref=ref: same_rows(out, ref, True))

    print("DISTINCT")
    for i, cols in enumerate([["criticName"], ["id", "scoreSentiment"], ["genre"]]):
        rows = movies if cols == ["genre"] else reviews
        ref = [rows[j] for j in rows_df(rows).drop_duplicates(subset=cols, keep="first").index]
        for impl in REGISTRY["DISTINCT"]:
            b.run("DISTINCT", impl, f"distinct_q{i}", [rows], b.ctx({"DISTINCT": {"columns": cols}}),
                  lambda out, ref=ref: same_rows(out, ref))

    print("JOIN")
    left = [{"movie_id": m["id"], "title": m["title"]} for m in movies]
    right = [{"id": r["id"], "reviewId": r["reviewId"]} for r in reviews]
    merged = rows_df(left).merge(rows_df(right), left_on="movie_id", right_on="id")
    ref = merged[["movie_id", "title", "id", "reviewId"]].to_dict("records")
    for impl in REGISTRY["JOIN"]:
        b.run("JOIN", impl, "join_movie_reviews", [left, right],
              b.ctx({"JOIN": {"on": ["movie_id", "id"]}}), lambda out, ref=ref: same_rows(out, ref))

    print("UNION")
    halves = [reviews[:1000], reviews[1000:]]
    ref = pd.concat([rows_df(h) for h in halves]).to_dict("records")
    b.run("UNION", "UNION_ALL", "union_halves", halves, b.ctx({}),
          lambda out, ref=ref: same_rows(out, ref, True))

    print("VISUALIZE")
    spec = {"chart": "bar", "x": "title", "y": "audienceScore"}
    vctx = b.ctx({"VISUALIZE": spec})
    b.run("VISUALIZE", "MATERIALIZE", "materialize_movies", [movies], vctx,
          lambda out: same_rows(out, movies, True) and vctx.metadata["chart"] == spec)

    print("EXPORT")
    tmp = Path(tempfile.mkdtemp(prefix="microbench_export_"))
    for impl, ext in (("CSV_EXPORT", ".csv"), ("JSON_EXPORT", ".json")):
        path = tmp / f"reviews{ext}"

        def score(out, path=path, ext=ext):
            back = read_csv(path) if ext == ".csv" else json.loads(path.read_text())
            return same_rows(back, reviews, True)
        b.run("EXPORT", impl, f"export{ext}", [reviews], b.ctx({"EXPORT": {"path": str(path)}}), score)


def ref_aggregate(rows, spec, kind):
    df = rows_df(rows)
    name = spec.get("as") or spec["function"]
    if kind == "global":
        return [{name: float(len(df))}]
    key = spec["group_by"][0]
    if kind == "count":
        g = df.groupby(key, sort=False).size()
    elif kind == "share":
        g = (df[spec["attribute"]].str.lower() == spec["condition"]["value"].lower()) \
            .astype(float).groupby(df[key], sort=False).mean()
    else:
        vals = pd.to_numeric(df[spec["attribute"]].str.replace(",", ""), errors="coerce")
        g = getattr(vals.groupby(df[key], sort=False), kind)()
    return [{key: k, name: (None if pd.isna(v) else float(v))} for k, v in g.items()]


def agg_equal(out, ref) -> bool:
    def norm(rows):
        return sorted(json.dumps({k: (round(v, 9) if isinstance(v, float) else v)
                                  for k, v in r.items()}, sort_keys=True) for r in rows)
    return norm([{k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                  for k, v in r.items()} for r in out]) == norm(ref)


def ref_sort(rows, key, desc):
    df = rows_df(rows)
    num = pd.to_numeric(df[key].str.replace(",", ""), errors="coerce")
    by = num if num.notna().all() else df[key]
    order = by.sort_values(ascending=not desc, kind="stable").index
    return [rows[i] for i in order]


# ---------------------------------------------------------------- model-based tasks
def positive(label: str) -> bool:
    return label.strip().upper() == "POSITIVE"


def bench_filter(b: Bench) -> None:
    print("FILTER (semantic: positive review)")
    rows = [{"reviewText": r["reviewText"], "reviewId": r["reviewId"]}
            for r in sample([r for r in b.data["reviews"] if r["reviewText"].strip()],
                            N_FILTER, "filter")]
    gold = {r["reviewId"]: positive(label_of(b, r["reviewId"])) for r in rows}
    spec = {"semantic": "the review expresses a positive opinion of the movie",
            "columns": ["reviewText"]}

    def score(out):
        kept = {r["reviewId"] for r in out}
        return sum((rid in kept) == g for rid, g in gold.items()) / len(gold)
    for impl in REGISTRY["FILTER"]:
        b.run("FILTER", impl, "filter_positive_reviews", [rows], b.ctx({"FILTER": spec}),
              score, model=True, family="semantic")


def label_of(b: Bench, review_id: str) -> str:
    if not hasattr(b, "_labels"):
        b._labels = {r["reviewId"]: r["scoreSentiment"] for r in b.data["reviews"]}
    return b._labels[review_id]


def bench_content_extract(b: Bench) -> None:
    print("CONTENT_EXTRACT (text sentiment)")
    # the values TiMetaGraph precomputed, on the documents it covers
    pre = {k: v for meta in b.graph.modalities.values()
           for k, v in (((meta.raw_metadata or {}).get("doc_attributes") or {})
                        .get("sentiment") or {}).items()}
    text = [r for r in b.data["reviews"] if r["reviewText"].strip()]
    covered = [r for r in text if doc_key(r["reviewText"]) in pre]
    tasks = {"METADATA_LOOKUP": covered,
             "TOOL_EXTRACT": sample(text, N_SENTIMENT, "sentiment"),
             "LLM_EXTRACT": sample(text, N_SENTIMENT, "sentiment")}
    spec = {"attribute": "sentiment", "columns": ["reviewText"], "type": "number",
            "description": "the sentiment of the review as a number from -1 (negative) "
                           "to 1 (positive)"}
    for impl, revs in tasks.items():
        rows = [{"reviewText": r["reviewText"]} for r in revs]
        gold = [positive(r["scoreSentiment"]) for r in revs]

        def score(out, gold=gold):
            return sum(((pe._number(o["sentiment"]) or 0.0) > 0) == g
                       for o, g in zip(out, gold)) / len(gold)
        b.run("CONTENT_EXTRACT", impl, "sentiment_text", [rows],
              b.ctx({"CONTENT_EXTRACT": spec}, doc_attributes={"sentiment": pre}), score,
              model=impl == "LLM_EXTRACT")

    print("CONTENT_EXTRACT (image category)")
    products = sample(b.data["products"], N_IMAGES, "images")
    cats = sorted({p["masterCategory"] for p in b.data["products"]})
    rows = [{"image": p["image"]} for p in products]
    gold = [p["masterCategory"] for p in products]
    spec = {"attribute": "category", "columns": ["image"], "type": "text", "values": cats,
            "description": f"the product's category, one of {', '.join(cats)}"}

    def score(out):
        return sum(str(o["category"]).strip().strip(".").lower() == g.lower()
                   for o, g in zip(out, gold)) / len(gold)
    for impl in ("TOOL_EXTRACT", "LLM_EXTRACT"):
        b.run("CONTENT_EXTRACT", impl, "category_image", [rows],
              b.ctx({"CONTENT_EXTRACT": spec}), score, model=impl == "LLM_EXTRACT")


def match_rows(b: Bench):
    """Movies (title and descriptive columns) and the reviews TiMetaGraph
    processed, with the gold movie of every review."""
    docs = document_entities(b.graph)
    movies = [{k: m[k] for k in ("title", "genre", "director", "writer", "releaseDateTheaters")}
              | {"_movie": m["id"]} for m in b.data["movies"]]
    reviews = [r for r in b.data["reviews"] if doc_key(r["reviewText"]) in docs]
    return movies, reviews


def bench_cross_modal_match(b: Bench) -> None:
    print("CROSS_MODAL_MATCH (review -> movie)")
    movies, reviews = match_rows(b)
    ctx_meta = {"doc_entities": document_entities(b.graph),
                "entity_ids_by_surface": entity_ids_by_surface(b.graph)}
    tasks = {"ENTITY_LINK_MATCH": reviews, "LLM_MATCH": sample(reviews, N_MATCH, "match")}
    for impl, revs in tasks.items():
        rows = [{"reviewText": r["reviewText"], "_review": r["reviewId"]} for r in revs]
        gold = {r["reviewId"]: r["id"] for r in revs}
        lrows = [{k: v for k, v in m.items() if k != "_movie"} for m in movies]
        rrows = [{"reviewText": r["reviewText"]} for r in rows]
        by_title = {m["title"]: m["_movie"] for m in movies}
        by_text = {r["reviewText"]: r["_review"] for r in rows}

        def score(out, gold=gold, by_title=by_title, by_text=by_text):
            got: Dict[str, set] = {}
            for o in out:
                got.setdefault(by_text[o["reviewText"]], set()).add(by_title[o["title"]])
            return sum(got.get(rid, set()) == {mid} for rid, mid in gold.items()) / len(gold)
        b.run("CROSS_MODAL_MATCH", impl, "review_to_movie", [lrows, rrows],
              b.ctx({"CROSS_MODAL_MATCH": {"condition": "the review is about the movie"}},
                    **ctx_meta), score, model=impl == "LLM_MATCH")


def bench_similarity_join(b: Bench) -> None:
    print("SIMILARITY_JOIN (reviews of the same movie)")
    reviews = [r for r in b.data["reviews"] if r["reviewText"].strip()]
    per_movie: Dict[str, list] = {}
    for r in reviews:
        per_movie.setdefault(r["id"], []).append(r)
    probes = sample([r for r in reviews if len(per_movie[r["id"]]) >= 2], N_SIMJOIN_PROBES, "simjoin")
    probe_ids = {r["reviewId"] for r in probes}
    # candidates: another review of each probe's movie plus random reviews (120 rows)
    cands = {}
    for r in probes:
        other = next(x for x in per_movie[r["id"]] if x["reviewId"] not in probe_ids)
        cands[other["reviewId"]] = other
    for r in sample([r for r in reviews if r["reviewId"] not in probe_ids], 400, "simjoin-c"):
        if len(cands) >= 120:
            break
        cands.setdefault(r["reviewId"], r)
    left = [{"candidate": r["reviewText"]} for r in cands.values()]
    right = [{"probe": r["reviewText"]} for r in probes]
    movie_of = {r["reviewText"]: r["id"] for r in list(cands.values()) + probes}

    def score(out):
        got: Dict[str, set] = {}
        for o in out:
            got.setdefault(o["probe"], set()).add(movie_of[o["candidate"]])
        return sum(got.get(p["probe"], set()) == {movie_of[p["probe"]]} for p in right) / len(right)
    spec = {"condition": "the two reviews are about the same movie",
            "columns": ["candidate", "probe"]}
    for impl in REGISTRY["SIMILARITY_JOIN"]:
        b.run("SIMILARITY_JOIN", impl, "same_movie_reviews", [left, right],
              b.ctx({"SIMILARITY_JOIN": spec}), score, model=impl == "SEMANTIC_JOIN")


def bench_semantic_search(b: Bench) -> None:
    print("SEMANTIC_SEARCH (products by article type)")
    products = b.data["products"]
    rows = [{"productDisplayName": p["productDisplayName"], "_type": p["articleType"]}
            for p in products]
    counts: Dict[str, int] = {}
    for p in products:
        counts[p["articleType"]] = counts.get(p["articleType"], 0) + 1
    queries = sorted(counts, key=lambda t: (-counts[t], t))[:5]
    for impl in REGISTRY["SEMANTIC_SEARCH"]:
        for q in queries:
            spec = {"query": q, "columns": ["productDisplayName"], "k": counts[q]}
            b.run("SEMANTIC_SEARCH", impl, f"search_{q.replace(' ', '_')}", [rows],
                  b.ctx({"SEMANTIC_SEARCH": spec}),
                  lambda out, q=q: sum(o["_type"] == q for o in out) / len(out))


def bench_feature_transform(b: Bench) -> None:
    print("FEATURE_TRANSFORM (zero-shot category from the embedding)")
    import numpy as np
    products = sample(b.data["products"], N_IMAGES, "images")
    cats = sorted({p["masterCategory"] for p in b.data["products"]})
    prompts = pe._embed_texts([f"a photo of a {c.lower()} product" for c in cats], pe.CLIP_MODEL)
    gold = [p["masterCategory"] for p in products]

    def score(out):
        v = np.array([o["vec"] for o in out])
        return float(np.mean([cats[i] == g for i, g in zip((v @ prompts.T).argmax(1), gold)]))
    for impl, col in (("CLIP_IMAGE_ENCODER", "image"), ("CLIP_TEXT_ENCODER", "productDisplayName")):
        rows = [{col: p[col]} for p in products]
        b.run("FEATURE_TRANSFORM", impl, f"category_from_{col}", [rows],
              b.ctx({"FEATURE_TRANSFORM": {"columns": [col], "as": "vec"}}), score)


# ---------------------------------------------------------------- profiles
def profile(impl: str, recs: List[dict]) -> Dict[str, Any]:
    if not recs:
        raise RuntimeError(f"{impl} was not measured")
    # every task family weighs equally (e.g. structured and semantic filters)
    families: Dict[str, List[dict]] = {}
    for r in recs:
        families.setdefault(r["family"], []).append(r)

    def mean_over_families(f):
        return statistics.mean(statistics.mean(f(r) for r in rs) for rs in families.values())
    return {
        "impl_type": impl,
        "base_cost": round(mean_over_families(
            lambda r: r["seconds"] / max(r["input_rows"], 1) * 1000), 6),
        "base_confidence": round(mean_over_families(lambda r: r["accuracy"]), 4),
        "gpu_memory_mb": round(_device_mb(DEVICE_MODELS.get(impl, []))
                               + max(r["accel_mb"] for r in recs), 1),
        "cpu_cores": round(max(statistics.mean(r["cpu_cores"] for r in recs), 0.01), 2),
        "memory_mb": round(max(max(r["peak_mb"] for r in recs), 0.1), 1),
        "tasks": [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()
                   if k not in ("op", "impl")} for r in recs],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--graph", default=str(DEFAULT_GRAPH))
    ap.add_argument("--refresh", action="store_true", help="re-measure model-based runs")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    import pickle
    from core.query_planner.llm_query_analyzer import create_provider
    with open(args.graph, "rb") as f:
        graph = pickle.load(f)
    graph = graph[0] if isinstance(graph, tuple) else graph
    data = load_data()
    llm = create_provider("dashscope", LLM_MODEL)
    vision = create_provider("dashscope", VISION_MODEL)
    load_s = {}
    for name in (pe.TEXT_MODEL, pe.CLIP_MODEL):   # loaded before any timing
        t0 = time.perf_counter()
        pe._encoder(name)
        pe._embed_texts(["warm up"], name)
        load_s[name] = round(time.perf_counter() - t0, 3)

    b = Bench(data, graph, llm, vision, args.refresh)
    bench_relational(b)
    bench_filter(b)
    bench_content_extract(b)
    bench_cross_modal_match(b)
    bench_similarity_join(b)
    bench_semantic_search(b)
    bench_feature_transform(b)

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                                text=True).stdout.strip()
    except OSError:
        commit = ""
    import torch
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    doc = {
        "description": "Profiles of the registered physical implementations, measured by "
                       "scripts/microbenchmark_impls.py; default_impl_registry.py loads them.",
        "units": {
            "base_cost": "wall-clock seconds per 1,000 input rows",
            "base_confidence": "accuracy on clean labeled inputs (relational: share of queries "
                               "whose output equals the pandas reference)",
            "gpu_memory_mb": "accelerator memory: on-device encoder parameters + allocator growth",
            "cpu_cores": "process CPU time / wall time",
            "memory_mb": "peak Python/numpy allocations (tracemalloc)",
        },
        "provenance": {
            "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": commit,
            "machine": {"platform": platform.platform(), "processor": platform.processor(),
                        "python": platform.python_version(), "cpu_count": os.cpu_count(),
                        "accelerator": device},
            "llm": {"provider": "dashscope", "text": LLM_MODEL, "vision": VISION_MODEL,
                    "temperature": 0.0},
            "encoders": {"text": pe.TEXT_MODEL, "clip": pe.CLIP_MODEL,
                         "load_seconds": load_s},
            "data": {"movie": str(MOVIE.relative_to(REPO)), "ecomm": str(ECOMM.relative_to(REPO)),
                     "metadata_graph": str(Path(args.graph).resolve().relative_to(REPO))
                     if Path(args.graph).resolve().is_relative_to(REPO) else args.graph},
            "seed": SEED, "repeats": REPEATS,
            "samples": {"sentiment": N_SENTIMENT, "semantic_filter": N_FILTER,
                        "llm_match_reviews": N_MATCH, "similarity_join_probes": N_SIMJOIN_PROBES,
                        "images": N_IMAGES},
            "external_sort_run_rows": EXTERNAL_SORT_RUN_ROWS,
            "index_build_seconds": {k: round(v, 6) for k, v in b.index_build_seconds.items()},
        },
        "profiles": {op: [profile(i, b.results[i]) for i in impls] for op, impls in REGISTRY.items()},
    }
    Path(args.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(f"\nwrote {args.out}")
    for op, impls in doc["profiles"].items():
        conf = statistics.mean(p["base_confidence"] for p in impls)
        print(f"{op:18s} c0={conf:.3f}  " + "  ".join(
            f"{p['impl_type']}(cost={p['base_cost']:.4g}, conf={p['base_confidence']:.3f})"
            for p in impls))


if __name__ == "__main__":
    main()

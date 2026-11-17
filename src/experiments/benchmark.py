"""Benchmark runner implementing the evaluation protocol of the paper (Sec. VI).

Items are read from a JSONL file, one query per line:

    {"id": "estate/q1/cross_modal_referential", "dataset": "nirvana/estate",
     "question": "<query as posed, possibly ambiguous>",
     "hidden_intent": "<what the user means; drives the user simulator>",
     "gold_answer": "<gold answer>", "ambiguity_type": "...", "base_id": "estate/q1"}

Protocol
- TiMMInsight: the metadata graph of each dataset is built once offline (its
  cost is reported separately and excluded from query latency).  Each query is
  clarified by TiQC, whose questions are answered by an LLM user simulator that
  sees only the hidden intent; the clarified query is then planned and executed.
  The simulator runs on its own provider, so its tokens and time are excluded
  (it stands in for the human).
- Baselines receive the query as posed (``--baseline-input clarified`` gives them
  TiMMInsight's clarified query instead, as in Fig. 4(g)).
- Ablations: ``timminsight-no-tiqc`` (K = 0), ``timminsight-no-metagraph`` (no
  metadata graph for TiQC and TiTSP), ``timminsight-no-titsp`` (one-shot LLM
  plan generation with the same backbone).
- Each query is executed ``--runs`` times; accuracy is judged by GPT-4.1 and
  averaged over all executions; latency is the median over all executions.
"""
from __future__ import annotations

import argparse
import csv
import faulthandler
import json
import logging
import os
import pickle
import re
import signal
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from core.query_planner.llm_query_analyzer import create_provider  # noqa: E402
from core.query_planner.plan_executor import PlanExecutionError  # noqa: E402
from core.query_planner.query_clarifier import TiQCClarifier  # noqa: E402
from core.query_planner.query_planner import BASELINE_STRATEGIES, QueryPlanner  # noqa: E402
from experiments.judge import JUDGE_MODEL, create_judge, grade  # noqa: E402

Table = List[Dict[str, Any]]
NIRVANA = REPO / "data" / "nirvana_repo" / "nirvana-main" / "testdata"
# SemBench scenarios staged by scripts/build_benchmarks.py in SemBench's own
# layout (<scenario>/data/sf_<N>)
SEMBENCH = Path(os.environ.get("SEMBENCH_DATA_ROOT", REPO / "data" / "sembench" / "files"))
METHODS = ("timminsight", "timminsight-no-tiqc", "timminsight-no-metagraph",
           "timminsight-no-titsp") + tuple(sorted(BASELINE_STRATEGIES))


# ---------------------------------------------------------------- datasets
@dataclass
class Dataset:
    tables: Dict[str, Table]
    text_columns: Dict[str, List[str]] = field(default_factory=dict)
    image_columns: Dict[str, List[str]] = field(default_factory=dict)


def _csv(path: Path, limit: Optional[int] = None) -> Table:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def _parquet(path: Path, limit: Optional[int] = None) -> Table:
    """Rows of a Parquet file; nested struct columns are flattened into
    dotted columns (e.g. productDescriptors.description.value)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    while any(pa.types.is_struct(f.type) for f in t.schema):
        t = t.flatten()
    return (t.slice(0, limit) if limit else t).to_pylist()


def _long_text_columns(rows: Table, min_avg_chars: int = 80, min_avg_words: int = 8) -> List[str]:
    """Free-text columns: long natural-language values (many words), not URLs
    or file paths."""
    url = re.compile(r"^\s*(?:[a-z][a-z0-9+.-]*://|/|[A-Za-z]:\\)\S*\s*$", re.I)
    cols = []
    for c in (rows[0] if rows else {}):
        vals = [r.get(c) for r in rows[:50] if isinstance(r.get(c), str) and r.get(c).strip()]
        if not vals or sum(bool(url.match(v)) for v in vals) > len(vals) / 2:
            continue
        if (sum(len(v) for v in vals) / len(vals) >= min_avg_chars
                and sum(len(v.split()) for v in vals) / len(vals) >= min_avg_words):
            cols.append(c)
    return cols


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff")


def _image_columns(rows: Table, bases: Tuple[Path, ...], sample: int = 50) -> List[str]:
    """Columns whose cells are images: raw bytes, {"bytes": ...}, data URIs,
    image URLs, or paths of image files.  Local paths are resolved against
    ``bases`` and replaced by the file bytes, so every component reads them."""
    cols = []
    for c in (rows[0] if rows else {}):
        vals = [r.get(c) for r in rows[:sample] if r.get(c) not in (None, "")]
        if not vals:
            continue

        def is_image(v):
            if isinstance(v, (bytes, bytearray)) or (isinstance(v, dict) and "bytes" in v):
                return True
            if not isinstance(v, str):
                return False
            if v.startswith("data:image"):
                return True
            return v.lower().split("?")[0].endswith(IMAGE_SUFFIXES)
        if sum(map(is_image, vals)) < 0.8 * len(vals):
            continue
        cols.append(c)
    # resolve local image paths (SemBench keeps files in an images/ folder)
    bases = tuple(bases) + tuple(b / "images" for b in bases)
    local: Dict[str, List[str]] = {}
    for c in cols:
        if not all(isinstance(r.get(c), str) and not r[c].startswith(("http", "data:image"))
                   for r in rows if r.get(c) not in (None, "")):
            continue
        paths = []
        for r in rows:
            v = r.get(c)
            if v in (None, ""):
                paths.append("")
                continue
            path = next((b / v for b in bases if (b / v).is_file()), None)
            if path is None and Path(v).is_file():
                path = Path(v)
            if path is None:
                raise FileNotFoundError(f"image {v!r} of column {c!r} not found under {bases}")
            paths.append(str(path.resolve()))
        local[c] = paths
    # one column per image: drop a local column repeating an earlier one, and
    # URL columns when the table holds the images locally
    keep, seen = [], []
    for c in cols:
        if c in local:
            if local[c] in seen:
                continue
            seen.append(local[c])
        elif local:
            continue
        keep.append(c)
    for c in keep:
        if c in local:
            for r, path in zip(rows, local[c]):
                if path:
                    r[c] = {"bytes": Path(path).read_bytes(), "path": path}
    return keep


def parse_dataset_name(name: str) -> Tuple[str, Optional[int]]:
    """``<dataset>[:<n>]``: the dataset, restricted to its first n rows per
    table when a suffix is given (e.g. nirvana/steam:1000)."""
    base, sep, n = name.rpartition(":")
    if sep and n.isdigit() and base not in ("", "dir"):
        return base, int(n)
    return name, None


def load_dataset(name: str, limit: Optional[int] = None) -> Dataset:
    """Real benchmark data: Nirvana-Bench (estate, imdb, steam), SemBench
    scenarios (``sembench/<scenario>`` or ``sembench/<scenario>/sf_<N>`` for one
    scale factor of <scenario>/data), or ``dir:<path>`` for any directory of
    CSV/Parquet tables (e.g. the Agriculture dataset).  A ``:<n>`` suffix keeps
    the first n rows of every table."""
    name, rows_cap = parse_dataset_name(name)
    if rows_cap is not None:
        limit = min(limit, rows_cap) if limit else rows_cap
    if name == "nirvana/estate":
        rows = _parquet(NIRVANA / "multimodal_real_estate.parquet", limit)
        return Dataset({"estate": rows}, {"estate": ["Details"]}, {"estate": ["image"]})
    if name == "nirvana/imdb":
        rows = _csv(NIRVANA / "movie_data.csv", limit)
        return Dataset({"movie": rows}, {"movie": ["Plot"]}, {"movie": ["Poster"]})
    if name == "nirvana/steam":
        rows = _csv(NIRVANA / "steam_games.csv", limit)
        return Dataset({"game": rows}, {"game": ["description"]}, {"game": ["image"]})
    if name.startswith("sembench/") or name.startswith("dir:"):
        if name.startswith("sembench/"):
            scenario, _, sf = name.split("/", 1)[1].partition("/")
            root = SEMBENCH / scenario / "data" / sf if sf else SEMBENCH / scenario / "data"
        else:
            root = Path(name[4:])
        files = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".csv", ".parquet"))
        if not files:
            raise FileNotFoundError(f"no tables under {root}")
        tables: Dict[str, Table] = {}
        image_cols: Dict[str, List[str]] = {}
        origin: Dict[str, Path] = {}
        for p in files:
            if p.stem in origin:
                raise ValueError(f"two tables named {p.stem!r}: {origin[p.stem]} and {p}")
            origin[p.stem] = p
            rows = _csv(p, limit) if p.suffix.lower() == ".csv" else _parquet(p, limit)
            tables[p.stem], image_cols[p.stem] = rows, _image_columns(rows, (p.parent, root))
        # images held as local files make the URL image columns of other
        # tables redundant (e.g. the same product photos at other resolutions)
        local = {n: [c for c in cols if isinstance(tables[n][0].get(c), dict)]
                 for n, cols in image_cols.items() if tables[n]}
        if any(local.values()):
            image_cols = {n: local.get(n, []) for n in image_cols}
        return Dataset(tables, {n: [c for c in _long_text_columns(r) if c not in image_cols[n]]
                                for n, r in tables.items()}, image_cols)
    raise ValueError(f"unknown dataset {name!r}")


# ---------------------------------------------------------------- metadata graph (offline)
GRAPH_WORKERS = 16   # concurrent LLM calls of the offline TiMetaGraph build (--graph-workers)


def metadata_graph(name: str, data: Dataset, llm: Any, vision: Any, cache_dir: Path,
                   limit: Optional[int] = None) -> Tuple[Any, Dict]:
    """Build TiMetaGraph once per dataset and cache it with its offline cost."""
    from core.query_planner.processors import extract_metadata_graph_tables
    path = cache_dir / (name.replace("/", "_").replace(":", "_")
                        + (f"_limit{limit}" if limit else "") + ".pkl")
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    before, t0 = llm.usage.snapshot(), time.time()
    # per-item results are checkpointed, so an interrupted build resumes
    checkpoints = cache_dir / "items" / path.stem
    resumed = checkpoints.exists() and any(checkpoints.rglob("*.json"))
    graph = extract_metadata_graph_tables(data.tables, llm, vision, max_workers=GRAPH_WORKERS,
                                          checkpoint_dir=str(checkpoints),
                                          text_cols=data.text_columns,
                                          image_cols=data.image_columns, graph_id=name)
    # a resumed build reports only the time and tokens spent after resuming
    cost = {"resumed": resumed, "seconds": time.time() - t0,
            "tokens": llm.usage.total_tokens - before.total_tokens}
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump((graph, cost), f)
    return graph, cost


def sources_for(graph: Any, data: Dataset) -> Dict[str, Table]:
    """Every source of the metadata graph reads the rows of its table."""
    out = {}
    for mid in graph.modalities:
        # longest matching table name, so "movie" does not capture "movie_review_*"
        matches = [t for t in data.tables if mid == t or mid.startswith(f"{t}_")]
        if not matches:
            raise KeyError(f"metadata-graph source {mid!r} matches no table of the dataset")
        out[mid] = data.tables[max(matches, key=len)]
    return out


# ---------------------------------------------------------------- one query
@dataclass
class Record:
    method: str
    item: str
    dataset: str
    run: int
    question: str
    gold_answer: Optional[str] = None
    benchmark: str = ""
    clarified: str = ""
    answer: str = ""
    correct: Optional[bool] = None
    latency_s: float = 0.0
    clarify_s: float = 0.0
    tokens: int = 0
    error: str = ""
    rows: List[Dict[str, Any]] = field(default_factory=list)   # the full result, for grading
    rows_truncated: bool = False
    metric: Optional[float] = None    # F1, row-match rate, or Spearman rho (None: scalar)


class TimedSimulator:
    """User simulator on its own provider; records the time it spends."""

    def __init__(self, tiqc: TiQCClarifier, hidden_intent: str):
        self._ask = tiqc.make_user_simulator(hidden_intent)
        self.seconds = 0.0

    def __call__(self, ambiguity, options):
        t0 = time.time()
        try:
            return self._ask(ambiguity, options)
        finally:
            self.seconds += time.time() - t0


def run_timminsight(item: Dict, data: Dataset, graph: Any, llm: Any, sim_llm: Any,
                    vision: Any, method: str, history: Any = None,
                    sim_clock: Optional[Dict[str, float]] = None) -> Tuple[Any, str, float, float]:
    """Returns (ExecutableResult, clarified query, TiQC seconds without the
    simulator, seconds spent by the simulator)."""
    use_graph = None if method == "timminsight-no-metagraph" else graph
    question = item["question"]
    clarify_s = sim_s = 0.0
    if method != "timminsight-no-tiqc" and CLARIFY_ROUNDS > 0 and not item.get("hidden_intent"):
        # the user simulator answers from the hidden intent; without one TiQC
        # cannot be driven and the item runs unclarified
        logging.getLogger(__name__).warning(
            "item %s has no hidden_intent: TiQC is skipped for it", item.get("id"))
    if method != "timminsight-no-tiqc" and CLARIFY_ROUNDS > 0 and item.get("hidden_intent"):
        tiqc = TiQCClarifier(llm, metadata_graph=use_graph, round_budget=CLARIFY_ROUNDS)
        sim_tiqc = TiQCClarifier(sim_llm, metadata_graph=use_graph)
        ask = TimedSimulator(sim_tiqc, item["hidden_intent"])
        t0 = time.time()
        try:
            question = tiqc.clarify_query(question, ask)["clarified_query"]
        finally:
            sim_s = ask.seconds
            if sim_clock is not None:
                sim_clock["seconds"] = sim_s
        clarify_s = time.time() - t0 - sim_s
    strategy = "titsp-oneshot" if method == "timminsight-no-titsp" else "titsp"
    planner = QueryPlanner(llm_provider=llm, metadata_graph=use_graph, planning_strategy=strategy,
                           history_manager=history)
    sources = sources_for(use_graph, data) if use_graph is not None else data.tables
    result = planner.plan_and_execute(question, sources, vision_provider=vision)
    return result, question, clarify_s, sim_s


def run_baseline(item: Dict, data: Dataset, llm: Any, method: str, question: str) -> Any:
    """Baselines get the raw tables with their text-column typing only (no TiMetaGraph)."""
    planner = QueryPlanner(llm_provider=llm, planning_strategy=method)
    table_info = {t: {"text_columns": data.text_columns.get(t, [])} for t in data.tables}
    result = planner.plan_and_execute(question, data.tables, table_info=table_info)
    if not result.baseline.ok:
        raise RuntimeError("; ".join(result.baseline.errors)[-500:])
    return result


# K, the round budget of TiQC (Sec. IV); --rounds sweeps it (K = 0: no clarification)
CLARIFY_ROUNDS = 3

# endpoint overrides of the backbone provider (--base-url, --api-key-env)
BACKBONE_ENDPOINT: Dict[str, Any] = {}

MAX_RESULT_ROWS = 20000


def result_rows(result: Any) -> Tuple[List[Dict[str, Any]], bool]:
    """The rows a method returned (images and embeddings dropped), capped at
    MAX_RESULT_ROWS; the flag tells whether rows were cut."""
    rows = list(getattr(getattr(result, "execution", None), "data", None) or [])

    def keep(v: Any) -> bool:
        if isinstance(v, (bytes, bytearray)) or (isinstance(v, dict) and "bytes" in v):
            return False
        if isinstance(v, str) and v.startswith("data:image"):
            return False
        return not (isinstance(v, list) and len(v) > 64 and all(isinstance(x, float) for x in v))
    clean = [{k: v for k, v in r.items() if keep(v)} if isinstance(r, dict) else {"value": r}
             for r in rows[:MAX_RESULT_ROWS]]
    return json.loads(json.dumps(clean, default=str)), len(rows) > MAX_RESULT_ROWS


# ---------------------------------------------------------------- main loop
def run(items: List[Dict], methods: List[str], model: str, runs: int, judge_model: str,
        baseline_input: str, limit: Optional[int], cache_dir: Path, out_path: Path,
        vision_model: Optional[str] = None, judge_provider: str = "openai",
        provider: str = "dashscope", vision_provider: str = "dashscope") -> List[Record]:
    llm = create_provider(provider, model, **BACKBONE_ENDPOINT)
    sim_llm = create_provider(provider, model, **BACKBONE_ENDPOINT)
    vision = create_provider(vision_provider, vision_model) if vision_model else None
    judge_llm = None if judge_model == "none" else create_judge(judge_provider, judge_model)
    datasets: Dict[str, Tuple[Dataset, Any]] = {}
    offline: Dict[str, Dict] = {}
    clarified: Dict[Tuple[str, int], str] = {}
    records: List[Record] = []
    # offline phase: every dataset and its TiMetaGraph before any query runs
    for name in dict.fromkeys(item["dataset"] for item in items):
        data = load_dataset(name, limit)
        graph, cost = metadata_graph(name, data, llm, vision, cache_dir, limit)
        datasets[name], offline[name] = (data, graph), cost
    for run_id in range(runs):
        for method in methods:
            for item in items:
                name = item["dataset"]
                data, graph = datasets[name]
                rec = Record(method, item["id"], name, run_id, item["question"],
                             item.get("gold_answer"), benchmark_of(item))
                before, t0 = llm.usage.snapshot(), time.time()
                vision_before = vision.usage.total_tokens if vision is not None else 0
                sim_clock = {"seconds": 0.0}
                try:
                    if method.startswith("timminsight"):
                        result, rec.clarified, rec.clarify_s, sim_s = run_timminsight(
                            item, data, graph, llm, sim_llm, vision, method, sim_clock=sim_clock)
                        rec.answer = result.answer
                        rec.rows, rec.rows_truncated = result_rows(result)
                        if method == "timminsight":
                            clarified[(item["id"], run_id)] = rec.clarified
                        # the user simulator stands in for the human: its time is excluded
                        rec.latency_s = time.time() - t0 - sim_s
                    else:
                        q = item["question"]
                        if baseline_input == "clarified":
                            if (item["id"], run_id) not in clarified:
                                raise RuntimeError("no clarified query: TiMMInsight failed on this item")
                            q = clarified[(item["id"], run_id)]
                        rec.clarified = q
                        result = run_baseline(item, data, llm, method, q)
                        rec.latency_s = time.time() - t0
                        rec.answer = result.answer
                        rec.rows, rec.rows_truncated = result_rows(result)
                except Exception as exc:  # recorded, the run continues
                    rec.error = f"{type(exc).__name__}: {exc}"[:500]
                    rec.latency_s = time.time() - t0 - sim_clock["seconds"]
                rec.tokens = llm.usage.total_tokens - before.total_tokens
                if vision is not None:
                    rec.tokens += vision.usage.total_tokens - vision_before
                if judge_llm is not None and item.get("gold_answer") and not rec.error:
                    try:
                        rec.correct, rec.metric = grade(judge_llm, item, rec.answer, rec.rows)
                    except ValueError as exc:   # ungraded, not counted as wrong
                        rec.error = f"judge: {exc}"[:500]
                records.append(rec)
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
    with open(out_path.with_suffix(".offline.json"), "w") as f:
        json.dump(offline, f, indent=2)
    return records


# ---------------------------------------------------------------- calibration (Table II)
def final_plan_confidence(result: Any) -> float:
    """Predicted success probability of the final (physical) plan: f_conf of
    the selected assignment (Table II)."""
    return result.plan.selected_physical.objectives.confidence


def calibration_metrics(pairs: List[Tuple[float, bool]], bins: int = 10) -> Dict[str, float]:
    """ECE over ``bins`` equal-width confidence bins and the Brier score."""
    n = len(pairs)
    brier = sum((p - float(y)) ** 2 for p, y in pairs) / n
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        inside = [(p, y) for p, y in pairs if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if inside:
            acc = sum(float(y) for _, y in inside) / len(inside)
            conf = sum(p for p, _ in inside) / len(inside)
            ece += len(inside) / n * abs(acc - conf)
    return {"ece10": ece, "brier": brier, "n": n}


def run_calibration(items: List[Dict], model: str, judge_model: str, judge_provider: str,
                    limit: Optional[int], cache_dir: Path, vision_model: Optional[str],
                    provider: str = "dashscope", vision_provider: str = "dashscope") -> Dict:
    """Cold start vs warm calibration with leave-one-family-out history: a query's
    warm history excludes every execution record of its query family (the base
    query and all of its ambiguity variants)."""
    from core.query_planner.execution_history import ExecutionHistoryManager
    llm = create_provider(provider, model, **BACKBONE_ENDPOINT)
    sim_llm = create_provider(provider, model, **BACKBONE_ENDPOINT)
    vision = create_provider(vision_provider, vision_model) if vision_model else None
    judge_llm = create_judge(judge_provider, judge_model)
    history = ExecutionHistoryManager()
    loaded: Dict[str, Tuple[Dataset, Any]] = {}

    def run_item(item, hist):
        """(plan, operator results, failed operator ids, (predicted, correct)).
        A plan whose execution fails is an incorrect answer of its predicted
        confidence, and the failed operator is reported for the history."""
        name = item["dataset"]
        if name not in loaded:
            data = load_dataset(name, limit)
            loaded[name] = (data, metadata_graph(name, data, llm, vision, cache_dir, limit)[0])
        data, graph = loaded[name]
        try:
            result, _, _, _ = run_timminsight(item, data, graph, llm, sim_llm, vision,
                                              "timminsight", hist)
        except PlanExecutionError as err:
            return (err.plan, err.operator_results, [err.node_id],
                    (err.plan.selected_physical.objectives.confidence, False))
        correct, _ = grade(judge_llm, item, result.answer, result_rows(result)[0])
        return (result.plan, result.execution.operator_results, [],
                (final_plan_confidence(result), correct))

    graded = [it for it in items if it.get("gold_answer")]
    cold, warm, skipped = {}, {}, []

    def attempt(item, hist, phase):
        """None when no final plan was produced or graded (no prediction to
        score); the item is reported in "skipped" instead of ending the run."""
        try:
            return run_item(item, hist)
        except Exception as err:    # noqa: BLE001
            skipped.append({"id": item.get("id"), "phase": phase, "error": str(err)[:300]})
            print(f"[calibration] {phase} {item.get('id')}: skipped ({str(err)[:200]})",
                  flush=True)
            return None

    for item in graded:
        family = item.get("base_id") or item["id"]
        out = attempt(item, None, "cold")
        if out is None:
            continue
        plan, operator_results, failed, pair = out
        cold.setdefault(benchmark_of(item), []).append(pair)
        history.record_from_execution(plan.best_logical_plan, operator_results,
                                      query_id=family, failed=failed,
                                      bindings=plan.query_context.bindings)
    for item in graded:
        family = item.get("base_id") or item["id"]
        out = attempt(item, history.snapshot_excluding(family), "warm")
        if out is not None:
            warm.setdefault(benchmark_of(item), []).append(out[3])
    report = {name: {"cold_start": calibration_metrics(cold[name]),
                     "warm_calibrated": calibration_metrics(warm[name])}
              for name in cold if name in warm}
    return {"datasets": report, "skipped": skipped}


def benchmark_of(item: Dict) -> str:
    """The benchmark a query belongs to, the unit of the paper's tables:
    an explicit "benchmark" field, else Nirvana-Bench for nirvana/*, SemBench
    for sembench/*, and the directory name for dir:<path>."""
    if item.get("benchmark"):
        return str(item["benchmark"])
    name = item["dataset"]
    if name.startswith("nirvana/"):
        return "Nirvana-Bench"
    if name.startswith("sembench/"):
        return "SemBench"
    if name.startswith("dir:"):
        return Path(name[4:]).name
    return name


def summarize(records: List[Record]) -> Dict[str, Dict[str, Any]]:
    """Accuracy micro-averaged over all executions, median latency, and mean
    tokens, per method and benchmark (e.g. the three Nirvana-Bench domains
    together)."""
    out: Dict[str, Dict[str, Any]] = {}
    for key in sorted({(r.method, r.benchmark) for r in records}):
        rs = [r for r in records if (r.method, r.benchmark) == key]
        # system errors count as wrong; judge failures are left ungraded
        judged = [r for r in rs if r.correct is not None
                  or (r.error and not r.error.startswith("judge:"))]
        out[" / ".join(key)] = {
            "executions": len(rs),
            "accuracy": (sum(bool(r.correct) for r in judged) / len(judged)) if judged else None,
            "median_latency_s": statistics.median(r.latency_s for r in rs),
            "mean_tokens": statistics.mean(r.tokens for r in rs),
            "errors": sum(bool(r.error) for r in rs),
        }
    return out


def main() -> None:
    global GRAPH_WORKERS, CLARIFY_ROUNDS
    # `kill -USR1 <pid>` prints every thread's Python stack (diagnosing stalls)
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    # progress of the offline TiMetaGraph build and of the planner
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s: %(message)s")
    logging.getLogger("core").setLevel(logging.INFO)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("items", help="benchmark JSONL")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--model", required=True, help="backbone model id")
    ap.add_argument("--provider", default="dashscope",
                    help="backbone provider: dashscope, openai, anthropic, deepseek, or custom "
                         "(any OpenAI-compatible endpoint, with --base-url); bedrock runs "
                         "TiMMInsight only, as the baselines need an OpenAI-compatible endpoint")
    ap.add_argument("--base-url", default=None,
                    help="backbone endpoint (required for --provider custom)")
    ap.add_argument("--api-key-env", default=None,
                    help="environment variable holding the backbone API key "
                         "(default: the provider's, LLM_API_KEY for custom)")
    ap.add_argument("--vision-provider", default="dashscope")
    ap.add_argument("--vision-model", default=None, help="vision model for image predicates")
    ap.add_argument("--runs", type=int, default=5,
                    help="executions per query (Sec. VI: five)")
    ap.add_argument("--judge-model", default=JUDGE_MODEL, help="'none' to skip judging")
    ap.add_argument("--judge-provider", default="openai",
                    help="the paper uses OpenAI GPT-4.1; other providers only for smoke tests")
    ap.add_argument("--rounds", type=int, default=CLARIFY_ROUNDS,
                    help="K, the maximum number of TiQC clarification rounds (default 3; "
                         "0 disables clarification)")
    ap.add_argument("--baseline-input", choices=("asked", "clarified"), default="asked")
    ap.add_argument("--limit", type=int, default=None, help="rows per table (smoke tests)")
    ap.add_argument("--cache", default="runs/metadata_graphs")
    ap.add_argument("--graph-workers", type=int, default=GRAPH_WORKERS,
                    help="concurrent LLM calls when building TiMetaGraph offline")
    ap.add_argument("--out", default="runs/results.jsonl")
    ap.add_argument("--calibration", action="store_true",
                    help="Table II: cold start vs leave-one-family-out warm calibration")
    args = ap.parse_args()
    GRAPH_WORKERS = args.graph_workers
    if args.rounds < 0:
        ap.error("--rounds must be >= 0")
    CLARIFY_ROUNDS = args.rounds
    if args.provider == "custom" and not args.base_url:
        ap.error("--provider custom needs --base-url")
    if args.provider != "bedrock":
        BACKBONE_ENDPOINT.update(base_url=args.base_url, api_key_env=args.api_key_env)
    with open(args.items, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    methods = [m.strip() for m in args.methods.split(",")]
    if args.calibration and args.judge_model == "none":
        ap.error("--calibration needs a judge")
    if args.baseline_input == "clarified":
        if "timminsight" not in methods:
            ap.error("--baseline-input clarified needs the timminsight method")
        methods = ["timminsight"] + [m for m in methods if m != "timminsight"]
    if args.calibration:
        # one LLM per run; Table II macro-averages the per-LLM files
        # (python -m src.experiments.calibration_table <file> ...)
        result = {"model": args.model,
                  **run_calibration(items, args.model, args.judge_model,
                                    args.judge_provider, args.limit, Path(args.cache),
                                    args.vision_model, args.provider,
                                    args.vision_provider)}
        out = Path(args.out).with_suffix(".calibration.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    records = run(items, methods, args.model, args.runs,
                  args.judge_model, args.baseline_input, args.limit, Path(args.cache), out,
                  args.vision_model, args.judge_provider, args.provider, args.vision_provider)
    print(json.dumps(summarize(records), indent=2))


if __name__ == "__main__":
    main()

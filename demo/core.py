"""Bridge between the demo server and the TiMMInsight system.

Everything here calls the real system: datasets come from the benchmark
loaders (src/experiments/benchmark.py), TiMetaGraph is built (or read from the
same cache) by its offline path, TiQC runs Algorithm 1 one round per request
with TiQCClarifier, and TiTSP plans and executes with QueryPlanner.  The
functions below only turn the system's objects into JSON for the browser.
"""
from __future__ import annotations

import base64
import io
import logging
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from core.query_planner.llm_query_analyzer import create_provider  # noqa: E402
from core.query_planner.query_clarifier import Ambiguity, TiQCClarifier  # noqa: E402
from core.query_planner.query_planner import QueryPlanner  # noqa: E402
from experiments import benchmark as bench  # noqa: E402

BACKBONE = "deepseek-v4.1-flash-1017"
VISION = "qwen3-vl-plus"
BUILTIN_DATASETS = ["sembench/movie", "nirvana/estate", "nirvana/imdb", "nirvana/steam",
                    "sembench/ecomm", "sembench/mmqa"]
# paper Sec. V-A: the cross-modal operator types (shaded in Fig. 1)
CROSS_MODAL = {"SEMANTIC_SEARCH", "CROSS_MODAL_MATCH", "CONTENT_EXTRACT", "SIMILARITY_JOIN",
               "FEATURE_TRANSFORM"}
# graph caches read: the demo's own (under the data root), then the benchmark's
BENCH_CACHES = [REPO / "runs" / "metadata_graphs", REPO / "runs" / "smoke_graphs"]


def graph_caches(data_root: Path) -> List[Path]:
    return [data_root / "graphs"] + BENCH_CACHES
MAX_GRAPH_ENTITIES = 160
MAX_GRAPH_RELATIONS = 240
SAMPLE_ROWS = 5


# ---------------------------------------------------------------- models
def providers(api_key: str, backbone: str = BACKBONE, vision: str = VISION):
    """The backbone LLM and the vision model on DashScope."""
    if not api_key:
        raise PermissionError("a DashScope API key is needed for live queries")
    return (create_provider("dashscope", backbone, api_key=api_key),
            create_provider("dashscope", vision, api_key=api_key))


# ---------------------------------------------------------------- datasets
def dataset_key(name: str, limit: Optional[int]) -> str:
    return f"{name}@{limit}" if limit else name


def load(name: str, limit: Optional[int]) -> "bench.Dataset":
    return bench.load_dataset(name, limit)


def _cell(v: Any) -> Any:
    """A table cell for the browser: images become small JPEG data URIs,
    embeddings and long texts are shortened."""
    img = _image_bytes(v)
    if img is not None:
        return {"image": _thumbnail(img)}
    if isinstance(v, (list, tuple)) and len(v) > 16:
        return f"[{len(v)} values]"
    if isinstance(v, str) and len(v) > 400:
        return v[:400] + "…"
    if isinstance(v, float):
        return round(v, 6)
    return v if isinstance(v, (str, int, bool)) or v is None else str(v)


def _image_bytes(v: Any) -> Optional[bytes]:
    if isinstance(v, (bytes, bytearray)):
        return bytes(v)
    if isinstance(v, dict) and isinstance(v.get("bytes"), (bytes, bytearray)):
        return bytes(v["bytes"])
    if isinstance(v, str) and v.startswith("data:image"):
        try:
            return base64.b64decode(v.split(",", 1)[1])
        except ValueError:
            return None
    return None


def _thumbnail(data: bytes, side: int = 160) -> Optional[str]:
    from PIL import Image, UnidentifiedImageError
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, ValueError):
        return None
    img = img.convert("RGB")
    img.thumbnail((side, side))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode()


def preview(data: "bench.Dataset", rows: int = 20) -> Dict[str, Any]:
    """Tables of a dataset with their text and image columns and first rows."""
    return {"tables": [
        {"name": name, "rows": len(table), "columns": list(table[0].keys()) if table else [],
         "text_columns": data.text_columns.get(name, []),
         "image_columns": data.image_columns.get(name, []),
         "sample": [{k: _cell(v) for k, v in r.items()} for r in table[:rows]]}
        for name, table in data.tables.items()]}


# ---------------------------------------------------------------- TiMetaGraph
def cached_graph(name: str, limit: Optional[int], dirs: List[Path]):
    """The cached metadata graph of a dataset, if any (same file naming as the
    benchmark's cache)."""
    import pickle
    file = (name.replace("/", "_").replace(":", "_") + (f"_limit{limit}" if limit else "") + ".pkl")
    for d in dirs:
        if (d / file).exists():
            with open(d / file, "rb") as f:
                return pickle.load(f)[0]
    return None


class _ProgressHandler(logging.Handler):
    def __init__(self, sink: Callable[[str], None]):
        super().__init__(logging.INFO)
        self.sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        self.sink(record.getMessage())


def build_graph(name: str, data: "bench.Dataset", limit: Optional[int], llm: Any, vision: Any,
                progress: Callable[[str], None], data_root: Path):
    """Build TiMetaGraph offline through the benchmark's cached path (or read a
    cache), reporting the build's progress messages to ``progress``."""
    graph = cached_graph(name, limit, graph_caches(data_root))
    if graph is not None:
        progress("Loaded the cached metadata graph.")
        return graph
    logger = logging.getLogger("core")
    handler = _ProgressHandler(progress)
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        graph, cost = bench.metadata_graph(name, data, llm, vision, data_root / "graphs", limit)
        progress(f"Built in {cost['seconds']:.0f} s, {cost['tokens']} tokens.")
        return graph
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


def _build_child(spec: Dict[str, Any], api_key: str, events) -> None:
    """Child process of a cancellable build: load the data, build (or read)
    the graph into the cache, report progress."""
    try:
        if spec["upload"]:
            from demo import uploads
            data = uploads.load(Path(spec["upload_root"]), spec["upload"], spec["limit"])
        else:
            data = load(spec["dataset"], spec["limit"])
        llm, vision = providers(api_key)
        build_graph(spec["dataset"], data, spec["limit"], llm, vision,
                    lambda m: events.put(("progress", m)), Path(spec["data_root"]))
        events.put(("done", None))
    except BaseException as exc:  # reported to the parent, which raises it
        events.put(("error", f"{type(exc).__name__}: {exc}"))


class GraphBuild:
    """A TiMetaGraph build in a separate process, so it can be cancelled.
    The finished graph is read from the cache the child wrote."""

    def __init__(self, spec: Dict[str, Any], api_key: str):
        import multiprocessing
        ctx = multiprocessing.get_context("spawn")
        self.spec = spec
        self.events = ctx.Queue()
        self.process = ctx.Process(target=_build_child, args=(spec, api_key, self.events),
                                   daemon=True)
        self.cancelled = False

    def start(self) -> "GraphBuild":
        self.process.start()
        return self

    def cancel(self) -> None:
        self.cancelled = True
        if self.process.is_alive():
            self.process.terminate()
        self.events.put(("cancelled", None))

    def graph(self):
        return cached_graph(self.spec["dataset"], self.spec["limit"],
                            graph_caches(Path(self.spec["data_root"])))


def graph_view(graph: Any) -> Dict[str, Any]:
    """Sources, the most mentioned entities, their mentions per source, and the
    strongest relationship edges of the metadata graph."""
    sources = [{"id": f"src:{mid}", "label": mid, "kind": "source",
                "modality": m.modality_type.value, "records": (m.content_summary or {}).get("n_rows")}
               for mid, m in graph.modalities.items()]
    ents = sorted(graph.entities.values(), key=lambda e: -len(e.mentions))[:MAX_GRAPH_ENTITIES]
    keep = {e.entity_id for e in ents}
    nodes = sources + [
        {"id": e.entity_id, "label": e.canonical_name, "kind": "entity",
         "type": e.entity_type.value, "mentions": len(e.mentions),
         "cross_modal": bool((e.properties or {}).get("cross_modal")),
         "modalities": sorted({m.modality_type.value for m in e.mentions})}
        for e in ents]
    links = []
    for e in ents:
        for mid in sorted({m.modality_id for m in e.mentions}):
            links.append({"source": e.entity_id, "target": f"src:{mid}", "kind": "mention"})
    rels = sorted((r for r in graph.relations.values()
                   if r.source_modality in keep and r.target_modality in keep),
                  key=lambda r: -r.confidence)[:MAX_GRAPH_RELATIONS]
    links += [{"source": r.source_modality, "target": r.target_modality, "kind": "relation",
               "type": r.relation_type.value, "weight": round(r.confidence, 4)} for r in rels]
    links += [{"source": f"src:{r.source_modality}", "target": f"src:{r.target_modality}",
               "kind": "foreign_key", "type": f"{(r.evidence or {}).get('fk_column')} -> "
               f"{(r.evidence or {}).get('pk_column')}"}
              for r in graph.relations.values()
              if (r.properties or {}).get("relation_category") == "table_relationship"
              and r.source_modality in graph.modalities and r.target_modality in graph.modalities]
    return {"nodes": nodes, "links": links,
            "stats": {"sources": len(sources), "entities": len(graph.entities),
                      "relations": len(graph.relations),
                      "cross_modal_entities": sum(bool((e.properties or {}).get("cross_modal"))
                                                  for e in graph.entities.values())}}


# ---------------------------------------------------------------- TiQC, one round per step
@dataclass
class Clarification:
    """Algorithm 1 driven step by step: ``next_round`` runs lines 4-7 (detect,
    filter by theta_amb, pick the most severe, generate options); ``answer``
    runs line 9 (rewrite) with the user's choice."""
    clarifier: TiQCClarifier
    query: str
    original: str
    rounds: List[Dict[str, Any]] = field(default_factory=list)
    pending: Optional[Ambiguity] = None
    done: bool = False

    def next_round(self) -> Optional[Dict[str, Any]]:
        if self.done or len(self.rounds) >= self.clarifier.round_budget:
            self.done = True
            return None
        found = [a for a in self.clarifier.detect_ambiguities(self.query)
                 if a.severity >= self.clarifier.theta_amb]
        if not found:
            self.done = True
            return None
        a_star = max(found, key=lambda a: a.severity)
        options = self.clarifier.generate_options(self.query, a_star)
        self.pending = a_star
        return {"round": len(self.rounds) + 1, "ambiguity_type": a_star.ambiguity_type,
                "severity": a_star.severity, "description": a_star.description,
                "question": a_star.question, "options": options,
                "detected": [{"ambiguity_type": a.ambiguity_type, "severity": a.severity,
                              "description": a.description} for a in found]}

    def answer(self, reply: str) -> str:
        if self.pending is None:
            raise ValueError("no clarification question is open")
        a_star, self.pending = self.pending, None
        before = self.query
        self.query = self.clarifier.rewrite_query(before, a_star, reply)
        self.rounds.append({**a_star.to_dict(), "round": len(self.rounds) + 1, "answer": reply,
                            "query_before": before, "query_after": self.query})
        return self.query


# ---------------------------------------------------------------- TiTSP + execution
def run_query(llm: Any, vision: Any, graph: Any, data: "bench.Dataset", query: str) -> Dict[str, Any]:
    """Plan and execute the clarified query; the full trace as JSON."""
    planner = QueryPlanner(llm_provider=llm, metadata_graph=graph)
    t0 = time.time()
    sources = bench.sources_for(graph, data) if graph is not None else data.tables
    result = planner.plan_and_execute(query, sources, vision_provider=vision)
    out = run_view(result)
    out["seconds"] = round(time.time() - t0, 2)
    return out


def _rows(rows: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    return [{k: _cell(v) for k, v in r.items()} for r in rows[:n]]


def run_view(result: Any) -> Dict[str, Any]:
    """Φ, the MCTS candidates, the selected logical plan with propagated
    confidences, the Pareto front with the Tchebycheff choice, the per-operator
    execution trace, the answer, and the chart of an ExecutableResult."""
    plan = result.plan
    ctx = plan.query_context
    umap = plan.uncertainty_map
    assignment = plan.selected_physical.assignment
    trace = result.execution.operator_results

    def tree(n) -> Dict[str, Any]:
        est = umap.get(n.node_id)
        res = trace.get(n.node_id)
        return {"id": n.node_id, "op": n.operator_type, "source": n.source_name,
                "n": n.source_cardinality, "cross_modal": n.operator_type in CROSS_MODAL,
                "confidence": round(est.confidence, 4) if est else None,
                "impl": assignment.get(n.node_id), "rows": res.row_count if res else None,
                "children": [tree(c) for c in n.children]}

    def objectives(c) -> Dict[str, float]:
        o = c.objectives
        return {"latency": round(o.latency, 4), "resource": round(o.resource_cost, 4),
                "confidence": round(o.confidence, 4)}

    def describe(n) -> str:
        return n.operator_type + (f"({', '.join(describe(c) for c in n.children)})" if n.children else "")

    pareto = [{**objectives(c), "assignment": c.assignment,
               "selected": c.assignment == assignment} for c in plan.pareto_front]
    if not any(p["selected"] for p in pareto):
        pareto.append({**objectives(plan.selected_physical), "assignment": assignment,
                       "selected": True})
    return {
        "phi": {"query": ctx.query_text,
                "required": sorted(ctx.required_operators),
                "optional": sorted(ctx.optional_operators),
                "root": ctx.root_operator_type,
                "modalities": sorted(m.value for m in ctx.target_modalities),
                "bindings": ctx.bindings or {},
                "source_sizes": [list(s) for s in (ctx.source_sizes or [])]},
        "candidates": [{"plan": describe(c.plan), "q": round(c.quality_score, 4),
                        "cost": round(c.estimated_cost, 3),
                        "confidence": round(c.estimated_confidence, 4),
                        "ocg": round(c.ocg_score, 4),
                        "selected": c.plan.node_id == plan.best_logical_plan.node_id}
                       for c in plan.logical_candidates[:12]],
        "plan": tree(plan.best_logical_plan),
        "pareto": pareto,
        "selected": {**objectives(plan.selected_physical), "assignment": assignment},
        "trace": [{"id": nid, "op": r.operator_type, "impl": r.impl_type, "rows": r.row_count,
                   "sample": _rows(r.data, SAMPLE_ROWS)} for nid, r in trace.items()],
        "result": {"rows": result.execution.row_count, "sample": _rows(result.execution.data, 50)},
        "answer": result.answer,
        "chart": result.chart or {},
        "timing": {k: round(v, 1) for k, v in vars(plan.timing).items()
                   if isinstance(v, (int, float))} if plan.timing else {},
    }


# ---------------------------------------------------------------- sessions
@dataclass
class Session:
    id: str
    dataset: str
    limit: Optional[int]
    data: Any
    graph: Any
    llm: Any
    vision: Any
    clarification: Clarification
    run: Optional[Dict[str, Any]] = None


class Sessions:
    """In-memory sessions of the live mode (one per question)."""

    def __init__(self):
        self._items: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self, **kw) -> Session:
        s = Session(id=uuid.uuid4().hex[:12], **kw)
        with self._lock:
            self._items[s.id] = s
        return s

    def get(self, sid: str) -> Session:
        with self._lock:
            if sid not in self._items:
                raise KeyError(sid)
            return self._items[sid]

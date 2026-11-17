"""
Physical Plan Executor.

Executes a physical plan tree bottom-up against a tabular data source.
Each logical operator type maps to concrete execution logic; the physical
implementation assignment selects the specific algorithm variant.

Data model: rows are ``List[Dict[str, Any]]`` (no pandas dependency).
Operators read their parameters from the binding environment B; relational
operators run deterministically, semantic operators run the LLM (or the vision
model on image columns) over all their input rows.  Missing parameters or
unparsable model output raise ExecutionError.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .llm_query_analyzer import complete_json
from .mcts_plan_search import LogicalPlanNode
from .metadata_graph import doc_key


Row = Dict[str, Any]
Table = List[Row]


@dataclass
class ExecutionContext:
    """Shared context passed to every operator during execution."""
    data_source: Table = field(default_factory=list)
    llm_provider: Any = None
    query_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    # vision model for operators over image columns
    vision_provider: Any = None
    # what the running implementation reports about its execution (index
    # columns, sort runs, ...); recorded in its ExecutionResult
    notes: Dict[str, Any] = field(default_factory=dict)


def _binding(ctx: "ExecutionContext", op_type: str) -> Dict[str, Any]:
    """Parameters the binding environment B attaches to an operator type."""
    return (ctx.metadata.get("bindings") or {}).get(op_type) or {}


_COMPARE = {
    "=": lambda a, b: a == b, "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
    ">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
    "in": lambda a, b: a in b,
}


def _holds(row: dict, cond: Dict[str, Any]) -> bool:
    """A structured condition {"column", "op", "value"} on one row; numeric
    targets compare numerically (CSV cells arrive as strings), string targets
    case-insensitively, and "contains" tests substring membership."""
    value, target = row.get(cond.get("column")), cond.get("value")
    op_name = str(cond.get("op", "=")).strip().lower()
    if value is None:
        return False
    if op_name == "contains":
        return str(target).lower() in str(value).lower()
    op = _COMPARE.get(op_name)
    if op is None:
        raise ExecutionError(f"unknown comparison {op_name!r}")
    if op_name == "in":
        targets = target if isinstance(target, list) else [target]
        return str(value).lower() in {str(x).lower() for x in targets}
    if isinstance(target, (int, float)) and not isinstance(target, bool):
        number = _number(value)
        return number is not None and bool(op(number, float(target)))
    return bool(op(str(value).lower(), str(target).lower()))


def _vader_scores(texts: List[Any]) -> List[float]:
    """VADER compound sentiment in [-1, 1] per text (the TOOL_EXTRACT tool)."""
    from .extraction_tools import VaderSentimentTool
    tool = VaderSentimentTool()
    return [tool.run(str(x or ""))["compound"] for x in texts]


class ExecutionError(RuntimeError):
    """An operator could not be executed with the parameters it was given."""


class PlanExecutionError(ExecutionError):
    """An operator of a plan failed.  Carries the failed operator instance and
    the results of the operators that completed before it, so the failure can
    be recorded in the execution history (a_r = 0, paper Eq. 11)."""

    def __init__(self, message: str, node_id: str, operator_type: str,
                 operator_results: Dict[str, "ExecutionResult"]):
        super().__init__(message)
        self.node_id = node_id
        self.operator_type = operator_type
        self.operator_results = operator_results
        self.plan = None    # the PlanResult, attached by QueryPlanner.plan_and_execute


def _require_llm(ctx: "ExecutionContext", op: str) -> Any:
    if ctx.llm_provider is None:
        raise ExecutionError(f"{op} needs the LLM provider")
    return ctx.llm_provider


def _llm_json(llm: Any, system: str, user: str, expect: type, what: str,
              validate=None) -> Any:
    """One LLM call returning JSON of type ``expect``; retried once on a parse
    failure or an incomplete reply, then raised (never silently replaced)."""
    return complete_json(llm, [{"role": "system", "content": system},
                               {"role": "user", "content": user}],
                         expect, validate, what, ExecutionError)


def _cell_text(row: dict, columns: Optional[List[str]], maxlen: int = 1500) -> str:
    """Text of the columns an operator reads (all non-binary columns by default)."""
    cols = columns or [c for c, v in row.items() if not isinstance(v, (bytes, bytearray, dict))]
    parts = []
    for c in cols:
        v = row.get(c)
        if isinstance(v, (bytes, bytearray)) or (isinstance(v, dict) and "bytes" in v):
            continue
        s = str(v)
        parts.append(f"{c}: {s[:maxlen]}")
    return "; ".join(parts)


def _chunks(n: int, size: int) -> List[range]:
    return [range(s, min(n, s + size)) for s in range(0, n, size)]


def _is_image_column(rows: Table, col: Optional[str]) -> bool:
    if not col or not rows or col not in rows[0]:
        return False
    v = rows[0].get(col)
    return (isinstance(v, (bytes, bytearray)) or (isinstance(v, dict) and "bytes" in v)
            or (isinstance(v, str) and v.startswith(("http", "data:image"))
                and v.lower().split("?")[0].endswith((".jpg", ".jpeg", ".png", ".webp", ".gif"))))


def _image_data_url(cell: Any) -> str:
    """Data URL of an image cell (raw bytes, a {bytes,path} struct, or an
    http/data URL string); empty if the cell holds no image."""
    import base64
    if isinstance(cell, dict) and cell.get("bytes"):
        return "data:image/jpeg;base64," + base64.b64encode(cell["bytes"]).decode()
    if isinstance(cell, (bytes, bytearray)):
        return "data:image/jpeg;base64," + base64.b64encode(bytes(cell)).decode()
    s = str(cell or "")
    return s if s.startswith(("http", "data:")) else ""


def _vision_answer(provider: Any, instruction: str, cell: Any) -> str:
    url = _image_data_url(cell)
    if not url:
        raise ExecutionError("image cell holds no image")
    return str(provider.complete([{"role": "user", "content": [
        {"type": "text", "text": instruction},
        {"type": "image_url", "image_url": {"url": url}}]}])).strip()


def _vision_map(provider: Any, instruction: str, cells: List[Any], workers: int = 12) -> List[str]:
    """One vision-model answer per image (concurrent HTTP calls)."""
    from concurrent.futures import ThreadPoolExecutor
    if provider is None:
        raise ExecutionError("an image operator needs the vision model")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda c: _vision_answer(provider, instruction, c), cells))


def _llm_row_values(llm: Any, rows: Table, columns: Optional[List[str]], task: str,
                    what: str, chunk: int = 25) -> List[Any]:
    """One LLM value per row for ``task`` (chunked over all rows)."""
    out: List[Any] = [None] * len(rows)
    for rng in _chunks(len(rows), chunk):
        items = {str(i): _cell_text(rows[i], columns) for i in rng}

        def complete(got: dict, rng=rng) -> None:
            missing = [i for i in rng if str(i) not in got]
            if missing:
                raise ValueError(f"omitted rows {missing[:10]}")

        got = _llm_json(llm, "You process data rows. Output JSON only.",
                        f"{task}\nRows (index -> content):\n{json.dumps(items, ensure_ascii=False)}\n"
                        'Return a JSON object mapping every index to its value.', dict, what,
                        complete)
        for i in rng:
            out[i] = got[str(i)]
    return out


def _yes(value: Any) -> bool:
    return str(value).strip().upper().startswith(("Y", "TRUE", "1"))


# ---------------------------------------------------------------------------
# Execution results
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """Result of executing a single operator node."""
    data: Table
    operator_type: str
    impl_type: str      # the implementation that executed (never substituted)
    row_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.row_count = len(self.data)


# ---------------------------------------------------------------------------
# Operator Executors
#
# Every operator reads its parameters from the binding environment B
# (ctx.metadata["bindings"][<operator type>], paper Sec. V).  Relational
# operators are deterministic routines; semantic operators run the LLM (or the
# vision model for image columns) over all of their input rows.  An operator
# whose parameters are missing, or whose model output cannot be parsed after a
# retry, raises ExecutionError instead of guessing.
# ---------------------------------------------------------------------------

class OperatorExecutor(ABC):
    """Base class for operator execution logic.

    ``IMPLS`` lists the physical implementations the executor runs, each a
    distinct algorithm; the plan executor rejects any other implementation."""

    IMPLS: Tuple[str, ...] = ()

    @abstractmethod
    def execute(
        self,
        inputs: List[Table],
        ctx: "ExecutionContext",
        node: LogicalPlanNode,
        impl_type: str,
    ) -> Table:
        """Execute the operator on input data tables."""


class _HashIndex:
    """Hash index of one column of a source: row ids by the equality key of
    ``_holds`` (lower-cased text, and the number for numeric cells)."""

    def __init__(self, rows: Table, column: str):
        self.by_key: Dict[tuple, List[int]] = {}
        for i, r in enumerate(rows):
            v = r.get(column)
            if v is None:
                continue
            self.by_key.setdefault(("s", str(v).lower()), []).append(i)
            n = _number(v)
            if n is not None:
                self.by_key.setdefault(("n", n), []).append(i)

    def lookup(self, cond: Dict[str, Any]) -> set:
        """Row ids satisfying an equality/IN condition, exactly as ``_holds``."""
        op, target = str(cond.get("op", "=")).strip().lower(), cond.get("value")
        if op == "in":
            targets = target if isinstance(target, list) else [target]
            keys = [("s", str(x).lower()) for x in targets]
        elif isinstance(target, (int, float)) and not isinstance(target, bool):
            keys = [("n", float(target))]
        else:
            keys = [("s", str(target).lower())]
        return {i for k in keys for i in self.by_key.get(k, [])}


class ScanExecutor(OperatorExecutor):
    """Reads the source bound to this leaf by the binding environment B
    (``ctx.metadata["sources"][node.source_name]``), else ``ctx.data_source``.

    TABLE_SCAN  returns every row of the source.
    INDEX_SCAN  pushes the equality/IN conditions of its parent FILTER (bound
                to the same source) into the scan: each is answered by a hash
                index on its column, built once per source and column and kept
                for later queries, and only the rows satisfying all of them
                are read, in source order.  The FILTER above still applies
                every condition, so both implementations give the same result.
    """

    IMPLS = ("TABLE_SCAN", "INDEX_SCAN")
    _indexes: Dict[tuple, _HashIndex] = {}

    def execute(self, inputs, ctx, node, impl_type):
        sources = ctx.metadata.get("sources") or {}
        name = getattr(node, "source_name", None)
        rows = sources[name] if name in sources else ctx.data_source
        if impl_type == "TABLE_SCAN":
            return list(rows)
        from .applicability import indexable_conditions
        fb = _binding(ctx, "FILTER")
        parent = node.parent
        if parent is None or parent.operator_type != "FILTER" or name is None \
                or fb.get("source") != name:
            raise ExecutionError("INDEX_SCAN needs a parent FILTER bound to the same source")
        if not rows:
            return []
        conds = indexable_conditions(fb, set(rows[0]))
        if not conds:
            raise ExecutionError(f"INDEX_SCAN: the FILTER on {name!r} has no equality/IN "
                                 "condition on a column of the source")
        hit: Optional[set] = None
        for c in conds:
            index = self.index(name, rows, c["column"])
            ids = index.lookup(c)
            hit = ids if hit is None else hit & ids
        ctx.notes["index_columns"] = [c["column"] for c in conds]
        ctx.notes["rows_read"] = len(hit)
        return [rows[i] for i in sorted(hit)]

    @classmethod
    def index(cls, source: str, rows: Table, column: str) -> _HashIndex:
        key = (source, column, id(rows), len(rows))
        if key not in cls._indexes:
            cls._indexes[key] = _HashIndex(rows, column)
        return cls._indexes[key]


# THRESHOLD_FILTER keeps a row when its score for the semantic condition is at
# least this value (B may override it with "threshold")
DEFAULT_FILTER_THRESHOLD = 0.5


class FilterExecutor(OperatorExecutor):
    """Structured conditions are applied exactly by both implementations; they
    differ on a semantic condition, judged per row by the LLM (or by the
    vision model on an image column).

    PREDICATE_FILTER  asks for a YES/NO verdict per row and keeps the YES rows.
    THRESHOLD_FILTER  asks for a score in [0, 1] of how well the row satisfies
                      the condition and keeps scores >= B's "threshold"
                      (default 0.5); it applies only to a semantic condition.
    """

    IMPLS = ("PREDICATE_FILTER", "THRESHOLD_FILTER")

    def execute(self, inputs, ctx, node, impl_type):
        data = inputs[0] if inputs else []
        b = _binding(ctx, "FILTER")
        conditions = b.get("conditions") or []
        semantic = b.get("semantic")
        if not conditions and not semantic:
            raise ExecutionError("FILTER needs 'conditions' or a 'semantic' condition in B")
        if impl_type == "THRESHOLD_FILTER" and not semantic:
            raise ExecutionError("THRESHOLD_FILTER needs a 'semantic' condition in B")
        out = [r for r in data if all(_holds(r, c) for c in conditions)]
        if not semantic or not out:
            return out
        columns = b.get("columns") or None
        image_col = next((c for c in (columns or []) if _is_image_column(out, c)), None)
        if impl_type == "PREDICATE_FILTER":
            if image_col:
                answers = _vision_map(
                    ctx.vision_provider,
                    f'Does this image satisfy: "{semantic}"? Answer strictly YES or NO.',
                    [r.get(image_col) for r in out])
            else:
                answers = _llm_row_values(
                    _require_llm(ctx, "FILTER"), out, columns,
                    f'For each row, answer YES if it satisfies "{semantic}", else NO.',
                    "FILTER")
            return [r for r, a in zip(out, answers) if _yes(a)]
        threshold = float(b.get("threshold", DEFAULT_FILTER_THRESHOLD))
        if image_col:
            scores = _vision_map(
                ctx.vision_provider,
                f'How well does this image satisfy: "{semantic}"? Answer with a number '
                'from 0 (not at all) to 1 (fully) only.', [r.get(image_col) for r in out])
        else:
            scores = _llm_row_values(
                _require_llm(ctx, "FILTER"), out, columns,
                f'For each row, score from 0 (not at all) to 1 (fully) how well it '
                f'satisfies "{semantic}".', "FILTER")
        ctx.notes["threshold"] = threshold
        return [r for r, s in zip(out, scores) if (_number(s) or 0.0) >= threshold]


class ProjectExecutor(OperatorExecutor):
    IMPLS = ("COLUMN_PROJECT",)

    def execute(self, inputs, ctx, node, impl_type):
        data = inputs[0] if inputs else []
        cols = _binding(ctx, "PROJECT").get("columns")
        if not cols:
            raise ExecutionError("PROJECT needs 'columns' in B")
        return [{k: r.get(k) for k in cols} for r in data]


def _order_token(v: Any) -> tuple:
    """Total-order key whose equality is Python equality of cell values
    (1 == 1.0 == True; "1" != 1), so sorting places equal values adjacently."""
    if v is None:
        return (2, "")
    if isinstance(v, (bool, int, float)):
        return (0, float(v)) if v == v else (3, "nan")
    if isinstance(v, str):
        return (1, v)
    return (4, repr(v))


class AggregateExecutor(OperatorExecutor):
    """Deterministic group-by with the key, function and attribute of B.  An
    optional ``condition`` {"op", "value"} on the attribute aggregates its
    indicator (1 if it holds, else 0), e.g. the share of positive reviews as
    avg of [scoreSentiment = POSITIVE].

    HASH_AGGREGATE    groups the rows in a hash table in one pass (groups in
                      order of first appearance).
    STREAM_AGGREGATE  sorts the rows by the group key and aggregates each run
                      of equal keys as it streams past (groups in key order).
    """

    IMPLS = ("HASH_AGGREGATE", "STREAM_AGGREGATE")
    FUNCTIONS = {"count", "avg", "mean", "average", "sum", "min", "max"}

    def execute(self, inputs, ctx, node, impl_type):
        data = inputs[0] if inputs else []
        spec = _binding(ctx, "AGGREGATE")
        fn = str(spec.get("function", "")).lower()
        if fn not in self.FUNCTIONS or (fn != "count" and not spec.get("attribute")):
            raise ExecutionError("AGGREGATE needs 'function' (count/avg/sum/min/max) and "
                                 "'attribute' in B")
        keys = list(spec.get("group_by") or [])
        attr = spec.get("attribute")
        name = spec.get("as") or (f"{fn}_{attr}" if attr else fn)
        if not data:
            return [{name: 0}] if (fn == "count" and not keys) else []
        cond = spec.get("condition")
        if impl_type == "HASH_AGGREGATE":
            groups: Dict[tuple, List[dict]] = {}
            for r in data:
                groups.setdefault(tuple(r.get(k) for k in keys), []).append(r)
            runs = groups.items()
        else:
            import itertools
            token = lambda r: tuple(_order_token(r.get(k)) for k in keys)  # noqa: E731
            runs = ((tuple(rows[0].get(k) for k in keys), rows) for rows in
                    (list(g) for _, g in itertools.groupby(sorted(data, key=token), key=token)))
        return [{**dict(zip(keys, key)), name: self._aggregate(fn, attr, cond, rows)}
                for key, rows in runs]

    @staticmethod
    def _aggregate(fn: str, attr: Optional[str], cond: Optional[dict], rows: Table) -> Any:
        raw: List[Any] = []
        if cond and attr:
            vals = [1.0 if _holds(r, {**cond, "column": attr}) else 0.0 for r in rows]
        elif attr:
            raw = [r.get(attr) for r in rows if r.get(attr) not in (None, "")]
            vals = [v for v in map(_number, raw) if v is not None]
            if fn != "count" and raw and not vals:
                raise ExecutionError(f"AGGREGATE {fn} over non-numeric {attr!r}; give a "
                                     "'condition' to aggregate its indicator")
        else:
            vals = []
        if fn == "count":   # rows whose attribute is present (or satisfies the condition)
            return (sum(vals) if cond else len(raw)) if attr else len(rows)
        if not vals:
            return None
        if fn in ("avg", "mean", "average"):
            return sum(vals) / len(vals)
        if fn == "sum":
            return sum(vals)
        return min(vals) if fn == "min" else max(vals)


_TRUE = {"yes", "true", "y", "1", "present", "visible"}
_FALSE = {"no", "false", "n", "0", "absent", "none", "not visible"}


def _typed(v: Any, kind: str) -> Any:
    """An extracted value in the type the binding declares: boolean answers
    (yes/no, true/false) become True/False, numbers become floats; a value that
    does not fit the type becomes None (missing), never a guessed value."""
    if kind == "boolean":
        if isinstance(v, bool):
            return v
        t = str(v).strip().strip(".!").lower()
        return True if t in _TRUE else False if t in _FALSE else None
    if kind == "number":
        return _number(v)
    return _number(v) if _number(v) is not None else v


def _number(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


# EXTERNAL_SORT spills sorted runs of this many rows (B "run_rows" overrides it)
DEFAULT_SORT_RUN_ROWS = 50_000


class SortExecutor(OperatorExecutor):
    """Sorts on B's key (numerically if every present value is a number),
    descending by default; rows missing the key come last.  Both
    implementations are stable and give the same order.

    MEMORY_SORT    sorts all rows in memory.
    EXTERNAL_SORT  sorts runs of at most B's "run_rows" rows, spills each run
                   to a temporary file, and k-way merges the runs (heap merge),
                   holding one row per run in memory during the merge.
    """

    IMPLS = ("MEMORY_SORT", "EXTERNAL_SORT")

    def execute(self, inputs, ctx, node, impl_type):
        data = inputs[0] if inputs else []
        spec = _binding(ctx, "SORT")
        key = spec.get("key")
        if not key:
            raise ExecutionError("SORT needs 'key' in B")
        rev = bool(spec.get("descending", True))
        present = [r for r in data if r.get(key) is not None]
        missing = [r for r in data if r.get(key) is None]
        num = all(_number(r[key]) is not None for r in present)
        sort_key = (lambda r: _number(r[key])) if num else (lambda r: str(r[key]))
        if impl_type == "MEMORY_SORT":
            return sorted(present, key=sort_key, reverse=rev) + missing
        run_rows = int(spec.get("run_rows") or DEFAULT_SORT_RUN_ROWS)
        if run_rows <= 0:
            raise ExecutionError("EXTERNAL_SORT needs a positive 'run_rows'")
        return self._external(present, sort_key, rev, run_rows, ctx) + missing

    @staticmethod
    def _external(rows: Table, sort_key, rev: bool, run_rows: int, ctx) -> Table:
        import heapq
        import pickle
        import tempfile

        def read(path):
            with open(path, "rb") as f:
                while True:
                    try:
                        yield pickle.load(f)
                    except EOFError:
                        return

        with tempfile.TemporaryDirectory(prefix="external_sort_") as tmp:
            runs = []
            for n, start in enumerate(range(0, len(rows), run_rows)):
                path = f"{tmp}/run{n}.pkl"
                with open(path, "wb") as f:
                    for r in sorted(rows[start:start + run_rows], key=sort_key, reverse=rev):
                        pickle.dump(r, f, protocol=pickle.HIGHEST_PROTOCOL)
                runs.append(path)
            ctx.notes["runs"] = len(runs)
            # ties go to the earlier run, so the merge is stable like sorted()
            return list(heapq.merge(*(read(p) for p in runs), key=sort_key, reverse=rev))


def limit_k(binding: Optional[Dict[str, Any]]) -> int:
    """Number of rows a LIMIT keeps: the positive integer 'count' of B."""
    v = (binding or {}).get("count")
    if isinstance(v, str) and v.strip().isdigit():
        v = int(v.strip())
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
        raise ExecutionError(f"LIMIT needs a positive integer 'count' in B, got {v!r}")
    return v


class LimitExecutor(OperatorExecutor):
    IMPLS = ("STREAMING_LIMIT",)

    def execute(self, inputs, ctx, node, impl_type):
        data = inputs[0] if inputs else []
        return data[:limit_k(_binding(ctx, "LIMIT"))]


class DistinctExecutor(OperatorExecutor):
    """Keeps the first row of every distinct value of B's columns (all columns
    by default).

    HASH_DISTINCT    one pass with a hash set of the keys seen (input order).
    STREAM_DISTINCT  stable sort by the key, then drops each row equal to its
                     predecessor (key order).
    """

    IMPLS = ("HASH_DISTINCT", "STREAM_DISTINCT")

    def execute(self, inputs, ctx, node, impl_type):
        data = inputs[0] if inputs else []
        cols = _binding(ctx, "DISTINCT").get("columns")

        def key(row):
            return json.dumps({k: row.get(k) for k in (cols or sorted(row))},
                              sort_keys=True, default=str)

        out: Table = []
        if impl_type == "HASH_DISTINCT":
            seen = set()
            for row in data:
                k = key(row)
                if k not in seen:
                    seen.add(k)
                    out.append(row)
            return out
        previous = None
        for k, row in sorted(((key(r), r) for r in data), key=lambda p: p[0]):
            if k != previous:
                out.append(row)
                previous = k
        return out


class JoinExecutor(OperatorExecutor):
    """Equality join on the columns bound in B ("on": [left, right]) or, if
    none are bound, on the single column the inputs share.  Both
    implementations emit the pairs in the same order (left-major).

    HASH_JOIN         builds a hash table on the right input and probes it
                      with every left row.
    NESTED_LOOP_JOIN  compares every left row with every right row.
    """

    IMPLS = ("HASH_JOIN", "NESTED_LOOP_JOIN")

    def execute(self, inputs, ctx, node, impl_type):
        if len(inputs) != 2:
            raise ExecutionError("JOIN needs two inputs")
        left, right = inputs
        on = _binding(ctx, "JOIN").get("on")
        if on and len(on) == 2:
            lk, rk = on
        else:
            shared = sorted(set(left[0]) & set(right[0])) if left and right else []
            if len(shared) != 1:
                raise ExecutionError(f"JOIN needs 'on' in B (shared columns: {shared})")
            lk = rk = shared[0]
        if impl_type == "NESTED_LOOP_JOIN":
            right_keys = [_join_key(r.get(rk)) for r in right]
            out: Table = []
            for l in left:
                k = _join_key(l.get(lk))
                if k is None:
                    continue
                out.extend({**l, **r} for r, rkey in zip(right, right_keys) if rkey == k)
            return out
        index: Dict[Any, List[dict]] = {}
        for r in right:
            key = _join_key(r.get(rk))
            if key is not None:
                index.setdefault(key, []).append(r)
        return [{**l, **r} for l in left
                for r in index.get(_join_key(l.get(lk)), []) if _join_key(l.get(lk)) is not None]


def _join_key(v: Any) -> Any:
    """Equality key of a join cell: NULL never matches, numbers compare by value
    (1 = "1.0", as CSV cells arrive as strings), other values as trimmed text."""
    if v is None or (isinstance(v, float) and v != v) or (isinstance(v, str) and not v.strip()):
        return None
    n = _number(v)
    return ("n", n) if n is not None else ("s", str(v).strip())


class UnionExecutor(OperatorExecutor):
    IMPLS = ("UNION_ALL",)

    def execute(self, inputs, ctx, node, impl_type):
        return [r for inp in inputs for r in inp]


class ContentExtractExecutor(OperatorExecutor):
    """Extracts an attribute from the text or image columns bound in B.

    Implementations (paper Sec. V-D):
      METADATA_LOOKUP  reads the per-document value precomputed by TiMetaGraph
      TOOL_EXTRACT     runs a specialized model at query time: VADER for text
                       sentiment; CLIP zero-shot classification for an image
                       column, over yes/no for a boolean attribute or over B's
                       candidate "values" otherwise
      LLM_EXTRACT      extracts the attribute with the LLM (the vision-language
                       model for an image column)
    The planner offers METADATA_LOOKUP only for precomputed attributes and
    TOOL_EXTRACT only for text sentiment or an image column with a boolean
    type or candidate values; an implementation that does not apply raises.
    """

    IMPLS = ("METADATA_LOOKUP", "TOOL_EXTRACT", "LLM_EXTRACT")

    def execute(self, inputs, ctx, node, impl_type):
        data = inputs[0] if inputs else []
        b = _binding(ctx, "CONTENT_EXTRACT")
        attr = b.get("attribute")
        columns = b.get("columns")
        if not attr or not columns:
            raise ExecutionError("CONTENT_EXTRACT needs 'attribute' and 'columns' in B")
        if not data:
            return data
        missing = [c for c in columns if c not in data[0]]
        if missing:
            raise ExecutionError(f"CONTENT_EXTRACT: columns {missing} not in the input")
        description = b.get("description") or attr
        image_col = next((c for c in columns if _is_image_column(data, c)), None)
        # the document as TiMetaGraph keyed it: text columns in sorted order
        text = [" ".join(str(r.get(c) or "") for c in sorted(columns)).strip() for r in data]
        kind = b.get("type") or "text"
        answer_form = {"boolean": "Answer yes or no.", "number": "Answer with a number only."
                       }.get(kind, "Answer with the value only.")
        prompt = f"From this image, extract: {description}. {answer_form}"
        if impl_type == "METADATA_LOOKUP":
            table = (ctx.metadata.get("doc_attributes") or {}).get(attr) or {}
            values = [table.get(doc_key(t)) for t in text]
            if image_col or any(v is None for v in values):
                raise ExecutionError(f"METADATA_LOOKUP: {attr!r} is not precomputed for every document")
        elif impl_type == "TOOL_EXTRACT":
            if image_col:
                values = _clip_zero_shot([r.get(image_col) for r in data], attr, description,
                                         kind, b.get("values"))
            elif attr == "sentiment":
                values = _vader_scores(text)
            else:
                raise ExecutionError(f"TOOL_EXTRACT: no tool extracts {attr!r} from text")
        elif impl_type == "LLM_EXTRACT":
            if image_col:
                values = _vision_map(ctx.vision_provider, prompt, [r.get(image_col) for r in data])
            else:
                values = _llm_row_values(_require_llm(ctx, "CONTENT_EXTRACT"), data, columns,
                                         f"For each row, extract: {description}. "
                                         + {"boolean": "Give yes or no.",
                                            "number": "Give a number only."
                                            }.get(kind, "Give a number when the attribute is numeric."),
                                         "CONTENT_EXTRACT")
        else:
            raise ExecutionError(f"CONTENT_EXTRACT: unknown implementation {impl_type!r}")
        return [{**r, attr: _typed(v, kind)} for r, v in zip(data, values)]


# Sentence encoders (sentence-transformers): text rows are embedded with
# MiniLM; images, and text that must share their space, with CLIP ViT-B/32.
TEXT_MODEL = "all-MiniLM-L6-v2"
CLIP_MODEL = "clip-ViT-B-32"
_MODELS: Dict[str, Any] = {}


def _encoder(name: str):
    if name not in _MODELS:
        from sentence_transformers import SentenceTransformer
        _MODELS[name] = SentenceTransformer(name)
    return _MODELS[name]


def _embed_texts(texts: List[str], model: str = TEXT_MODEL):
    """Unit-norm embeddings of texts (CLIP truncates to its 77-token window)."""
    enc = _encoder(model)
    if model == CLIP_MODEL:
        tok = enc[0].processor.tokenizer
        texts = [tok.decode(tok(t, truncation=True, max_length=75)["input_ids"][1:-1])
                 for t in texts]
    return enc.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)


def _embed_images(cells: List[Any]):
    import io
    from PIL import Image
    images = [Image.open(io.BytesIO(_image_bytes_of(c))).convert("RGB") for c in cells]
    return _encoder(CLIP_MODEL).encode(images, normalize_embeddings=True, show_progress_bar=False)


def _embed_rows(rows: Table, columns: Optional[List[str]]):
    """Unit-norm row embeddings and the model used: CLIP on an image column,
    MiniLM on the text of the columns otherwise."""
    image_col = next((c for c in (columns or []) if _is_image_column(rows, c)), None)
    if image_col:
        return _embed_images([r.get(image_col) for r in rows]), CLIP_MODEL
    return _embed_texts([_cell_text(r, columns, 2000) for r in rows]), TEXT_MODEL


def _spherical_kmeans(x, k: int, seed: int, iters: int = 10):
    """k unit-norm centroids of the unit-norm rows of x (Lloyd iterations)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    centroids = x[rng.choice(len(x), size=k, replace=False)].copy()
    for _ in range(iters):
        assign = np.argmax(x @ centroids.T, axis=1)
        for c in range(k):
            members = x[assign == c]
            if len(members):
                m = members.sum(axis=0)
                centroids[c] = m / max(float(np.linalg.norm(m)), 1e-12)
    return centroids, np.argmax(x @ centroids.T, axis=1)


def _clip_zero_shot(cells: List[Any], attr: str, description: str, kind: str,
                    values: Optional[List[Any]]) -> List[Any]:
    """CLIP zero-shot label of each image: yes/no for a boolean attribute,
    else the most similar of the candidate values."""
    if kind == "boolean":
        labels, prompts = ["yes", "no"], [f"a photo with {description}",
                                          f"a photo without {description}"]
    elif values:
        labels = list(values)
        prompts = [f"a photo whose {attr} is {v}" for v in labels]
    else:
        raise ExecutionError("TOOL_EXTRACT on images needs a boolean attribute or "
                             "candidate 'values' in B")
    sims = _embed_images(cells) @ _embed_texts(prompts, CLIP_MODEL).T
    return [labels[int(i)] for i in sims.argmax(axis=1)]


class SemanticSearchExecutor(OperatorExecutor):
    """Ranks rows by the cosine similarity of their embedding (B's columns; an
    image column with CLIP) to B's query and keeps the top k (all rows by
    default), with the similarity as "relevance".

    VECTOR_SCAN    exact: scores every row.
    VECTOR_SEARCH  approximate (IVF): clusters the rows into nlist =
                   round(sqrt(n)) spherical k-means cells (seed 0), probes the
                   nprobe = ceil(sqrt(nlist)) cells whose centroids are closest
                   to the query (more while they hold fewer than k rows), and
                   scores only their rows.
    """

    IMPLS = ("VECTOR_SEARCH", "VECTOR_SCAN")

    def execute(self, inputs, ctx, node, impl_type):
        import numpy as np
        data = inputs[0] if inputs else []
        b = _binding(ctx, "SEMANTIC_SEARCH")
        query = b.get("query")
        if not query:
            raise ExecutionError("SEMANTIC_SEARCH needs 'query' in B")
        if not data:
            return data
        k = min(int(b.get("k") or len(data)), len(data))
        vectors, model = _embed_rows(data, b.get("columns"))
        q = _embed_texts([str(query)], model)[0]
        if impl_type == "VECTOR_SCAN":
            candidates = np.arange(len(data))
        else:
            n = len(data)
            nlist = max(1, int(round(n ** 0.5)))
            nprobe = max(1, int(np.ceil(nlist ** 0.5)))
            centroids, assign = _spherical_kmeans(vectors, nlist, seed=0)
            chosen: List[int] = []
            for probed, c in enumerate(np.argsort(-(centroids @ q)), start=1):
                chosen.extend(np.flatnonzero(assign == c).tolist())
                if probed >= nprobe and len(chosen) >= k:
                    break
            candidates = np.array(sorted(chosen))
            ctx.notes.update(nlist=nlist, nprobe=probed, scored=len(candidates))
        scores = vectors[candidates] @ q
        order = candidates[np.argsort(-scores, kind="stable")][:k]
        sims = vectors[order] @ q
        return [{**data[i], "relevance": float(s)} for i, s in zip(order, sims)]


class CrossModalMatchExecutor(OperatorExecutor):
    """Matches the rows of two inputs that refer to the same entity.

    ENTITY_LINK_MATCH uses the CrossModalEntity links of the metadata graph;
    LLM_MATCH asks the LLM which candidate rows each probe row refers to,
    over all rows of both inputs (in chunks)."""

    IMPLS = ("ENTITY_LINK_MATCH", "LLM_MATCH")

    def execute(self, inputs, ctx, node, impl_type):
        if len(inputs) != 2:
            raise ExecutionError("CROSS_MODAL_MATCH needs two inputs")
        left, right = inputs
        if not left or not right:
            return []
        if impl_type == "ENTITY_LINK_MATCH":
            return self._entity_link_match(left, right, ctx,
                                           self._entity_links(left, right, ctx))
        return llm_match(_require_llm(ctx, "CROSS_MODAL_MATCH"), left, right,
                         _binding(ctx, "CROSS_MODAL_MATCH").get("condition")
                         or "the two rows refer to the same entity", "CROSS_MODAL_MATCH")

    @staticmethod
    def _entity_links(left: Table, right: Table, ctx: "ExecutionContext"):
        """CrossModalEntity links between the two inputs, from the metadata
        graph: (text side, row side, document column, the rows denoting each
        entity, the entities each document mentions).  None when the graph
        holds no links or neither input holds documents TiMetaGraph processed."""
        doc_entities = ctx.metadata.get("doc_entities") or {}
        by_surface = ctx.metadata.get("entity_ids_by_surface") or {}
        if not doc_entities or not by_surface:
            return None

        def document_column(rows: Table):
            """The column whose cells are documents TiMetaGraph processed
            (exact doc_key hits), with its number of hits."""
            best = (None, 0)
            for col in rows[0]:
                hits = sum(isinstance(r.get(col), str) and doc_key(r.get(col)) in doc_entities
                           for r in rows)
                if hits > best[1]:
                    best = (col, hits)
            return best

        (lcol, lhits), (rcol, rhits) = document_column(left), document_column(right)
        if lhits == 0 and rhits == 0:
            return None
        text_side, row_side, text_col = (left, right, lcol) if lhits >= rhits else (right, left, rcol)
        rows_by_entity: Dict[str, Dict[int, dict]] = {}
        for k, row in enumerate(row_side):
            for v in row.values():
                eid = by_surface.get(re.sub(r"[^a-z0-9]", "", str(v).lower()))
                if eid:
                    # a row whose columns name the same entity twice joins once
                    rows_by_entity.setdefault(eid, {})[k] = row
        doc_eids = [list(dict.fromkeys(doc_entities.get(doc_key(doc.get(text_col)), [])))
                    for doc in text_side]
        return text_side, row_side, text_col, rows_by_entity, doc_eids

    @staticmethod
    def _entity_link_match(left: Table, right: Table, ctx: "ExecutionContext", links) -> Table:
        """Join text documents to the rows whose values denote an entity the
        document mentions, using the CrossModalEntity links of the metadata graph."""
        if links is None:
            if not (ctx.metadata.get("doc_entities") and ctx.metadata.get("entity_ids_by_surface")):
                raise ExecutionError("ENTITY_LINK_MATCH needs the entity links of the metadata graph")
            raise ExecutionError("ENTITY_LINK_MATCH: neither input holds documents "
                                 "processed by TiMetaGraph")
        text_side, _, _, rows_by_entity, doc_eids = links
        out: Table = []
        for doc, eids in zip(text_side, doc_eids):
            matched: Dict[int, dict] = {}
            for eid in eids:
                matched.update(rows_by_entity.get(eid, {}))
            out.extend({**row, **doc} for row in matched.values())
        return out


def llm_match(llm: Any, left: Table, right: Table, condition: str, what: str,
              probe_chunk: int = 15, candidate_block: int = 120) -> Table:
    """For every right row, the LLM picks the left rows satisfying ``condition``
    with it; all rows of both inputs are covered (candidates in blocks)."""
    out: Table = []
    for block in _chunks(len(left), candidate_block):
        candidates = {str(i): _cell_text(left[i], None, 200) for i in block}
        for rng in _chunks(len(right), probe_chunk):
            probes = {str(j): _cell_text(right[j], None, 600) for j in rng}
            got = _llm_json(
                llm, "You match rows of two tables. Output JSON only.",
                f'Condition: {condition}.\nCandidates (index -> content):\n'
                f'{json.dumps(candidates, ensure_ascii=False)}\nProbe rows (index -> content):\n'
                f'{json.dumps(probes, ensure_ascii=False)}\n'
                'Return a JSON object mapping every probe index to the list of candidate '
                'indices it satisfies the condition with ([] if none).', dict, what)
            for j, idxs in got.items():
                if not (j.isdigit() and int(j) in rng) or not isinstance(idxs, list):
                    continue
                for i in idxs:
                    if str(i) in candidates:
                        out.append({**left[int(i)], **right[int(j)]})
    return out


# INTRA_MODAL_JOIN pairs a right row with its most similar left rows whose
# cosine similarity is at least this value (B "threshold", "k" override)
DEFAULT_SIMILARITY_THRESHOLD = 0.5


class SimilarityJoinExecutor(OperatorExecutor):
    """Joins two inputs on a semantic similarity condition bound in B.

    INTRA_MODAL_JOIN  embeds both inputs with one encoder of their common
                      modality (B "columns": [left, right], whole rows by
                      default; CLIP for image columns) and pairs every right
                      row with its top-k (default 1) left rows of cosine
                      similarity >= B's "threshold" (default 0.5).
    SEMANTIC_JOIN     the LLM decides the condition for the pairs (llm_match).
    """

    IMPLS = ("INTRA_MODAL_JOIN", "SEMANTIC_JOIN")

    def execute(self, inputs, ctx, node, impl_type):
        import numpy as np
        if len(inputs) != 2:
            raise ExecutionError("SIMILARITY_JOIN needs two inputs")
        left, right = inputs
        b = _binding(ctx, "SIMILARITY_JOIN")
        condition = b.get("condition")
        if not condition:
            raise ExecutionError("SIMILARITY_JOIN needs 'condition' in B")
        if not left or not right:
            return []
        if impl_type == "SEMANTIC_JOIN":
            return llm_match(_require_llm(ctx, "SIMILARITY_JOIN"), left, right, condition,
                             "SIMILARITY_JOIN")
        cols = list(b.get("columns") or [])
        lcols, rcols = ([cols[0]], [cols[1]]) if len(cols) == 2 else (None, None)
        lv, lmodel = _embed_rows(left, lcols)
        rv, rmodel = _embed_rows(right, rcols)
        if lmodel != rmodel:
            raise ExecutionError("INTRA_MODAL_JOIN needs two inputs of the same modality")
        threshold = float(b.get("threshold", DEFAULT_SIMILARITY_THRESHOLD))
        k = int(b.get("k") or 1)
        sims = rv @ lv.T
        out: Table = []
        for j, row in enumerate(sims):
            for i in np.argsort(-row, kind="stable")[:k]:
                if row[i] >= threshold:
                    out.append({**left[int(i)], **right[j]})
        ctx.notes["threshold"] = threshold
        return out


class FeatureTransformExecutor(OperatorExecutor):
    """Embeds one column into a vector column named by B with CLIP ViT-B/32,
    whose image and text towers share one space.

    CLIP_IMAGE_ENCODER  the image tower, for an image column.
    CLIP_TEXT_ENCODER   the text tower, for a text column (77-token window).
    """

    IMPLS = ("CLIP_IMAGE_ENCODER", "CLIP_TEXT_ENCODER")

    def execute(self, inputs, ctx, node, impl_type):
        data = inputs[0] if inputs else []
        b = _binding(ctx, "FEATURE_TRANSFORM")
        cols = b.get("columns") or []
        if len(cols) != 1:
            raise ExecutionError("FEATURE_TRANSFORM needs one input column in 'columns'")
        if not data:
            return data
        col, name = cols[0], b.get("as") or f"{cols[0]}_embedding"
        image = _is_image_column(data, col)
        if image != (impl_type == "CLIP_IMAGE_ENCODER"):
            raise ExecutionError(f"{impl_type} does not encode the "
                                 f"{'image' if image else 'text'} column {col!r}")
        if image:
            vectors = _embed_images([r.get(col) for r in data])
        else:
            vectors = _embed_texts([str(r.get(col) or "") for r in data], CLIP_MODEL)
        return [{**r, name: [float(x) for x in v]} for r, v in zip(data, vectors)]


def _image_bytes_of(cell: Any) -> bytes:
    import base64
    import urllib.request
    if isinstance(cell, dict) and cell.get("bytes"):
        return cell["bytes"]
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    s = str(cell or "")
    if s.startswith("data:"):
        return base64.b64decode(s.split(",", 1)[1])
    if s.startswith("http"):
        with urllib.request.urlopen(s, timeout=60) as resp:
            return resp.read()
    raise ExecutionError("cell holds no image")


class VisualizeExecutor(OperatorExecutor):
    """Passes the result through and records the chart specification of B
    ({"chart": bar|line|scatter|pie|table, "x": ..., "y": ...}) for the chart
    recommendation step."""

    IMPLS = ("MATERIALIZE",)

    def execute(self, inputs, ctx, node, impl_type):
        data = inputs[0] if inputs else []
        spec = _binding(ctx, "VISUALIZE")
        if not spec.get("chart"):
            raise ExecutionError("VISUALIZE needs 'chart' in B")
        ctx.metadata["chart"] = {k: spec.get(k) for k in ("chart", "x", "y")}
        return data


class ExportExecutor(OperatorExecutor):
    """Writes the result to the path bound in B; the path's extension decides
    the implementation (CSV_EXPORT for .csv, JSON_EXPORT for .json)."""

    IMPLS = ("CSV_EXPORT", "JSON_EXPORT")

    def execute(self, inputs, ctx, node, impl_type):
        from .applicability import export_implementation
        data = inputs[0] if inputs else []
        path = _binding(ctx, "EXPORT").get("path")
        if not path:
            raise ExecutionError("EXPORT needs 'path' in B")
        try:
            wanted = export_implementation(path)
        except ValueError as e:
            raise ExecutionError(str(e)) from None
        if impl_type != wanted:
            raise ExecutionError(f"{impl_type} cannot write {path!r} ({wanted} does)")
        if impl_type == "CSV_EXPORT":
            import csv as _csv
            cols: List[str] = []
            for r in data:
                cols += [c for c in r if c not in cols]
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = _csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                w.writerows(data)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, default=str)
        return data


# ---------------------------------------------------------------------------
# Executor Registry
# ---------------------------------------------------------------------------

EXECUTOR_REGISTRY: Dict[str, OperatorExecutor] = {
    "SCAN": ScanExecutor(),
    "FILTER": FilterExecutor(),
    "PROJECT": ProjectExecutor(),
    "AGGREGATE": AggregateExecutor(),
    "SORT": SortExecutor(),
    "LIMIT": LimitExecutor(),
    "DISTINCT": DistinctExecutor(),
    "JOIN": JoinExecutor(),
    "UNION": UnionExecutor(),
    "SEMANTIC_SEARCH": SemanticSearchExecutor(),
    "CROSS_MODAL_MATCH": CrossModalMatchExecutor(),
    "CONTENT_EXTRACT": ContentExtractExecutor(),
    "SIMILARITY_JOIN": SimilarityJoinExecutor(),
    "FEATURE_TRANSFORM": FeatureTransformExecutor(),
    "VISUALIZE": VisualizeExecutor(),
    "EXPORT": ExportExecutor(),
}


# ---------------------------------------------------------------------------
# Plan Executor
# ---------------------------------------------------------------------------

@dataclass
class PlanExecutionResult:
    """Complete result of executing a physical plan."""
    data: Table
    operator_results: Dict[str, ExecutionResult]
    root_operator: str
    row_count: int = 0

    def __post_init__(self):
        self.row_count = len(self.data)


class PlanExecutor:
    """Executes a physical plan tree bottom-up against data.

    Usage::

        executor = PlanExecutor(data_source=rows, llm_provider=provider)
        result = executor.execute(logical_plan, physical_assignment)
        print(result.data)  # final output rows
    """

    def __init__(
        self,
        data_source: Table,
        llm_provider: Any = None,
        query_text: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        vision_provider: Any = None,
    ):
        self._ctx = ExecutionContext(
            data_source=data_source,
            llm_provider=llm_provider,
            query_text=query_text,
            metadata=metadata or {},
            vision_provider=vision_provider,
        )
        self._operator_results: Dict[str, ExecutionResult] = {}

    def execute(
        self,
        plan: LogicalPlanNode,
        assignment: Dict[str, str],
    ) -> PlanExecutionResult:
        """Execute the full plan tree and return results."""
        self._operator_results.clear()
        data = self._execute_node(plan, assignment)

        return PlanExecutionResult(
            data=data,
            operator_results=dict(self._operator_results),
            root_operator=plan.operator_type,
        )

    def _execute_node(
        self,
        node: LogicalPlanNode,
        assignment: Dict[str, str],
    ) -> Table:
        """Recursively execute a node after its children."""
        child_outputs = []
        for child in node.children:
            child_data = self._execute_node(child, assignment)
            child_outputs.append(child_data)

        # assignments are per operator instance (paper Sec. V-D)
        impl_type = assignment.get(node.node_id)
        if impl_type is None:
            raise ExecutionError(f"no implementation assigned to {node.operator_type} "
                                 f"({node.node_id})")
        executor = EXECUTOR_REGISTRY.get(node.operator_type)
        if executor is None:
            raise ExecutionError(f"no executor for operator type {node.operator_type!r}")
        if impl_type not in executor.IMPLS:
            raise ExecutionError(f"unknown implementation {impl_type!r} of {node.operator_type} "
                                 f"(known: {list(executor.IMPLS)})")
        self._ctx.notes = {}
        try:
            data = executor.execute(child_outputs, self._ctx, node, impl_type)
        except Exception as exc:
            raise PlanExecutionError(
                f"{node.operator_type} ({impl_type}) failed: {exc}", node.node_id,
                node.operator_type, dict(self._operator_results)) from exc

        result = ExecutionResult(
            data=data,
            operator_type=node.operator_type,
            impl_type=impl_type,
            metadata={"executed_impl": impl_type, **self._ctx.notes},
        )
        self._operator_results[node.node_id] = result

        return data

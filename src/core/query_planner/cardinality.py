"""Cardinality estimation of logical plans (paper Sec. V-B, Eq. 7).

Costs are charged per record, so TiTSP estimates the output cardinality n^(n)
of every operator instance bottom-up from TiMetaGraph statistics.  A SCAN
outputs the exact record count n_o of its source; for an inner operator with
child estimates n^_1 (and n^_2):

    n^_1                          record-preserving (every other unary operator)
    min(k, n^_1)                  LIMIT, SEMANTIC_SEARCH
    s_p * n^_1                    FILTER
    min(d_G, n^_1)                AGGREGATE, DISTINCT
    n^_1 * n^_2 / max(d_1, d_2)   binary matching (JOIN, CROSS_MODAL_MATCH,
                                  SIMILARITY_JOIN; containment assumption)
    n^_1 + n^_2                   UNION

k is the bound count; d_G the number of distinct group keys; d_1, d_2 the
numbers of distinct join-key values or, for the cross-modal operators, of
linked CrossModalEntity nodes.  The selectivity s_p is 1/d_c for an equality on
a column with d_c distinct values, |V|/d_c for membership in a set V, the
covered fraction of the value range for a range predicate, their product for a
conjunction, and the default 1/3 for a predicate without statistics, such as a
semantic condition.  Distinct counts and value ranges come from the column
summaries and entity links of TiMetaGraph and are capped by the input
cardinality.

Every node also records whether its estimate is grounded in statistics; an
estimate that relies on the default selectivity (or lacks the statistics it
needs) is not, and carries no calibration signal (Eq. 11).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

DEFAULT_SELECTIVITY = 1.0 / 3.0     # predicate without statistics (Selinger et al.)

LIMITING = ("LIMIT", "SEMANTIC_SEARCH")
GROUPING = ("AGGREGATE", "DISTINCT")
BINARY_MATCHING = ("JOIN", "CROSS_MODAL_MATCH", "SIMILARITY_JOIN")
CROSS_MODAL_MATCHING = ("CROSS_MODAL_MATCH", "SIMILARITY_JOIN")

# Operators realized by pre-programmed deterministic routines (paper Sec. V);
# a FILTER is deterministic unless B binds a semantic condition.
DETERMINISTIC_OPERATORS = frozenset({
    "SCAN", "PROJECT", "AGGREGATE", "SORT", "LIMIT", "DISTINCT", "JOIN", "UNION",
    "VISUALIZE", "EXPORT",
})


def is_deterministic(operator_type: str, bindings: Optional[Dict[str, Dict]] = None) -> bool:
    """Whether the operator runs a deterministic routine, exact by construction."""
    if operator_type == "FILTER":
        return not ((bindings or {}).get("FILTER") or {}).get("semantic")
    return operator_type in DETERMINISTIC_OPERATORS


def source_statistics(graph: Any) -> Dict[str, Any]:
    """Statistics of TiMetaGraph used by the estimates: per source, the
    distinct count and value range of every column (column summaries), and the
    number of distinct CrossModalEntity nodes its records are linked to."""
    columns: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for mid, meta in graph.modalities.items():
        stats = (meta.content_summary or {}).get("column_stats") or {}
        cols = {}
        for c, st in stats.items():
            if not isinstance(st, dict) or not st.get("n_distinct"):
                continue
            lo, hi = st.get("min"), st.get("max")
            numeric = all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (lo, hi))
            cols[str(c)] = {"n_distinct": int(st["n_distinct"]),
                            "min": float(lo) if numeric else None,
                            "max": float(hi) if numeric else None}
        if cols:
            columns[mid] = cols
    linked: Dict[str, Set[str]] = {}
    for eid, entity in graph.entities.items():
        for m in entity.mentions:
            linked.setdefault(m.modality_id, set()).add(eid)
    return {"columns": columns, "entities": {mid: len(ids) for mid, ids in linked.items()}}


def _number(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


class _Estimator:
    def __init__(self, bindings: Dict[str, Dict], statistics: Dict[str, Any]):
        self._bindings = bindings or {}
        self._columns: Dict[str, Dict[str, Dict[str, Any]]] = (statistics or {}).get("columns") or {}
        self._entities: Dict[str, int] = (statistics or {}).get("entities") or {}

    # -- statistics lookups --------------------------------------------------
    def _column(self, column: Any, sources: List[str]) -> Optional[Dict[str, Any]]:
        for s in sources:
            st = (self._columns.get(s) or {}).get(str(column))
            if st:
                return st
        return None

    def _distinct(self, column: Any, sources: List[str], n_in: float) -> Optional[float]:
        """d_c capped by the input cardinality; None without statistics."""
        st = self._column(column, sources)
        return None if st is None else max(1.0, min(float(st["n_distinct"]), n_in))

    def _condition(self, cond: Dict[str, Any], sources: List[str], n_in: float) -> Optional[float]:
        """Selectivity of one structured condition; None without statistics."""
        op, value = str(cond.get("op", "=")).strip().lower(), cond.get("value")
        d = self._distinct(cond.get("column"), sources, n_in)
        if d is None:
            return None
        if op in ("=", "=="):
            return 1.0 / d
        if op == "!=":
            return 1.0 - 1.0 / d
        if op == "in":
            members = value if isinstance(value, list) else [value]
            return min(1.0, len({str(x).lower() for x in members}) / d)
        if op in (">", ">=", "<", "<="):
            st, t = self._column(cond.get("column"), sources), _number(value)
            lo, hi = st.get("min"), st.get("max")
            if t is None or lo is None or hi is None:
                return None
            if hi == lo:
                holds = {">": lo > t, ">=": lo >= t, "<": lo < t, "<=": lo <= t}[op]
                return 1.0 if holds else 0.0
            share = (hi - t) / (hi - lo) if op in (">", ">=") else (t - lo) / (hi - lo)
            return max(0.0, min(1.0, share))
        return None     # e.g. substring match: no statistics

    # -- Eq. 7 ---------------------------------------------------------------
    def estimate(self, node: Any) -> Tuple[float, bool, List[str]]:
        """(n^, grounded, sources below) of ``node``; annotates the subtree."""
        children = [self.estimate(c) for c in node.children]
        op = node.operator_type
        b = self._bindings.get(op) or {}
        if not children:
            n_o = getattr(node, "source_cardinality", None)
            known = isinstance(n_o, (int, float)) and n_o >= 0
            out, grounded = (float(n_o) if known else 1.0), bool(known)
            sources = [node.source_name] if getattr(node, "source_name", None) else []
            n_in = out
        else:
            n1, g1, s1 = children[0]
            sources = [s for _, _, ss in children for s in ss]
            grounded = all(g for _, g, _ in children)
            n_in = sum(n for n, _, _ in children)
            if op == "UNION":
                out = n_in
            elif len(children) >= 2 and op in BINARY_MATCHING:
                n2, _, s2 = children[1]
                d1, d2 = self._match_keys(op, b, (n1, s1), (n2, s2))
                if d1 is None or d2 is None:
                    grounded = False
                    d1 = n1 if d1 is None else d1
                    d2 = n2 if d2 is None else d2
                denom = max(d1, d2)
                out = n1 * n2 / denom if denom > 0 else 0.0
            elif op in LIMITING:
                k = _number(b.get("count") if op == "LIMIT" else b.get("k"))
                out = min(k, n1) if k is not None and k > 0 else n1
            elif op == "FILTER":
                sel = 1.0
                conditions = b.get("conditions") or []
                for cond in conditions:
                    s = self._condition(cond, sources, n1)
                    if s is None:
                        s, grounded = DEFAULT_SELECTIVITY, False
                    sel *= s
                if b.get("semantic") or not conditions:
                    sel, grounded = sel * DEFAULT_SELECTIVITY, False
                out = sel * n1
            elif op in GROUPING:
                keys = list(b.get("group_by" if op == "AGGREGATE" else "columns") or [])
                if op == "AGGREGATE" and not keys:
                    out = min(1.0, n1)
                else:
                    d_g = 1.0 if keys else n1
                    for key in keys:
                        d = self._distinct(key, sources, n1)
                        if d is None:
                            d, grounded = n1, False
                        d_g *= d
                    out = min(d_g, n1)
            else:
                out = n1                    # record-preserving
        node.estimated_cardinality = out
        node.input_cardinality = n_in
        node.estimate_grounded = grounded
        return out, grounded, sources

    def _match_keys(self, op: str, b: Dict[str, Any], left: Tuple[float, List[str]],
                    right: Tuple[float, List[str]]) -> Tuple[Optional[float], Optional[float]]:
        """d_1, d_2 of a binary matching operator, capped by its inputs."""
        (n1, s1), (n2, s2) = left, right
        if op in CROSS_MODAL_MATCHING:
            def linked(n: float, sources: List[str]) -> Optional[float]:
                total = sum(self._entities.get(s, 0) for s in sources)
                return max(1.0, min(float(total), n)) if total > 0 else None
            return linked(n1, s1), linked(n2, s2)
        on = b.get("on")
        if isinstance(on, list) and len(on) == 2:
            lk, rk = on
        else:
            cols1 = {c for s in s1 for c in (self._columns.get(s) or {})}
            cols2 = {c for s in s2 for c in (self._columns.get(s) or {})}
            shared = sorted(cols1 & cols2)
            if len(shared) != 1:
                return None, None
            lk = rk = shared[0]
        return self._distinct(lk, s1, n1), self._distinct(rk, s2, n2)


def estimate_cardinalities(root: Any, bindings: Optional[Dict[str, Dict]] = None,
                           statistics: Optional[Dict[str, Any]] = None) -> float:
    """Annotate every node of the plan with its estimated output cardinality
    (``estimated_cardinality``), its estimated number of input records
    (``input_cardinality``; n_o for a SCAN), and whether the estimate is
    grounded in statistics (``estimate_grounded``).  Returns n^(root)."""
    return _Estimator(bindings or {}, statistics or {}).estimate(root)[0]

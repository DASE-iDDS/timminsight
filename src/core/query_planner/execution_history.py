"""Execution history store for warm-start calibration (paper Eq. 4 / Eq. 10-11).

This is the concrete, standard-library-only history manager that activates the
history-based calibration already wired into the planner:

  * OCG edge-weight calibration (Eq. 4) reads ``get_edge_records``.
  * Operator-confidence calibration (Eq. 10-11) reads ``get_records_by_operator``.

At cold start no manager is attached (``history_manager=None``), so the planner
falls back to the priors w0 / c0. Attaching a populated manager turns
on warm start: recent per-operator and per-edge outcomes shrink the priors
toward realized behaviour through the Bayesian blend inside the calibrators
(``lambda`` prior pseudo-count, ``kappa`` history cap).

Every record is tagged with the query that produced it, so a leak-free
leave-one-out evaluation can build a warm history that excludes the query
currently under test (see ``snapshot_excluding``).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .cardinality import is_deterministic
from .uncertainty_propagation import HistoricalRecord, record_fidelity


class ExecutionHistoryManager:
    """Accumulates per-operator and per-edge execution outcomes and serves them
    to the planner's calibrators.

    Read interface (consumed by the planner):
      * ``get_edge_records(parent, child, limit)`` -> ``List[float]`` (Eq. 4)
      * ``get_records_by_operator(op, limit)`` -> ``List[HistoricalRecord]`` (Eq. 10-11)

    Write interface (called by the experiment harness or a closed loop):
      * ``record_operator(record, query_id=)``
      * ``record_edge_outcome(parent, child, score, query_id=)``
      * ``record_from_execution(plan_root, operator_results, query_id=, failed=, bindings=)``
    """

    def __init__(self) -> None:
        # newest-last, grouped by operator type
        self._op_records: Dict[str, List[Tuple[HistoricalRecord, Optional[str]]]] = defaultdict(list)
        # newest-last, grouped by (parent_type, child_type)
        self._edge_records: Dict[Tuple[str, str], List[Tuple[float, Optional[str]]]] = defaultdict(list)

    # -- write side ---------------------------------------------------------

    def record_operator(self, record: HistoricalRecord, query_id: Optional[str] = None) -> None:
        self._op_records[record.operator_type].append((record, query_id))

    def record_edge_outcome(self, parent_type: str, child_type: str,
                            score: float, query_id: Optional[str] = None) -> None:
        score = max(0.0, min(1.0, float(score)))
        self._edge_records[(parent_type, child_type)].append((score, query_id))

    def record_from_execution(self, plan_root, operator_results: Dict[str, Any],
                              query_id: Optional[str] = None,
                              failed: Iterable[str] = (),
                              bindings: Optional[Dict[str, Dict]] = None) -> int:
        """Harvest per-operator and per-edge outcomes from one executed plan.

        Each plan node is joined to its ``ExecutionResult`` by ``node_id``.
        n_act is the observed output count and n_est the estimate n^ of Eq. 7
        attached to the node by the planner.  Nodes listed in ``failed`` raised
        during execution and are recorded as failures (a_r = 0).  Two kinds of
        successful records carry no calibration signal and are not stored
        (Eq. 11): records of deterministic implementations, which are exact by
        construction, so that a deviation reflects the error of n^; and
        records whose n^ uses the default selectivity instead of TiMetaGraph
        statistics.  ``bindings`` is the binding environment B (a FILTER is
        deterministic unless B binds a semantic condition).  Each edge
        e = (u, v) is scored by the fidelity of the parent u on the executions
        in which its child had type v (Eq. 4).  Returns the number of operator
        records added.
        """
        nodes = plan_root.get_all_operators()
        failed = set(failed)
        n_added = 0
        fidelity: Dict[str, Optional[float]] = {}
        for node in nodes:
            er = operator_results.get(node.node_id)
            if node.node_id in failed:
                rec = HistoricalRecord(node.operator_type, None, None, success=False)
            elif er is not None:
                signal = (getattr(node, "estimate_grounded", False)
                          and not is_deterministic(node.operator_type, bindings))
                rec = HistoricalRecord(
                    operator_type=node.operator_type,
                    estimated_cardinality=(getattr(node, "estimated_cardinality", None)
                                           if signal else None),
                    actual_cardinality=int(er.row_count),
                    success=True,
                )
            else:
                continue
            fidelity[node.node_id] = record_fidelity(rec)
            if fidelity[node.node_id] is None:
                continue
            self.record_operator(rec, query_id)
            n_added += 1
        for node in nodes:
            score = fidelity.get(node.node_id)
            if score is None:
                continue
            for child in node.children:
                self.record_edge_outcome(node.operator_type, child.operator_type, score, query_id)
        return n_added

    # -- read side (planner calibrators) ------------------------------------

    def get_edge_records(self, parent_type: str, child_type: str, limit: int = 20) -> List[float]:
        recs = self._edge_records.get((parent_type, child_type), [])
        return [s for (s, _qid) in recs[-limit:]]

    def get_records_by_operator(self, operator_type: str, limit: int = 20) -> List[HistoricalRecord]:
        recs = self._op_records.get(operator_type, [])
        return [r for (r, _qid) in recs[-limit:]]

    # -- leave-one-out support ---------------------------------------------

    def snapshot_excluding(self, query_id: str) -> "ExecutionHistoryManager":
        """Return a new manager with every record tagged ``query_id`` removed,
        so that query can be evaluated warm without its own outcome leaking in.
        """
        clone = ExecutionHistoryManager()
        for op, recs in self._op_records.items():
            kept = [(r, q) for (r, q) in recs if q != query_id]
            if kept:
                clone._op_records[op] = list(kept)
        for edge, recs in self._edge_records.items():
            kept = [(s, q) for (s, q) in recs if q != query_id]
            if kept:
                clone._edge_records[edge] = list(kept)
        return clone

    # -- introspection ------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        return {
            "operator_types": len(self._op_records),
            "operator_records": sum(len(v) for v in self._op_records.values()),
            "edge_types": len(self._edge_records),
            "edge_records": sum(len(v) for v in self._edge_records.values()),
        }

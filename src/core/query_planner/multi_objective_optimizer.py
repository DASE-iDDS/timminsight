"""
Multi-Objective Physical Optimizer with Pareto Selection.

Given a fixed logical plan (from the MCTS logical search), enumerates the
assignments of a physical implementation to every operator instance, scores
them on three objectives (latency, resource cost, confidence), extracts the
Pareto front F_0 (the non-dominated assignments), and selects the final plan
with the weighted Tchebycheff scalarization (paper Sec. V-D).

The confidence objective f_conf(pi) = q^pi(root) is obtained by re-running
the bottom-up uncertainty propagation per assignment, substituting each
assigned implementation's intrinsic confidence for the type-level prior.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .cardinality import estimate_cardinalities
from .mcts_plan_search import (
    LogicalPlanNode,
    PhysicalImplInfo,
    PlanObjectives,
    ResourceConstraints,
    UserPreference,
    per_record_cost,
)
from .uncertainty_propagation import (
    HistoricalCalibrator,
    apply_propagation_rule,
)


# ---------------------------------------------------------------------------
# Physical Optimization Config
# ---------------------------------------------------------------------------

@dataclass
class PhysicalOptConfig:
    """Configuration for physical-level optimization."""
    max_combinations: int = 10000
    random_seed: Optional[int] = 0     # fixed: sampled assignments are reproducible


# ---------------------------------------------------------------------------
# Objective Evaluator
# ---------------------------------------------------------------------------

class ObjectiveEvaluator:
    """Evaluates a physical implementation assignment on 3 objectives.

    Latency and resource cost are additive over the assigned implementations.
    The confidence objective f_conf(pi) = q^pi(root) is computed by re-running
    the bottom-up propagation rules (LEAF/PIPELINE/JOIN/UNION) over the
    logical plan tree, with the type-level prior c0(o) replaced by the
    assigned implementation's intrinsic confidence conf_pi(o); historical
    calibration applies unchanged, keyed by the logical type tau(o) (Eq. 10).
    The logical stage scores by the *expected* confidence over I(o); the
    physical stage resolves that expectation with the actual assignment.
    """

    def __init__(
        self,
        impl_registry: Dict[str, List[PhysicalImplInfo]],
        logical_plan: LogicalPlanNode,
        history_manager=None,
        constraints: Optional[ResourceConstraints] = None,
    ):
        self._impl_registry = impl_registry
        self._logical_plan = logical_plan
        self._calibrator = HistoricalCalibrator(history_manager)
        self._constraints = constraints or ResourceConstraints()
        nodes = logical_plan.get_all_operators()
        self._node_types: Dict[str, str] = {n.node_id: n.operator_type for n in nodes}
        # n^_in(o), the estimated input cardinality (Eq. 7), computed once per plan
        if any(n.input_cardinality is None for n in nodes):
            estimate_cardinalities(logical_plan)     # a plan built outside the planner
        self._inputs: Dict[str, float] = {n.node_id: n.input_cardinality for n in nodes}

    def evaluate(self, assignment: Dict[str, str]) -> PlanObjectives:
        """Score an assignment {node_id: impl_type} on the three objectives.

        f_lat  = sum_o cost_pi(o) * n^_in(o)
        f_res  = sum_o cost_pi(o) * n^_in(o) * (gpu_pi(o)/gpu_max + cpu_pi(o)/cpu_max)
        f_conf = q^pi(root(P)), re-propagated with the assigned implementations
        """
        total_latency = 0.0
        total_resource = 0.0
        for node_id, impl_type in assignment.items():
            impl = self._find_impl(self._node_types[node_id], impl_type)
            cost = per_record_cost(impl.base_cost) * self._inputs[node_id]
            total_latency += cost
            total_resource += cost * (
                impl.gpu_memory_mb / max(self._constraints.max_gpu_memory_mb, 1)
                + impl.cpu_cores / max(self._constraints.max_cpu_cores, 1)
            )

        return PlanObjectives(
            latency=total_latency,
            resource_cost=total_resource,
            confidence=self._repropagate(self._logical_plan, assignment),
        )

    def _repropagate(
        self, node: LogicalPlanNode, assignment: Dict[str, str]
    ) -> float:
        """Post-order propagation (Eq. 12) with implementation-specific priors."""
        child_confs = [
            self._repropagate(child, assignment)
            for child in (node.children or [])
        ]
        op_type = node.operator_type
        impl = self._find_impl(op_type, assignment[node.node_id])
        # c_0(tau(o)) is replaced by conf_pi(o); the calibration of Eq. 10
        # applies unchanged, i.e. over the execution history of tau(o).
        calibrated, _ = self._calibrator.calibrate(op_type, impl.base_confidence)
        return apply_propagation_rule(op_type, calibrated, child_confs)

    def _find_impl(self, op_type: str, impl_type: str) -> PhysicalImplInfo:
        for impl in self._impl_registry.get(op_type, []):
            if impl.impl_type == impl_type:
                return impl
        raise KeyError(f"{impl_type!r} is not a registered implementation of {op_type}")


# ---------------------------------------------------------------------------
# Constraint Checker
# ---------------------------------------------------------------------------

class ConstraintChecker:
    """Feasibility: the peak requirement in every resource dimension must not
    exceed the system capacity (paper Sec. V-D)."""

    def __init__(self, constraints: ResourceConstraints):
        self._constraints = constraints

    def is_feasible(
        self,
        assignment: Dict[str, str],
        impl_registry: Dict[str, List[PhysicalImplInfo]],
        node_types: Dict[str, str],
    ) -> bool:
        impls = [
            impl
            for node_id, impl_type in assignment.items()
            for impl in impl_registry[node_types[node_id]]
            if impl.impl_type == impl_type
        ]
        c = self._constraints
        return (
            max(i.gpu_memory_mb for i in impls) <= c.max_gpu_memory_mb
            and max(i.cpu_cores for i in impls) <= c.max_cpu_cores
            and max(i.memory_mb for i in impls) <= c.max_memory_mb
        )


# ---------------------------------------------------------------------------
# Physical Plan Enumerator
# ---------------------------------------------------------------------------

def instance_implementations(op: LogicalPlanNode,
                             impl_registry: Dict[str, List[PhysicalImplInfo]]) -> List[str]:
    """The implementations of I(tau(o)) the physical stage may assign to the
    instance ``op``: all of them, or those the planner found applicable."""
    impls = [i.impl_type for i in impl_registry.get(op.operator_type) or []]
    if not impls:
        raise KeyError(f"no physical implementation registered for {op.operator_type}")
    if op.applicable_impls is None:
        return impls
    allowed = [i for i in impls if i in op.applicable_impls]
    if not allowed:
        raise ValueError(f"no registered implementation of {op.operator_type} applies to "
                         f"instance {op.node_id} (applicable: {list(op.applicable_impls)})")
    return allowed


class PhysicalPlanEnumerator:
    """Enumerates physical implementation combinations for a logical plan."""

    def __init__(
        self,
        logical_plan: LogicalPlanNode,
        impl_registry: Dict[str, List[PhysicalImplInfo]],
        config: Optional[PhysicalOptConfig] = None,
    ):
        self._plan = logical_plan
        self._impl_registry = impl_registry
        self._config = config or PhysicalOptConfig()

    def enumerate(self) -> List[Dict[str, str]]:
        """Assignments {node_id: impl_type} over the operator instances of P.

        Exhaustive when there are at most ``max_combinations`` (10^4) of them;
        otherwise ``max_combinations`` distinct assignments are drawn uniformly
        at random (paper Sec. V-D).  Every operator type needs I(o) != {}; an
        instance whose ``applicable_impls`` the planner set is offered only
        those (preconditions per instance, see applicability.py).
        """
        node_ids: List[str] = []
        impl_options: List[List[str]] = []
        for op in self._plan.get_all_operators():
            node_ids.append(op.node_id)
            impl_options.append(instance_implementations(op, self._impl_registry))

        total = 1
        for opts in impl_options:
            total *= len(opts)

        if total <= self._config.max_combinations:
            combos = itertools.product(*impl_options)
        else:
            combos = self._sample_distinct(impl_options, self._config.max_combinations)
        return [dict(zip(node_ids, combo)) for combo in combos]

    def _sample_distinct(self, impl_options: List[List[str]], k: int) -> List[Tuple[str, ...]]:
        rng = random.Random(self._config.random_seed)
        seen: set = set()
        while len(seen) < k:
            seen.add(tuple(rng.choice(opts) for opts in impl_options))
        return list(seen)


# ---------------------------------------------------------------------------
# Plan Candidate for Physical Optimization
# ---------------------------------------------------------------------------

@dataclass
class PlanCandidate:
    """A physical plan candidate with its objectives."""
    assignment: Dict[str, str]
    objectives: PlanObjectives


# ---------------------------------------------------------------------------
# Pareto Candidate
# ---------------------------------------------------------------------------

@dataclass
class ParetoCandidate:
    """An assignment on the Pareto front F_0 with its objectives."""
    assignment: Dict[str, str]
    objectives: PlanObjectives


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------

class ParetoSorter:
    """Extracts the Pareto front F_0 (minimize latency and resource cost,
    maximize confidence)."""

    @staticmethod
    def pareto_front(candidates: List[PlanCandidate]) -> List[int]:
        """Indices of the non-dominated candidates.

        Candidates are swept in lexicographic order (latency, resource cost,
        -confidence); a candidate can only be dominated by one earlier in that
        order, so each is checked against the front found so far, which is
        typically far smaller than the candidate set.
        """
        key = lambda i: (candidates[i].objectives.latency,
                         candidates[i].objectives.resource_cost,
                         -candidates[i].objectives.confidence)
        front: List[int] = []
        for i in sorted(range(len(candidates)), key=key):
            if not any(ParetoSorter._dominates(candidates[j], candidates[i]) for j in front):
                front.append(i)
        return front

    @staticmethod
    def _dominates(a: PlanCandidate, b: PlanCandidate) -> bool:
        a_obj = a.objectives
        b_obj = b.objectives
        at_least_one_better = False

        if a_obj.latency > b_obj.latency:
            return False
        if a_obj.latency < b_obj.latency:
            at_least_one_better = True

        if a_obj.resource_cost > b_obj.resource_cost:
            return False
        if a_obj.resource_cost < b_obj.resource_cost:
            at_least_one_better = True

        if a_obj.confidence < b_obj.confidence:
            return False
        if a_obj.confidence > b_obj.confidence:
            at_least_one_better = True

        return at_least_one_better


# ---------------------------------------------------------------------------
# Plan Selector (Tchebycheff Scalarization)
# ---------------------------------------------------------------------------

class PlanSelector:
    """Selects from the Pareto front using Weighted Tchebycheff scalarization."""

    def __init__(self, preference: UserPreference):
        self._preference = preference

    def select(
        self,
        candidates: List[ParetoCandidate],
    ) -> Tuple[ParetoCandidate, int]:
        if not candidates:
            raise ValueError("Empty candidate list")
        if len(candidates) == 1:
            return candidates[0], 0

        lats = [c.objectives.latency for c in candidates]
        costs = [c.objectives.resource_cost for c in candidates]
        confs = [c.objectives.confidence for c in candidates]

        ideal_lat = min(lats)
        ideal_cost = min(costs)
        ideal_conf = max(confs)

        range_lat = max(max(lats) - ideal_lat, 1e-10)
        range_cost = max(max(costs) - ideal_cost, 1e-10)
        range_conf = max(ideal_conf - min(confs), 1e-10)

        best_idx = 0
        best_score = float("inf")

        for i, c in enumerate(candidates):
            tcheb = max(
                self._preference.latency_weight * abs(c.objectives.latency - ideal_lat) / range_lat,
                self._preference.cost_weight * abs(c.objectives.resource_cost - ideal_cost) / range_cost,
                self._preference.confidence_weight * abs(c.objectives.confidence - ideal_conf) / range_conf,
            )
            if tcheb < best_score:
                best_score = tcheb
                best_idx = i

        return candidates[best_idx], best_idx


# ---------------------------------------------------------------------------
# Multi-Objective Physical Optimizer
# ---------------------------------------------------------------------------

class MultiObjectivePhysicalOptimizer:
    """Physical optimization: enumerate -> evaluate -> Pareto front -> select.

    Usage:
        optimizer = MultiObjectivePhysicalOptimizer(logical_plan, impl_registry)
        selected, pareto_front = optimizer.optimize()
    """

    def __init__(
        self,
        logical_plan: LogicalPlanNode,
        impl_registry: Dict[str, List[PhysicalImplInfo]],
        constraints: Optional[ResourceConstraints] = None,
        preference: Optional[UserPreference] = None,
        config: Optional[PhysicalOptConfig] = None,
        history_manager=None,
    ):
        self._logical_plan = logical_plan
        self._impl_registry = impl_registry
        self._constraints = constraints or ResourceConstraints()
        self._preference = preference or UserPreference()
        self._config = config or PhysicalOptConfig()
        self._history_manager = history_manager

    def optimize(self) -> Tuple[ParetoCandidate, List[ParetoCandidate]]:
        """Run full physical optimization pipeline.

        Returns:
            (selected_plan, pareto_front)

        Raises:
            ValueError: if no feasible assignments exist.
        """
        # Step 1: Enumerate
        enumerator = PhysicalPlanEnumerator(
            self._logical_plan, self._impl_registry, self._config
        )
        assignments = enumerator.enumerate()

        # Step 2: Evaluate + filter (confidence via per-assignment
        # re-propagation over the logical plan tree)
        evaluator = ObjectiveEvaluator(
            self._impl_registry,
            logical_plan=self._logical_plan,
            history_manager=self._history_manager,
            constraints=self._constraints,
        )
        checker = ConstraintChecker(self._constraints)
        node_types = {n.node_id: n.operator_type for n in self._logical_plan.get_all_operators()}

        candidates: List[PlanCandidate] = []
        for assignment in assignments:
            if not checker.is_feasible(assignment, self._impl_registry, node_types):
                continue
            objectives = evaluator.evaluate(assignment)
            candidates.append(PlanCandidate(assignment=assignment, objectives=objectives))

        if not candidates:
            raise ValueError(
                "No feasible physical plans found. Check resource constraints."
            )

        # Step 3: Pareto front F_0 (assignments are distinct by construction)
        pareto_front = [
            ParetoCandidate(candidates[i].assignment, candidates[i].objectives)
            for i in ParetoSorter.pareto_front(candidates)
        ]

        # Step 4: Tchebycheff selection
        selector = PlanSelector(self._preference)
        selected, _ = selector.select(pareto_front)

        return selected, pareto_front

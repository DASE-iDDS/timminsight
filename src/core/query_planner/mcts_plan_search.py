"""
MCTS Logical Plan Search.

Monte Carlo Tree Search over logical plan structures constrained by the
Operator Compatibility Graph (OCG).  Each MCTS iteration builds a logical
plan tree top-down: the root operator is fixed by the query context, and
subsequent operators are selected from OCG-valid children until all branches
terminate at SCAN leaves.

The search discovers diverse valid logical plans; Evaluate(P) admits only
OCG-valid plans that cover Sigma_req within the depth/operator budgets and
scores them with Q(P) = beta/(1+cost/C_Phi) + gamma*conf + delta*ocg (Eq. 8),
where cost(P) charges every operator per estimated input record (Eq. 7).
"""

from __future__ import annotations

import math
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .cardinality import estimate_cardinalities
from .operator_compatibility_graph import (
    OperatorCompatibilityGraph,
    QueryContext,
)

# The profiles give base_cost in seconds per COST_UNIT_RECORDS input records
# (impl_profiles.json); costs are charged per record (paper Sec. V-B).
COST_UNIT_RECORDS = 1000.0


def per_record_cost(base_cost: float) -> float:
    """Latency per input record of a profile cost."""
    return base_cost / COST_UNIT_RECORDS


# ---------------------------------------------------------------------------
# Shared data structures (also used by the physical optimizer)
# ---------------------------------------------------------------------------

@dataclass
class PlanObjectives:
    """Three objectives for plan evaluation."""
    latency: float = 0.0
    resource_cost: float = 0.0
    confidence: float = 0.0


@dataclass
class ResourceConstraints:
    """Hard resource constraints for physical planning."""
    max_gpu_memory_mb: int = 16384
    max_cpu_cores: int = 16
    max_memory_mb: int = 65536


@dataclass
class UserPreference:
    """User preference weights for multi-objective selection."""
    latency_weight: float = 0.4
    cost_weight: float = 0.3
    confidence_weight: float = 0.3

    def __post_init__(self):
        total = self.latency_weight + self.cost_weight + self.confidence_weight
        if total > 0:
            self.latency_weight /= total
            self.cost_weight /= total
            self.confidence_weight /= total


@dataclass
class PhysicalImplInfo:
    """Descriptor for a physical implementation option."""
    impl_type: str
    base_cost: float = 1.0
    base_confidence: float = 0.8
    gpu_memory_mb: int = 0
    cpu_cores: int = 1
    memory_mb: int = 256


# ---------------------------------------------------------------------------
# Logical Plan Node
# ---------------------------------------------------------------------------

@dataclass
class LogicalPlanNode:
    """A node in a logical plan tree built by MCTS."""
    node_id: str
    operator_type: str
    children: List["LogicalPlanNode"] = field(default_factory=list)
    parent: Optional["LogicalPlanNode"] = None
    source_cardinality: Optional[int] = None  # n_o, exact source size (SCAN leaves only)
    source_name: Optional[str] = None         # source bound to a SCAN leaf by B
    estimated_cost: float = 1.0               # per-record cost c(o), Eq. (6)
    # Eq. 7: estimated output cardinality n^, estimated number of input records
    # n^_in (n_o for a SCAN), and whether the estimate rests on statistics only
    estimated_cardinality: Optional[float] = None
    input_cardinality: Optional[float] = None
    estimate_grounded: bool = False
    # implementations whose precondition holds for this instance (set by the
    # planner before the physical stage, see applicability.py); None = all of I(o)
    applicable_impls: Optional[Tuple[str, ...]] = None

    def clone(self) -> "LogicalPlanNode":
        """Deep-copy the subtree rooted at this node."""
        new_node = LogicalPlanNode(
            node_id=self.node_id,
            operator_type=self.operator_type,
            source_cardinality=self.source_cardinality,
            source_name=self.source_name,
            estimated_cost=self.estimated_cost,
            estimated_cardinality=self.estimated_cardinality,
            input_cardinality=self.input_cardinality,
            estimate_grounded=self.estimate_grounded,
            applicable_impls=self.applicable_impls,
        )
        for child in self.children:
            child_clone = child.clone()
            child_clone.parent = new_node
            new_node.children.append(child_clone)
        return new_node

    def get_all_operators(self) -> List["LogicalPlanNode"]:
        """DFS traversal returning all nodes."""
        result = [self]
        for child in self.children:
            result.extend(child.get_all_operators())
        return result

    def get_edges(self) -> List[Tuple[str, str]]:
        """Return all (parent_type, child_type) edges in this subtree."""
        edges: List[Tuple[str, str]] = []
        for child in self.children:
            edges.append((self.operator_type, child.operator_type))
            edges.extend(child.get_edges())
        return edges

    def get_operator_types(self) -> List[str]:
        """Return operator types in DFS order."""
        return [n.operator_type for n in self.get_all_operators()]

    def depth(self) -> int:
        d = 0
        node = self
        while node.parent is not None:
            d += 1
            node = node.parent
        return d

    def max_depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(c.max_depth() for c in self.children)


# ---------------------------------------------------------------------------
# Cardinality / cost annotation
# ---------------------------------------------------------------------------

def annotate_plan_estimates(
    root: LogicalPlanNode,
    query_context: Optional[QueryContext] = None,
    cost_lookup: Optional[Any] = None,
) -> None:
    """Bind SCAN leaves to their sources, estimate the cardinality of every
    operator (Eq. 7), and attach per-record costs.

    Each SCAN leaf receives the exact record count n_o of the source that the
    binding environment B assigns to it: the source bound to its nearest
    ancestor in ``query_context.bindings`` (operator type -> {"source": ...}).
    Leaves without such an ancestor take the sources of
    ``query_context.source_sizes`` in order.  The cardinalities of the inner
    operators are then estimated bottom-up from the TiMetaGraph statistics of
    the context.  ``cost_lookup`` maps an operator type to its per-record cost
    c(o) (Eq. 6).
    """
    sizes: List[Tuple[str, int]] = list(query_context.source_sizes) if query_context else []
    size_of = dict(sizes)
    bindings = query_context.bindings if query_context else {}
    unbound = 0
    for node in root.get_all_operators():
        if cost_lookup is not None:
            node.estimated_cost = cost_lookup(node.operator_type)
        if node.children or not sizes:
            continue
        source = None
        child, ancestor = node, node.parent
        while ancestor is not None and source is None:
            b = bindings.get(ancestor.operator_type) or {}
            if len(ancestor.children) > 1:
                # binary operators bind each input: {"sources": [left, right]}
                srcs = b.get("sources") or []
                idx = next(i for i, c in enumerate(ancestor.children) if c is child)
                source = srcs[idx] if idx < len(srcs) else None
            else:
                source = b.get("source")
            child, ancestor = ancestor, ancestor.parent
        if source is None or source not in size_of:
            source = sizes[unbound % len(sizes)][0]
            unbound += 1
        node.source_name, node.source_cardinality = source, size_of[source]
    estimate_cardinalities(root, bindings, query_context.statistics if query_context else None)


def reads_bound_sources(root: LogicalPlanNode, query_context: QueryContext) -> bool:
    """Whether every operator of an annotated plan reads the source B binds it
    to: the source of a unary operator must be scanned below it, and each input
    of a binary operator must scan the source bound to that input.  A plan that
    places an operator over the wrong source cannot be executed."""
    bindings = query_context.bindings or {}

    def below(node: LogicalPlanNode) -> Set[str]:
        return {n.source_name for n in node.get_all_operators() if not n.children and n.source_name}

    known = set(dict(query_context.source_sizes or []))
    for node in root.get_all_operators():
        b = bindings.get(node.operator_type) or {}
        if len(node.children) > 1:
            for child, source in zip(node.children, b.get("sources") or []):
                if source in known and source not in below(child):
                    return False
        elif node.children and b.get("source") in known and b["source"] not in below(node):
            return False
    return True


# ---------------------------------------------------------------------------
# Partial Plan (MCTS state)
# ---------------------------------------------------------------------------

@dataclass
class OpenPosition:
    """An unfilled child slot in the partial plan."""
    parent_node: LogicalPlanNode
    depth: int


class PartialPlan:
    """State in the MCTS search: a partially-built logical plan tree."""

    def __init__(
        self,
        ocg: OperatorCompatibilityGraph,
        root: Optional[LogicalPlanNode] = None,
    ):
        self.root = root
        self._ocg = ocg
        self.open_positions: List[OpenPosition] = []
        self.node_count = 1 if root else 0

    def clone(self) -> "PartialPlan":
        new_plan = PartialPlan(self._ocg)
        if self.root:
            new_plan.root = self.root.clone()
            new_plan.open_positions = new_plan._rebuild_open_positions(new_plan.root)
            new_plan.node_count = self.node_count
        return new_plan

    def is_terminal(self) -> bool:
        return len(self.open_positions) == 0

    def _rebuild_open_positions(self, root: LogicalPlanNode) -> List[OpenPosition]:
        positions: List[OpenPosition] = []
        self._scan_open_positions(root, positions)
        return positions

    def _scan_open_positions(
        self, node: LogicalPlanNode, positions: List[OpenPosition]
    ):
        spec = self._ocg.get_spec(node.operator_type)
        for _ in range(spec.min_children - len(node.children)):
            positions.append(OpenPosition(node, node.depth() + 1))
        for child in node.children:
            self._scan_open_positions(child, positions)


def feasible_children(ocg: OperatorCompatibilityGraph, context: QueryContext,
                      parent_type: str) -> List[Tuple[str, float]]:
    """A(s) of Eq. 3: the OCG-feasible children of ``parent_type``, sorted by
    compatibility."""
    return list(ocg.get_valid_children(parent_type))


def is_bound(context: QueryContext, op: str) -> bool:
    """Whether B binds the parameters of ``op`` (SCAN leaves are bound to
    sources when the plan is annotated).  A plan with an unbound operator
    cannot be executed, so Evaluate(P) rejects it."""
    from .llm_query_analyzer import binding_problem
    return op == "SCAN" or binding_problem(op, (context.bindings or {}).get(op) or {}) is None


# ---------------------------------------------------------------------------
# MCTS Configuration
# ---------------------------------------------------------------------------

@dataclass
class MCTSLogicalConfig:
    """Configuration for logical-level MCTS.

    Defaults are the paper's values; the environment variables MCTS_ITERS,
    MCTS_UCT and MCTS_W_COST/CONF/OCG override them for parameter-sensitivity
    sweeps.  The search itself is deterministic.
    """
    num_iterations: int = field(default_factory=lambda: int(os.environ.get("MCTS_ITERS", "200")))      # T
    exploration_constant: float = field(default_factory=lambda: float(os.environ.get("MCTS_UCT", "0.50")))  # c_uct
    max_plan_depth: int = 8         # depth budget
    max_plan_operators: int = 12    # operator-count budget
    # Q(P) weights (beta, gamma, delta), Eq. 7; coverage of Sigma_req is a hard
    # requirement checked by Evaluate, not a weighted term.
    cost_weight: float = field(default_factory=lambda: float(os.environ.get("MCTS_W_COST", "0.35")))
    confidence_weight: float = field(default_factory=lambda: float(os.environ.get("MCTS_W_CONF", "0.35")))
    ocg_weight: float = field(default_factory=lambda: float(os.environ.get("MCTS_W_OCG", "0.30")))


# ---------------------------------------------------------------------------
# Logical Plan Candidate (result)
# ---------------------------------------------------------------------------

@dataclass
class LogicalPlanCandidate:
    """A complete logical plan discovered by MCTS with its quality score."""
    plan: LogicalPlanNode
    quality_score: float
    completeness: float
    estimated_cost: float
    estimated_confidence: float
    ocg_score: float
    operator_sequence: List[str]


# ---------------------------------------------------------------------------
# MCTS Node (search tree node)
# ---------------------------------------------------------------------------

class MCTSLogicalNode:
    """Node in the MCTS search tree over logical plan structures."""

    def __init__(
        self,
        partial_plan: PartialPlan,
        action_taken: Optional[str] = None,
        parent: Optional["MCTSLogicalNode"] = None,
    ):
        self.partial_plan = partial_plan
        self.action_taken = action_taken
        self.parent = parent
        self.children: Dict[str, "MCTSLogicalNode"] = {}
        self.visit_count: int = 0
        self.total_reward: float = 0.0
        self._feasible_actions: Optional[List[str]] = None

    @property
    def average_reward(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_reward / self.visit_count

    @property
    def is_terminal(self) -> bool:
        return self.partial_plan.is_terminal()

    def get_feasible_actions(self, ocg: OperatorCompatibilityGraph,
                             context: QueryContext) -> List[str]:
        if self._feasible_actions is not None:
            return self._feasible_actions

        if self.is_terminal:
            self._feasible_actions = []
            return []

        pos = self.partial_plan.open_positions[0]
        parent_type = pos.parent_node.operator_type
        valid = feasible_children(ocg, context, parent_type)
        self._feasible_actions = [op_type for op_type, _ in valid]
        return self._feasible_actions

    @property
    def is_fully_expanded(self) -> bool:
        if self._feasible_actions is None:
            return False
        return len(self.children) >= len(self._feasible_actions)

    def get_unexplored_actions(self, ocg: OperatorCompatibilityGraph,
                               context: QueryContext) -> List[str]:
        feasible = self.get_feasible_actions(ocg, context)
        return [a for a in feasible if a not in self.children]


# ---------------------------------------------------------------------------
# Logical Cost Model
# ---------------------------------------------------------------------------

class LogicalCostModel:
    """Estimates cost of a logical plan.

    c(o) is the mean per-record cost of the operator's registered physical
    implementations (the default registry when none is given), so both
    planning stages share one source of truth, and
    cost(P) = sum_n c(tau(n)) * n^_in(n) charges every operator for its
    estimated input records (Eq. 7).  Every logical type needs I(o) != {}; an
    unknown type raises KeyError.
    """

    def __init__(
        self,
        impl_registry: Optional[Dict[str, List[PhysicalImplInfo]]] = None,
    ):
        if not impl_registry:
            from .default_impl_registry import build_default_impl_registry
            impl_registry = build_default_impl_registry()
        # c(o): mean latency over the registered implementations (Eq. 6)
        self._costs = {
            op: sum(per_record_cost(i.base_cost) for i in impls) / len(impls)
            for op, impls in impl_registry.items()
            if impls
        }

    def operator_cost(self, operator_type: str) -> float:
        """Per-record cost c(o), the registry mean (Eq. 6)."""
        if operator_type not in self._costs:
            raise KeyError(f"no physical implementation registered for {operator_type}")
        return self._costs[operator_type]

    def estimate_plan_cost(self, plan_root: LogicalPlanNode) -> float:
        """cost(P) of a plan annotated by ``annotate_plan_estimates``."""
        total = 0.0
        for node in plan_root.get_all_operators():
            if node.input_cardinality is None:
                raise ValueError("the plan carries no cardinality estimates")
            total += self.operator_cost(node.operator_type) * node.input_cardinality
        return total

    def reference_cost(self, context: QueryContext) -> float:
        """C_Phi = N_Phi * sum_{o in Sigma_req} c(o): the cost of applying
        every required operator to all N_Phi records of the bound sources."""
        size_of = dict(context.source_sizes or [])
        bound = {s for b in (context.bindings or {}).values()
                 for s in [b.get("source")] + list(b.get("sources") or []) if s in size_of}
        n_phi = sum(size_of[s] for s in bound) if bound else sum(size_of.values())
        return n_phi * sum(self.operator_cost(o) for o in context.required_operators)


class LogicalConfidenceEstimator:
    """Estimates confidence at the logical level.

    c_hat(o) is the mean base_confidence of the operator's registered physical
    implementations (Eq. 6), over the default registry when none is given;
    an unknown type raises KeyError.
    """

    def __init__(
        self,
        impl_registry: Optional[Dict[str, List[PhysicalImplInfo]]] = None,
    ):
        if not impl_registry:
            from .default_impl_registry import build_default_impl_registry
            impl_registry = build_default_impl_registry()
        self._confs = {
            op: sum(i.base_confidence for i in impls) / len(impls)
            for op, impls in impl_registry.items()
            if impls
        }

    def estimate_plan_confidence(self, plan_root: LogicalPlanNode) -> float:
        """Structural confidence conf(P) = prod_n c_hat(tau(n)) (paper Sec. V-B)."""
        confidence = 1.0
        for node in plan_root.get_all_operators():
            op = node.operator_type
            if op not in self._confs:
                raise KeyError(f"no physical implementation registered for {op}")
            confidence *= self._confs[op]
        return confidence


# ---------------------------------------------------------------------------
# Rollout Policy
# ---------------------------------------------------------------------------

class LogicalRolloutPolicy:
    """Simulation: greedily fills the open slots along OCG edges (paper Sec. V-B)."""

    def __init__(
        self,
        ocg: OperatorCompatibilityGraph,
        query_context: QueryContext,
        max_depth: int = 10,
        max_operators: int = 12,
    ):
        self._ocg = ocg
        self._context = query_context
        self._max_depth = max_depth
        self._max_operators = max_operators

    def complete_plan(self, partial: PartialPlan) -> Optional[LogicalPlanNode]:
        """Complete the plan along OCG edges; None when no valid completion
        within the depth/operator budgets exists (the plan earns reward 0)."""
        plan = partial.clone()
        present = set(plan.root.get_operator_types())
        remaining_required = self._context.required_operators - present

        while not plan.is_terminal():
            if plan.node_count >= self._max_operators:
                return None
            pos = plan.open_positions[0]
            parent_type = pos.parent_node.operator_type
            valid = [(t, w) for t, w in feasible_children(self._ocg, self._context, parent_type)
                     if is_bound(self._context, t)]

            if not valid:
                return None
            if pos.depth >= self._max_depth:
                if "SCAN" not in {t for t, _ in valid}:
                    return None
                chosen_type = "SCAN"
            else:
                chosen_type = self._select_operator(valid, remaining_required)

            new_node = LogicalPlanNode(
                node_id=str(uuid.uuid4())[:8],
                operator_type=chosen_type,
                parent=pos.parent_node,
            )
            pos.parent_node.children.append(new_node)
            plan.open_positions.pop(0)
            plan.node_count += 1
            remaining_required.discard(chosen_type)

            for _ in range(self._ocg.get_spec(chosen_type).min_children):
                plan.open_positions.append(OpenPosition(new_node, pos.depth + 1))

        return plan.root

    @staticmethod
    def _select_operator(valid: List[Tuple[str, float]], remaining_required: Set[str]) -> str:
        """The highest-compatibility required-or-valid operator (Sec. V-B):
        a missing required operator first; while required operators are still
        missing, the most compatible other operator that leads towards them;
        otherwise SCAN to terminate the branch.  ``valid`` holds the feasible
        operators that B binds, sorted by compatibility."""
        types = [t for t, _ in valid]
        required = [t for t in types if t in remaining_required and t != "SCAN"]
        if required:
            return required[0]
        if remaining_required - {"SCAN"}:
            inner = [t for t in types if t != "SCAN"]
            if inner:
                return inner[0]
        return "SCAN" if "SCAN" in types else types[0]


# ---------------------------------------------------------------------------
# MCTS Logical Plan Search
# ---------------------------------------------------------------------------

class MCTSLogicalPlanSearch:
    """MCTS over logical plan structures using OCG constraints.

    Each MCTS iteration builds a logical plan top-down. The tree is
    explored via UCT, expanded by placing operators at open positions,
    and evaluated with a composite quality function.
    """

    def __init__(
        self,
        ocg: OperatorCompatibilityGraph,
        query_context: QueryContext,
        config: Optional[MCTSLogicalConfig] = None,
        impl_registry: Optional[Dict[str, List[PhysicalImplInfo]]] = None,
    ):
        self._ocg = ocg
        self._context = query_context
        self._config = config or MCTSLogicalConfig()
        self._cost_model = LogicalCostModel(impl_registry)
        self._reference_cost = self._cost_model.reference_cost(query_context)   # C_Phi
        self._confidence_estimator = LogicalConfidenceEstimator(impl_registry)
        self._rollout_policy = LogicalRolloutPolicy(
            ocg, query_context, self._config.max_plan_depth,
            self._config.max_plan_operators,
        )

    def greedy(self) -> List[LogicalPlanCandidate]:
        """Ablation (w/o MCTS): a single OCG-guided greedy completion from the
        root rho, scored with the same Evaluate(P) as the search."""
        root = self._rollout_policy.complete_plan(self._create_initial_plan())
        if root is None:
            return []
        _, candidate = self._evaluate(root)
        return [candidate] if candidate is not None else []

    def search(self) -> List[LogicalPlanCandidate]:
        """Run MCTS and return discovered logical plan candidates."""
        initial_plan = self._create_initial_plan()
        root_node = MCTSLogicalNode(partial_plan=initial_plan)

        discovered: Dict[tuple, LogicalPlanCandidate] = {}

        for _ in range(self._config.num_iterations):
            # Selection
            node = self._select(root_node)

            # Expansion
            if not node.is_terminal:
                node = self._expand(node)

            # Simulation
            if node.partial_plan.root is None:
                continue
            completed_root = self._rollout_policy.complete_plan(node.partial_plan)

            # Evaluation
            if completed_root is None:
                reward, candidate = 0.0, None
            else:
                reward, candidate = self._evaluate(completed_root)

            # Store unique plans
            if candidate is not None:
                key = tuple(candidate.operator_sequence)
                if key not in discovered or candidate.quality_score > discovered[key].quality_score:
                    discovered[key] = candidate

            # Backpropagation
            self._backpropagate(node, reward)

        # Algorithm 2: the distinct cached plans, ranked by Q.
        return sorted(discovered.values(), key=lambda c: -c.quality_score)

    def _create_initial_plan(self) -> PartialPlan:
        root_type = self._context.root_operator_type
        root_node = LogicalPlanNode(
            node_id=str(uuid.uuid4())[:8],
            operator_type=root_type,
        )
        plan = PartialPlan(self._ocg, root=root_node)
        for _ in range(self._ocg.get_spec(root_type).min_children):
            plan.open_positions.append(OpenPosition(root_node, 1))

        return plan

    def _select(self, root: MCTSLogicalNode) -> MCTSLogicalNode:
        node = root
        while not node.is_terminal and node.is_fully_expanded and node.children:
            node = self._uct_select(node)
        return node

    def _uct_select(self, node: MCTSLogicalNode) -> MCTSLogicalNode:
        c = self._config.exploration_constant
        log_parent = math.log(max(node.visit_count, 1))

        best_child = None
        best_ucb = -float("inf")

        for child in node.children.values():
            if child.visit_count == 0:
                return child
            exploit = child.average_reward
            explore = c * math.sqrt(log_parent / child.visit_count)
            ucb = exploit + explore
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child

        return best_child or node

    def _expand(self, node: MCTSLogicalNode) -> MCTSLogicalNode:
        unexplored = node.get_unexplored_actions(self._ocg, self._context)
        if not unexplored:
            return node

        present_types = set(
            node.partial_plan.root.get_operator_types()
            if node.partial_plan.root else []
        )
        remaining_req = self._context.required_operators - present_types

        # Prefer not-yet-present required non-leaf operators so that Sigma_req
        # is covered early; SCAN is placed last to terminate a branch.
        non_leaf_req = [op for op in unexplored if op in remaining_req and op != "SCAN"]
        non_leaf = [op for op in unexplored if op != "SCAN"]
        chosen = (non_leaf_req or non_leaf or unexplored)[0]

        new_plan = node.partial_plan.clone()
        pos = new_plan.open_positions[0]

        new_node = LogicalPlanNode(
            node_id=str(uuid.uuid4())[:8],
            operator_type=chosen,
            parent=pos.parent_node,
        )
        pos.parent_node.children.append(new_node)
        new_plan.open_positions.pop(0)
        new_plan.node_count += 1

        for _ in range(self._ocg.get_spec(chosen).min_children):
            new_plan.open_positions.append(OpenPosition(new_node, pos.depth + 1))

        child_node = MCTSLogicalNode(
            partial_plan=new_plan,
            action_taken=chosen,
            parent=node,
        )
        node.children[chosen] = child_node
        return child_node

    def _evaluate(
        self, plan_root: LogicalPlanNode
    ) -> Tuple[float, Optional[LogicalPlanCandidate]]:
        ops = plan_root.get_operator_types()
        edges = plan_root.get_edges()

        # Evaluate(P): (false, 0) for plans outside the depth/operator budgets
        # or not OCG-valid (Sec. V-A).
        if plan_root.max_depth() > self._config.max_plan_depth:
            return 0.0, None
        if len(ops) > self._config.max_plan_operators:
            return 0.0, None
        if self._ocg.plan_violations(plan_root):
            return 0.0, None
        # every operator must be bound by B to be executable
        if not all(is_bound(self._context, op) for op in set(ops)):
            return 0.0, None

        present = set(ops)
        required = self._context.required_operators
        completeness = len(present & required) / max(len(required), 1)
        # Evaluate(P): only complete plans (covering every type in Sigma_req)
        # are rewarded and cached; the others return (false, 0).
        if not required <= present:
            return 0.0, None

        # bind the SCAN leaves and estimate cardinalities (Eq. 7), then charge
        # every operator per estimated input record
        annotate_plan_estimates(plan_root, query_context=self._context,
                                cost_lookup=self._cost_model.operator_cost)
        # ... and every operator must read the source B binds it to
        if not reads_bound_sources(plan_root, self._context):
            return 0.0, None
        cost = self._cost_model.estimate_plan_cost(plan_root)
        confidence = self._confidence_estimator.estimate_plan_confidence(plan_root)
        ocg_score = self._ocg.compute_plan_ocg_score(edges) if edges else 1.0

        cfg = self._config
        # the cost term is unit-free: cost(P) relative to the reference C_Phi
        relative_cost = cost / self._reference_cost if self._reference_cost > 0 else cost
        quality = (
            cfg.cost_weight * (1.0 / (1.0 + relative_cost))
            + cfg.confidence_weight * confidence
            + cfg.ocg_weight * ocg_score
        )

        candidate = LogicalPlanCandidate(
            plan=plan_root.clone(),
            quality_score=quality,
            completeness=completeness,
            estimated_cost=cost,
            estimated_confidence=confidence,
            ocg_score=ocg_score,
            operator_sequence=ops,
        )

        return quality, candidate

    def _backpropagate(self, node: MCTSLogicalNode, reward: float):
        current: Optional[MCTSLogicalNode] = node
        while current is not None:
            current.visit_count += 1
            current.total_reward += reward
            current = current.parent

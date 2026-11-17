"""
Operator Compatibility Graph (OCG) for Multimodal Query Planning.

Defines a directed graph G = (V, E, w, sigma) (paper Sec. V-A): V = the 16
logical operator types, E = valid parent->child transitions, w: E -> [0,1]
compatibility scores, and sigma: V -> structural spec (arity, input/output
modalities, leaf/terminal placement).  A plan is OCG-valid iff (i) every
parent-child pair is in E and (ii) every operator satisfies its spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Modality Types
# ---------------------------------------------------------------------------

class ModalityType(Enum):
    TABULAR = "tabular"
    IMAGE = "image"
    TEXT = "text"
    EMBEDDING = "embedding"
    MIXED = "mixed"
    ANY = "any"


# ---------------------------------------------------------------------------
# Operator Specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OperatorSpec:
    """Structural specification for a logical operator type."""
    operator_type: str
    min_children: int
    max_children: int
    input_modalities: FrozenSet[ModalityType]
    output_modality: ModalityType
    is_leaf: bool
    is_terminal: bool
    category: str  # "access", "transform", "cross_modal", "output"


# ---------------------------------------------------------------------------
# OCG Edge
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OCGEdge:
    """A directed edge in the OCG: parent_type → child_type."""
    parent_type: str
    child_type: str
    compatibility: float


# ---------------------------------------------------------------------------
# Query Context
# ---------------------------------------------------------------------------

@dataclass
class QueryContext:
    """Query context Phi = (Sigma_req, Sigma_opt, rho, M, B) (paper Sec. V).

    ``required_operators``/``optional_operators`` are Sigma_req/Sigma_opt
    (the search must cover Sigma_req and prefers Sigma_opt over other
    operators when it adds one), ``root_operator_type`` is rho,
    ``target_modalities`` is M (the modalities holding the referenced data;
    the analyzer grounds them in B), and
    ``bindings`` is the binding environment B (operator -> source, attributes,
    predicates).  ``source_sizes`` lists the bound sources with their exact
    record counts n_o, attached to SCAN leaves, and ``statistics`` holds the
    TiMetaGraph statistics (column summaries, entity links) from which the
    cardinality of every operator is estimated (Eq. 7).
    """
    query_text: str
    required_operators: Set[str]
    optional_operators: Set[str]
    root_operator_type: str
    target_modalities: Set[ModalityType]
    bindings: Dict[str, Dict] = field(default_factory=dict)
    source_sizes: List[Tuple[str, int]] = field(default_factory=list)
    statistics: Dict[str, Dict] = field(default_factory=dict)

    @classmethod
    def from_query_llm(
        cls, query_text: str, provider, metadata_context: Optional[str] = None,
        sources: Optional[List[str]] = None,
        source_columns: Optional[Dict[str, Set[str]]] = None,
        source_attributes: Optional[Dict[str, Set[str]]] = None,
        source_values: Optional[Dict[str, Dict[str, Set[str]]]] = None,
        source_tables: Optional[Dict[str, str]] = None,
        source_modalities: Optional[Dict[str, str]] = None,
    ) -> "QueryContext":
        """Derive query context using an LLM provider for semantic understanding.

        Args:
            query_text: Natural language query.
            provider: An LLMProvider instance (from llm_query_analyzer).
            metadata_context: Optional metadata-graph summary (modality
                bindings, cross-modal entities) used to ground operator
                extraction.

        """
        from .llm_query_analyzer import LLMQueryAnalyzer

        analyzer = LLMQueryAnalyzer(provider, metadata_context=metadata_context,
                                    sources=sources, source_columns=source_columns,
                                    source_attributes=source_attributes,
                                    source_values=source_values,
                                    source_tables=source_tables,
                                    source_modalities=source_modalities)
        return analyzer.to_query_context(query_text)


# ---------------------------------------------------------------------------
# Operator Compatibility Graph
# ---------------------------------------------------------------------------

class OperatorCompatibilityGraph:
    """Directed graph of valid logical operator transitions.

    G = (V, E, w, sigma) where:
      V = set of 16 logical operator type nodes
      E = set of directed edges (parent -> child)
      w: E -> [0, 1] compatibility scoring function
      sigma: V -> OperatorSpec structural constraints
    """

    def __init__(self, history_manager=None, unconstrained: bool = False):
        self._specs: Dict[str, OperatorSpec] = {}
        self._edges: Dict[Tuple[str, str], OCGEdge] = {}
        self._adjacency: Dict[str, List[Tuple[str, float]]] = {}
        self._prior_weights: Dict[Tuple[str, str], float] = {}
        # Ablation: when unconstrained, the OCG imposes no structural constraint —
        # every operator is a valid child of every operator (titsp-no-ocg).
        self._unconstrained = unconstrained
        self._build_default_graph()
        if history_manager is not None:
            self.calibrate_edge_weights(history_manager)

    # ---- Public API ----

    def get_spec(self, operator_type: str) -> OperatorSpec:
        """sigma(o); raises KeyError for an operator type outside V."""
        if operator_type not in self._specs:
            raise KeyError(f"unknown logical operator type: {operator_type}")
        return self._specs[operator_type]

    @property
    def all_operator_types(self) -> List[str]:
        return list(self._specs.keys())

    def get_valid_children(
        self, parent_type: str
    ) -> List[Tuple[str, float]]:
        """Return [(child_type, compatibility)] sorted by compatibility desc."""
        if self._unconstrained:
            return [(op, 1.0) for op in self._specs if op != parent_type]
        children = self._adjacency.get(parent_type, [])
        return sorted(children, key=lambda x: -x[1])

    def is_valid_edge(self, parent_type: str, child_type: str) -> bool:
        return (parent_type, child_type) in self._edges

    def get_compatibility(self, parent_type: str, child_type: str) -> float:
        edge = self._edges.get((parent_type, child_type))
        return edge.compatibility if edge else 0.0

    @staticmethod
    def _modality_ok(child_out: ModalityType, parent_in: FrozenSet[ModalityType]) -> bool:
        return (child_out == ModalityType.ANY or ModalityType.ANY in parent_in
                or child_out in parent_in)

    def plan_violations(self, root) -> List[str]:
        """OCG-validity of a plan tree (paper Sec. V-A): (i) every parent-child
        pair is in E (waived for the w/o-OCG ablation) and (ii) every operator
        satisfies sigma: child count in [min_ch, max_ch], child output modality
        accepted by the parent, leaves only at leaf-typed operators, and
        terminal operators only at the root.  Returns the violations."""
        errors: List[str] = []
        stack = [(root, True)]
        while stack:
            node, is_root = stack.pop()
            spec = self.get_spec(node.operator_type)
            n = len(node.children)
            if not spec.min_children <= n <= spec.max_children:
                errors.append(f"{node.operator_type}: {n} children")
            if spec.is_terminal and not is_root:
                errors.append(f"{node.operator_type}: terminal below the root")
            for child in node.children:
                cspec = self.get_spec(child.operator_type)
                if not self._unconstrained and not self.is_valid_edge(node.operator_type, child.operator_type):
                    errors.append(f"invalid edge {node.operator_type} -> {child.operator_type}")
                if not self._modality_ok(cspec.output_modality, spec.input_modalities):
                    errors.append(f"modality {cspec.output_modality.value} into {node.operator_type}")
                stack.append((child, False))
        return errors

    def compute_plan_ocg_score(
        self, plan_edges: List[Tuple[str, str]]
    ) -> float:
        """Average compatibility score across all edges in a plan."""
        if not plan_edges:
            return 1.0
        total = sum(self.get_compatibility(p, c) for p, c in plan_edges)
        return total / len(plan_edges)

    # ---- Edge Weight Calibration ----

    def calibrate_edge_weights(
        self, history_manager, prior_weight: float = 2.0, kappa: int = 10
    ):
        """Edge calibration (Eq. 4) from historical execution records.

        w(e) = (w0(e)*lambda + s_bar(e)*min(k,kappa)) / (lambda + min(k,kappa)),
        where s_bar(e), e = (u, v), is the mean fidelity a_r of the parent u
        over the executions in which its child had type v.
        """
        for (parent, child), prior_w in self._prior_weights.items():
            records = history_manager.get_edge_records(parent, child, limit=20)
            if not records:
                continue
            hist_avg = sum(records) / len(records)
            hist_weight = min(len(records), kappa)
            calibrated = (
                prior_w * prior_weight + hist_avg * hist_weight
            ) / (prior_weight + hist_weight)
            calibrated = max(0.0, min(1.0, calibrated))
            self._edges[(parent, child)] = OCGEdge(parent, child, calibrated)
        self._adjacency.clear()
        for (parent, child), edge in self._edges.items():
            self._adjacency.setdefault(parent, []).append(
                (child, edge.compatibility)
            )

    # ---- Graph Construction ----

    def _build_default_graph(self):
        """Register the 16 operator specs and their valid edges."""
        self._register_operators()
        self._register_edges()

    def _register_operators(self):
        ALL = frozenset(ModalityType)
        specs = [
            OperatorSpec("SCAN", 0, 0, frozenset(), ModalityType.ANY,
                         is_leaf=True, is_terminal=False, category="access"),
            OperatorSpec("FILTER", 1, 1, ALL, ModalityType.ANY,
                         is_leaf=False, is_terminal=False, category="transform"),
            OperatorSpec("PROJECT", 1, 1, ALL, ModalityType.ANY,
                         is_leaf=False, is_terminal=False, category="transform"),
            OperatorSpec("AGGREGATE", 1, 1, ALL, ModalityType.TABULAR,
                         is_leaf=False, is_terminal=False, category="transform"),
            OperatorSpec("SORT", 1, 1,
                         frozenset({ModalityType.TABULAR, ModalityType.MIXED, ModalityType.ANY}),
                         ModalityType.ANY,
                         is_leaf=False, is_terminal=False, category="transform"),
            OperatorSpec("LIMIT", 1, 1, ALL, ModalityType.ANY,
                         is_leaf=False, is_terminal=False, category="transform"),
            OperatorSpec("DISTINCT", 1, 1, ALL, ModalityType.ANY,
                         is_leaf=False, is_terminal=False, category="transform"),
            OperatorSpec("JOIN", 2, 2, ALL, ModalityType.MIXED,
                         is_leaf=False, is_terminal=False, category="transform"),
            OperatorSpec("UNION", 2, 4, ALL, ModalityType.ANY,
                         is_leaf=False, is_terminal=False, category="transform"),
            OperatorSpec("SEMANTIC_SEARCH", 1, 1,
                         frozenset({ModalityType.TEXT, ModalityType.IMAGE, ModalityType.ANY}),
                         ModalityType.EMBEDDING,
                         is_leaf=False, is_terminal=False, category="cross_modal"),
            OperatorSpec("CROSS_MODAL_MATCH", 2, 2,
                         frozenset({ModalityType.IMAGE, ModalityType.TEXT, ModalityType.EMBEDDING, ModalityType.ANY}),
                         ModalityType.MIXED,
                         is_leaf=False, is_terminal=False, category="cross_modal"),
            OperatorSpec("CONTENT_EXTRACT", 1, 1,
                         frozenset({ModalityType.IMAGE, ModalityType.TEXT, ModalityType.MIXED, ModalityType.ANY}),
                         ModalityType.TABULAR,
                         is_leaf=False, is_terminal=False, category="cross_modal"),
            OperatorSpec("SIMILARITY_JOIN", 2, 2,
                         frozenset({ModalityType.EMBEDDING, ModalityType.ANY}),
                         ModalityType.MIXED,
                         is_leaf=False, is_terminal=False, category="cross_modal"),
            OperatorSpec("FEATURE_TRANSFORM", 1, 1,
                         frozenset({ModalityType.IMAGE, ModalityType.TEXT, ModalityType.TABULAR, ModalityType.ANY}),
                         ModalityType.EMBEDDING,
                         is_leaf=False, is_terminal=False, category="cross_modal"),
            OperatorSpec("VISUALIZE", 1, 1, ALL, ModalityType.MIXED,
                         is_leaf=False, is_terminal=True, category="output"),
            OperatorSpec("EXPORT", 1, 1, ALL, ModalityType.TABULAR,
                         is_leaf=False, is_terminal=True, category="output"),
        ]
        for spec in specs:
            self._specs[spec.operator_type] = spec

    def _register_edges(self):
        edge_defs: List[Tuple[str, str, float]] = [
            # VISUALIZE children
            ("VISUALIZE", "AGGREGATE", 1.0),
            ("VISUALIZE", "SORT", 0.9),
            ("VISUALIZE", "PROJECT", 0.85),
            ("VISUALIZE", "FILTER", 0.8),
            ("VISUALIZE", "DISTINCT", 0.8),
            ("VISUALIZE", "CONTENT_EXTRACT", 0.75),
            ("VISUALIZE", "FEATURE_TRANSFORM", 0.7),
            ("VISUALIZE", "LIMIT", 0.9),
            ("VISUALIZE", "SIMILARITY_JOIN", 0.7),

            # EXPORT children
            ("EXPORT", "AGGREGATE", 1.0),
            ("EXPORT", "PROJECT", 0.9),
            ("EXPORT", "SORT", 0.9),
            ("EXPORT", "FILTER", 0.85),
            ("EXPORT", "DISTINCT", 0.85),
            ("EXPORT", "LIMIT", 0.85),

            # SORT children
            ("SORT", "FILTER", 1.0),
            ("SORT", "AGGREGATE", 0.9),
            ("SORT", "PROJECT", 0.85),
            ("SORT", "SCAN", 0.85),
            ("SORT", "JOIN", 0.85),
            ("SORT", "SEMANTIC_SEARCH", 0.8),
            ("SORT", "UNION", 0.75),
            ("SORT", "SIMILARITY_JOIN", 0.75),
            ("SORT", "CONTENT_EXTRACT", 0.8),   # rank by a value extracted from text/images

            # LIMIT children
            ("LIMIT", "SORT", 1.0),
            ("LIMIT", "FILTER", 0.8),
            ("LIMIT", "SCAN", 0.7),
            ("LIMIT", "SIMILARITY_JOIN", 0.7),

            # DISTINCT children
            ("DISTINCT", "FILTER", 1.0),
            ("DISTINCT", "SCAN", 0.9),
            ("DISTINCT", "PROJECT", 0.85),
            ("DISTINCT", "JOIN", 0.8),
            ("DISTINCT", "UNION", 0.85),

            # AGGREGATE children
            ("AGGREGATE", "FILTER", 1.0),
            ("AGGREGATE", "SCAN", 0.9),
            ("AGGREGATE", "SEMANTIC_SEARCH", 0.85),
            ("AGGREGATE", "JOIN", 0.8),
            ("AGGREGATE", "CONTENT_EXTRACT", 0.75),
            ("AGGREGATE", "CROSS_MODAL_MATCH", 0.7),
            ("AGGREGATE", "PROJECT", 0.85),
            ("AGGREGATE", "UNION", 0.8),
            ("AGGREGATE", "SIMILARITY_JOIN", 0.75),

            # PROJECT children
            ("PROJECT", "FILTER", 1.0),
            ("PROJECT", "SCAN", 1.0),
            ("PROJECT", "JOIN", 0.9),
            ("PROJECT", "AGGREGATE", 0.85),
            ("PROJECT", "SEMANTIC_SEARCH", 0.8),
            ("PROJECT", "UNION", 0.8),
            ("PROJECT", "SIMILARITY_JOIN", 0.75),
            ("PROJECT", "CONTENT_EXTRACT", 0.8),  # list values extracted from text/images

            # FILTER children
            ("FILTER", "SCAN", 1.0),
            ("FILTER", "SEMANTIC_SEARCH", 0.9),
            ("FILTER", "CONTENT_EXTRACT", 0.85),
            ("FILTER", "JOIN", 0.85),
            ("FILTER", "FEATURE_TRANSFORM", 0.8),
            ("FILTER", "PROJECT", 0.8),
            ("FILTER", "UNION", 0.75),
            ("FILTER", "SIMILARITY_JOIN", 0.75),

            # JOIN children (requires 2)
            ("JOIN", "SCAN", 1.0),
            ("JOIN", "FILTER", 0.9),
            ("JOIN", "SEMANTIC_SEARCH", 0.85),
            ("JOIN", "PROJECT", 0.8),

            # SIMILARITY_JOIN children (requires 2)
            ("SIMILARITY_JOIN", "SEMANTIC_SEARCH", 1.0),
            ("SIMILARITY_JOIN", "SCAN", 0.9),
            ("SIMILARITY_JOIN", "FEATURE_TRANSFORM", 0.9),
            ("SIMILARITY_JOIN", "FILTER", 0.85),

            # UNION children (requires 2+)
            ("UNION", "SCAN", 0.9),
            ("UNION", "FILTER", 0.85),
            ("UNION", "PROJECT", 0.8),

            # CROSS_MODAL_MATCH children (requires 2)
            ("CROSS_MODAL_MATCH", "CONTENT_EXTRACT", 1.0),
            ("CROSS_MODAL_MATCH", "SEMANTIC_SEARCH", 0.9),
            ("CROSS_MODAL_MATCH", "FEATURE_TRANSFORM", 0.9),
            ("CROSS_MODAL_MATCH", "SCAN", 0.85),
            ("CROSS_MODAL_MATCH", "FILTER", 0.85),   # match against a filtered table

            # CONTENT_EXTRACT children
            ("CONTENT_EXTRACT", "SCAN", 1.0),
            ("CONTENT_EXTRACT", "FILTER", 0.8),
            ("CONTENT_EXTRACT", "CROSS_MODAL_MATCH", 0.85),  # extract from matched records

            # SEMANTIC_SEARCH children
            ("SEMANTIC_SEARCH", "SCAN", 1.0),
            ("SEMANTIC_SEARCH", "FILTER", 0.7),

            # FEATURE_TRANSFORM children
            ("FEATURE_TRANSFORM", "SCAN", 1.0),
            ("FEATURE_TRANSFORM", "FILTER", 0.85),
            ("FEATURE_TRANSFORM", "CONTENT_EXTRACT", 0.8),
        ]

        for parent, child, compat in edge_defs:
            edge = OCGEdge(parent, child, compat)
            self._edges[(parent, child)] = edge
            self._adjacency.setdefault(parent, []).append((child, compat))
            self._prior_weights[(parent, child)] = compat

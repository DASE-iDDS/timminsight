"""
Uncertainty Propagation for Multimodal Query Plans (paper Sec. V-C).

Bottom-up confidence propagation through a logical plan tree with historical
calibration.  Every operator o receives

    q_o = c~(o)                        LEAF         (Eq. 12)
    q_o = c~(o) * q_{o'}               PIPELINE     one input
    q_o = c~(o) * min_{o'} q_{o'}      MULTI-INPUT  join, cross-modal match, union

q_o is the probability that a record in the output of o is correct, and
c~(o), the prior c_0(o) = c_hat(o) (Eq. 6) calibrated toward the historical
accuracy (Eq. 10-11), the probability that o emits a correct record given
correct input records.  Source records are taken as given.  A multi-input
output record combines one record of each input and is correct only if all of
them are; child errors are taken as comonotone (Frechet upper bound).  The
root confidence is a predictor of answer correctness, not its probability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class UncertaintyEstimate:
    """Uncertainty annotation q_o of a single operator in the plan tree."""
    confidence: float           # q_o in [0, 1]
    calibration_source: str     # "prior" | "historical" | "uniform"

    def __post_init__(self):
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class OperatorUncertaintyProfile:
    """Prior of an operator type: its base confidence c_0(o)."""
    base_confidence: float


class PropagationRule(Enum):
    LEAF = "leaf"               # no children: c~
    PIPELINE = "pipeline"       # one input: chain rule
    MULTI_INPUT = "multi_input" # several inputs, all of which must be correct


# ---------------------------------------------------------------------------
# Priors c_0(o) = c_hat(o)
# ---------------------------------------------------------------------------

# The prior of a logical type is the mean intrinsic confidence of its
# registered physical implementations, c_0(o) = c_hat(o) (Eq. 6).  The
# registry import is deferred to avoid a circular module dependency.
_DEFAULT_LOGICAL_PROFILES: Optional[Dict[str, OperatorUncertaintyProfile]] = None


def get_logical_operator_profiles(
    impl_registry: Optional[Dict[str, Any]] = None,
) -> Dict[str, OperatorUncertaintyProfile]:
    """Logical priors derived from ``impl_registry`` (default registry if None)."""
    global _DEFAULT_LOGICAL_PROFILES
    if impl_registry is None and _DEFAULT_LOGICAL_PROFILES is not None:
        return _DEFAULT_LOGICAL_PROFILES
    from .default_impl_registry import build_default_impl_registry

    registry = impl_registry if impl_registry is not None else build_default_impl_registry()
    derived = {
        op_type: OperatorUncertaintyProfile(
            sum(i.base_confidence for i in impls) / len(impls))
        for op_type, impls in registry.items() if impls
    }
    if impl_registry is None:
        _DEFAULT_LOGICAL_PROFILES = derived
    return derived


def logical_confidence_prior(operator_type: str) -> float:
    """c_0(o) = c_hat(o) of a logical operator type (default registry)."""
    profiles = get_logical_operator_profiles()
    if operator_type not in profiles:
        raise KeyError(f"no physical implementation registered for {operator_type}")
    return profiles[operator_type].base_confidence


# ---------------------------------------------------------------------------
# Historical Calibrator
# ---------------------------------------------------------------------------

@dataclass
class HistoricalRecord:
    """An execution record r of an operator, as used by Eq. 10-11."""
    operator_type: str
    estimated_cardinality: Optional[float]   # n_est, reported by the implementation
    actual_cardinality: Optional[int]      # n_act, observed during execution
    success: bool                          # the operator executed without error


def cardinality_fidelity(
    estimated: Optional[float], actual: Optional[float], success: bool = True,
) -> Optional[float]:
    """Per-record fidelity a_r (paper Eq. 11).

    0 for a failed record; exp(-|ln(n_act / n_est)|) when both counts are
    known; ``None`` when a successful record does not report both counts, in
    which case it carries no calibration signal and is excluded.

    exp(-|ln(a / e)|) = min(a, e) / max(a, e), which also covers the boundary:
    1 when both counts are zero (they coincide) and 0 when exactly one is.
    """
    if not success:
        return 0.0
    if estimated is None or actual is None or estimated < 0 or actual < 0:
        return None
    hi = max(estimated, actual)
    return 1.0 if hi == 0 else min(estimated, actual) / hi


def record_fidelity(record: HistoricalRecord) -> Optional[float]:
    return cardinality_fidelity(
        record.estimated_cardinality, record.actual_cardinality, record.success)


class HistoricalCalibrator:
    """Calibrates a prior toward the realized accuracy (Eq. 10):

        c~(o) = (lambda*c_0(o) + min(|H_o|,kappa)*a_bar(o)) / (lambda + min(|H_o|,kappa))

    over the (up to 20) most recent records H_o of the logical type.  The
    historical accuracy a_bar(o) depends only on the type, so it is computed
    once per type and cached for the lifetime of the calibrator.
    """

    def __init__(self, history_manager=None, prior_weight: float = 2.0, kappa: int = 10):
        self._history_manager = history_manager
        self._prior_weight = prior_weight      # lambda
        self._kappa = kappa
        self._accuracy: Dict[str, Optional[Tuple[float, int]]] = {}

    def historical_accuracy(self, operator_type: str) -> Optional[Tuple[float, int]]:
        """(a_bar(o), |H_o|) or None without usable history."""
        if self._history_manager is None:
            return None
        if operator_type not in self._accuracy:
            records = self._history_manager.get_records_by_operator(operator_type, limit=20)
            scores = [s for s in (record_fidelity(r) for r in records) if s is not None]
            self._accuracy[operator_type] = (
                (sum(scores) / len(scores), len(scores)) if scores else None)
        return self._accuracy[operator_type]

    def calibrate(self, operator_type: str, prior_confidence: float) -> Tuple[float, str]:
        """Return (c~(o), source); the prior unchanged without history."""
        hist = self.historical_accuracy(operator_type)
        if hist is None:
            return prior_confidence, "prior"
        accuracy, k = hist
        w = min(k, self._kappa)
        calibrated = (prior_confidence * self._prior_weight + accuracy * w) / (self._prior_weight + w)
        return max(0.0, min(1.0, calibrated)), "historical"


# ---------------------------------------------------------------------------
# Uncertainty Prior
# ---------------------------------------------------------------------------

class UncertaintyPrior:
    """Priors c_0(o) of the logical operator types."""

    def __init__(self, custom_profiles: Optional[Dict[str, OperatorUncertaintyProfile]] = None,
                 impl_registry: Optional[Dict[str, Any]] = None):
        # c_0(o) = c_hat(o) over the same registry the search uses (Eq. 6).
        self._profiles = dict(get_logical_operator_profiles(impl_registry))
        if custom_profiles:
            self._profiles.update(custom_profiles)

    def get_base_confidence(self, operator_type: str) -> float:
        if operator_type not in self._profiles:
            raise KeyError(f"no prior for operator type {operator_type}")
        return self._profiles[operator_type].base_confidence


# ---------------------------------------------------------------------------
# Annotated Plan
# ---------------------------------------------------------------------------

@dataclass
class AnnotatedPlan:
    """A logical plan with its uncertainty map U = {o -> q_o}, keyed by node id."""
    logical_plan: Any
    uncertainty_map: Dict[str, UncertaintyEstimate] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Propagation rules (Eq. 12)
# ---------------------------------------------------------------------------

def resolve_propagation_rule(operator_type: str, children_count: int) -> PropagationRule:
    if children_count == 0:
        return PropagationRule.LEAF
    if children_count == 1:
        return PropagationRule.PIPELINE
    return PropagationRule.MULTI_INPUT


def apply_propagation_rule(operator_type: str, calibrated: float,
                           children: List[float]) -> float:
    """q_o from c~(o) and the children's confidences (Eq. 12).  q_o is the
    probability that a record in the output of o is correct; source records
    are taken as given, so a leaf has q_o = c~(o)."""
    rule = resolve_propagation_rule(operator_type, len(children))
    if rule == PropagationRule.LEAF:
        return calibrated
    if rule == PropagationRule.PIPELINE:
        return calibrated * children[0]
    return calibrated * min(children)


# ---------------------------------------------------------------------------
# Main Propagator
# ---------------------------------------------------------------------------

class UncertaintyPropagator:
    """Propagates confidence bottom-up through a logical plan tree.

    For each operator in post-order:
        1. prior c_0(o) of its type (Eq. 6)
        2. historical calibration c~(o) (Eq. 10-11)
        3. combination with the children's confidences by the dataflow rule
           of the operator (Eq. 12)
    """

    def __init__(
        self,
        prior: Optional[UncertaintyPrior] = None,
        history_manager=None,
        prior_weight: float = 2.0,
    ):
        self._prior = prior or UncertaintyPrior()
        self._calibrator = HistoricalCalibrator(history_manager, prior_weight)

    def propagate(self, root) -> AnnotatedPlan:
        """Uncertainty map of the plan rooted at ``root`` (a LogicalPlanNode)."""
        uncertainty_map: Dict[str, UncertaintyEstimate] = {}
        self._propagate_recursive(root, uncertainty_map)
        return AnnotatedPlan(logical_plan=root, uncertainty_map=uncertainty_map)

    def _propagate_recursive(self, node, uncertainty_map: Dict[str, UncertaintyEstimate]) -> float:
        children = [self._propagate_recursive(c, uncertainty_map) for c in node.children]
        calibrated, source = self._calibrator.calibrate(
            node.operator_type, self._prior.get_base_confidence(node.operator_type))
        confidence = apply_propagation_rule(node.operator_type, calibrated, children)
        uncertainty_map[node.node_id] = UncertaintyEstimate(confidence, source)
        return confidence

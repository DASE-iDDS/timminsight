"""
Default Physical Implementation Registry.

One to three physical implementations per logical operator type (paper
Sec. V-D), each a distinct algorithm in plan_executor.py.  Their profiles
(cost, conf, gpu, cpu, mem) are measured by offline micro-benchmarks
(scripts/microbenchmark_impls.py) and read from impl_profiles.json; nothing
is set by hand.  The logical prior of each type is the mean confidence of its
implementations, c_0(o) = c_hat(o) (paper Eq. 6), so this registry is the
single source of truth for both planning stages.  Used as the default when no
custom registry is provided to the QueryPlanner.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .mcts_plan_search import PhysicalImplInfo

PROFILES_PATH = Path(__file__).with_name("impl_profiles.json")
_FIELDS = ("base_cost", "base_confidence", "gpu_memory_mb", "cpu_cores", "memory_mb")


def build_default_impl_registry() -> Dict[str, List[PhysicalImplInfo]]:
    """The registry of all 16 logical operators with their measured profiles."""
    if not PROFILES_PATH.exists():
        raise FileNotFoundError(
            f"{PROFILES_PATH} is missing: measure the implementation profiles with "
            "`python -m scripts.microbenchmark_impls`")
    profiles = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))["profiles"]
    registry: Dict[str, List[PhysicalImplInfo]] = {}
    for op, impls in profiles.items():
        for p in impls:
            missing = [f for f in ("impl_type",) + _FIELDS if f not in p]
            if missing:
                raise ValueError(f"{PROFILES_PATH.name}: {op} profile lacks {missing}")
        registry[op] = [PhysicalImplInfo(p["impl_type"], **{f: p[f] for f in _FIELDS})
                        for p in impls]
    return registry

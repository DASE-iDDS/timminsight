"""Table II: calibration quality macro-averaged over LLMs.

Each input is the ``<out>.calibration.json`` of one ``benchmark.py
--calibration`` run (one LLM, leave-one-family-out per LLM).  For every dataset
and setting (cold start, warm calibrated), ECE10 and Brier are averaged over
the LLMs with equal weight.

    python -m src.experiments.calibration_table runs/kimi.calibration.json ...
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from statistics import mean
from typing import Dict, List

SETTINGS = ("cold_start", "warm_calibrated")
METRICS = ("ece10", "brier")


def macro_average(runs: List[Dict]) -> Dict[str, Dict[str, Dict[str, float]]]:
    models = [r["model"] for r in runs]
    if len(set(models)) != len(models):
        raise ValueError(f"each LLM must appear once, got {models}")
    datasets = set(runs[0]["datasets"])
    for r in runs:
        if set(r["datasets"]) != datasets:
            raise ValueError(f"{r['model']} covers {sorted(r['datasets'])}, expected {sorted(datasets)}")
    table: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for ds in sorted(datasets):
        for setting in SETTINGS:
            table[ds][setting] = {m: mean(r["datasets"][ds][setting][m] for r in runs)
                                  for m in METRICS}
    return dict(table)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="per-LLM <out>.calibration.json files")
    args = ap.parse_args()
    runs = [json.load(open(f)) for f in args.files]
    print(json.dumps({"models": [r["model"] for r in runs], "table": macro_average(runs)}, indent=2))


if __name__ == "__main__":
    main()

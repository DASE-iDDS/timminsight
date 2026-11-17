"""Blind human audit of the LLM judge (paper Sec. VI-A).

1. ``sample``: draw a random sample of judged answers from the results JSONL
   written by benchmark.py and write one blind sheet per annotator
   (question, gold answer, system answer; the judge's verdict is hidden).
2. Each annotator fills the ``consistent`` column with 1 or 0.
3. ``score``: compare the judge's verdicts with the annotators' majority label
   and report the agreement rate and Cohen's kappa.

    python -m src.experiments.judge_audit sample runs/results.jsonl --n 300 --out audit/
    python -m src.experiments.judge_audit score audit/ a1.csv a2.csv a3.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from typing import Dict, List, Sequence

FIELDS = ["item", "question", "gold_answer", "system_answer", "consistent"]


def sample(results_path: str, n: int, out_dir: str, annotators: int = 3, seed: int = 0) -> None:
    with open(results_path, encoding="utf-8") as f:
        rows = [r for r in (json.loads(line) for line in f if line.strip())
                if r.get("correct") is not None]
    picked = random.Random(seed).sample(rows, min(n, len(rows)))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "judge_labels.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "judge"])
        for i, r in enumerate(picked):
            w.writerow([i, int(bool(r["correct"]))])
    for a in range(1, annotators + 1):
        order = list(range(len(picked)))
        random.Random(seed + a).shuffle(order)          # independent order per annotator
        with open(os.path.join(out_dir, f"annotator_{a}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for i in order:
                r = picked[i]
                w.writerow({"item": i, "question": r.get("question", ""),
                            "gold_answer": r.get("gold_answer", ""),
                            "system_answer": r.get("answer", ""), "consistent": ""})


def cohen_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def score(audit_dir: str, annotation_files: List[str]) -> Dict[str, float]:
    with open(os.path.join(audit_dir, "judge_labels.csv"), encoding="utf-8") as f:
        judge = {int(r["item"]): int(r["judge"]) for r in csv.DictReader(f)}
    votes: Dict[int, List[int]] = {i: [] for i in judge}
    for path in annotation_files:
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                item = int(r["item"])
                if item not in votes:
                    raise ValueError(f"{path}: item {item} is not in judge_labels.csv")
                if r["consistent"].strip() not in ("0", "1"):
                    raise ValueError(f"{path}: item {item} needs consistent=0 or 1, got {r['consistent']!r}")
                votes[item].append(int(r["consistent"]))
    items = sorted(i for i, v in votes.items() if len(v) == len(annotation_files))
    if not items:
        raise ValueError("no item is labeled by every annotator")
    majority = [int(sum(votes[i]) * 2 > len(votes[i])) for i in items]
    judged = [judge[i] for i in items]
    agreement = sum(m == j for m, j in zip(majority, judged)) / len(items)
    return {"items": len(items), "agreement": agreement, "cohen_kappa": cohen_kappa(majority, judged)}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("results")
    s.add_argument("--n", type=int, default=300)
    s.add_argument("--out", default="audit")
    s.add_argument("--annotators", type=int, default=3)
    s.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("score")
    c.add_argument("audit_dir")
    c.add_argument("annotations", nargs="+")
    args = ap.parse_args()
    if args.cmd == "sample":
        sample(args.results, args.n, args.out, args.annotators, args.seed)
    else:
        print(json.dumps(score(args.audit_dir, args.annotations), indent=2))


if __name__ == "__main__":
    main()

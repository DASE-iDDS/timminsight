"""Apply the item-by-item review of the ambiguity variants to the benchmarks.

Each reviewed item (benchmarks/review/<ambiguity_type>.jsonl) gives a verdict
(keep / rewrite), the final question, the resolution that states the analyst's
reading, a reason, and a confidence.  The benchmark item takes the final
question; its hidden intent is the exact base question plus the resolution, so
the gold answer is unchanged.  Items whose reviewer confidence is low stay
marked ``needs_review`` for the authors.

Usage: python -m scripts.apply_review
"""
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmarks"
REVIEW = BENCH / "review"
TYPES = ("cross_modal_referential", "intra_modal_attribute", "multimodal_intent",
         "schema_structure", "value_content", "temporal_spatial")
FILES = ("nirvana_bench.jsonl", "sembench.jsonl")


def main() -> None:
    bases = {json.loads(l)["base_id"]: json.loads(l)
             for l in open(BENCH / "base_queries.jsonl", encoding="utf-8")}
    reviews = {}
    for t in TYPES:
        for line in open(REVIEW / f"{t}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r.get("verdict") not in ("keep", "rewrite") or not r.get("question") \
                    or not r.get("resolution"):
                raise ValueError(f"incomplete review of {r.get('id')}")
            reviews[r["id"]] = r
    rows_all = []
    for name in FILES:
        rows = [json.loads(l) for l in open(BENCH / name, encoding="utf-8")]
        for it in rows:
            r = reviews.get(it["id"])
            if r is None:
                raise KeyError(f"{it['id']} was not reviewed")
            base = bases[it["base_id"]]
            it["question"] = r["question"].strip()
            # the gold of the base query (rebuilt deterministically)
            for k in ("dataset", "gold_answer", "gold_kind", "gold_source"):
                if k in base:
                    it[k] = base[k]
            it["hidden_intent"] = f"{base['question']} ({r['resolution'].strip()})"
            it["review"] = {"verdict": r["verdict"], "reason": r.get("reason", ""),
                            "confidence": r.get("confidence", "medium")}
            it["needs_review"] = r.get("confidence") == "low"
        with open(BENCH / name, "w", encoding="utf-8") as f:
            for it in rows:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        rows_all += rows
    if len(reviews) != len(rows_all):
        raise ValueError(f"{len(reviews)} reviews for {len(rows_all)} items")
    with open(BENCH / "review_checklist.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "ambiguity_type", "base_question", "variant_question", "hidden_intent",
                    "gold_answer", "gold_source", "review_verdict", "review_confidence",
                    "review_reason", "note", "ok (y/n)", "comment"])
        for it in rows_all:
            b = bases[it["base_id"]]
            w.writerow([it["id"], it["ambiguity_type"], b["question"], it["question"],
                        it["hidden_intent"], str(it["gold_answer"])[:500], it["gold_source"],
                        it["review"]["verdict"], it["review"]["confidence"],
                        it["review"]["reason"], b.get("note", ""), "", ""])
    low = sum(it["needs_review"] for it in rows_all)
    print(f"applied {len(reviews)} reviews; {low} low-confidence items marked needs_review")


if __name__ == "__main__":
    main()

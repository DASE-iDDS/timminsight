"""Build benchmarks/smoke.jsonl: a few real queries on the first LIMIT rows of
SemBench Movie and Nirvana-Bench Estate, with gold answers computed on exactly
those rows (SemBench's gold SQL; deterministic rules for Estate).
Usage: python -m scripts.make_smoke_benchmark [LIMIT]"""
import json
import sqlite3
import sys
from pathlib import Path

from src.experiments.benchmark import load_dataset

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "benchmarks" / "smoke.jsonl"


def sembench_gold(sql, limit):
    data = load_dataset("sembench/movie", limit)
    con = sqlite3.connect(":memory:")
    for name, rows in data.tables.items():
        cols = list(rows[0])
        con.execute(f'CREATE TABLE "{name}" ({", ".join(f"[{c}]" for c in cols)})')
        con.executemany(f'INSERT INTO "{name}" VALUES ({", ".join("?" * len(cols))})',
                        [[r[c] for c in cols] for r in rows])
    return con.execute(sql).fetchall()


def main(limit: int = 120) -> None:
    items = []
    n_pos = sembench_gold("SELECT COUNT(*) FROM Reviews WHERE id='ant_man_and_the_wasp_quantumania' AND scoreSentiment='POSITIVE'", limit)[0][0]
    n_all = sembench_gold("SELECT COUNT(*) FROM Reviews WHERE id='ant_man_and_the_wasp_quantumania'", limit)[0][0]
    items.append({"id": "sembench/movie/Q3", "dataset": "sembench/movie",
                  "question": 'Count of positive reviews for movie "ant_man_and_the_wasp_quantumania".',
                  "gold_answer": str(n_pos)})
    items.append({"id": "sembench/movie/Q4", "dataset": "sembench/movie",
                  "question": 'Positivity ratio of reviews for movie "ant_man_and_the_wasp_quantumania".',
                  "gold_answer": f"{n_pos}/{n_all} = {n_pos / n_all:.3f}"})
    estate = load_dataset("nirvana/estate", limit).tables["estate"]
    ajah = sorted(r["Title"] for r in estate if "Ajah, Lagos" in (r["Location"] or ""))
    items.append({"id": "estate/q3/value_content", "dataset": "nirvana/estate",
                  "question": "How many houses are located in Ajah?",
                  "hidden_intent": "the location is specifically 'Ajah, Lagos'.",
                  "ambiguity_type": "value_content", "base_id": "nirvana/estate/q3",
                  "gold_answer": str(len(ajah))})
    items.append({"id": "estate/q1", "dataset": "nirvana/estate",
                  "question": "How many houses have a yard, judging from their pictures?"})
    with open(OUT, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    print(json.dumps(items, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120)

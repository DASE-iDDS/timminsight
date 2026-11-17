"""ThalamusDB baseline worker (official PyPI package thalamusdb==0.1.15, module `tdb`).

Protocol: see src/baselines/common.py.  The NL question is compiled by the
backbone LLM into one ThalamusDB SQL query (DuckDB SQL plus NLfilter/NLjoin
predicates) with the fixed SYSTEM_PROMPT below, then executed with the
library's own ExecutionEngine, exactly as its console entry point
(`tdb.console:run_console`) does: Database(dbfile) -> Query(db, sql) ->
ExecutionEngine(db, dop=20, model_config_path).run(query, Constraints()).
Constraints use the library defaults (max_seconds=600, max_calls=1000,
max_tokens=1e6, max_error=0, i.e. run until the result is exact).  Queries
without NL predicates are run as plain SQL, as in the console.

LLM: ThalamusDB calls litellm.completion(**kwargs) with kwargs taken from a
models.json file ({"models": [{"modalities", "priority", "kwargs":
{"filter": {...}, "join": {...}}}]}).  We write one entry routing to the
request's OpenAI-compatible endpoint ("openai/<model>", api_base, api_key,
temperature 0, enable_thinking False for DashScope).  Execution token usage is
read from the TdbCounters returned by ExecutionEngine.run.
"""
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.baselines import common  # noqa: E402

from tdb.data.relational import Database  # noqa: E402
from tdb.execution.constraints import Constraints  # noqa: E402
from tdb.execution.engine import ExecutionEngine  # noqa: E402
from tdb.queries.query import Query  # noqa: E402

DOP = 20  # console default degree of parallelism

SYSTEM_PROMPT = """You translate a natural-language question into ONE ThalamusDB query.
ThalamusDB executes DuckDB SQL extended with two natural-language predicates that may
appear only as conjuncts in the WHERE clause (combined with AND, not OR/NOT):
- NLfilter(table.column, 'condition'): true for rows whose cell satisfies the condition,
  judged by an LLM (yes/no). The column must be a text column; if it holds an image path
  (.png/.jpg/.jpeg) the image is judged.
- NLjoin(t1.column1, t2.column2, 'condition'): true for row pairs satisfying the condition,
  judged by an LLM; use it to match rows when no exact join key exists (e.g. free text
  that refers to an entity named in another table).
Rules:
- NL predicates return only true/false; there is no NL function that extracts or scores
  values. Derive quantities with SQL aggregates over NL-filtered rows (e.g. count the
  rows that satisfy a condition).
- Qualify every column with its table name or alias. List tables in FROM separated by
  commas; put all join conditions in WHERE. Write conditions as short, self-contained
  English statements. Use plain SQL for anything structured (numbers, dates, exact values).
- Prefer simple queries: a single SELECT ... FROM ... WHERE ... [GROUP BY ...]
  [ORDER BY ...] [LIMIT n]; avoid subqueries over NL predicates.
Examples:
Question: Which pictures taken in 2021 show a dog?
```sql
SELECT P.id FROM photos P WHERE P.year = 2021 AND NLfilter(P.image, 'The picture shows a dog')
```
Question: For each product, how many complaints mention it?
```sql
SELECT P.name, COUNT(*) AS complaints FROM products P, tickets T
WHERE NLjoin(P.name, T.text, 'The ticket text refers to this product')
GROUP BY P.name ORDER BY complaints DESC
```
Return only the SQL query in one ```sql``` block."""


def _sql_safe_columns(tables, directory):
    """ThalamusDB qualifies columns with sqlglot against a schema whose names
    sqlglot lower-cases (a quoted "Location" is never found), and writes column
    names unquoted into the DDL of its temporary tables ("Unnamed: 0" breaks
    it).  Tables are therefore given to ThalamusDB, and described to the
    translator, with lower-case SQL identifiers as column names (data
    unchanged)."""
    import csv
    import re
    csv.field_size_limit(1 << 30)
    out = {}
    for name, t in tables.items():
        with open(t["path"], newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        header, seen = [], set()
        for c in rows[0] if rows else []:
            ident = re.sub(r"[^0-9a-z_]+", "_", c.lower()).strip("_") or "col"
            ident = f"c_{ident}" if ident[0].isdigit() else ident
            low, k = ident, 2
            while low in seen:
                low, k = f"{ident}_{k}", k + 1
            seen.add(low)
            header.append(low)
        rename = dict(zip(rows[0], header)) if rows else {}
        path = os.path.join(directory, f"{name}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows([header] + rows[1:])
        out[name] = {**t, "path": path,
                     "text_columns": [rename.get(c, c) for c in t.get("text_columns", [])],
                     "image_columns": [rename.get(c, c) for c in t.get("image_columns", [])]}
    return out


def _load(db, tables):
    for name, t in tables.items():
        path = os.path.abspath(t["path"])
        cols = {c: "VARCHAR" for c in t.get("text_columns", []) + t.get("image_columns", [])}
        types = f", types={json.dumps(cols).replace('\"', chr(39))}" if cols else ""
        db.con.execute(f"CREATE TABLE \"{name}\" AS SELECT * FROM read_csv_auto('{path}'{types})")
        base = os.path.dirname(path)
        for c in t.get("image_columns", []):  # ThalamusDB resolves relative paths against the DB dir
            db.con.execute(f"UPDATE \"{name}\" SET \"{c}\" = '{base}/' || \"{c}\" "
                           f"WHERE \"{c}\" IS NOT NULL AND NOT starts_with(\"{c}\", '/')")


def _model_config(cfg, path):
    kw = {"model": f"openai/{cfg.model}", "api_base": cfg.base_url, "api_key": cfg.api_key,
          "temperature": 0, **common.litellm_options(cfg.model, cfg.base_url)}
    conf = {"models": [{"modalities": ["text", "image"], "priority": 10,
                        "kwargs": {"filter": kw, "join": kw}}]}
    Path(path).write_text(json.dumps(conf))


def run(req):
    tmp = tempfile.TemporaryDirectory()
    req = {**req, "tables": _sql_safe_columns(req["tables"], tmp.name)}
    db = Database(os.path.join(tmp.name, "db.duckdb"))
    _load(db, req["tables"])
    conf_path = os.path.join(tmp.name, "models.json")
    _model_config(common.LLMConfig.from_request(req), conf_path)
    engine = ExecutionEngine(db, DOP, conf_path)

    def execute(sql, cfg, usage):
        sql = sql.strip().rstrip(";")
        query = Query(db, sql)
        if not query.semantic_predicates:
            return db.execute2df(sql)
        created = []
        create = engine._create_operators

        def track(q):  # keep the operators so a failed run is still charged
            ops = create(q)
            created.extend(ops)
            return ops
        engine._create_operators = track
        try:
            with contextlib.redirect_stdout(sys.stderr):  # progress output -> stderr
                result, _ = engine.run(query, Constraints())
        finally:
            engine._create_operators = create
            counters = engine._aggregate_counters(created)
            usage.add({"prompt_tokens": counters.total_input_tokens(),
                       "completion_tokens": counters.total_output_tokens()})
        return result

    try:
        return common.translate_and_run("thalamusdb", req, SYSTEM_PROMPT, execute, lang="sql")
    finally:
        db.con.close()
        tmp.cleanup()


if __name__ == "__main__":
    common.main(run, "thalamusdb")

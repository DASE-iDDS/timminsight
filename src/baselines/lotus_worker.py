"""LOTUS baseline worker (official PyPI package lotus-ai==1.2.4, litellm 1.109.1).

Translation protocol (see src/baselines/common.py): the NL question plus table
schemas/samples is compiled by the backbone LLM, with the fixed SYSTEM_PROMPT
below (temperature 0), into a Python program using LOTUS semantic operators over
pandas DataFrames.  The program is exec'd in a fresh namespace with ``pd`` and
``lotus`` pre-imported and the LOTUS LM already configured; it must assign the
answer to ``result``.  On failure the error is fed back to the translator
(up to max_attempts).  Execution tokens are read from LOTUS's own
``lm.stats.physical_usage`` (the counters behind ``lm.print_total_usage()``).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd  # noqa: E402
import lotus  # noqa: E402
from lotus.models import LM  # noqa: E402

from src.baselines import common  # noqa: E402

SYSTEM_PROMPT = r'''You translate a natural-language question over CSV tables into a Python program that uses LOTUS (lotus-ai 1.2.4), a library of LLM-powered semantic operators on pandas DataFrames.

Environment: `pd` (pandas) and `lotus` are already imported and the LOTUS language model is already configured -- do NOT call lotus.settings.configure or create models. Load every table you need with pd.read_csv(<given path>). Assign the final answer to a variable named `result` (preferably a DataFrame with only the columns that answer the question). Use ordinary pandas for structured work (comparisons on numeric columns, groupby, mean, sort, merge on exact keys); use semantic operators only where language understanding is needed.

Semantic operators (DataFrame accessors). Natural-language instructions reference columns in single curly braces, e.g. "{abstract}"; each instruction must reference at least one column.
- df.sem_filter(user_instruction) -> rows of df for which the claim is true. E.g. df.sem_filter("{abstract} is about databases").
- df.sem_map(user_instruction, suffix="_map") -> df plus one new string column named by `suffix` holding the LLM output per row.
- df.sem_extract(input_cols, output_cols) -> DataFrame with the new columns; input_cols: list[str]; output_cols: dict mapping new column name -> description (or None). E.g. df.sem_extract(["abstract"], {"venue_type": "conference or journal"}). Outputs are strings; convert with pd.to_numeric(..., errors="coerce") before arithmetic.
- df1.sem_join(df2, join_instruction, how="inner") -> joined rows of df1 x df2 where the instruction holds. Reference columns of both frames, e.g. "{name} is the company mentioned in {text}". If both frames share a column name, write "{name:left}" / "{name:right}"; such shared columns appear in the output as "name:left" / "name:right".
- df.sem_agg(user_instruction, all_cols=False, suffix="_output", group_by=None) -> one row per group (or one row) with the aggregate text in column `suffix` (plus group_by columns).
- df.sem_topk(user_instruction, K) -> the K rows that best satisfy the ranking instruction, best first.

Example 1. Tables: paper(title, abstract, year) at /data/paper.csv. Question: Which papers since 2020 are about query optimization?
```python
paper = pd.read_csv("/data/paper.csv")
recent = paper[paper["year"] >= 2020]
result = recent.sem_filter("{abstract} is about query optimization")[["title", "year"]]
```

Example 2. Tables: company(name, country) at /d/company.csv; news(nid, text) at /d/news.csv (news items mention companies by name). Question: How many news items mention each German company?
```python
company = pd.read_csv("/d/company.csv")
news = pd.read_csv("/d/news.csv")
german = company[company["country"] == "Germany"]
pairs = german.sem_join(news, "{text} mentions the company {name}")
result = pairs.groupby("name", as_index=False).size().rename(columns={"size": "mentions"})
```

Output only the program in one ```python``` block.'''

_LM = None


def _lm(cfg):
    global _LM
    if _LM is None:
        extra = common.litellm_options(cfg.model, cfg.base_url)   # top_p 1, thinking off
        _LM = LM(model=f"openai/{cfg.model}", temperature=0.0, max_tokens=1024,
                 api_base=cfg.base_url, api_key=cfg.api_key, **extra)
        lotus.settings.configure(lm=_LM)
    return _LM


def execute(program, cfg, usage):
    lm = _lm(cfg)
    lm.reset_stats()
    ns = {"pd": pd, "lotus": lotus}
    try:
        exec(compile(program, "<lotus_program>", "exec"), ns)
    finally:  # charge LOTUS's LLM calls even when the program fails
        u = lm.stats.physical_usage
        usage.add({"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens})
    if "result" not in ns:
        raise NameError("the program did not assign `result`")
    return ns["result"]


def run(req):
    return common.translate_and_run("lotus", req, SYSTEM_PROMPT, execute)


if __name__ == "__main__":
    common.main(run, "lotus")

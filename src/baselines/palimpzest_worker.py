"""Palimpzest baseline worker (official PyPI package palimpzest==1.5.3).

Protocol: see src/baselines/common.py.  The NL question is compiled by the
backbone LLM (fixed SYSTEM_PROMPT, temperature 0) into a Palimpzest program,
which is executed by the official library; failures are fed back to the
translator.  Palimpzest is pointed at the request's model through its supported
custom-endpoint path: ``pz.Model("openai/<model>", api_base=...)`` (marked as a
"vLLM" model, so litellm receives ``api_base``/``api_key``/extra kwargs; the
key is read from ``VLLM_API_KEY``).  Execution tokens come from Palimpzest's own
``ExecutionStats`` (input_text_tokens + cache_read_tokens, output_text_tokens).  PZ 1.5.3 only
records prompt tokens for models whose provider it recognises (openai/azure), so for the custom
endpoint the prompt-token count falls back to the litellm responses observed during execution.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.baselines import common  # noqa: E402

SYSTEM_PROMPT = '''You translate a natural-language question over CSV tables into a Palimpzest (pz) program.

Environment: `import palimpzest as pz` and `import pandas as pd` are already done. A ready QueryProcessorConfig
named `config` is defined; always execute with `dataset.run(config)` (never build your own config or models).

Palimpzest API (version 1.5.3):
- Load a table: `pz.MemoryDataset(id="<unique id>", vals=pd.read_csv("<path>"))` (one row = one record; columns become fields).
- `ds.sem_filter("<natural-language condition>", depends_on=["col", ...])`  keep records the LLM judges true.
- `ds.filter(lambda r: <bool>, depends_on=[...])`  Python filter; `r` is a dict of the record's fields.
- `ds.sem_map(cols=[{"name": "x", "type": str|int|float|bool, "desc": "..."}], depends_on=[...])`  LLM computes new columns per record.
- `ds.map(udf=lambda r: {"x": ...}, cols=[{"name": "x", "type": ..., "desc": "..."}])`  Python computes new columns.
- `ds.sem_join(other, condition="<NL condition relating the two records>", depends_on=[...])`  LLM-judged join.
- `ds.join(other, on="col")`  equality join on a shared column name.
- `ds.groupby(pz.GroupBySig(group_by_fields=[...], agg_funcs=["count"|"average"|"sum"|"min"|"max"], agg_fields=[...]))`.
- `ds.count()`, `ds.average()`, `ds.sum()`, `ds.min()`, `ds.max()`, `ds.project([...])`, `ds.distinct([...])`, `ds.limit(n)`.
- `ds.sem_agg(col={"name": "x", "type": str, "desc": "..."}, agg="<NL aggregation instruction>", depends_on=[...])`.
- `out = ds.run(config)` returns a DataRecordCollection; `out.to_df()` gives a pandas DataFrame.
Prefer exact Python operators for structured conditions and semantic operators for conditions over free text.
You may post-process the DataFrame with pandas (sorting, grouping, idxmax). Assign the final answer
(a DataFrame or a list of dicts) to a variable named `result`.

Example 1. Tables: product(name, category, price) at /d/product.csv. Question: "Which cheap kitchen products are eco-friendly?"
```python
products = pz.MemoryDataset(id="product", vals=pd.read_csv("/d/product.csv"))
products = products.filter(lambda r: r["category"] == "Kitchen" and r["price"] < 20, depends_on=["category", "price"])
products = products.sem_filter("The product is eco-friendly", depends_on=["name"])
result = products.project(["name", "price"]).run(config).to_df()
```

Example 2. Tables: company(name, country) at /d/company.csv, news(nid, text) at /d/news.csv (news items mention companies by name).
Question: "How many news items mention each German company?"
```python
companies = pz.MemoryDataset(id="company", vals=pd.read_csv("/d/company.csv"))
companies = companies.filter(lambda r: r["country"] == "Germany", depends_on=["country"])
news = pz.MemoryDataset(id="news", vals=pd.read_csv("/d/news.csv"))
pairs = companies.sem_join(news, condition="The news item mentions this company", depends_on=["name", "text"])
pairs = pairs.groupby(pz.GroupBySig(group_by_fields=["name"], agg_funcs=["count"], agg_fields=["nid"]))
result = pairs.run(config).to_df()
```
'''


def make_config(cfg: common.LLMConfig):
    import palimpzest as pz
    os.environ["VLLM_API_KEY"] = cfg.api_key  # Palimpzest reads the custom-endpoint key from here
    extra = common.litellm_options(cfg.model, cfg.base_url)   # top_p 1, thinking off
    model = pz.Model(f"openai/{cfg.model}", api_base=cfg.base_url, **extra)
    # Single model, no critic/mixture/RAG/split variants: Palimpzest runs its optimizer over
    # physical plans for this one model only (temperature 0 is Palimpzest's default).
    return pz.QueryProcessorConfig(
        available_models=[model], execution_strategy="parallel", progress=False, verbose=False,
        allow_mixtures=False, allow_critic=False, allow_rag_reduction=False, allow_split_merge=False,
        max_workers=8)


def execute(program: str, cfg: common.LLMConfig, usage: common.Usage):
    import pandas as pd
    import palimpzest as pz
    from palimpzest.core.data.dataset import Dataset

    import litellm

    stats, seen = [], {"prompt": 0}
    orig_run, orig_completion = Dataset.run, litellm.completion

    def counted_completion(*a, **kw):  # instrumentation only: observe provider-reported usage
        resp = orig_completion(*a, **kw)
        seen["prompt"] += int(getattr(getattr(resp, "usage", None), "prompt_tokens", 0) or 0)
        return resp

    def counted_run(self, config=None, **kw):  # instrumentation only: record ExecutionStats
        out = orig_run(self, config, **kw)
        if getattr(out, "execution_stats", None) is not None:
            stats.append(out.execution_stats)
        return out

    Dataset.run, litellm.completion = counted_run, counted_completion
    ns = {"pz": pz, "pd": pd, "config": make_config(cfg)}
    try:
        exec(compile(program, "<palimpzest_program>", "exec"), ns)
    finally:
        Dataset.run, litellm.completion = orig_run, orig_completion
        pz_prompt = int(sum(s.input_text_tokens + s.cache_read_tokens for s in stats))
        usage.add({"prompt_tokens": max(pz_prompt, seen["prompt"]),
                   "completion_tokens": int(sum(s.output_text_tokens for s in stats))})
    if "result" not in ns:
        raise NameError("program did not assign `result`")
    result = ns["result"]
    if hasattr(result, "to_df") and not hasattr(result, "to_json"):
        result = result.to_df()
    return result


def run(req):
    return common.translate_and_run("palimpzest", req, SYSTEM_PROMPT, execute)


if __name__ == "__main__":
    common.main(run, "palimpzest")

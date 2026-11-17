"""Nirvana baseline worker (Zhu et al., PACMMOD 2026; official source commit
7ef95e614a97836c37899a23db7f94e4fa836a35, package nirvana-ai 1.3.2).

Protocol: see src/baselines/common.py.  The NL question is compiled into a
Nirvana lazy-API program (nv.DataFrame.semantic_* + optimize_and_execute, i.e.
Nirvana's logical plan optimizer) with the fixed SYSTEM_PROMPT below.

Adapter notes (no change to Nirvana's algorithms):
* Nirvana 1.3.2 routes each model name to a hard-coded provider URL and calls
  the OpenAI *Responses* API.  We redirect the client to the request's
  OpenAI-compatible base_url and translate responses.create -> chat.completions
  (temperature 0, DashScope thinking off), recording token usage there.
* nirvana.dataframe.arrays.utils.infer_dtype samples 10 values without
  replacement, so nv.DataFrame crashes on string columns with < 10 rows; we
  sample min(10, len) values instead (same logic otherwise).
* DataFrame.semantic_join passes input_left_fields/input_right_fields to
  NodeFields, whose fields are left_input_fields/right_input_fields (TypeError
  on every lazy join); we map the names.  LineageNode then joins with
  `left.join(right, on="keys")`, which always fails because both sides carry a
  `keys` column; we perform the intended equi-merge on `keys`.
* LLMClient reuses one AsyncOpenAI client across asyncio.run() loops ("Event
  loop is closed"); our facade opens a fresh client per call.
* The non_llm_replace rule parses LLM-written UDFs with the regex
  `func=([^,]+)`, so any UDF containing a comma raises ValueError and aborts
  the query; we treat that as "no rewrite" (the plan stays unchanged).
* Physical optimization is disabled: it chooses among several models, while the
  experiment fixes one backbone model.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.baselines import common  # noqa: E402
from src.baselines.common import LLMConfig, Usage, translate_and_run  # noqa: E402

SYSTEM_PROMPT = r'''You translate a natural-language data question into a Python program that uses
Nirvana, a semantic data-analytics library whose operators are executed by an LLM.

API (already imported in the program as `import pandas as pd`, `import nirvana as nv`;
the LLM backbone is already configured, do not call nv.configure_llm_backbone):
- df = nv.DataFrame(pd.read_csv(path))   # lazy DataFrame; semantic_* calls only record a plan
- df.semantic_filter(user_instruction: str, input_columns: list[str])
      keep rows for which the natural-language predicate holds
- df.semantic_map(user_instruction: str, input_columns: list[str], output_columns: list[str])
      add new columns computed by the LLM from the input columns
- df.semantic_rank(user_instruction: str, input_column: str, descend: bool = True, limit: int | None = None)
      order rows by an LLM-judged criterion over one column
- df.semantic_reduce(user_instruction: str, input_column: str)
      aggregate one column of all rows into a single LLM answer
- df.semantic_join(other: nv.DataFrame, user_instruction: str, left_on: str, right_on: str, how="inner")
      join rows of two tables when the natural-language condition on left_on/right_on holds
Each semantic_* method mutates df and returns it.  Execute the recorded plan with
    out, cost, runtime = df.optimize_and_execute(optim_config=OPTIM_CONFIG)
(OPTIM_CONFIG is predefined; it enables Nirvana's logical plan optimizer).  `out` is a
pandas DataFrame; use pandas on it for exact operations (numeric comparison, groupby,
mean, sort, column selection).  Use pandas for exact
conditions on structured columns and semantic_* operators for anything requiring
understanding of text.  Never call an LLM yourself.

The program must load the tables from the given CSV paths and assign the final
answer (a pandas DataFrame with only the columns answering the question, or a scalar)
to a variable named `result`.

Example 1. Question: "Which products with price under 50 are eco-friendly?"
```python
products = pd.read_csv("/data/product.csv")
df = nv.DataFrame(products[products["price"] < 50].reset_index(drop=True))
df.semantic_filter(user_instruction="The product description says it is eco-friendly.", input_columns=["description"])
out, cost, runtime = df.optimize_and_execute(optim_config=OPTIM_CONFIG)
result = out[["name"]]
```

Example 2. Question: "How many news items mention each German company?"
```python
company = pd.read_csv("/data/company.csv")
company = nv.DataFrame(company[company["country"] == "Germany"].reset_index(drop=True))
news = nv.DataFrame(pd.read_csv("/data/news.csv"))
company.semantic_join(news, user_instruction="The news item mentions this company.",
                      left_on="name", right_on="text")
out, cost, runtime = company.optimize_and_execute(optim_config=OPTIM_CONFIG)
result = out.groupby("name", as_index=False).size().rename(columns={"size": "mentions"})
```
'''

_USAGE = [Usage()]  # current execution usage sink


class _Responses:
    """Responses-API facade over chat.completions of an OpenAI-compatible endpoint."""

    def __init__(self, cfg):
        self._cfg = cfg

    @staticmethod
    def _messages(inp):
        if isinstance(inp, str):
            return [{"role": "user", "content": inp}]
        msgs = []
        for m in inp:
            content = m["content"]
            if isinstance(content, list):
                content = [{"type": "text", "text": p["text"]} if p.get("type") == "input_text"
                           else {"type": "image_url", "image_url": {"url": p["image_url"]}}
                           if p.get("type") == "input_image" else p for p in content]
            msgs.append({"role": m["role"], "content": content})
        return msgs

    async def create(self, model, input, max_output_tokens=None, temperature=0, **_):
        from openai import AsyncOpenAI
        # temperature 0, top_p 1, thinking off (Sec. VI), as for every system
        opts = common.request_options(model, self._cfg.base_url)
        cap = "max_completion_tokens" if common.is_openai_reasoning(model) else "max_tokens"
        if max_output_tokens is not None:
            opts[cap] = max_output_tokens
        async with AsyncOpenAI(api_key=self._cfg.api_key, base_url=self._cfg.base_url) as client:
            r = await client.chat.completions.create(
                model=model, messages=self._messages(input), extra_body=opts)
        u = r.usage
        _USAGE[0].add({"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens})
        return SimpleNamespace(
            output_text=r.choices[0].message.content or "", model=model,
            usage=SimpleNamespace(input_tokens=u.prompt_tokens, output_tokens=u.completion_tokens,
                                  input_tokens_details=SimpleNamespace(cached_tokens=0)))


def configure(cfg: LLMConfig):
    import nirvana as nv
    from nirvana.executors import llm_backbone as lb
    from openai import AsyncOpenAI

    lb.MODEL_PRICING.setdefault(cfg.model, {"Input": 0.0, "Cache": 0.0, "Output": 0.0})
    lb._get_openai_compatible_provider_info = lambda model_name, api_key=None, base_url=None: (
        cfg.api_key, cfg.base_url)

    def _create_client(api_key, **kwargs):
        client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        client.responses = _Responses(cfg)
        return client

    lb._create_client = _create_client
    from nirvana.dataframe.arrays import utils as au
    _infer = au.infer_dtype

    def infer_dtype(col):
        return _infer(col if len(col) >= 10 else col.sample(10, replace=True, random_state=0))
    au.infer_dtype = infer_dtype
    from nirvana.lineage import abstractions as ab
    _nf_init = ab.NodeFields.__init__

    def nf_init(self, **kw):
        for a, b in (("input_left_fields", "left_input_fields"), ("input_right_fields", "right_input_fields")):
            if a in kw:
                kw[b] = kw.pop(a)
        _nf_init(self, **kw)
    ab.NodeFields.__init__ = nf_init
    _run = ab.LineageNode.run

    async def run_node(self, input=None):
        if self.op_name != "join":
            return await _run(self, input)
        o = await self.operator.execute(left_data=input[0], right_data=input[1])
        out = input[0].assign(keys=o.left_join_keys).merge(
            input[1].assign(keys=o.right_join_keys), on="keys", how=self.operator.how)
        return ab.NodeOutput(output=out.drop(columns="keys").reset_index(drop=True), cost=o.cost)
    ab.LineageNode.run = run_node
    from collections import deque
    from nirvana.optim.rules import non_llm_replace as nlr
    _extract = nlr.extract_udfs_from_code

    def extract_udfs(code):
        try:
            return _extract(code)
        except ValueError:
            return deque()
    nlr.extract_udfs_from_code = extract_udfs
    lb.LLMClient.config.temperature = 0.0
    lb.LLMClient.config.max_tokens = 2048
    nv.configure_llm_backbone(model_name=cfg.model, api_key=cfg.api_key, base_url=cfg.base_url)


def execute(program: str, cfg: LLMConfig, usage: Usage):
    import pandas as pd
    import nirvana as nv
    _USAGE[0] = usage
    optim = nv.optim.OptimizeConfig(do_logical_optimization=True, do_physical_optimization=False,
                                    available_models=[cfg.model])
    ns = {"pd": pd, "nv": nv, "OPTIM_CONFIG": optim, "__name__": "__nirvana_program__"}
    exec(compile(program, "<nirvana_program>", "exec"), ns)
    if "result" not in ns:
        raise NameError("the program did not assign `result`")
    result = ns["result"]
    return result.to_pandas() if isinstance(result, nv.DataFrame) else result


def run(req):
    configure(LLMConfig.from_request(req))
    return translate_and_run("nirvana", req, SYSTEM_PROMPT, execute)


if __name__ == "__main__":
    common.main(run, "nirvana")

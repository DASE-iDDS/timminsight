"""Answer and chart generation (paper Sec. II, final step).

The LLM analyzes the execution result, recommends a suitable chart type, and
writes a natural-language summary of the key findings.  When the plan has a
VISUALIZE operator, the chart the user asked for (its bound specification) is
given to the recommendation as a constraint: the chart type the user named is
kept, and the recommendation completes it from the result.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

from .llm_query_analyzer import complete_json
from .plan_executor import ExecutionError, PlanExecutionResult, Table

CHART_TYPES = ("bar", "line", "scatter", "pie", "histogram", "table")
MAX_ROWS_IN_PROMPT = 50


def _clean_for_answer(data: Table) -> Table:
    """Drop image cells and shorten long text so the prompt carries content only."""
    out = []
    for row in data:
        clean = {}
        for k, v in row.items():
            if isinstance(v, (bytes, bytearray)) or (isinstance(v, dict) and "bytes" in v):
                continue
            if isinstance(v, str):
                if v.startswith("data:image"):
                    continue
                if len(v) > 300:
                    v = v[:300] + "..."
            if isinstance(v, list) and len(v) > 16:   # embeddings
                continue
            clean[k] = v
        out.append(clean)
    return out


class AnswerGenerator:
    """Generates the final answer and chart for an executed query."""

    def __init__(self, llm_provider: Any):
        if llm_provider is None:
            raise ValueError("answer generation needs the LLM provider")
        self._llm = llm_provider

    def generate(self, query_text: str, execution_result: PlanExecutionResult,
                 chart: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """Return (answer, chart specification).  ``chart`` is the chart the
        user asked for (the binding of a VISUALIZE operator), or None."""
        # what the user explicitly asked for (the VISUALIZE binding), if anything
        asked = {k: v for k, v in (chart or {}).items() if k in ("chart", "x", "y") and v}
        data = _clean_for_answer(execution_result.data)
        shown = data[:MAX_ROWS_IN_PROMPT]
        prompt = (
            f"Question: {query_text}\n"
            f"The query returned {execution_result.row_count} rows"
            f"{'' if len(shown) == len(data) else f' (first {len(shown)} shown)'}:\n"
            f"{json.dumps(shown, default=str, ensure_ascii=False)}\n\n"
            "Answer the question from these rows only: state the entities and numbers "
            "directly and summarize the key findings. "
            f"Also analyze the result and recommend the chart that best presents it, one of "
            f"{list(CHART_TYPES)}, with its x and y columns. "
            + (f"The user asked for this chart: {json.dumps(asked, ensure_ascii=False)}; "
               "keep what the user named and complete the rest from the result. "
               if asked else "")
            + 'Return JSON: {"answer": str, "chart": {"chart": str, "x": str|null, "y": str|null}}'
        )
        def answer_ok(out: Dict[str, Any]) -> None:
            if not isinstance(out.get("answer"), str) or not out["answer"].strip():
                raise ValueError("missing 'answer'")
            if (not isinstance(out.get("chart"), dict)
                    or out["chart"].get("chart") not in CHART_TYPES):
                raise ValueError(f"'chart.chart' must be one of {list(CHART_TYPES)}")

        out = complete_json(self._llm, [
            {"role": "system", "content": "You are a data analyst. Output JSON only."},
            {"role": "user", "content": prompt}], dict, answer_ok, "answer generation",
            ExecutionError)
        recommended = {k: out["chart"].get(k) for k in ("chart", "x", "y")}
        # the chart type and axes the user named take precedence
        return out["answer"].strip(), {**recommended, **asked}

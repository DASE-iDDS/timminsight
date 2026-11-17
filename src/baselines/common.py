"""Shared protocol for running the baseline systems end to end from NL questions.

Only CAESURA accepts a natural-language question in its official release.
LOTUS, Palimpzest, ThalamusDB, and Nirvana take programs (semantic-operator
pipelines, declarative pipelines, or SQL with NL predicates).  To evaluate them
on the same NL questions as TiMMInsight, each question is compiled into the
system's input program by the same backbone LLM with a fixed, system-specific
translation prompt (temperature 0).  A program that fails to execute is sent
back to the translator together with the error, up to ``MAX_ATTEMPTS`` times.
The translation cost (tokens and time) is charged to the baseline.  No baseline
receives TiMetaGraph.

Each system runs in its own virtual environment as a subprocess (its official
dependencies conflict with each other).  A worker reads one JSON request from
stdin and writes one JSON response to stdout:

request  = {"question": str,
            "tables": {name: {"path": str, "description": str,
                              "text_columns": [...], "image_columns": [...]}},
            "llm": {"model": str, "base_url": str, "api_key_env": str},
            "max_attempts": int}
response = {"system": str, "rows": [...], "program": str, "attempts": int,
            "errors": [str], "translation_tokens": {"prompt": int, "completion": int},
            "execution_tokens": {"prompt": int, "completion": int},
            "translation_seconds": float, "execution_seconds": float, "ok": bool}

This module uses the standard library only, so every worker environment can
import it.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

MAX_ATTEMPTS = 3


@dataclass
class Usage:
    prompt: int = 0
    completion: int = 0

    def add(self, other: Dict[str, Any]) -> None:
        self.prompt += int(other.get("prompt_tokens", 0) or 0)
        self.completion += int(other.get("completion_tokens", 0) or 0)

    def as_dict(self) -> Dict[str, int]:
        return {"prompt": self.prompt, "completion": self.completion}


@dataclass
class LLMConfig:
    model: str
    base_url: str
    api_key_env: str = "DASHSCOPE_API_KEY"

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")

    @classmethod
    def from_request(cls, req: Dict[str, Any]) -> "LLMConfig":
        llm = req["llm"]
        return cls(llm["model"], llm["base_url"].rstrip("/"), llm.get("api_key_env", "DASHSCOPE_API_KEY"))


_CLAUDE = re.compile(r"claude-(?:opus|sonnet|haiku|fable|mythos)-(\d+)(?:-(\d{1,2}))?(?!\d)")


def fixed_sampling(model: str) -> bool:
    """Claude 4.7 and later reject requests that set ``temperature`` or
    ``top_p`` (HTTP 400): they sample with their fixed defaults."""
    m = _CLAUDE.search(model.lower())
    return bool(m) and (int(m.group(1)), int(m.group(2) or 0)) >= (4, 7)


def is_openai_reasoning(model: str) -> bool:
    """OpenAI GPT-5 models: reasoning is switched off with ``reasoning_effort``
    and the output cap is ``max_completion_tokens``."""
    return model.lower().split("/")[-1].startswith("gpt-5")


def request_options(model: str, base_url: str) -> Dict[str, Any]:
    """Request fields of the protocol of Sec. VI, the same for TiMMInsight and
    every baseline: temperature 0, top_p 1, thinking mode off.

    * DashScope models (Qwen, Kimi, GLM, DeepSeek): ``enable_thinking: false``.
    * OpenAI GPT-5 models: ``reasoning_effort: "none"``.
    * Claude 4.7 and later (Claude-Opus-4.8): the sampling fields are omitted,
      as the API rejects them; these models run without thinking unless it is
      requested, so nothing is sent for it.
    """
    opts: Dict[str, Any] = {} if fixed_sampling(model) else {"temperature": 0, "top_p": 1}
    if "dashscope" in base_url:
        opts["enable_thinking"] = False
    elif is_openai_reasoning(model):
        opts["reasoning_effort"] = "none"
    return opts


def extra_options(model: str, base_url: str) -> Dict[str, Any]:
    """``request_options`` without the sampling fields (for clients that take
    the temperature as their own argument)."""
    return {k: v for k, v in request_options(model, base_url).items()
            if k not in ("temperature", "top_p")}


def litellm_options(model: str, base_url: str) -> Dict[str, Any]:
    """Keyword arguments for systems that call ``litellm.completion`` with their
    own ``temperature=0``: top_p 1 and thinking off, and for a model with fixed
    sampling the instruction to drop the sampling fields from the request."""
    if fixed_sampling(model):
        opts: Dict[str, Any] = {"additional_drop_params": ["temperature", "top_p"]}
    else:
        opts = {"top_p": 1}
    extra = extra_options(model, base_url)
    if extra:
        opts["extra_body"] = extra
    return opts


def chat(cfg: LLMConfig, messages: List[Dict[str, str]], usage: Usage,
         timeout: int = 180) -> str:
    """One chat completion against an OpenAI-compatible endpoint (temperature 0,
    top_p 1, thinking off: ``request_options``)."""
    body = {"model": cfg.model, "messages": messages,
            **request_options(cfg.model, cfg.base_url)}
    req = urllib.request.Request(
        f"{cfg.base_url}/chat/completions", data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {cfg.api_key}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    usage.add(out.get("usage") or {})
    return out["choices"][0]["message"]["content"] or ""


def extract_code(text: str, lang: str = "") -> str:
    """Return the first fenced code block (or the whole reply if none)."""
    m = re.search(r"```(?:%s|[a-zA-Z]*)?\s*\n(.*?)```" % re.escape(lang), text, re.S)
    return (m.group(1) if m else text).strip()


def describe_tables(tables: Dict[str, Dict[str, Any]], preview_rows: int = 3) -> str:
    """Schema and a few sample rows of each table, as given to every translator."""
    import csv
    parts = []
    for name, t in tables.items():
        with open(t["path"], newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        cols = list(rows[0].keys()) if rows else []
        parts.append(
            f"Table `{name}` ({len(rows)} rows), file {t['path']}\n"
            f"  columns: {cols}\n"
            f"  text columns: {t.get('text_columns', [])}; image columns: {t.get('image_columns', [])}\n"
            f"  description: {t.get('description', '')}\n"
            f"  sample rows: {json.dumps(rows[:preview_rows], ensure_ascii=False)}")
    return "\n".join(parts)


def translate_and_run(
    system: str,
    req: Dict[str, Any],
    system_prompt: str,
    execute: Callable[[str, LLMConfig, Usage], List[Dict[str, Any]]],
    lang: str = "python",
) -> Dict[str, Any]:
    """Compile the NL question into a program with the fixed translation prompt
    and execute it, retrying with the error message on failure."""
    cfg = LLMConfig.from_request(req)
    max_attempts = int(req.get("max_attempts", MAX_ATTEMPTS))
    t_usage, x_usage = Usage(), Usage()
    t_sec = x_sec = 0.0
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content":
            f"Tables:\n{describe_tables(req['tables'])}\n\nQuestion: {req['question']}\n\n"
            f"Return only the program in one ```{lang}``` block."},
    ]
    errors: List[str] = []
    program = ""
    for attempt in range(1, max_attempts + 1):
        t0 = time.time()
        reply = chat(cfg, messages, t_usage)
        t_sec += time.time() - t0
        program = extract_code(reply, lang)
        t0 = time.time()
        try:
            rows = execute(program, cfg, x_usage)
            x_sec += time.time() - t0
            return _response(system, rows, program, attempt, errors, t_usage, x_usage, t_sec, x_sec, True)
        except Exception as exc:  # the error is fed back to the translator
            x_sec += time.time() - t0
            err = f"{type(exc).__name__}: {exc}"
            errors.append(err[:2000])
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": f"Executing the program failed:\n{err[:2000]}\n"
                                                      f"Fix it and return the full program."}]
    return _response(system, [], program, max_attempts, errors, t_usage, x_usage, t_sec, x_sec, False)


def _response(system, rows, program, attempts, errors, t_usage, x_usage, t_sec, x_sec, ok):
    return {"system": system, "rows": to_records(rows), "program": program, "attempts": attempts,
            "errors": errors, "translation_tokens": t_usage.as_dict(),
            "execution_tokens": x_usage.as_dict(), "translation_seconds": round(t_sec, 3),
            "execution_seconds": round(x_sec, 3), "ok": ok}


def to_records(result: Any) -> List[Dict[str, Any]]:
    """Normalize a system's result (DataFrame, list of dicts, scalar) to rows."""
    if result is None:
        return []
    if hasattr(result, "to_dict"):
        try:
            return json.loads(result.to_json(orient="records", default_handler=str))
        except Exception:
            return result.to_dict(orient="records")
    if isinstance(result, dict):
        return [result]
    if isinstance(result, (list, tuple)):
        return [r if isinstance(r, dict) else {"value": r} for r in result]
    return [{"value": result}]


def main(run: Callable[[Dict[str, Any]], Dict[str, Any]], system: str) -> None:
    """Worker entry point: JSON request on stdin, JSON response on stdout.  A
    run that raises may attach the tokens it already spent as ``exc.usage``."""
    try:
        req = json.load(sys.stdin)
        resp = run(req)
    except Exception as exc:
        resp = {"system": system, "rows": [], "ok": False,
                "errors": [traceback.format_exc()[-4000:]]}
        if isinstance(getattr(exc, "usage", None), dict):
            resp["execution_tokens"] = exc.usage
    sys.stdout.write("\n__BASELINE_RESPONSE__" + json.dumps(resp, ensure_ascii=False, default=str) + "\n")

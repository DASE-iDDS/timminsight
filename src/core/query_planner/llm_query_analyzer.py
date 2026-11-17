"""
LLM-based Query Analyzer for Multimodal Query Planning.

Replaces regex-based keyword matching with LLM inference to extract
operator requirements from natural language queries. Supports multiple
LLM providers via a unified OpenAI-compatible chat completions interface.

Supported providers:
  - DashScope (Qwen, Kimi, etc.)
  - OpenAI (GPT-4o, GPT-5, etc.)
  - AWS Bedrock (Claude, etc.)
  - Any OpenAI-compatible endpoint
"""

from __future__ import annotations

import json
import logging
import random
import os
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

from .operator_compatibility_graph import (
    ModalityType,
    OperatorCompatibilityGraph,
    QueryContext,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSON output of LLM calls (shared by all modules that call an LLM)
# ---------------------------------------------------------------------------

class LLMOutputError(RuntimeError):
    """An LLM reply could not be parsed into the expected JSON in three attempts."""


def parse_json(text: Any, expect: type = dict) -> Any:
    """The JSON value of type ``expect`` in an LLM reply (code fences and
    surrounding prose are tolerated), or None if there is none."""
    t = re.sub(r"```[a-zA-Z]*", "", str(text)).replace("```", "").strip()
    try:
        value = json.loads(t)
        if isinstance(value, expect):
            return value
    except ValueError:
        pass
    opener, closer = ("[", "]") if expect is list else ("{", "}")
    s, e = t.find(opener), t.rfind(closer)
    if s != -1 and e > s:
        try:
            value = json.loads(t[s:e + 1])
            if isinstance(value, expect):
                return value
        except ValueError:
            pass
    # the first complete JSON value, ignoring trailing characters (e.g. a
    # stray closing brace); an incomplete value still yields None
    decoder = json.JSONDecoder()
    while s != -1:
        try:
            value, end = decoder.raw_decode(t, s)
            if isinstance(value, expect):
                # objects separated only by whitespace form one reply
                # (e.g. {"thought": ...} {"action": ...})
                while isinstance(value, dict):
                    nxt = len(t) - len(t[end:].lstrip())
                    if not t.startswith("{", nxt):
                        break
                    try:
                        more, end = decoder.raw_decode(t, nxt)
                    except ValueError:
                        break
                    if not isinstance(more, dict):
                        break
                    value = {**value, **more}
                return value
        except ValueError:
            pass
        s = t.find(opener, s + 1)
    return None


def complete_json(llm: Any, messages: List[Dict[str, Any]], expect: type = dict,
                  validate: Optional[Callable[[Any], None]] = None, what: str = "LLM",
                  error: type = LLMOutputError) -> Any:
    """One LLM call whose reply must be JSON of type ``expect`` (and pass
    ``validate``, which raises ValueError on an incomplete reply).  A rejected
    reply is returned to the model with the reason, up to three attempts in
    all; then ``error`` is raised.  Output is never silently replaced."""
    if llm is None:
        raise error(f"{what}: needs the LLM provider")
    conversation = list(messages)
    last, problem = "", "unparsable"
    for _ in range(3):
        last = str(llm.complete(conversation))
        value = parse_json(last, expect)
        if value is None:
            problem = "unparsable"
        else:
            try:
                if validate is not None:
                    validate(value)
                return value
            except ValueError as exc:
                problem = str(exc)
        reason = ("the reply is not valid JSON of the requested form" if problem == "unparsable"
                  else problem)
        conversation = conversation + [
            {"role": "assistant", "content": last},
            {"role": "user", "content": f"Your reply was rejected: {reason}. "
                                        "Return the corrected JSON only."}]
    raise error(f"{what}: {problem} LLM output: {last[:300]}")


# ---------------------------------------------------------------------------
# Operator Description Registry
# ---------------------------------------------------------------------------

OPERATOR_DESCRIPTIONS: Dict[str, str] = {
    "SCAN": "Read raw data from a table or collection (always required as leaf)",
    "FILTER": "Keep rows satisfying structured conditions and/or a semantic condition on text or images",
    "PROJECT": "Select specific columns/fields",
    "AGGREGATE": "Group-by and aggregate (COUNT, SUM, AVG, etc.)",
    "SORT": "Order results by one or more keys",
    "LIMIT": "Restrict output to top-N rows",
    "DISTINCT": "Remove duplicate rows",
    "JOIN": "Combine two data sources on a key",
    "UNION": "Merge multiple result sets",
    "SEMANTIC_SEARCH": "Rank rows by relevance to a natural-language query (LLM-rated) and keep the top k",
    "CROSS_MODAL_MATCH": "Match rows of two sources that refer to the same entity without an exact join key (e.g. reviews to the movies they mention)",
    "CONTENT_EXTRACT": "Derive a new attribute (e.g. sentiment, a count of objects) from text or image columns",
    "SIMILARITY_JOIN": "Join two sources on a semantic similarity condition between their rows",
    "FEATURE_TRANSFORM": "Embed a text or image column into a vector column",
    "VISUALIZE": "Produce charts, plots, or visual output",
    "EXPORT": "Save or download results to a file",
}


def build_prompt_from_ocg(
    ocg: Optional[OperatorCompatibilityGraph] = None,
    custom_descriptions: Optional[Dict[str, str]] = None,
) -> str:
    """Dynamically build the system prompt from OCG operator specs.

    If an OCG is provided, the operator list is generated from its registered
    specs. Otherwise falls back to a default OCG. Custom descriptions can
    override or extend the built-in ones.
    """
    if ocg is None:
        ocg = OperatorCompatibilityGraph()

    descriptions = dict(OPERATOR_DESCRIPTIONS)
    if custom_descriptions:
        descriptions.update(custom_descriptions)

    categories: Dict[str, List[str]] = {}
    category_labels = {
        "access": "Data Access",
        "transform": "Transforms",
        "cross_modal": "Cross-Modal",
        "output": "Output",
    }

    for op_type in ocg.all_operator_types:
        spec = ocg.get_spec(op_type)
        if spec is None:
            continue
        cat = spec.category
        categories.setdefault(cat, []).append(op_type)

    operator_section_lines = []
    for cat_key in ("access", "transform", "cross_modal", "output"):
        ops = categories.get(cat_key, [])
        if not ops:
            continue
        label = category_labels.get(cat_key, cat_key)
        operator_section_lines.append(f"**{label}:**")
        for op in ops:
            desc = descriptions.get(op, op)
            operator_section_lines.append(f"- {op}: {desc}")
        operator_section_lines.append("")

    operator_section = "\n".join(operator_section_lines).strip()
    num_ops = len(ocg.all_operator_types)

    leaf_ops = [
        t for t in ocg.all_operator_types
        if (s := ocg.get_spec(t)) and s.is_leaf
    ]

    leaf_str = ", ".join(leaf_ops) if leaf_ops else "SCAN"

    modalities = ", ".join(f'"{m.value}"' for m in ModalityType)

    return f"""\
You are a multimodal query planner. Given a natural language query about \
a multimodal dataset (tables, images, text), determine which logical \
operators are needed and how they should be structured.

## Available Operators ({num_ops} types)

{operator_section}

## Rules

1. {leaf_str} is always required (leaf operator for data access).
2. Required operators MUST appear in the plan. Optional operators MAY improve it.
3. Identify target modalities from the query content.

## Output Format

Return a JSON object (no markdown fences, no extra text):
{{
  "required_operators": ["{leaf_str}", ...],
  "optional_operators": [...],
  "target_modalities": ["tabular", ...],
  "bindings": {{"<OPERATOR>": {{<parameters of that operator, as listed below>}}}}
}}

"bindings" gives every required operator the parameters it executes with
(column names must be columns of the bound sources):
  FILTER:            {{"source": ..., "conditions": [{{"column": ..., "op": "=|!=|>|>=|<|<=|in|contains", "value": ...}}],
                       "semantic": "<condition needing language or image understanding, if any>",
                       "columns": [<columns the semantic condition reads>]}}
  PROJECT:           {{"columns": [...]}}
  DISTINCT:          {{"columns": [...]}}
  AGGREGATE:         {{"group_by": [...], "function": "avg|sum|count|min|max", "attribute": ..., "as": ..., "condition": {{"op": ..., "value": ...}} (optional: aggregate the 0/1 indicator of a condition on the attribute, e.g. a share or ratio)}}
  SORT:              {{"key": ..., "descending": true|false}}
  LIMIT:             {{"count": <number of rows the question asks for, e.g. 1 for "which movie">}}
  JOIN:              {{"sources": [<left>, <right>], "on": [<left column>, <right column>]}}
  UNION:             {{"sources": [...]}}
  CONTENT_EXTRACT:   {{"source": ..., "attribute": <new column>, "description": <what to extract>, "columns": [<text or image column>], "type": "boolean|number|text"}}
  SEMANTIC_SEARCH:   {{"query": ..., "k": <rows to keep>, "columns": [...]}}
  CROSS_MODAL_MATCH: {{"sources": [<left>, <right>], "condition": <when two rows match>}}
  SIMILARITY_JOIN:   {{"sources": [<left>, <right>], "condition": ...}}
  FEATURE_TRANSFORM: {{"columns": [<one text or image column>], "as": <vector column>}}
  VISUALIZE:         {{"chart": "bar|line|scatter|pie|histogram|table", "x": ..., "y": ...}}
  EXPORT:            {{"path": ...}}
An attribute that must be derived from free text or images (e.g. the sentiment of
review texts) requires CONTENT_EXTRACT on that source; rows of different sources
that refer to the same entity without an exact join key require CROSS_MODAL_MATCH.

Valid modalities: {modalities}
"""


# ---------------------------------------------------------------------------
# LLM Provider Abstraction
# ---------------------------------------------------------------------------

@dataclass
class TokenUsage:
    """Cumulative LLM token usage across calls on a single provider."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0

    def add(self, prompt: int, completion: int, total: int) -> None:
        self.prompt_tokens += int(prompt or 0)
        self.completion_tokens += int(completion or 0)
        # Fall back to prompt+completion if the API omits a total.
        self.total_tokens += int(total or (int(prompt or 0) + int(completion or 0)))
        self.calls += 1

    def reset(self) -> None:
        self.prompt_tokens = self.completion_tokens = self.total_tokens = 0
        self.calls = 0

    def snapshot(self) -> "TokenUsage":
        return TokenUsage(
            self.prompt_tokens, self.completion_tokens,
            self.total_tokens, self.calls,
        )


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Every provider exposes a ``usage`` counter (``TokenUsage``) that
    accumulates token consumption across ``complete`` calls, so experiment
    harnesses can snapshot/reset it around a query.
    """

    @abstractmethod
    def complete(self, messages: List[Dict[str, str]]) -> str:
        """Send chat messages and return the assistant's response text."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name."""


def _request_rules():
    """``request_options`` and ``is_openai_reasoning`` of src/baselines/common.py,
    whether this package is imported as ``src.core`` or as ``core``."""
    try:
        from ...baselines.common import is_openai_reasoning, request_options
    except ImportError:
        from baselines.common import is_openai_reasoning, request_options
    return request_options, is_openai_reasoning


class OpenAICompatibleProvider(LLMProvider):
    """Provider for any OpenAI-compatible chat completions API.

    Works with: OpenAI, DashScope, vLLM, Ollama, LiteLLM, etc.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: int = 75,
        extra_body: Optional[Dict[str, Any]] = None,
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._extra_body = extra_body or {}
        self.usage = TokenUsage()

    @property
    def provider_name(self) -> str:
        return f"openai-compatible ({self._base_url})"

    def complete(self, messages: List[Dict[str, str]]) -> str:
        url = f"{self._base_url}/chat/completions"
        # Sec. VI: temperature 0, top_p 1, thinking mode off -- the request
        # fields shared with the baselines (baselines/common.request_options)
        request_options, is_openai_reasoning = _request_rules()
        options = request_options(self._model, self._base_url)
        if "temperature" in options:
            options["temperature"] = self._temperature
        cap = "max_completion_tokens" if is_openai_reasoning(self._model) else "max_tokens"
        payload = {"model": self._model, "messages": messages, cap: self._max_tokens, **options}
        # caller overrides (extra_body=...)
        payload.update(self._extra_body)
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        import http.client
        import threading
        import time
        last_exc = None

        def _do_request(box: Dict[str, Any]) -> None:
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    box["body"] = json.loads(resp.read().decode("utf-8"))
            except BaseException as exc:          # handed to the caller below
                box["error"] = exc

        # Hard wall-clock cap per attempt: urllib's ``timeout`` is bypassed under
        # a SOCKS proxy (the socket can freeze), so the request runs in a daemon
        # thread that is abandoned if it overruns (a daemon thread never blocks
        # interpreter exit, unlike a ThreadPoolExecutor worker).
        hard = 100
        timeouts = 0
        attempt = 0          # server and network failures: up to 6 attempts
        throttled = 0        # rate limiting (HTTP 429): up to 8 patient retries
        while attempt < 6 and throttled <= 8:
            box: Dict[str, Any] = {}
            worker = threading.Thread(target=_do_request, args=(box,), daemon=True)
            worker.start()
            worker.join(hard)
            if worker.is_alive():
                last_exc = TimeoutError(f"request stalled >{hard}s")
                # a stalled connection rarely recovers: give up after two stalls
                timeouts += 1
                attempt += 1
                if timeouts >= 2:
                    break
                continue
            exc = box.get("error")
            if exc is None:
                body = box["body"]
                usage = body.get("usage") or {}
                self.usage.add(
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                    usage.get("total_tokens", 0),
                )
                try:
                    return body["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError) as err:
                    raise RuntimeError(f"LLM request failed: malformed reply {str(body)[:200]}") from err
            last_exc = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 429:
                # rate limited: wait as the server asks (Retry-After) or back off
                # exponentially up to 2 minutes, with jitter so that concurrent
                # workers do not retry in lockstep
                throttled += 1
                retry_after = (exc.headers or {}).get("Retry-After") if hasattr(exc, "headers") else None
                try:
                    wait = float(retry_after)
                except (TypeError, ValueError):
                    wait = min(120.0, 2.0 * (2 ** throttled))
                time.sleep(wait * (1.0 + 0.25 * random.random()))
                continue
            attempt += 1
            if isinstance(exc, urllib.error.HTTPError):
                # transient server errors: exponential backoff
                if exc.code in (500, 502, 503, 504):
                    time.sleep(min(60.0, 2.0 * (2 ** attempt)))
                    continue
                raise RuntimeError(f"LLM request failed: HTTP {exc.code}") from exc
            if isinstance(exc, (urllib.error.URLError, http.client.HTTPException,
                                ConnectionError, TimeoutError, json.JSONDecodeError)):
                # transient network failure (e.g. connection reset): back off, retry
                time.sleep(min(60.0, 2.0 * (2 ** attempt)))
                continue
            raise exc
        raise RuntimeError(f"LLM request failed after retries: {last_exc}")


class BedrockProvider(LLMProvider):
    """Provider for AWS Bedrock (boto3 Converse API), e.g. Claude-Opus-4.8
    (``anthropic.claude-opus-4-8``).  AWS credentials and the region
    (``AWS_REGION``) come from the environment.

    Request settings follow Sec. VI: temperature 0 and top_p 1 where the model
    accepts sampling parameters.  Claude 4.7 and later reject them, so none is
    sent for these models; they run without thinking unless it is requested.
    Messages may carry OpenAI-style content parts (text and image_url)."""

    def __init__(
        self,
        model_id: str = "anthropic.claude-opus-4-8",
        region: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ):
        self._model_id = model_id
        self._region = region or os.environ.get("AWS_REGION") \
            or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = None
        self.usage = TokenUsage()

    @property
    def provider_name(self) -> str:
        return f"bedrock ({self._model_id})"

    @staticmethod
    def _content(content: Any) -> List[Dict[str, Any]]:
        """Converse content blocks of a message content (a string, or
        OpenAI-style parts: text and image_url with a data or http URL)."""
        if isinstance(content, str):
            return [{"text": content}]
        blocks: List[Dict[str, Any]] = []
        for part in content or []:
            if isinstance(part, str):
                blocks.append({"text": part})
            elif part.get("type") == "text":
                blocks.append({"text": str(part.get("text", ""))})
            elif part.get("type") == "image_url":
                url = part["image_url"]["url"] if isinstance(part.get("image_url"), dict) \
                    else part.get("image_url")
                if str(url).startswith("data:"):
                    import base64
                    header, data = str(url).split(",", 1)
                    raw, mime = base64.b64decode(data), header[5:].split(";")[0]
                else:
                    with urllib.request.urlopen(str(url), timeout=60) as resp:
                        raw, mime = resp.read(), resp.headers.get_content_type()
                fmt = {"jpg": "jpeg"}.get(mime.split("/")[-1], mime.split("/")[-1])
                blocks.append({"image": {"format": fmt if fmt in ("jpeg", "png", "gif", "webp")
                                         else "jpeg", "source": {"bytes": raw}}})
        return blocks or [{"text": ""}]

    def _request(self, messages: List[Dict[str, Any]],
                 max_tokens: Optional[int] = None) -> Dict[str, Any]:
        request_options, _ = _request_rules()
        inference: Dict[str, Any] = {"maxTokens": int(max_tokens or self._max_tokens)}
        if "temperature" in request_options(self._model_id, ""):
            # top_p is left at its default of 1 (Sec. VI)
            inference["temperature"] = self._temperature
        system = [m for m in messages if m["role"] in ("system", "developer")]
        kwargs: Dict[str, Any] = {
            "modelId": self._model_id,
            "messages": [{"role": m["role"], "content": self._content(m["content"])}
                         for m in messages if m["role"] in ("user", "assistant")],
            "inferenceConfig": inference,
        }
        if system:
            kwargs["system"] = [{"text": "\n".join(
                b["text"] for m in system for b in self._content(m["content"]) if "text" in b)}]
        return kwargs

    def complete_with_usage(self, messages: List[Dict[str, Any]],
                            max_tokens: Optional[int] = None):
        """(reply text, {"prompt", "completion"} tokens) of one request; the
        tokens are also added to ``self.usage``."""
        if self._client is None:
            try:
                import boto3
            except ImportError:
                raise RuntimeError("boto3 is required for BedrockProvider: pip install boto3")
            self._client = boto3.client("bedrock-runtime", region_name=self._region)
        try:
            response = self._client.converse(**self._request(messages, max_tokens))
        except Exception as exc:
            raise RuntimeError(f"Bedrock request failed: {exc}") from exc
        u = response.get("usage") or {}
        prompt, completion = u.get("inputTokens", 0), u.get("outputTokens", 0)
        self.usage.add(prompt, completion, u.get("totalTokens", prompt + completion))
        text = "".join(b.get("text", "") for b in response["output"]["message"]["content"])
        return text, {"prompt": prompt, "completion": completion}

    def complete(self, messages: List[Dict[str, Any]]) -> str:
        return self.complete_with_usage(messages)[0]


# ---------------------------------------------------------------------------
# Provider Factory
# ---------------------------------------------------------------------------

PROVIDER_PRESETS: Dict[str, Dict[str, str]] = {
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_key": "DASHSCOPE_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "env_key": "DEEPSEEK_API_KEY",
    },
    # Claude through the Claude API's OpenAI-compatible endpoint, so that
    # TiMMInsight and the baselines (which only speak the OpenAI protocol)
    # call Claude-Opus-4.8 through one and the same endpoint
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "env_key": "ANTHROPIC_API_KEY",
    },
    # any other OpenAI-compatible endpoint: pass base_url (and the key in
    # LLM_API_KEY or api_key_env)
    "custom": {
        "env_key": "LLM_API_KEY",
    },
}


def create_provider(
    provider_type: str,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key_env: Optional[str] = None,
    **kwargs,
) -> LLMProvider:
    """Create an LLM provider by type.

    Args:
        provider_type: "dashscope", "openai", "deepseek", "anthropic",
                       "bedrock", or "custom" for any other OpenAI-compatible
                       endpoint (needs base_url).
        model: Model name (e.g., "qwen-plus", "gpt-4o", "claude-opus-4-8").
        api_key: API key. If None, reads from environment.
        base_url: Override base URL. If None, uses preset.
        api_key_env: environment variable holding the key (overrides the preset's).
    """
    if provider_type == "bedrock":
        return BedrockProvider(model_id=model, **kwargs)

    if provider_type not in PROVIDER_PRESETS:
        raise ValueError(
            f"Unknown provider '{provider_type}'. "
            f"Use one of: {list(PROVIDER_PRESETS.keys())} or 'bedrock'."
        )
    preset = PROVIDER_PRESETS[provider_type]
    resolved_url = base_url or preset.get("base_url")
    if not resolved_url:
        raise ValueError(f"provider '{provider_type}' needs base_url (--base-url)")

    env_key = api_key_env or preset["env_key"]
    resolved_key = api_key or os.environ.get(env_key, "")
    if not resolved_key:
        raise ValueError(
            f"API key not found. Set the {env_key} "
            f"environment variable or pass api_key directly."
        )

    # Temperature 0, top_p 1 and thinking mode OFF for every backbone (Sec. VI)
    # are set per request by OpenAICompatibleProvider.complete; callers may
    # override fields via extra_body=...
    return OpenAICompatibleProvider(
        api_key=resolved_key,
        model=model,
        base_url=resolved_url,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# LLM Query Analyzer
# ---------------------------------------------------------------------------

def binding_problem(op: str, b: Dict[str, Any]) -> Optional[str]:
    """What is missing from the binding of an operator (paper Sec. V-A: B maps
    every operator to its source, attributes, and predicates), or None."""
    def cols(key: str = "columns", exactly: Optional[int] = None) -> bool:
        v = b.get(key)
        return (isinstance(v, list) and bool(v) and all(isinstance(c, str) for c in v)
                and (exactly is None or len(v) == exactly))

    # B maps a binary operator to one source per input (Sec. V-A)
    if op in ("JOIN", "CROSS_MODAL_MATCH", "SIMILARITY_JOIN", "UNION"):
        srcs = b.get("sources")
        if not (isinstance(srcs, list) and len(srcs) >= 2
                and all(isinstance(x, str) and x for x in srcs)):
            return "needs 'sources', one per input"
    if op == "FILTER":
        conds = b.get("conditions") or []
        if not conds and not b.get("semantic"):
            return "needs 'conditions' or 'semantic'"
        if not isinstance(conds, list) or any(
                not isinstance(c, dict) or not c.get("column") or "value" not in c for c in conds):
            return "every condition needs 'column', 'op', 'value'"
    elif op in ("PROJECT", "DISTINCT") and not cols():
        return "needs 'columns'"
    elif op == "AGGREGATE":
        fn = str(b.get("function", "")).lower()
        cond = b.get("condition")
        if cond is not None and (not isinstance(cond, dict) or "value" not in cond):
            return "'condition' must be {\"op\": ..., \"value\": ...}"
        if fn not in {"count", "avg", "mean", "average", "sum", "min", "max"}:
            return "needs 'function' in count/avg/sum/min/max"
        if fn != "count" and not b.get("attribute"):
            return "needs 'attribute'"
    elif op == "SORT" and not b.get("key"):
        return "needs 'key'"
    elif op == "LIMIT":
        v = b.get("count")
        if isinstance(v, str) and v.strip().isdigit():
            v = int(v)
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
            return "needs a positive integer 'count'"
    elif op == "JOIN" and not cols("on", 2):
        # the executor joins on these columns; without them the plan cannot run
        return "needs 'on' as [left column, right column]"
    elif op == "CONTENT_EXTRACT" and b.get("type") not in (None, "boolean", "number", "text"):
        return "'type' must be boolean, number, or text"
    elif op == "CONTENT_EXTRACT" and (not b.get("attribute") or not cols()):
        return "needs 'attribute' and 'columns'"
    elif op == "SEMANTIC_SEARCH" and not b.get("query"):
        return "needs 'query'"
    elif op == "SIMILARITY_JOIN" and not b.get("condition"):
        return "needs 'condition'"
    elif op == "FEATURE_TRANSFORM" and not cols(exactly=1):
        return "needs exactly one column in 'columns'"
    elif op == "VISUALIZE" and not b.get("chart"):
        return "needs 'chart'"
    elif op == "EXPORT" and not b.get("path"):
        return "needs 'path'"
    return None


class LLMQueryAnalyzer:
    """Analyzes natural language queries using an LLM to extract operator requirements."""

    def __init__(
        self,
        provider: LLMProvider,
        ocg: Optional[OperatorCompatibilityGraph] = None,
        custom_descriptions: Optional[Dict[str, str]] = None,
        metadata_context: Optional[str] = None,
        sources: Optional[List[str]] = None,
        source_columns: Optional[Dict[str, Set[str]]] = None,
        source_attributes: Optional[Dict[str, Set[str]]] = None,
        source_values: Optional[Dict[str, Dict[str, Set[str]]]] = None,
        source_tables: Optional[Dict[str, str]] = None,
        source_modalities: Optional[Dict[str, str]] = None,
    ):
        self._provider = provider
        self._source_tables: Dict[str, str] = dict(source_tables or {})
        # modality of each source in the metadata graph ("table", "text",
        # "image"); when not passed, read from the source lines of the
        # metadata context ("Source '<id>' (<modality>, ...")
        self._source_modalities: Dict[str, str] = dict(source_modalities or {}) or dict(
            re.findall(r"Source '([^']+)' \((table|text|image),", metadata_context or ""))
        # string values of each source column (from TiMetaGraph's column
        # statistics); used only to validate equality predicates
        self._source_values = {s: {c: {str(v).casefold() for v in vs} for c, vs in cols.items()}
                               for s, cols in (source_values or {}).items()}
        self._operator_types = list((ocg or OperatorCompatibilityGraph()).all_operator_types)
        self._source_columns: Dict[str, Set[str]] = dict(source_columns or {})
        self._source_attributes: Dict[str, Set[str]] = dict(source_attributes or {})
        self._system_prompt = build_prompt_from_ocg(ocg, custom_descriptions)
        if sources:
            # the raw schema (source and column names), which every system sees
            listed = "; ".join(
                f"{name} (columns: {', '.join(sorted(self._source_columns[name]))})"
                if self._source_columns.get(name) else name for name in sources)
            self._system_prompt += (
                "\n\nAvailable data sources (use these exact source and column names in "
                "\"bindings\"): " + listed + "\n"
            )
        if metadata_context:
            # Ground operator extraction in the metadata graph: knowing which
            # modality holds an attribute decides whether cross-modal
            # operators (CONTENT_EXTRACT, CROSS_MODAL_MATCH, ...) are required
            # or a plain relational FILTER suffices.
            self._system_prompt += (
                "\n\nDataset context (from the metadata graph):\n"
                f"{metadata_context}\n"
                "Use this context to decide which operators are required: an "
                "attribute that resides in images or free text (rather than a "
                "table column) requires the corresponding cross-modal "
                "operator, whereas table-resident attributes need only "
                "relational operators."
            )

    def analyze(self, query_text: str) -> Dict[str, Any]:
        """Send the query to the LLM and return its validated analysis; an
        invalid reply is retried once, then LLMOutputError is raised."""
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": query_text},
        ]
        return complete_json(self._provider, messages, dict, self._validate,
                             "query analysis")

    def _validate(self, result: Dict[str, Any]) -> None:
        """Raise ValueError unless the reply is a well-formed query context."""
        known = set(self._operator_types)
        required = result.get("required_operators")
        if not isinstance(required, list) or not required:
            raise ValueError("'required_operators' must be a non-empty list")
        optional = result.get("optional_operators") or []
        if not isinstance(optional, list):
            raise ValueError("'optional_operators' must be a list")
        unknown = {str(o).upper() for o in required + optional} - known
        if unknown:
            raise ValueError(f"unknown operator types {sorted(unknown)}")
        modalities = result.get("target_modalities")
        valid = {m.value for m in ModalityType}
        if (not isinstance(modalities, list) or not modalities
                or any(m not in valid for m in modalities)):
            raise ValueError(f"'target_modalities' must be a non-empty list over {sorted(valid)}")
        bindings = result.get("bindings") or {}
        if not isinstance(bindings, dict):
            raise ValueError("'bindings' must be an object")
        for op in {str(o).upper() for o in required} - {"SCAN"}:
            b = bindings.get(op, bindings.get(op.lower()))
            if not isinstance(b, dict):
                # CONTENT_EXTRACT / binary operators may be completed by the repairs
                if op in ("CONTENT_EXTRACT",) or op in self.BINARY:
                    continue
                raise ValueError(f"required operator {op} has no binding")
            problem = binding_problem(op, b)
            if problem:
                raise ValueError(f"binding of {op}: {problem}")
        # a sort key is a source column, an extractable attribute, or the
        # name of the aggregate's output
        b_up = {str(k).upper(): v for k, v in bindings.items() if isinstance(v, dict)}
        key = (b_up.get("SORT") or {}).get("key")
        agg = b_up.get("AGGREGATE") or {}
        names = set().union(*self._source_columns.values()) if self._source_columns else set()
        names |= set().union(*self._source_attributes.values()) if self._source_attributes else set()
        names |= {agg.get("as"), agg.get("attribute")}
        if key and agg and names - {None} and key not in names:
            raise ValueError(f"SORT key {key!r} names no column, attribute, or the "
                             "aggregate's 'as' output")
        self._check_against_sources(b_up)

    def _check_against_sources(self, bindings: Dict[str, Dict[str, Any]]) -> None:
        """B refers only to existing sources and columns, and an equality on a
        string column names a value that occurs in it; an EXPORT path names
        its format by extension."""
        if bindings.get("EXPORT"):
            from .applicability import export_implementation
            export_implementation(bindings["EXPORT"].get("path"))
        if not self._source_columns:
            return
        derived = {b.get("attribute") for b in bindings.values()} | {b.get("as") for b in bindings.values()}
        everything = set().union(*self._source_columns.values()) | derived
        everything |= set().union(*self._source_attributes.values()) if self._source_attributes else set()
        # operators over intermediate rows name columns of some source (rows
        # carry plain column names, so "table.column" never matches)
        for op, key in (("AGGREGATE", "group_by"), ("SORT", "key"), ("DISTINCT", "columns"),
                        ("PROJECT", "columns")):
            b = bindings.get(op) or {}
            if b.get("source"):
                continue
            names = b.get(key) or []
            names = [names] if isinstance(names, str) else names
            unknown = [c for c in names if isinstance(c, str) and c not in everything]
            if unknown:
                raise ValueError(f"binding of {op}: {key} {unknown} names no column; use the plain "
                                 "column names of the sources (no table prefix)")
        for op, b in bindings.items():
            for src in [b.get("source")] + list(b.get("sources") or []):
                if src is not None and src not in self._source_columns:
                    raise ValueError(f"binding of {op}: unknown source {src!r}; use one of "
                                     f"{sorted(self._source_columns)}")
            fn = str(b.get("function", "")).lower()
            if op == "AGGREGATE" and fn in ("avg", "mean", "average", "sum", "min", "max") \
                    and not b.get("condition"):
                attr = b.get("attribute")
                srcs = [b["source"]] if b.get("source") else list(self._source_values)
                if any(self._source_values.get(x, {}).get(attr) for x in srcs):
                    raise ValueError(f"binding of AGGREGATE: {fn} over the categorical column "
                                     f"{attr!r}; give a 'condition' to aggregate its 0/1 "
                                     "indicator (e.g. a share of a value)")
            ce = bindings.get("CONTENT_EXTRACT") or {}
            if op == "AGGREGATE" and fn in ("avg", "mean", "average", "sum", "min", "max") \
                    and not b.get("condition") and ce.get("attribute") == b.get("attribute") \
                    and ce.get("type") not in ("boolean", "number"):
                raise ValueError(f"binding of AGGREGATE: {fn} over the extracted attribute "
                                 f"{b.get('attribute')!r}; set CONTENT_EXTRACT 'type' to boolean "
                                 "or number, or give a 'condition'")
            src = b.get("source")
            if src is None:
                continue
            known = self._source_columns[src] | self._source_attributes.get(src, set()) | derived
            cols = list(b.get("columns") or []) if op in ("CONTENT_EXTRACT", "PROJECT", "DISTINCT",
                                                          "FEATURE_TRANSFORM") else []
            cols += [c.get("column") for c in (b.get("conditions") or []) if isinstance(c, dict)]
            missing = [c for c in cols if c and c not in known]
            if missing:
                raise ValueError(f"binding of {op}: columns {missing} are not in {src!r}; its "
                                 f"columns are {sorted(self._source_columns[src])}")
            values = self._source_values.get(src, {})
            for c in b.get("conditions") or []:
                if not isinstance(c, dict) or str(c.get("op", "=")).strip() not in ("=", "=="):
                    continue
                v, col_values = c.get("value"), values.get(c.get("column"))
                if isinstance(v, str) and col_values and v.casefold() not in col_values:
                    raise ValueError(f"binding of {op}: {c['column']!r} never equals {v!r}; use "
                                     "'contains' for part of a value, or one of its values")

    def to_query_context(self, query_text: str) -> QueryContext:
        """Analyze a query and return a QueryContext directly."""
        result = self.analyze(query_text)

        required = {str(o).upper() for o in result["required_operators"]}
        required.add("SCAN")
        optional = {str(o).upper() for o in (result.get("optional_operators") or [])}
        optional -= required

        bindings = {
            str(op).upper(): dict(b) for op, b in (result.get("bindings") or {}).items()
            if isinstance(b, dict)
        }
        root, required = self.consistent_root_and_operators(
            required, bindings, self._source_columns, self._source_attributes,
            self._source_tables)
        for op in required - {"SCAN"}:
            problem = binding_problem(op, bindings.get(op) or {})
            if problem:
                raise LLMOutputError(f"query analysis: binding of {op}: {problem}")

        # an optional operator is explorable only with a complete binding
        optional = {op for op in optional - required
                    if not binding_problem(op, bindings.get(op) or {})}
        # M: the modalities of the sources B binds (from the metadata graph);
        # the analyzer's own list is kept only when the graph gives none
        modalities = self.bound_modalities(bindings, self._source_modalities) \
            or {ModalityType(m) for m in result["target_modalities"]}

        return QueryContext(
            query_text=query_text,
            required_operators=required,
            optional_operators=optional,
            root_operator_type=root,
            target_modalities=modalities,
            bindings=bindings,
        )

    # Top-down order of operator types in the query algebra (cf. the OCG): the
    # root rho is the highest required type, so every other required operator
    # can appear below it.
    ROOT_ORDER = ["VISUALIZE", "EXPORT", "LIMIT", "SORT", "AGGREGATE", "DISTINCT",
                  "PROJECT", "UNION", "FILTER", "JOIN", "CROSS_MODAL_MATCH",
                  "SIMILARITY_JOIN", "CONTENT_EXTRACT", "SEMANTIC_SEARCH",
                  "FEATURE_TRANSFORM", "SCAN"]
    BINARY = {"JOIN", "CROSS_MODAL_MATCH", "SIMILARITY_JOIN", "UNION"}
    MODALITY = {"table": ModalityType.TABULAR, "text": ModalityType.TEXT,
                "image": ModalityType.IMAGE}

    @classmethod
    def bound_modalities(cls, bindings: Dict[str, Dict],
                         source_modalities: Dict[str, str]) -> Set[ModalityType]:
        """M of Phi, consistent with B: the modalities (in the metadata graph)
        of the sources the bindings name; of the only source when B names none.
        Empty when the modalities of the sources are unknown."""
        bound = [s for b in bindings.values()
                 for s in [b.get("source")] + list(b.get("sources") or []) if s]
        if not bound and len(source_modalities) == 1:
            bound = list(source_modalities)
        return {cls.MODALITY[source_modalities[s]] for s in bound
                if source_modalities.get(s) in cls.MODALITY}

    @staticmethod
    def referenced_attributes(bindings: Dict[str, Dict]) -> Set[str]:
        """Every attribute the bindings of the operators refer to: aggregated,
        sorted, grouped, projected, filtered, joined, and charted attributes."""
        used: Set[str] = set()
        for b in bindings.values():
            for key in ("attribute", "key", "x", "y"):
                if isinstance(b.get(key), str):
                    used.add(b[key])
            for key in ("group_by", "columns", "on"):
                v = b.get(key)
                used.update(x for x in ([v] if isinstance(v, str) else v or [])
                            if isinstance(x, str))
            used.update(c.get("column") for c in (b.get("conditions") or [])
                        if isinstance(c, dict) and isinstance(c.get("column"), str))
        return used

    @classmethod
    def consistent_root_and_operators(cls, required: Set[str], bindings: Dict[str, Dict],
                                      source_columns: Dict[str, Set[str]],
                                      source_attributes: Optional[Dict[str, Set[str]]] = None,
                                      source_tables: Optional[Dict[str, str]] = None):
        """Make Phi consistent with the metadata graph, deterministically.

        (i)   an attribute referred to by any binding that is no column of any
              source but is extractable from a text/image source requires
              CONTENT_EXTRACT bound to that source and attribute;
        (ii)  the bound sources are combined: every source is connected to the
              ones before it, by CROSS_MODAL_MATCH when it shares no column with
              them (a JOIN asked for such sources becomes CROSS_MODAL_MATCH) and
              by JOIN otherwise, unless a required binary operator already
              combines it;
        (iii) rho is the highest required type in ROOT_ORDER.

        B is keyed by operator type, so it holds one binding per type: a second
        extractable attribute, or a second binary operator of one type, cannot
        be bound and is reported with a warning.
        """
        required = set(required)
        source_attributes = source_attributes or {}
        all_columns = set().union(*source_columns.values()) if source_columns else set()
        used = cls.referenced_attributes(bindings)
        for attr in sorted(a for a in used if a and a not in all_columns):
            src = next((s for s, attrs in sorted(source_attributes.items()) if attr in attrs), None)
            if not src:
                continue
            bound = (bindings.get("CONTENT_EXTRACT") or {}).get("attribute")
            if "CONTENT_EXTRACT" not in required:
                required.add("CONTENT_EXTRACT")
                bindings["CONTENT_EXTRACT"] = {"source": src, "attribute": attr,
                                               "columns": sorted(source_columns.get(src, ()))}
            elif bound != attr:
                logger.warning(
                    "query context: attribute %r is extractable from %r but CONTENT_EXTRACT "
                    "is already bound to %r; B holds one binding per operator type, so %r "
                    "is left unbound", attr, src, bound, attr)
        # sources are tables: the views of one table (its structured, text and
        # image views) share its records and need no join
        table_of = lambda src: (source_tables or {}).get(src, src)  # noqa: E731
        for op in sorted(required & cls.BINARY):
            srcs = list((bindings.get(op) or {}).get("sources") or [])
            if len(set(srcs)) > 1 and len({table_of(x) for x in srcs}) == 1:
                required.discard(op)
                bindings.pop(op, None)
        sources, seen_tables = [], set()
        for b in bindings.values():
            for s in ([b.get("source")] + list(b.get("sources") or [])):
                if s and table_of(s) not in seen_tables:
                    seen_tables.add(table_of(s))
                    sources.append(s)
        def combines(op: str, src: str) -> bool:
            return table_of(src) in {table_of(x) for x in (bindings.get(op) or {}).get("sources") or []}

        # a required binary operator bound to no sources combines the first
        # source that nothing else combines
        unbound = [op for op in sorted(required & cls.BINARY)
                   if not (bindings.get(op) or {}).get("sources")]
        connected = sources[:1]
        for src in sources[1:]:
            partner = next((c for c in connected
                            if source_columns.get(c, set()) & source_columns.get(src, set())), None)
            need = "JOIN" if partner else "CROSS_MODAL_MATCH"
            pair = [partner or connected[0], src]
            holders = [op for op in sorted(required & cls.BINARY) if combines(op, src)]
            if not holders and unbound:
                holders = [unbound.pop(0)]
            if need == "CROSS_MODAL_MATCH" and "JOIN" in holders:
                # sources that share no column cannot be joined on one
                if "CROSS_MODAL_MATCH" not in required:
                    required.discard("JOIN")
                    required.add("CROSS_MODAL_MATCH")
                    bindings["CROSS_MODAL_MATCH"] = bindings.pop("JOIN", None) or {"sources": pair}
            elif not holders:
                if need in required:
                    logger.warning(
                        "query context: sources %s need a second %s, but B holds one binding "
                        "per operator type; %r is left uncombined", pair, need, src)
                else:
                    required.add(need)
                    bindings.setdefault(need, {"sources": pair})
            connected.append(src)
        root = next((op for op in cls.ROOT_ORDER if op in required), "SCAN")
        return root, required

"""Local OpenAI-compatible endpoint in front of a backbone provider.

The baselines call the backbone through the OpenAI chat-completions protocol
(``model``, ``base_url``, ``api_key``).  A backbone served through another API,
such as Claude-Opus-4.8 on AWS Bedrock (boto3 Converse), has no such endpoint,
so the benchmark starts this proxy on 127.0.0.1: every ``POST
/v1/chat/completions`` is answered by ``provider.complete(messages)``, i.e. by
the same API, model and request settings (Sec. VI) that TiMMInsight uses, and
the provider-reported token usage is returned to the baseline.

Standard library only; runs in the benchmark process.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

API_KEY = "local-proxy"     # the proxy listens on the loopback interface only


class ChatCompletionsProxy:
    """Serves chat completions from ``provider`` (an object with
    ``complete(messages, max_tokens=None)``, ``last_usage()`` and ``_model``)."""

    def __init__(self, provider: Any):
        self._provider = provider
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):       # keep the benchmark log clean
                pass

            def _send(self, status: int, body: Dict[str, Any]) -> None:
                data = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                if not self.path.rstrip("/").endswith("/chat/completions"):
                    return self._send(404, {"error": {"message": f"unknown path {self.path}"}})
                try:
                    req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                except ValueError:
                    return self._send(400, {"error": {"message": "invalid JSON body"}})
                if req.get("stream"):
                    return self._send(400, {"error": {"message": "streaming is not supported"}})
                try:
                    text, usage = proxy.complete(req)
                except Exception as exc:    # noqa: BLE001 - reported to the client
                    return self._send(500, {"error": {"message": str(exc)[:2000],
                                                      "type": type(exc).__name__}})
                self._send(200, {
                    "id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion",
                    "created": int(time.time()), "model": req.get("model") or proxy.model,
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": text}}],
                    "usage": usage})

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    @property
    def model(self) -> str:
        return getattr(self._provider, "_model", None) or getattr(self._provider, "_model_id")

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def complete(self, req: Dict[str, Any]):
        """(reply text, OpenAI usage) of one chat-completions request."""
        limit = req.get("max_completion_tokens") or req.get("max_tokens")
        text, usage = self._provider.complete_with_usage(req.get("messages") or [],
                                                         max_tokens=limit)
        prompt, completion = int(usage.get("prompt", 0)), int(usage.get("completion", 0))
        return text, {"prompt_tokens": prompt, "completion_tokens": completion,
                      "total_tokens": prompt + completion}

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


_PROXIES: Dict[int, ChatCompletionsProxy] = {}
_LOCK = threading.Lock()


def endpoint_for(provider: Any) -> ChatCompletionsProxy:
    """The proxy of ``provider`` (started once per provider, daemon thread)."""
    with _LOCK:
        if id(provider) not in _PROXIES:
            _PROXIES[id(provider)] = ChatCompletionsProxy(provider)
        return _PROXIES[id(provider)]

"""TiMMInsight demo server (FastAPI).

Questions need a DashScope key, from DASHSCOPE_API_KEY or pasted in the page
(kept in this process's memory only).

    GET  /api/config                     key set, models, datasets (built-in and uploaded)
    POST /api/key                        set the key for this server process
    GET  /api/datasets/{name}/preview    tables, text and image columns, rows
    POST /api/uploads                    import tables/images/texts/zip -> upload/<name>
    GET  /api/uploads/{name}             detected (or corrected) column roles + preview
    PUT  /api/uploads/{name}/roles       correct text/image column roles
    POST /api/graphs                     start building TiMetaGraph (or load the cache)
    GET  /api/graphs/{build}/events      build progress (server-sent events)
    POST /api/graphs/{build}/cancel      cancel the build
    POST /api/sessions                   ask a question: first TiQC round (or none)
    POST /api/sessions/{id}/answer       answer a round: rewrite, next round
    POST /api/sessions/{id}/run          TiTSP planning + execution
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from demo import core, uploads

STATIC = Path(__file__).resolve().parent / "static"


class KeyIn(BaseModel):
    key: str


class GraphIn(BaseModel):
    dataset: str


class AskIn(BaseModel):
    dataset: str
    question: str


class AnswerIn(BaseModel):
    answer: str


class RolesIn(BaseModel):
    roles: Dict[str, Dict[str, List[str]]]


def create_app(data_root: Path, providers=core.providers) -> FastAPI:
    """``providers(key) -> (backbone, vision)``: the real DashScope models; tests
    pass their own."""
    app = FastAPI(title="TiMMInsight demo")
    upload_root = data_root / "uploads"
    upload_root.mkdir(parents=True, exist_ok=True)
    state: Dict[str, Any] = {"key": os.environ.get("DASHSCOPE_API_KEY", ""),
                             "graphs": {}, "data": {}, "builds": {}}
    sessions = core.Sessions()

    def live() -> None:
        if not state["key"]:
            raise HTTPException(401, "set a DashScope API key to ask live questions")

    def datasets() -> List[str]:
        return core.BUILTIN_DATASETS + [uploads.dataset_name(n) for n in uploads.uploads(upload_root)]

    def known(name: str) -> str:
        if name in datasets():
            return name
        raise HTTPException(404, f"unknown dataset {name!r}")

    def dataset(name: str, limit: Optional[int], fresh: bool = False):
        key = core.dataset_key(known(name), limit)
        if fresh or key not in state["data"]:
            try:
                state["data"][key] = (uploads.load(upload_root, name.split("/", 1)[1], limit)
                                      if name.startswith("upload/") else core.load(name, limit))
            except FileNotFoundError as exc:
                raise HTTPException(404, f"dataset {name!r} is not available here: {exc}")
        return state["data"][key]

    def graph_of(name: str, limit: Optional[int]):
        key = core.dataset_key(name, limit)
        if key not in state["graphs"]:
            g = core.cached_graph(name, limit, core.graph_caches(data_root))
            if g is None:
                raise HTTPException(409, "build the metadata graph of this dataset first")
            state["graphs"][key] = g
        return state["graphs"][key]

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/config")
    def config():
        return {"has_key": bool(state["key"]),
                "backbone": core.BACKBONE, "vision": core.VISION, "datasets": datasets()}

    @app.post("/api/key")
    def set_key(body: KeyIn):
        key = body.key.strip()
        if not key or not key.isascii() or any(c.isspace() for c in key):
            raise HTTPException(400, "the key must be one token of visible ASCII characters")
        state["key"] = key
        return {"has_key": True}

    @app.get("/api/datasets/{name:path}/preview")
    def dataset_preview(name: str):
        """Every dataset is analysed with all rows of every table."""
        return core.preview(dataset(name, None))

    # ------------------------------------------------------------ user datasets
    @app.post("/api/uploads")
    async def upload(name: str = Form(...), files: List[UploadFile] = File(...)):
        try:
            manifest = uploads.import_files(upload_root, name,
                                            [(f.filename or "", await f.read()) for f in files])
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return {"dataset": uploads.dataset_name(name), "manifest": manifest,
                **_upload_view(name)}

    def _upload_view(name: str) -> Dict[str, Any]:
        data = dataset(uploads.dataset_name(name), None, fresh=True)
        tables = core.preview(data, rows=8)["tables"]
        for t in tables:
            t["all_columns"] = t["columns"]
        return {"tables": tables, "roles": {t["name"]: {"text_columns": t["text_columns"],
                                                       "image_columns": t["image_columns"]}
                                            for t in tables}}

    @app.get("/api/uploads/{name}")
    def upload_info(name: str):
        known(uploads.dataset_name(name))
        return {"dataset": uploads.dataset_name(name), **_upload_view(name)}

    @app.put("/api/uploads/{name}/roles")
    def upload_roles(name: str, body: RolesIn):
        known(uploads.dataset_name(name))
        try:
            uploads.set_roles(upload_root, name, body.roles)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return {"dataset": uploads.dataset_name(name), **_upload_view(name)}

    # ------------------------------------------------------------ TiMetaGraph
    @app.post("/api/graphs")
    def start_graph(body: GraphIn):
        """Start building TiMetaGraph; a cached graph is returned at once."""
        name, limit = known(body.dataset), None          # all rows
        try:
            return {"status": "ready", "graph": core.graph_view(graph_of(name, limit))}
        except HTTPException:
            pass
        live()
        spec = {"dataset": name, "limit": limit, "data_root": str(data_root),
                "upload": name.split("/", 1)[1] if name.startswith("upload/") else None,
                "upload_root": str(upload_root)}
        bid = uuid.uuid4().hex[:12]
        state["builds"][bid] = core.GraphBuild(spec, state["key"]).start()
        return {"status": "building", "build": bid}

    @app.get("/api/graphs/{bid}/events")
    async def graph_events(bid: str):
        build = state["builds"].get(bid)
        if build is None:
            raise HTTPException(404, "unknown build")

        async def stream():
            loop = asyncio.get_running_loop()
            while True:
                try:
                    kind, msg = await loop.run_in_executor(None, build.events.get, True, 5.0)
                except queue.Empty:
                    if not build.process.is_alive() and not build.cancelled:
                        yield _sse({"event": "error", "message": "the build process stopped"})
                        return
                    yield ": keep-alive\n\n"
                    continue
                if kind == "progress":
                    yield _sse({"event": "progress", "message": msg})
                elif kind == "done":
                    g = build.graph()
                    key = core.dataset_key(build.spec["dataset"], build.spec["limit"])
                    state["graphs"][key] = g
                    yield _sse({"event": "graph", "graph": core.graph_view(g)})
                    return
                else:   # error or cancelled
                    yield _sse({"event": kind, "message": msg or "cancelled"})
                    return

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/api/graphs/{bid}/cancel")
    def cancel_graph(bid: str):
        build = state["builds"].get(bid)
        if build is None:
            raise HTTPException(404, "unknown build")
        build.cancel()
        return {"status": "cancelled"}

    @app.get("/api/graphs/view")
    def graph_view(dataset_name: str):
        return core.graph_view(graph_of(known(dataset_name), None))

    # ------------------------------------------------------------ TiQC + TiTSP
    @app.post("/api/sessions")
    def ask(body: AskIn):
        live()
        data = dataset(body.dataset, None)
        graph = graph_of(body.dataset, None)
        llm, vision = providers(state["key"])
        clar = core.Clarification(core.TiQCClarifier(llm, metadata_graph=graph),
                                  body.question, body.question)
        s = sessions.create(dataset=body.dataset, limit=None, data=data, graph=graph,
                            llm=llm, vision=vision, clarification=clar)
        return {"session": s.id, "query": clar.query, "round": _system(clar.next_round)}

    def _get(sid: str):
        try:
            return sessions.get(sid)
        except KeyError:
            raise HTTPException(404, "unknown session")

    @app.get("/api/sessions/{sid}")
    def session(sid: str):
        """The session's transcript, so a reloaded page can restore it."""
        s = _get(sid)
        c = s.clarification
        open_round = None
        if c.pending is not None:
            a = c.pending
            open_round = {"round": len(c.rounds) + 1, "ambiguity_type": a.ambiguity_type,
                          "severity": a.severity, "description": a.description,
                          "question": a.question, "options": a.options, "detected": []}
        return {"session": sid, "dataset": s.dataset, "question": c.original,
                "query": c.query, "rounds": c.rounds, "round": open_round, "done": c.done,
                "run": s.run}

    @app.post("/api/sessions/{sid}/answer")
    def answer(sid: str, body: AnswerIn):
        live()
        s = _get(sid)
        if s.clarification.pending is None:
            raise HTTPException(409, "no clarification question is open")
        query = _system(lambda: s.clarification.answer(body.answer))
        return {"session": sid, "query": query, "round": _system(s.clarification.next_round)}

    @app.post("/api/sessions/{sid}/run")
    def run(sid: str):
        live()
        s = _get(sid)
        if s.clarification.pending is not None:
            raise HTTPException(409, "answer the open clarification question first")
        s.run = _system(lambda: core.run_query(s.llm, s.vision, s.graph, s.data,
                                               s.clarification.query))
        return {"session": sid, "clarified": s.clarification.query,
                "rounds": s.clarification.rounds, "run": s.run}

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def _system(step):
    """Run a step of the system; an error it raises (e.g. no valid plan, an
    unparsable model reply) goes to the page as 422 with its reason."""
    try:
        return step()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, f"{type(exc).__name__}: {exc}")


def _sse(obj: Dict[str, Any]) -> str:
    return f"data: {json.dumps(obj, default=str)}\n\n"

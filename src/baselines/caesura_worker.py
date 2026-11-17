"""CAESURA baseline worker (Urban & Binnig, CIDR 2024).

Official source: github.com/DataManagementLab/caesura, commit 00ebb18
(00ebb188bf9e83c36ad06e9504594a74f169a23c), cloned at .baselines/src/caesura and
installed into .baselines/venvs/caesura by scripts/setup/caesura.sh.

CAESURA takes the natural-language question directly (no translation step): its
Caesura.run executes discovery -> planning -> mapping -> runner phases over a
caesura.database.Database.  The worker
  * builds the Database the way caesura/scenarios.py does: one tabular table per
    CSV (add_tabular_table), TEXT columns marked via Table.text_columns, and for
    each image column a separate image table (add_image_table) linked with
    link_image, as in the artwork scenario;
  * points CAESURA's LLM (langchain 0.0.197 ChatOpenAI on openai==0.28) at the
    request's OpenAI-compatible endpoint via the OPENAI_API_BASE/OPENAI_API_KEY
    environment variables that ChatOpenAI reads, and swaps caesura.main.MyOpenAI
    for a subclass that adds ``enable_thinking: False`` (DashScope) and counts
    token usage.  CAESURA's per-model tables (retries, token limits, rate limits)
    are keyed by GPT model names; the request's model is registered with the
    gpt-4-0613 settings.  The LangChain SQLite response cache CAESURA enables at
    import is disabled so every run hits the LLM.
CAESURA's own local models (BART-large QA for text, BLIP for images) are used
unchanged for TEXT/IMAGE columns.  Usage: JSON request on stdin (see common.py).
"""
from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
import time
import contextlib
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.baselines import common  # noqa: E402

REF_MODEL = "gpt-4-0613"


def _configure(cfg: common.LLMConfig, usage: common.Usage):
    os.environ["OPENAI_API_BASE"] = cfg.base_url
    os.environ["OPENAI_API_KEY"] = cfg.api_key
    import openai
    openai.api_base, openai.api_key = cfg.base_url, cfg.api_key
    import langchain
    import caesura.main as cmain
    import caesura.model as cmodel
    langchain.llm_cache = None  # caesura.main enables an on-disk response cache
    for table in (cmodel.MAX_RPM, cmodel.MAX_TPM, cmodel.MAX_NUM_TOKENS_SOFT,
                  cmodel.MAX_NUM_TOKENS_HARD, cmain.MAX_NUM_RETRIES, cmain.MAX_NUM_ERRORS):
        table[cfg.model] = table[REF_MODEL]
    extra = common.extra_options(cfg.model, cfg.base_url)   # thinking off (Sec. VI)
    fixed = common.fixed_sampling(cfg.model)                # the model rejects sampling fields

    class CountingOpenAI(cmodel.MyOpenAI):
        def __init__(self, **kw):
            kw.setdefault("model_kwargs", dict(extra))
            kw.setdefault("openai_api_base", cfg.base_url)
            kw.setdefault("openai_api_key", cfg.api_key)
            super().__init__(**kw)

        def _generate(self, *args, **kwargs):
            result = super()._generate(*args, **kwargs)
            usage.add((result.llm_output or {}).get("token_usage") or {})
            return result

        @property
        def _default_params(self):
            params = dict(super()._default_params)
            if fixed:
                params.pop("temperature", None)
                params.pop("top_p", None)
            return params

        def get_num_tokens(self, text):  # tiktoken has no encoding for non-OpenAI names
            import tiktoken
            return len(tiktoken.get_encoding("cl100k_base").encode(text))

    cmain.MyOpenAI = CountingOpenAI
    return cmain


def _build_database(tables: Dict[str, Dict[str, Any]]):
    from caesura.database.database import Database
    db = Database()
    for name, t in tables.items():
        path = Path(t["path"]).resolve()
        img_cols = list(t.get("image_columns", []))
        db.add_tabular_table(name, str(path), t.get("description", ""), path_columns=tuple(img_cols))
        db._tables[name].text_columns = tuple(t.get("text_columns", []))
        # empty CSV cells come back as NaN; CAESURA's text QA tokenizer needs strings
        frame = db._tables[name].data_frame
        for col in frame.columns:
            if frame[col].dtype == object:
                frame[col] = frame[col].fillna("")
        for col in img_cols:
            # CAESURA's own layout (artwork scenario): the tabular table keeps
            # the image path, and an image table holds the IMAGE column
            # "image" (<IMAGE stored at '...'>) linked on that path.
            files = db.get_column_values(name, col).tolist()
            img_dir = os.path.commonpath([str(Path(f).parent) for f in files])
            img_table = f"{name}_{col}_images"
            path_col = f"{col}_path"
            db._tables[name].data_frame.rename(columns={col: path_col}, inplace=True)
            db.add_image_table(img_table, img_dir, f"images referenced by {name}.{path_col}",
                               file_paths=files)
            db._link(db._tables[name], db._tables[img_table], path_col, "img_path")
    return db


def run(req: Dict[str, Any]) -> Dict[str, Any]:
    cfg = common.LLMConfig.from_request(req)
    usage = common.Usage()
    errors = []
    workdir = tempfile.mkdtemp(prefix="caesura_")
    cwd = os.getcwd()
    os.chdir(workdir)  # CAESURA writes .prompts/, chroma and log files to cwd
    log = io.StringIO()
    t0 = time.time()
    captured = {}
    try:
        with contextlib.redirect_stdout(log):
            cmain = _configure(cfg, usage)
            db = _build_database(req["tables"])
            setup_sec = time.time() - t0
            agent = cmain.Caesura(db, model_name=cfg.model, interactive=False,
                                  log_path=Path(workdir) / "log")
            orig = agent.log_final_plan

            def capture(query, final_plan, final_result):
                captured["plan"] = final_plan.final_format(query)
                captured["result"] = final_result
                orig(query, final_plan, final_result)
            agent.log_final_plan = capture
            orig_restart = agent.restart_after_error

            def on_error(e):
                errors.append(f"{type(e).__name__}: {e}"[:2000])
                return orig_restart(e)
            agent.restart_after_error = on_error
            t1 = time.time()
            agent.run(req["question"])
            exec_sec = time.time() - t1
        result = captured.get("result")
        rows = common.to_records(result.data_frame) if result is not None else []
    except Exception as exc:  # charge the tokens already spent
        exc.usage = usage.as_dict()
        raise
    finally:
        os.chdir(cwd)
        shutil.rmtree(workdir, ignore_errors=True)
    ok = "plan" in captured and result is not None
    if not ok and not errors:
        errors.append("CAESURA produced no final result")
    return {"system": "caesura", "rows": rows, "program": captured.get("plan", ""), "attempts": 1,
            "errors": errors, "translation_tokens": {"prompt": 0, "completion": 0},
            "execution_tokens": usage.as_dict(), "translation_seconds": 0.0,
            "execution_seconds": round(exec_sec, 3), "setup_seconds": round(setup_sec, 3),
            "ok": ok}

if __name__ == "__main__":
    common.main(run, "caesura")

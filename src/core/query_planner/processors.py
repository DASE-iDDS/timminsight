"""TiProcessor family — LLM/VLM-driven metadata extraction under ReAct (paper Sec. III-A).

BaseProcessor runs a bounded ReAct loop (Thought -> Action(tool) -> Observation -> ...
-> Final) in which the LLM decides which external specialized tool to invoke. The three
concrete processors extract per-modality metadata into the unified graph, following the
paper's staging exactly:

  TableProcessor  3-stage: (1) LLM column characterization over a sampled scan +
                  full-column statistics (distinct-value set, most-common values);
                  (2) the LLM designates PK/FK from the characterizations, and entity
                  values are materialized; (3) coarse-to-fine table relationships.
  TextProcessor   3-stage: (1) split text into paragraphs; (2) ReAct PARALLEL extraction
                  per chunk, the LLM invoking NER (spaCy/Transformers/GLiNER), sentiment
                  (VADER) and topic (KeyBERT) tools and consolidating their outputs;
                  (3) merge across paragraphs per document.
  ImageProcessor  2-stage: (1) group images into batches; (2) PARALLEL extraction per
                  batch, the LLM invoking object detection (YOLO/Detectron2/MMDetection), scene
                  classification (Places365/timm), semantic features (timm ViT) and
                  OpenCV tools under ReAct.

Extracted entities become CrossModalEntity + EntityMention carrying the exact per-modality
signals evidence mass omega(e, m) of Eq. 2 needs (text word_count, image box area/saliency,
table row). Real inputs only; each tool is honestly marked available/unavailable.
"""
from __future__ import annotations
import logging
import os
import threading
import hashlib
import json
import random
import re
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence

from .llm_query_analyzer import complete_json
from .metadata_graph import (
    MultimodalMetadataGraph, ModalityMetadata, ModalityType, EntityType,
    CrossModalEntity, EntityMention, CrossModalRelation, RelationType, doc_key,
    TemporalInfo, SpatialInfo,
)
from . import extraction_tools as ET

logger = logging.getLogger(__name__)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def entity_id(*parts: Any, limit: int = 120) -> str:
    """Entity id from its parts; long ids keep a readable prefix and a hash of
    the full id, so two distinct entities never collide."""
    full = ":".join(str(p) for p in parts)
    if len(full) <= limit:
        return full
    return f"{full[:limit - 17]}#{hashlib.sha1(full.encode('utf-8')).hexdigest()[:16]}"


class ToolUnavailableError(RuntimeError):
    """A metadata-extraction tool the processor requires cannot be loaded."""


def _require(tools: Dict[str, Any], *names: str) -> List[Any]:
    out = []
    for n in names:
        t = tools.get(n)
        if t is None or not t.available:
            probe = t if t is not None else ET.all_tools().get(n)
            reason = probe._err if probe is not None and not probe.available else None
            raise ToolUnavailableError(
                f"metadata extraction needs the {n} tool"
                + (f" ({reason})" if reason else "") + "; install it (see requirements.txt)")
        out.append(t)
    return out


class _Observed:
    """A tool run on the current item, invoked only if the ReAct agent calls
    it; ``ran``/``value`` record whether it was invoked and its output."""

    def __init__(self, fn: Callable[[], Any], observation: Optional[Callable[[Any], Any]] = None):
        self._fn = fn
        self.ran = False
        self.value: Any = None
        if observation is not None:
            self.observation = observation

    def __call__(self) -> Any:
        if not self.ran:
            self.value = self._fn()
            self.ran = True
        return self.value


class BaseProcessor:
    """ReAct scaffolding shared by all modality processors (paper Figure 2)."""

    def __init__(self, provider: Any, vision_provider: Any = None,
                 tools: Optional[Dict[str, Any]] = None, max_steps: Optional[int] = None,
                 max_workers: int = 4, checkpoint_dir: Optional[str] = None):
        if provider is None:
            raise ValueError("metadata extraction needs the LLM provider")
        self.provider = provider
        self.vision_provider = vision_provider
        self.tools = tools or {}
        # ReAct step budget; None = one step per available tool plus one
        self.max_steps = max_steps
        self.max_workers = max_workers
        self.tool_calls: List[str] = []
        # per-item results are written as they complete, so an interrupted
        # build resumes where it stopped
        self.checkpoint_dir = checkpoint_dir

    def _checkpointed(self, kind: str, key: str, compute: Callable[[], Any]) -> Any:
        """compute() once per item: cached as JSON under checkpoint_dir/kind,
        keyed by the item and the model, so a rerun reuses finished items."""
        if not self.checkpoint_dir:
            return compute()
        import hashlib
        model = getattr(self.provider, "_model", "") or ""
        digest = hashlib.sha1(f"{model}\n{key}".encode("utf-8")).hexdigest()
        path = os.path.join(self.checkpoint_dir, kind, digest[:2], digest + ".json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        value = compute()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, default=str)
        os.replace(tmp, path)
        return value

    def _json(self, system: str, user: str, validate=None, what: str = "metadata extraction") -> Any:
        return complete_json(self.provider, [{"role": "system", "content": system},
                                             {"role": "user", "content": user}],
                             dict, validate, what)

    def _parallel(self, fn: Callable, items: List[Any]) -> List[Any]:
        """Run fn over items; parallel (ReAct extraction is I/O-bound on tool+LLM calls)."""
        if self.max_workers <= 1 or len(items) <= 1:
            return [fn(x) for x in items]
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(items))) as ex:
            return list(ex.map(fn, items))

    def react(self, goal: str, context: str, tool_runners: Dict[str, Callable],
              tool_descs: Optional[Dict[str, str]] = None,
              required: Sequence[Sequence[str]] = ()) -> Dict[str, Any]:
        """Bounded Thought/Action/Observation loop; `tool_runners[name]() -> observation`
        runs the specialized tool on the current item. LLM chooses tools, emits final.
        A tool failure propagates; an unparsable step is retried once, then raised."""
        tool_descs = tool_descs or {}
        tdesc = "\n".join(f"- {n}: {tool_descs.get(n, 'specialized tool')}"
                          for n in tool_runners) or "(no tools available)"
        obs: List[str] = []
        sys = ("You extract structured metadata under the ReAct paradigm. Each turn reply "
               "with STRICT JSON: either {\"thought\": str, \"action\": \"<tool_name>\"} to "
               "call a tool, or {\"thought\": str, \"final\": {...}} to finish with the "
               "structured metadata. Call a tool only when its observation would help.")

        def step_ok(out: Dict[str, Any]) -> None:
            if "final" in out and not isinstance(out["final"], dict):
                raise ValueError("'final' must be an object")
            if "final" not in out and "action" not in out:
                raise ValueError("a step needs 'action' or 'final'")

        # each group of ``required`` names tools of which one must have been
        # called before the agent may finish (the goal's required metadata)
        groups = [[n for n in grp if n in tool_runners] for grp in required]
        groups = [g for g in groups if g]
        called: set = set()
        steps = self.max_steps if self.max_steps is not None else len(tool_runners) + 1
        steps += 2 * len(groups)
        for _ in range(steps):
            prompt = (f"Goal: {goal}\nContext: {context}\nAvailable tools:\n{tdesc}\n"
                      "Observations so far:\n" + ("\n".join(obs) if obs else "(none)") +
                      "\nNext ReAct step as JSON.")
            out = self._json(sys, prompt, step_ok, "ReAct step")
            missing = [g for g in groups if not called & set(g)]
            if "final" in out and missing:
                obs.append("(the goal still needs: " + "; ".join(
                    " or ".join(g) for g in missing) + " -- call them before finishing)")
                continue
            if "final" in out:
                return {**{k: v for k, v in out.items() if k not in ("final", "thought", "action", "action_input")},
                        **out["final"]}
            action = out.get("action")
            if action in tool_runners:
                runner = tool_runners[action]
                result = runner()
                self.tool_calls.append(action)
                called.add(action)
                shown = getattr(runner, "observation", lambda v: v)(result)
                obs.append(f"{action} -> {json.dumps(shown, ensure_ascii=False, default=str)[:1000]}")
            else:
                obs.append(f"({action!r} is no available tool; finish now)")

        def final_ok(out: Dict[str, Any]) -> None:
            if not isinstance(out.get("final"), dict):
                raise ValueError("expected {\"final\": {...}}")

        out = self._json(sys, f"Goal: {goal}\nContext: {context}\nObservations:\n" + "\n".join(obs)
                         + "\nOutput ONLY {\"final\": {...}} with the structured metadata.",
                         final_ok, "ReAct final")
        # fields the model placed beside "final" belong to it
        return {**{k: v for k, v in out.items() if k != "final"}, **out["final"]}


def _as_number(v: str) -> Optional[float]:
    try:
        x = float(v)
    except ValueError:
        return None
    return x if x == x and x not in (float("inf"), float("-inf")) else None


def _as_date(v: str):
    import datetime as _dt
    try:
        return _dt.datetime.fromisoformat(v)
    except ValueError:
        return None


def column_statistics(rows_of_value: Dict[str, List[int]], top: int = 10) -> Dict[str, Any]:
    """Statistics of a full column given the rows holding each of its values:
    the distinct-value set, the most-common values, the number of distinct
    values, and min/max for numeric and date columns."""
    values = list(rows_of_value)
    freq = Counter({v: len(ix) for v, ix in rows_of_value.items()})
    stats: Dict[str, Any] = {"n_distinct": len(values),
                             "most_common": [v for v, _ in freq.most_common(top)]}
    nums = [_as_number(v) for v in values]
    if values and all(x is not None for x in nums):
        order = sorted(range(len(values)), key=lambda i: nums[i])
        stats["distinct_values"] = [values[i] for i in order]
        lo, hi = nums[order[0]], nums[order[-1]]
        stats["min"], stats["max"] = [int(x) if x.is_integer() else x for x in (lo, hi)]
        return stats
    stats["distinct_values"] = sorted(values)
    dates = [_as_date(v) for v in values]
    if values and all(d is not None for d in dates):
        stats["min"], stats["max"] = min(dates).isoformat(), max(dates).isoformat()
    return stats


# Stage 1 of table processing characterizes the columns over a sampled scan:
# SAMPLE_ROWS rows drawn uniformly at random (fixed seed, so a rebuild sees the
# same sample), each value cut to SAMPLE_VALUE_CHARS characters.  Wide tables
# are characterized in several calls of at most SAMPLE_PROMPT_CHARS characters
# of samples, so no column is cut from the prompt.
SAMPLE_ROWS = 8
SAMPLE_SEED = 0
SAMPLE_VALUE_CHARS = 40
SAMPLE_PROMPT_CHARS = 6000
# a categorical column is low-cardinality up to this many distinct values; its
# REPRESENTATIVE_VALUES most-common values are materialized as entities
LOW_CARDINALITY = 40
REPRESENTATIVE_VALUES = 6

_LOCATION_NAME = re.compile(
    r"(^|[^a-z])(lat|latitude|lon|lng|long|longitude|city|town|country|state|province|"
    r"county|region|address|location|zip|zipcode|postcode|postal)([^a-z]|$)")


def sample_rows(rows: List[Dict[str, Any]], k: int = SAMPLE_ROWS,
                seed: int = SAMPLE_SEED) -> List[Dict[str, Any]]:
    """The sampled scan: k rows drawn uniformly at random, in table order."""
    if len(rows) <= k:
        return list(rows)
    return [rows[i] for i in sorted(random.Random(seed).sample(range(len(rows)), k))]


def _column_batches(sample: Dict[str, List[str]], budget: int) -> List[List[str]]:
    """Columns grouped so the samples of a group fit one prompt of ``budget``
    characters (a column never shares a prompt it does not fit in)."""
    batches: List[List[str]] = [[]]
    used = 0
    for c, values in sample.items():
        size = len(json.dumps({c: values}, ensure_ascii=False))
        if batches[-1] and used + size > budget:
            batches.append([])
            used = 0
        batches[-1].append(c)
        used += size
    return [b for b in batches if b]


def _is_categorical(info: Dict[str, Any]) -> bool:
    return "categorical" in str(info.get("value_distribution") or "").lower()


def _is_temporal(info: Dict[str, Any], stats: Dict[str, Any]) -> bool:
    """A date/time column: characterized as such by the LLM, or holding only
    ISO dates."""
    described = f"{info.get('data_type') or ''} {info.get('value_distribution') or ''}".lower()
    if re.search(r"date|time|year|temporal", described):
        return True
    return isinstance(stats.get("min"), str)        # column_statistics: all values are dates


def _is_spatial(column: str, info: Dict[str, Any]) -> bool:
    """A location column: characterized as such by the LLM, or evidently named
    (latitude/longitude, city, country, address, ...)."""
    if info.get("is_location") is True:
        return True
    described = f"{info.get('data_type') or ''} {info.get('value_distribution') or ''}".lower()
    if re.search(r"geo|location|spatial", described):
        return True
    return bool(_LOCATION_NAME.search(re.sub(r"(?<=[a-z])(?=[A-Z])", "_", str(column)).lower()))


def _as_datetime(v: Any):
    """A column bound as a datetime: an ISO date, or a year."""
    import datetime as _dt
    if isinstance(v, str):
        return _as_date(v)
    if isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 9999:
        return _dt.datetime(v, 1, 1)
    return None


class TableProcessor(BaseProcessor):
    def process(self, dataset: str, rows: List[Dict[str, Any]],
                graph: MultimodalMetadataGraph, image_cols=None, text_cols=None) -> str:
        cols = list(rows[0].keys()) if rows else []
        image_cols, text_cols = image_cols or [], text_cols or []
        struct_cols = [c for c in cols if c not in image_cols and c not in text_cols]
        mid = f"{dataset}_table"

        # Stage 1 (paper Sec. III-A): the column summary.  Over a sampled scan
        # the LLM characterizes each column (data type, value distribution,
        # whether it denotes an entity); over the full column we compute its
        # distinct-value set and its most-common values.
        scan = sample_rows(rows)
        sample = {c: [str(r.get(c, ""))[:SAMPLE_VALUE_CHARS] for r in scan] for c in struct_cols}
        sys = ("Characterize the columns of a relational table from sampled rows. For each "
               "column give data_type (integer/float/string/boolean/date/datetime), a "
               "one-line semantic summary, value_distribution (one of categorical, numeric, "
               "datetime, identifier, high-cardinality, followed by a short description), "
               "is_entity (bool: its values name real-world entities such as titles, people "
               "or products), and is_location (bool: its values are places or coordinates). "
               "Return JSON: {\"columns\": {col: {\"data_type\": str, \"summary\": str, "
               "\"value_distribution\": str, \"is_entity\": bool, \"is_location\": bool}}}.")
        col_info: Dict[str, Dict[str, Any]] = {}
        for batch in _column_batches(sample, SAMPLE_PROMPT_CHARS):
            def columns_ok(info: Dict[str, Any], batch=batch) -> None:
                if not isinstance(info.get("columns"), dict):
                    raise ValueError("'columns' must map every column to its summary")
                missing = [c for c in batch if not isinstance(info["columns"].get(c), dict)]
                if missing:
                    raise ValueError(f"columns {missing[:10]} are not summarized")

            info = self._json(sys, f"Dataset: {dataset}\nColumns+samples: "
                                   + json.dumps({c: sample[c] for c in batch}, ensure_ascii=False),
                              columns_ok, "table column characterization")
            col_info.update({c: info["columns"][c] for c in batch})
        # one pass over the rows: the rows holding each value of each column
        rows_of: Dict[str, Dict[str, List[int]]] = {c: defaultdict(list) for c in struct_cols}
        for i, r in enumerate(rows):
            for c in struct_cols:
                v = str(r.get(c, "")).strip()
                if v:
                    rows_of[c][v].append(i)
        # full-column statistics: distinct-value set and most-common values
        column_stats = {c: column_statistics(rows_of[c]) for c in struct_cols}

        # Stage 2: the table summary.  The LLM designates the primary-key and
        # foreign-key columns from the column characterizations.
        pk, fk = self._designate_keys(dataset, struct_cols, col_info, column_stats, len(rows))

        # ... and the values of entity columns, and the most-common values of
        # low-cardinality categorical columns, become cross-modal entity nodes
        # with one mention per row holding the value.
        entity_cols = [c for c in struct_cols if col_info[c].get("is_entity") is True]
        reps: Dict[str, List[str]] = {}
        for c in struct_cols:
            numeric = isinstance(column_stats[c].get("min"), (int, float))
            if (c not in entity_cols and _is_categorical(col_info[c]) and not numeric
                    and 1 < column_stats[c]["n_distinct"] <= LOW_CARDINALITY):
                reps[c] = column_stats[c]["most_common"][:REPRESENTATIVE_VALUES]
        temporal, spatial = self._dimensions(struct_cols, col_info, column_stats)
        graph.add_modality(ModalityMetadata(
            modality_id=mid, modality_type=ModalityType.TABLE, source_path=dataset,
            content_summary={"columns": cols, "n_rows": len(rows), "column_info": col_info,
                             "column_stats": column_stats},
            semantic_features={"representative_values": reps},
            temporal_info=temporal, spatial_info=spatial,
            raw_metadata={"image_columns": image_cols, "text_columns": text_cols,
                          "primary_key_columns": pk, "foreign_key_columns": fk,
                          "entity_columns": entity_cols}))
        to_materialize: Dict[str, List[str]] = dict(reps)
        for c in entity_cols:
            to_materialize[c] = sorted(rows_of[c])
        for c, vals in to_materialize.items():
            for v in vals:
                mentions = [EntityMention(mid, ModalityType.TABLE, v, confidence=1.0,
                                          position={"row": i, "column": c}, context=c)
                            for i in rows_of[c][v]]
                graph.add_entity(CrossModalEntity(
                    entity_id=entity_id(dataset, c, v), entity_type=EntityType.CONCEPT,
                    canonical_name=v, mentions=mentions, properties={"column": c}))
        return mid

    def _designate_keys(self, dataset: str, struct_cols: List[str],
                        col_info: Dict[str, Dict[str, Any]], column_stats: Dict[str, Any],
                        n_rows: int) -> Any:
        """Stage 2: primary-key and foreign-key columns, designated by the LLM
        from the column characterizations (never from row samples)."""
        if not struct_cols:
            return [], []
        characterizations = {
            c: {"data_type": col_info[c].get("data_type"), "summary": col_info[c].get("summary"),
                "value_distribution": col_info[c].get("value_distribution"),
                "is_entity": col_info[c].get("is_entity"),
                "n_distinct": column_stats[c]["n_distinct"]}
            for c in struct_cols}
        sys = ("Designate the key columns of a relational table from its column "
               "characterizations. primary_key_columns: the column(s) that identify a row "
               "(empty if none does). foreign_key_columns: columns that hold the key of "
               "another table (empty if none). Use only the given column names. Return "
               "JSON: {\"primary_key_columns\": [..], \"foreign_key_columns\": [..]}.")

        def keys_ok(out: Dict[str, Any]) -> None:
            for key in ("primary_key_columns", "foreign_key_columns"):
                if not isinstance(out.get(key), list):
                    raise ValueError(f"'{key}' must be a list")
                unknown = [c for c in out[key] if c not in struct_cols]
                if unknown:
                    raise ValueError(f"{key}: {unknown[:10]} are no columns of the table")

        out = self._json(sys, f"Dataset: {dataset} ({n_rows} rows)\nColumn characterizations: "
                              + json.dumps(characterizations, ensure_ascii=False),
                         keys_ok, "table key designation")
        return list(out["primary_key_columns"]), list(out["foreign_key_columns"])

    @staticmethod
    def _dimensions(struct_cols: List[str], col_info: Dict[str, Dict[str, Any]],
                    column_stats: Dict[str, Any]) -> Any:
        """Optional temporal and spatial information of the table (paper
        Sec. III-A): its date/time columns with their value ranges, and its
        location columns with their most-common values."""
        t_cols: Dict[str, Dict[str, Any]] = {}
        bounds = []
        for c in struct_cols:
            st = column_stats[c]
            if "min" in st and _is_temporal(col_info[c], st):
                t_cols[c] = {"min": st["min"], "max": st["max"]}
                bounds += [d for d in (_as_datetime(st["min"]), _as_datetime(st["max"])) if d]
        temporal = TemporalInfo(
            start_time=min(bounds) if bounds else None, end_time=max(bounds) if bounds else None,
            time_expressions=[f"{c}: {r['min']} .. {r['max']}" for c, r in t_cols.items()],
            columns=t_cols) if t_cols else None
        s_cols = [c for c in struct_cols if c not in t_cols and _is_spatial(c, col_info[c])]
        places = [v for c in s_cols if not isinstance(column_stats[c].get("min"), (int, float))
                  for v in column_stats[c]["most_common"]]
        spatial = SpatialInfo(locations=list(dict.fromkeys(places)), columns=s_cols) if s_cols else None
        return temporal, spatial


_SENT = {"positive": "positive", "negative": "negative", "neutral": "neutral",
         "mixed": "mixed", "pos": "positive", "neg": "negative"}


def _split_paragraphs(text: str, max_words: int = 120) -> List[str]:
    """Stage 1 of text processing: split into paragraphs, further windowing long ones."""
    paras = [p.strip() for p in re.split(r"\n\s*\n|\r\n\r\n", text) if p.strip()]
    if not paras:
        paras = [text.strip()] if text.strip() else []
    out: List[str] = []
    for p in paras:
        words = p.split()
        if len(words) <= max_words:
            out.append(p)
        else:  # window long paragraphs by sentence into ~max_words chunks
            sents, buf, n = re.split(r"(?<=[.!?])\s+", p), [], 0
            for s in sents:
                buf.append(s); n += len(s.split())
                if n >= max_words:
                    out.append(" ".join(buf)); buf, n = [], 0
            if buf:
                out.append(" ".join(buf))
    return out


def document_text(row: Dict[str, Any], text_cols: List[str]) -> str:
    """The document a row holds: its text columns in sorted column order (the
    executor builds the same string to look documents up by ``doc_key``)."""
    return " ".join(str(row.get(c) or "") for c in sorted(text_cols)).strip()


TEXT_NER_TOOLS = ("spacy", "transformers_ner", "gliner")
TEXT_GOAL = ("Extract the text's required metadata: (1) named entities, including titles of "
             "works and products even if abbreviated, (2) sentiment, and (3) topics. Obtain "
             "each with the specialized tools: call a NER tool, the sentiment tool, and the "
             "topic tool. Then consolidate their outputs; in the final answer give "
             "{\"entities\": [{\"text\": str, \"label\": spaCy-style label, \"full_name\": "
             "str}], \"sentiment\": \"positive|negative|neutral|mixed\", \"topics\": [str]}, "
             "adding entities the tools missed or split. \"text\" is the mention exactly as "
             "written; give \"full_name\" only for an abbreviated or shortened mention whose "
             "full name the text itself states.")
# NER labels summarized as the temporal / spatial information of a text source
TEMPORAL_LABELS = ("DATE", "TIME")
SPATIAL_LABELS = ("GPE", "LOC", "FAC")
DIMENSION_VALUES = 50


def _overlaps(span: Any, spans: List[Any]) -> bool:
    return any(span[0] < e and s < span[1] for s, e in spans)


def _occurrences(text: str, chunk: str) -> List[Any]:
    """Character spans of every occurrence of ``text`` in ``chunk`` as whole
    words (case-insensitive)."""
    return [(m.start(), m.end()) for m in
            re.finditer(r"(?<!\w)" + re.escape(text.lower()) + r"(?!\w)", chunk.lower())]


def collect_occurrences(chunk: str, tool_outputs: List[List[Dict[str, Any]]],
                        llm_entities: List[Any]) -> List[Dict[str, Any]]:
    """The entity occurrences O(e, m) of one paragraph (paper Eq. 2): every
    occurrence a NER tool reports is kept -- an entity named twice has two
    occurrences -- while a span several tools detect counts once (the first
    tool's).  Entities the LLM adds in its consolidation are located in the
    text, and each occurrence the tools did not cover is added."""
    ents: List[Dict[str, Any]] = []
    spans: List[Any] = []
    by_name: Dict[str, List[Any]] = defaultdict(list)   # spans already counted per entity
    for output in tool_outputs:
        kept = []
        for e in output or []:
            text = str(e.get("text", ""))
            if not _norm(text):
                continue
            if e.get("start") is None or e.get("end") is None:
                kept.append((e, None))          # no offsets: nothing to compare
            elif not _overlaps((e["start"], e["end"]), spans):
                kept.append((e, (e["start"], e["end"])))
        for e, span in kept:                    # a tool never shadows itself
            ents.append(e)
            if span is not None:
                spans.append(span)
                by_name[_norm(e.get("text", ""))].append(span)
    for e in llm_entities or []:
        text = str(e.get("text", "") if isinstance(e, dict) else e).strip()
        if not _norm(text):
            continue
        label = str(e.get("label") or "WORK_OF_ART") if isinstance(e, dict) else "WORK_OF_ART"
        full = str(e.get("full_name") or "").strip() if isinstance(e, dict) else ""
        found = _occurrences(text, chunk)
        if any(e2.get("start") is None for e2 in ents if _norm(e2.get("text", "")) == _norm(text)):
            found = []                          # offsets unknown: already counted by a tool
        else:
            # an occurrence is dropped only when this same entity was already
            # counted there; an entity the tools missed is added even where
            # they found a fragment of it ("Alien" inside "Alien: Romulus")
            found = [sp for sp in found if not _overlaps(sp, by_name[_norm(text)])]
        for start, end in found:
            item = {"text": text, "label": label, "start": start, "end": end}
            if full and _norm(full) != _norm(text):
                item["full_name"] = full
            ents.append(item)
            by_name[_norm(text)].append((start, end))
    return ents
IMAGE_GOAL = ("Extract the image's required metadata: objects, scene, and semantic features. "
              "Obtain them with the specialized tools: call an object-detection tool, a "
              "scene-classification tool, the feature-extraction tool, and the OpenCV tool "
              "for visual properties. In the final answer give {\"description\": str} "
              "describing the salient content.")


class TextProcessor(BaseProcessor):
    def process(self, dataset: str, rows: List[Dict[str, Any]], text_cols: List[str],
                graph: MultimodalMetadataGraph, max_docs: Optional[int] = None) -> Optional[str]:
        text_cols = [c for c in text_cols if rows and c in rows[0]]
        if not text_cols:
            return None
        mid = f"{dataset}_text"
        # dedicated external tools (paper Sec. III-A): NER (spaCy, Transformers,
        # GLiNER), sentiment (VADER) and topics (KeyBERT); the LLM invokes them under ReAct.
        vader, keybert = _require(self.tools, "vader", "keybert")
        ner_tools = [t for t in (self.tools.get(n) for n in TEXT_NER_TOOLS)
                     if t is not None and t.available]
        if not ner_tools:
            _require(self.tools, TEXT_NER_TOOLS[0])            # raises with the reason
        # tools the user registered before query processing: further NER tools
        # join the built-in ones; the others are offered to the agent as well
        extra = ET.registered_tools(self.tools, "text")
        ner_tools = ner_tools + [t for t in extra if t.kind == "ner"]
        extra = [t for t in extra if t.kind != "ner"]
        docs = [document_text(r, text_cols) for r in (rows if max_docs is None else rows[:max_docs])]
        docs = [d for d in docs if d]
        doc_words = [len(d.split()) for d in docs]
        total_words = sum(doc_words)

        # Stage 1: split every doc into paragraphs (chunks), remembering the doc
        chunks: List[str] = []
        chunk_doc: List[int] = []
        for i, d in enumerate(docs):
            parts = _split_paragraphs(d)
            chunks.extend(parts)
            chunk_doc.extend([i] * len(parts))

        # Stage 2: ReAct PARALLEL extraction per chunk; metadata comes from the
        # tools the agent invoked and from its consolidation of their outputs
        def extract_chunk(chunk: str) -> Dict[str, Any]:
            runners = {t.name: _Observed(lambda t=t: t.run(chunk)) for t in ner_tools}
            runners[vader.name] = _Observed(lambda: vader.run(chunk))
            runners[keybert.name] = _Observed(lambda: keybert.run(chunk))
            for t in extra:
                runners[t.name] = _Observed(lambda t=t: t.run(chunk))
            descs = {t.name: t.description for t in ner_tools + [vader, keybert] + extra}
            final = self.react(goal=TEXT_GOAL, context=chunk[:1500],
                               tool_runners=runners, tool_descs=descs,
                               required=[[t.name for t in ner_tools], [vader.name],
                                         [keybert.name]])
            # every occurrence the invoked NER tools report, plus the entities
            # the LLM adds in its consolidation (must occur in the text)
            ents = collect_occurrences(
                chunk, [runners[t.name].value for t in ner_tools if runners[t.name].ran],
                final.get("entities"))
            vs, kb = runners[vader.name], runners[keybert.name]
            label = vs.value["sentiment"] if vs.ran else final.get("sentiment")
            if isinstance(label, dict):
                label = label.get("label")
            topics = ([t["topic"] for t in kb.value] if kb.ran
                      else [str(x) for x in final.get("topics") or [] if str(x).strip()])
            for t in extra:                     # registered topic tools add their topics
                if t.kind == "topic" and runners[t.name].ran:
                    topics += [str(x["topic"] if isinstance(x, dict) else x)
                               for x in runners[t.name].value or []]
            other = {t.name: runners[t.name].value for t in extra
                     if t.kind == "other" and runners[t.name].ran}
            return {"chunk": chunk, "words": len(chunk.split()), "ents": ents,
                    "sentiment": label, "compound": vs.value["compound"] if vs.ran else None,
                    "topics": topics, "other": other,
                    "tools": [n for n, r in runners.items() if r.ran]}

        tool_set = ",".join(sorted([t.name for t in ner_tools + extra]
                                   + [vader.name, keybert.name]))
        results = self._parallel(
            lambda c: self._checkpointed("text", f"{TEXT_GOAL}\n{tool_set}\n{c}",
                                         lambda: extract_chunk(c)), chunks)

        # Stage 3: merge across chunks (aggregate entities / VADER sentiment / KeyBERT topics).
        # Every entity occurrence becomes one mention of its document instance,
        # carrying the document length L for g(o) = 1/L (Eq. 2).  spaCy and the
        # LLM report no per-entity confidence, so c_o = 1.
        ent_agg: Dict[str, Dict[str, Any]] = {}
        topics: Counter = Counter()
        sentiments: Counter = Counter()
        times: Counter = Counter()
        places: Counter = Counter()
        other_outputs: Dict[str, Dict[str, List[Any]]] = defaultdict(lambda: defaultdict(list))
        invoked: Counter = Counter(n for res in results for n in res["tools"])
        for res, di in zip(results, chunk_doc):
            dkey = doc_key(docs[di])
            for name, value in (res.get("other") or {}).items():
                other_outputs[name][dkey].append(value)
            for e in res["ents"]:
                key = _norm(e.get("text", ""))
                if not key:
                    continue
                et = ET.SPACY_LABEL_MAP.get(e.get("label", ""), "CONCEPT")
                d = ent_agg.setdefault(key, {"name": e["text"], "type": et, "occ": [],
                                             "aliases": set()})
                # the full name of an abbreviated mention is registered as an
                # alias when the document itself states it
                full = e.get("full_name")
                if full and full.lower() in docs[di].lower():
                    d["aliases"].add(full)
                if e.get("label") in TEMPORAL_LABELS:
                    times[e["text"]] += 1
                elif e.get("label") in SPATIAL_LABELS:
                    places[e["text"]] += 1
                ctx = res["chunk"][max(0, (e.get("start") or 0) - 40): (e.get("end") or 0) + 40]
                d["occ"].append((dkey, doc_words[di], ctx, float(e.get("confidence", 1.0))))
            for tp in (res["topics"] or [])[:3]:
                topics[str(tp).lower().strip()] += 1
            sv = _SENT.get(str(res["sentiment"] or "").lower().strip())
            if sv:
                sentiments[sv] += 1
        dom_sent = sentiments.most_common(1)[0][0] if sentiments else None

        # Per-document metadata reused at query time (paper Sec. II / V-D):
        # VADER sentiment (word-weighted mean compound over the doc's chunks,
        # in [-1, 1]; only for documents whose every chunk ran VADER) and the
        # entity keys each document mentions.
        doc_comp: Dict[int, List[Any]] = defaultdict(list)
        doc_ent_keys: Dict[int, set] = defaultdict(set)
        for res, di in zip(results, chunk_doc):
            doc_comp[di].append((res["compound"], max(res["words"], 1)))
            for e in res["ents"]:
                if _norm(e.get("text", "")):
                    doc_ent_keys[di].add(_norm(e["text"]))
        doc_sentiment = {
            doc_key(docs[i]): sum(c * w for c, w in v) / sum(w for _, w in v)
            for i, v in doc_comp.items() if v and all(c is not None for c, _ in v)
        }
        doc_entities = {
            doc_key(docs[i]): sorted(entity_id(dataset, "text", k) for k in keys)
            for i, keys in doc_ent_keys.items()
        }

        graph.add_modality(ModalityMetadata(
            modality_id=mid, modality_type=ModalityType.TEXT, source_path=dataset,
            content_summary={"text_columns": sorted(text_cols), "word_count": total_words,
                             "n_docs": len(docs),
                             "n_distinct_docs": len({doc_key(d) for d in docs}),
                             "n_paragraph_chunks": len(chunks),
                             "top_topics": [t for t, _ in topics.most_common(5)],
                             "sentiment": dom_sent,
                             "sentiment_distribution": dict(sentiments),
                             "tool_invocations": dict(invoked)},
            semantic_features={"sentiment_distribution": dict(sentiments)},
            # optional temporal / spatial information (paper Sec. III-A): the
            # time expressions and the places the NER tools found
            temporal_info=TemporalInfo(time_expressions=[
                t for t, _ in times.most_common(DIMENSION_VALUES)]) if times else None,
            spatial_info=SpatialInfo(locations=[
                p for p, _ in places.most_common(DIMENSION_VALUES)]) if places else None,
            raw_metadata={"word_count": total_words,
                          "doc_attributes": {"sentiment": doc_sentiment},
                          "doc_entities": doc_entities,
                          **({"tool_outputs": {n: dict(v) for n, v in other_outputs.items()}}
                             if other_outputs else {})}))
        for key, d in ent_agg.items():
            et = EntityType[d["type"]] if d["type"] in EntityType.__members__ else EntityType.CONCEPT
            mentions = [EntityMention(mid, ModalityType.TEXT, d["name"], confidence=conf,
                                      position={"instance": dk, "words": words}, context=ctx)
                        for dk, words, ctx, conf in d["occ"]]
            properties: Dict[str, Any] = {"source": "text_ner"}
            if d["aliases"]:
                properties["aliases"] = sorted(d["aliases"])
            graph.add_entity(CrossModalEntity(entity_id(dataset, "text", key), et, d["name"],
                                              mentions=mentions, properties=properties))
        return mid


IMAGE_DETECTORS = ("yolo", "detectron2", "mmdetection")
# detections of one object by several detectors: same label, boxes overlapping
# by at least this intersection-over-union
SAME_OBJECT_IOU = 0.5


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / union if union > 0 else 0.0


def distinct_detections(per_detector: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """The object occurrences O(e, m) of one image (paper Eq. 2): every
    detection of a detector is kept, while an object several detectors find
    (same label, overlapping boxes) counts once (the first detector's)."""
    out: List[Dict[str, Any]] = []
    for dets in per_detector:
        kept = [o for o in dets or []
                if not any(_norm(o["label"]) == _norm(k["label"]) and o.get("bbox") and k.get("bbox")
                           and _iou(o["bbox"], k["bbox"]) >= SAME_OBJECT_IOU for k in out)]
        out.extend(kept)
    return out


class ImageProcessor(BaseProcessor):
    def process(self, dataset: str, rows: List[Dict[str, Any]], image_col: str,
                graph: MultimodalMetadataGraph, max_images: Optional[int] = None,
                batch_size: int = 4) -> Optional[str]:
        if not rows or image_col not in rows[0]:
            return None
        mid = f"{dataset}_image"
        # object detection (YOLO, Detectron2), scene classification (Places365,
        # timm), semantic features (timm ViT) and OpenCV, invoked under ReAct
        detectors = [t for t in (self.tools.get(n) for n in IMAGE_DETECTORS)
                     if t is not None and t.available]
        if not detectors:
            _require(self.tools, "yolo")          # raises with the reason
        places, timm_cls, vit, cv = _require(self.tools, "places365", "timm", "vit", "opencv")
        # tools the user registered before query processing: further detectors
        # join the built-in ones; the others are offered to the agent as well
        extra = ET.registered_tools(self.tools, "image")
        detectors = detectors + [t for t in extra if t.kind == "object_detection"]
        extra = [t for t in extra if t.kind != "object_detection"]
        scope = rows if max_images is None else rows[:max_images]
        # instance id of an image = its row index in the source
        images = [(i, r.get(image_col)) for i, r in enumerate(scope) if r.get(image_col)]
        if not images:
            return None

        # Stage 1: group images into batches
        batches = [images[i:i + batch_size] for i in range(0, len(images), batch_size)]

        # Stage 2: PARALLEL extraction per batch; each image is loaded once and
        # metadata comes from the tools the agent invoked
        tool_set = ",".join(sorted([d.name for d in detectors + extra] +
                                   [places.name, timm_cls.name, vit.name, cv.name]))

        def process_batch(batch: List[Any]) -> List[Dict[str, Any]]:
            out = []
            for inst, src in batch:
                raw = ET.image_bytes(src)
                key = f"{IMAGE_GOAL}\n{tool_set}\n{image_col}\n{hashlib.sha1(raw).hexdigest()}"
                item = self._checkpointed("image", key, lambda raw=raw: extract_image(raw))
                out.append({**item, "instance": inst})
            return out

        def extract_image(raw: bytes) -> Dict[str, Any]:
            img = ET.load_image(raw)
            runners = {d.name: _Observed(lambda d=d: d.run(img)) for d in detectors}
            runners[places.name] = _Observed(lambda: places.run(img))
            runners[timm_cls.name] = _Observed(lambda: timm_cls.run(img))
            runners[vit.name] = _Observed(lambda: vit.run(img), vit.observation)
            runners[cv.name] = _Observed(lambda: cv.run(raw))
            for t in extra:
                runners[t.name] = _Observed(lambda t=t: t.run(img))
            descs = {t.name: t.description
                     for t in detectors + [places, timm_cls, vit, cv] + extra}
            final = self.react(goal=IMAGE_GOAL, context=f"image from column {image_col}",
                               tool_runners=runners, tool_descs=descs,
                               required=[[d.name for d in detectors],
                                         [places.name, timm_cls.name], [vit.name]])
            objs = distinct_detections(
                [[{**o, "detector": d.name} for o in runners[d.name].value]
                 for d in detectors if runners[d.name].ran])
            scenes = (list(runners[places.name].value) if runners[places.name].ran else []) + \
                     ([{"scene": c["label"], "confidence": c["confidence"]}
                       for c in runners[timm_cls.name].value] if runners[timm_cls.name].ran else [])
            for t in extra:                     # registered scene classifiers add their scenes
                if t.kind == "scene" and runners[t.name].ran:
                    scenes += list(runners[t.name].value or [])
            other = {t.name: runners[t.name].value for t in extra
                     if t.kind == "other" and runners[t.name].ran}
            return {"objs": objs, "scenes": scenes, "other": other,
                    "features": runners[vit.name].value if runners[vit.name].ran else None,
                    "properties": runners[cv.name].value if runners[cv.name].ran else None,
                    "description": str(final.get("description") or ""),
                    "tools": [n for n, r in runners.items() if r.ran]}

        batch_results = [item for br in self._parallel(process_batch, batches) for item in br]

        # merge across batches: one mention per detection (g(o) = s_o * a_o / A, Eq. 2)
        obj_agg: Dict[str, Dict[str, Any]] = {}
        scenes: Counter = Counter()
        for item in batch_results:
            for o in item["objs"]:
                d = obj_agg.setdefault(_norm(o["label"]), {"name": o["label"], "dets": []})
                d["dets"].append((item["instance"], o))
            for sc in item["scenes"]:
                scenes[sc["scene"]] += 1

        graph.add_modality(ModalityMetadata(
            modality_id=mid, modality_type=ModalityType.IMAGE, source_path=dataset,
            content_summary={"image_column": image_col, "n_images": len(images),
                             "n_batches": len(batches), "batch_size": batch_size,
                             "top_scenes": [sc for sc, _ in scenes.most_common(5)],
                             "objects": sorted(d["name"] for d in obj_agg.values())},
            raw_metadata={"image_descriptions": {str(it["instance"]): it["description"]
                                                 for it in batch_results},
                          "image_features": {str(it["instance"]): it["features"]
                                             for it in batch_results if it["features"] is not None},
                          "image_properties": {str(it["instance"]): it["properties"]
                                               for it in batch_results if it["properties"] is not None},
                          "tool_invocations": dict(Counter(n for it in batch_results
                                                           for n in it["tools"])),
                          **({"tool_outputs": {
                              name: {str(it["instance"]): it["other"][name]
                                     for it in batch_results if name in (it.get("other") or {})}
                              for name in sorted({n for it in batch_results
                                                  for n in (it.get("other") or {})})}}
                             if any(it.get("other") for it in batch_results) else {})}))
        for key, d in obj_agg.items():
            mentions = [EntityMention(mid, ModalityType.IMAGE, d["name"],
                                      confidence=float(o["confidence"]),
                                      position={"instance": inst, "bbox": o.get("bbox"),
                                                "area": float(o["area"]),
                                                "saliency": float(o["saliency"])})
                        for inst, o in d["dets"]]
            graph.add_entity(CrossModalEntity(entity_id(dataset, "image", key), EntityType.OBJECT,
                                              d["name"], mentions=mentions,
                                              properties={"source": "image_detector"}))
        return mid


# ------------------------------------------------------------------ orchestration
def merge_cross_modal_entities(graph: MultimodalMetadataGraph, llm: Any = None,
                               theta_link: float = 0.85, batch_size: int = 20,
                               embed: Optional[Callable[[List[str]], Any]] = None,
                               max_workers: int = 1) -> int:
    """Cross-modal entity resolution (paper Sec. III-A).

    (i)   normalize mention names (case folding, punctuation removal);
    (ii)  merge entities whose normalized names or registered aliases coincide,
          regardless of the (possibly noisy) extractor-assigned entity types;
    (iii) block the remaining entities by type and, within a block, verify the
          pairs whose name embeddings reach cosine similarity theta_link with
          the LLM, in batches, which returns a same-entity decision with
          confidence v;
    (iv)  merge accepted pairs transitively by union-find.  The canonical name
          is the table value when one exists and the most frequent surface form
          otherwise; confidence_score is the mean of c_m * v_m over the mentions,
          where v_m is the verification confidence of the merge that attached m
          (1 for exact matches).
    Step (iii) is skipped when no LLM is given.  Per-document entity lists are
    remapped to the merged ids.  Returns the number of merged groups.
    """
    ents = dict(graph.entities)
    original_mentions = {eid: list(e.mentions) for eid, e in ents.items()}
    parent = {eid: eid for eid in ents}
    # confidence of the link from an entity to its parent; no path compression,
    # so the merge that attached each entity to the final group stays known
    link_conf = {eid: 1.0 for eid in ents}

    def find(x: str) -> str:
        while parent[x] != x:
            x = parent[x]
        return x

    def union(a: str, b: str, v: float) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            link_conf[rb] = v

    def attach_conf(x: str) -> float:
        """v_m: weakest verification on the path from x to its group root."""
        v = 1.0
        while parent[x] != x:
            v = min(v, link_conf[x])
            x = parent[x]
        return v

    def names(e: CrossModalEntity) -> set:
        forms = {e.canonical_name, *(m.mention_text for m in e.mentions),
                 *((e.properties or {}).get("aliases", []))}
        return {_norm(f) for f in forms if _norm(f)}

    # (ii) exact match on normalized names / aliases, across entity types
    by_name: Dict[str, str] = {}
    for eid in ents:
        for n in names(ents[eid]):
            if n in by_name:
                union(by_name[n], eid, 1.0)
            else:
                by_name[n] = eid

    blocks: Dict[Any, List[str]] = defaultdict(list)
    for eid, e in ents.items():
        blocks[e.entity_type].append(eid)

    for etype, ids in blocks.items():
        # (iii) embedding recall + batched LLM verification within a type block
        if llm is None or len(ids) < 2:
            continue
        reps = {}
        for eid in ids:
            reps.setdefault(find(eid), eid)
        roots = list(reps)
        if len(roots) < 2:
            continue
        vecs = (embed or _embed_names)([ents[reps[r]].canonical_name for r in roots])
        candidates = [(reps[roots[i]], reps[roots[j]])
                      for i, j in _similar_pairs(vecs, theta_link)]
        batches = [candidates[k:k + max(1, batch_size)]
                   for k in range(0, len(candidates), max(1, batch_size))]
        if batches:
            logger.info("TiMetaGraph: verifying %d candidate pairs (%s) in %d batches",
                        len(candidates), getattr(etype, "value", etype), len(batches))
        # batches are verified concurrently; the merges are applied in candidate
        # order, so the result equals sequential verification
        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(batches) or 1))) as ex:
            verdicts = list(ex.map(lambda b: _verify_same_entity(llm, ents, b), batches))
        for batch, got in zip(batches, verdicts):
            for (a, b), (same, v) in zip(batch, got):
                if same:
                    union(a, b, v)

    # (iv) materialize groups
    groups: Dict[str, List[str]] = defaultdict(list)
    for eid in ents:
        groups[find(eid)].append(eid)
    remap: Dict[str, str] = {}
    merged = 0
    for root, members in groups.items():
        keep = ents[root]
        for eid in members:
            remap[eid] = root
        if len(members) > 1:
            merged += 1
            for eid in members:
                if eid != root:
                    keep.mentions.extend(original_mentions[eid])
                    graph.entities.pop(eid, None)
        table_names = [m.mention_text for eid in members for m in original_mentions[eid]
                       if m.modality_type == ModalityType.TABLE]
        surface = Counter(m.mention_text for m in keep.mentions)
        if table_names:
            keep.canonical_name = Counter(table_names).most_common(1)[0][0]
        elif surface:
            keep.canonical_name = surface.most_common(1)[0][0]
        weights = [m.confidence * attach_conf(eid) for eid in members for m in original_mentions[eid]]
        keep.confidence_score = sum(weights) / len(weights) if weights else 0.0
        mods = {m.modality_type for m in keep.mentions}
        # the other surface forms of the entity, and the aliases its members
        # registered, are its aliases
        forms = {m.mention_text for m in keep.mentions} | {
            a for eid in members for a in (ents[eid].properties or {}).get("aliases", [])}
        aliases = sorted(f for f in forms if f and f != keep.canonical_name)
        keep.properties = dict(keep.properties or {})
        if aliases:
            keep.properties["aliases"] = aliases
        else:
            keep.properties.pop("aliases", None)
        keep.properties["modalities"] = sorted(m.value for m in mods)
        keep.properties["cross_modal"] = len(mods) > 1
    # keep the graph's indexes consistent with the merged entity set
    for etype, ids in list(graph.entities_by_type.items()):
        graph.entities_by_type[etype] = [e for e in dict.fromkeys(remap.get(i, i) for i in ids)
                                         if e in graph.entities]
    for eid in ents:
        if eid not in graph.entities and graph.entity_graph.has_node(eid):
            graph.entity_graph.remove_node(eid)
    for meta in graph.modalities.values():
        meta.entities = list(dict.fromkeys(remap.get(e, e) for e in meta.entities))
    for meta in graph.modalities.values():
        doc_entities = (meta.raw_metadata or {}).get("doc_entities")
        if doc_entities:
            meta.raw_metadata["doc_entities"] = {
                k: sorted({remap.get(e, e) for e in v}) for k, v in doc_entities.items()}
    return merged


def _cosine(a: Any, b: Any) -> float:
    num = sum(x * y for x, y in zip(a, b))
    den = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
    return num / den if den else 0.0


def _similar_pairs(vecs: Any, theta: float, block: int = 2048) -> List[Any]:
    """The index pairs (i < j) whose cosine similarity is >= theta, in order;
    computed blockwise as normalized matrix products."""
    import numpy as np
    m = np.asarray(vecs, dtype=np.float64)
    if m.ndim != 2 or len(m) < 2:
        return []
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    m = np.divide(m, norms, out=np.zeros_like(m), where=norms > 0)
    pairs = []
    for start in range(0, len(m), block):
        sims = m[start:start + block] @ m.T
        for r, c in zip(*np.nonzero(sims >= theta)):
            i = start + int(r)
            if int(c) > i:
                pairs.append((i, int(c)))
    pairs.sort()
    return pairs


_EMBEDDER_LOCK = threading.Lock()


def _embed_names(texts: List[str]) -> List[List[float]]:
    return [list(map(float, v)) for v in _get_embedder().encode(texts)]


def _verify_same_entity(llm: Any, ents: Dict[str, CrossModalEntity],
                        pairs: List[Any]) -> List[Any]:
    """Ask the LLM whether each candidate pair denotes the same entity."""
    items = []
    for k, (a, b) in enumerate(pairs):
        ea, eb = ents[a], ents[b]
        items.append({"id": k, "a": ea.canonical_name, "b": eb.canonical_name,
                      "a_context": str((ea.mentions[0].context if ea.mentions else "") or "")[:200],
                      "b_context": str((eb.mentions[0].context if eb.mentions else "") or "")[:200]})
    sys = ("Decide whether each pair of mentions refers to the same real-world entity, "
           "using the mentions and their contexts. Return ONLY a JSON list "
           "[{\"id\": int, \"same\": bool, \"confidence\": float}] covering every id.")

    def parse(reply: list) -> Dict[int, Any]:
        got = {}
        for o in reply:
            try:
                k, same, conf = int(o["id"]), o["same"], float(o["confidence"])
            except (KeyError, TypeError, ValueError):
                raise ValueError("every item needs id, same, confidence")
            if not isinstance(same, bool):
                raise ValueError("'same' must be a boolean")
            got[k] = (same, max(0.0, min(1.0, conf)))
        missing = set(range(len(pairs))) - got.keys()
        if missing:
            raise ValueError(f"omitted ids {sorted(missing)[:10]}")
        return got

    reply = complete_json(llm, [{"role": "system", "content": sys},
                                {"role": "user", "content": json.dumps(items, ensure_ascii=False)}],
                          list, parse, "entity verification")
    by_id = parse(reply)
    return [by_id[k] for k in range(len(pairs))]


_EMBEDDER = None


def _get_embedder():
    global _EMBEDDER
    with _EMBEDDER_LOCK:
        if _EMBEDDER is None:
            from sentence_transformers import SentenceTransformer
            _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


def _table_doc(m: ModalityMetadata) -> str:
    """A short textual summary of a table used for coarse-grained similarity search."""
    cols = m.content_summary.get("columns", [])
    info = m.content_summary.get("column_info", {}) or {}
    parts = []
    for c in cols:
        ci = info.get(c, {}) if isinstance(info, dict) else {}
        parts.append(f"{c} ({ci.get('summary') or ci.get('data_type') or ''})".strip())
    return f"Table {m.modality_id} with columns: " + "; ".join(parts)


def _coarse_grained(tables, target_mid: str, similar_count: int) -> List[str]:
    """Stage 1: retrieve the `similar_count` tables whose summaries are most similar to
    the target table by embedding similarity, pruning the O(n^2) pairwise comparison."""
    others = [mid for mid, _ in tables if mid != target_mid]
    if len(others) <= similar_count:
        return others
    import numpy as np
    docs = {mid: _table_doc(m) for mid, m in tables}
    vecs = _get_embedder().encode([docs[target_mid]] + [docs[o] for o in others],
                                  normalize_embeddings=True)
    sims = np.asarray(vecs[1:]) @ np.asarray(vecs[0])
    order = np.argsort(-sims)[:similar_count]
    return [others[i] for i in order]


def discover_table_relations(graph: MultimodalMetadataGraph, llm: Any,
                             similar_count: int = 5) -> List[CrossModalRelation]:
    """Table relationships BETWEEN table modalities via a two-stage coarse-to-fine
    procedure (paper Sec. III-A). Stage 1 (coarse-grained) retrieves the most similar
    candidate tables for each table by embedding similarity, reducing the pairwise cost
    from O(n^2) to O(n). Stage 2 (fine-grained) feeds the target table and its candidates
    to the LLM under a chain-of-thought prompt, which identifies each referential-integrity
    relationship as (referencing table, referenced table, foreign-key column, primary-key
    column, cardinality). No-op on single-table datasets."""
    if llm is None:
        raise ValueError("table relationship discovery needs the LLM")
    tables = [(mid, m) for mid, m in graph.modalities.items()
              if m.modality_type == ModalityType.TABLE]
    if len(tables) < 2:
        return []
    meta = dict(tables)
    rels: List[CrossModalRelation] = []

    for target_mid, tm in tables:
        candidates = _coarse_grained(tables, target_mid, similar_count)   # Stage 1
        if not candidates:
            continue
        # Stage 2: fine-grained LLM CoT over the target + retrieved candidate tables
        def schema(mid):
            m = meta[mid]
            info = m.content_summary.get("column_info") or {}
            stats = m.content_summary.get("column_stats") or {}
            summary = {c: {"data_type": (info.get(c) or {}).get("data_type"),
                           "summary": (info.get(c) or {}).get("summary"),
                           "n_distinct": (stats.get(c) or {}).get("n_distinct"),
                           "most_common": (stats.get(c) or {}).get("most_common", [])[:5]}
                       for c in m.content_summary.get("columns", []) if c in info or c in stats}
            return {"table": mid, "columns": m.content_summary.get("columns", []),
                    "primary_key": m.raw_metadata.get("primary_key_columns", []),
                    "foreign_key_columns": m.raw_metadata.get("foreign_key_columns", []),
                    "column_summaries": summary}
        payload = {"target": schema(target_mid),
                   "candidates": [schema(c) for c in candidates]}
        sys = ("You identify referential-integrity (foreign-key) relationships between "
               "relational tables. Given a TARGET table and CANDIDATE tables with their "
               "columns, primary keys, designated foreign-key columns, and column summaries, "
               "determine which column of the target table is a "
               "foreign key that references a primary/key column of a candidate table. "
               "Reason step by step, then return ONLY JSON {\"relationships\": "
               "[{\"referencing_table\": str, \"referenced_table\": str, \"fk_column\": str, "
               "\"pk_column\": str, \"cardinality\": \"1:1|1:N|N:1|N:M\"}]}, and an empty "
               "list if there is no relationship.")
        def rels_ok(out: Dict[str, Any]) -> None:
            if not isinstance(out.get("relationships"), list) or not all(
                    isinstance(r, dict) for r in out["relationships"]):
                raise ValueError("'relationships' must be a list of objects")

        out = complete_json(llm, [
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            dict, rels_ok, "table relationship discovery")
        for r in out["relationships"]:
            src = r.get("referencing_table") or target_mid
            tgt = r.get("referenced_table")
            if not tgt or tgt not in meta or src not in meta:
                continue
            rels.append(CrossModalRelation(
                relation_id=f"fk_{uuid.uuid4().hex[:8]}",
                relation_type=RelationType.REFERENCES,
                source_modality=src, target_modality=tgt, confidence=1.0,
                evidence={"fk_column": r.get("fk_column"), "pk_column": r.get("pk_column"),
                          "cardinality": r.get("cardinality"),
                          "candidates_searched": len(candidates),
                          "discovery_method": "coarse_to_fine_llm"},
                properties={"relation_category": "table_relationship"}))
    for r in rels:
        graph.add_relation(r)
    return rels


def build_tools(enable: bool = True) -> Dict[str, Any]:
    """The registered tools whose dependencies are installed (paper Fig. 2);
    an unavailable tool is left out and named by _require when it is needed."""
    if not enable:
        return {}
    return {name: t for name, t in ET.all_tools().items() if t.available}


def extract_metadata_graph_tables(tables: Dict[str, List[Dict[str, Any]]], provider: Any,
                                  vision_provider: Any = None, enable_tools: bool = True,
                                  text_cols: Optional[Dict[str, List[str]]] = None,
                                  image_cols: Optional[Dict[str, List[str]]] = None,
                                  max_workers: int = 4, graph_id: str = "meta",
                                  checkpoint_dir: Optional[str] = None) -> MultimodalMetadataGraph:
    """TiMetaGraph over a dataset of several tables (paper Sec. III): every table
    gets the table processor, its text columns the text processor, and its image
    columns the image processor; entities are then resolved across all sources
    and table and cross-modal relationships are discovered."""
    from .metadata_graph import MultimodalMetadataGraph as _G
    from .entity_relationship_discoverer import EntityRelationshipDiscoverer
    g = _G(graph_id=graph_id)
    tools = build_tools(enable_tools)
    kw = {"tools": tools, "max_workers": max_workers, "checkpoint_dir": checkpoint_dir}
    for name, rows in tables.items():
        if not rows:
            continue
        cols = list(rows[0].keys())
        tcols = [c for c in (text_cols or {}).get(name, []) if c in cols]
        icols = [c for c in (image_cols or {}).get(name, []) if c in cols]
        logger.info("TiMetaGraph %s: table processor (%d rows)", name, len(rows))
        TableProcessor(provider, **kw).process(name, rows, g, icols, tcols)
        if tcols:
            logger.info("TiMetaGraph %s: text processor (%s)", name, tcols)
            TextProcessor(provider, **kw).process(name, rows, tcols, g)
        for ic in icols:
            logger.info("TiMetaGraph %s: image processor (%s)", name, ic)
            ImageProcessor(provider, vision_provider, **kw).process(name, rows, ic, g)
    logger.info("TiMetaGraph: entity resolution over %d entities", len(g.entities))
    g.stats["cross_modal_entities_merged"] = merge_cross_modal_entities(g, llm=provider, max_workers=max_workers)
    logger.info("TiMetaGraph: table and entity relationship discovery")
    g.stats["table_fk_relations"] = len(discover_table_relations(g, llm=provider))
    entity_rels = EntityRelationshipDiscoverer().discover_entity_relations(g, llm=provider)
    for r in entity_rels:
        g.add_relation(r)
    g.stats["relation_edges"] = len(entity_rels)
    return g


def extract_metadata_graph(dataset: str, rows: List[Dict[str, Any]], provider: Any,
                           vision_provider: Any = None, enable_tools: bool = True,
                           image_cols=None, text_cols=None,
                           max_workers: int = 4) -> MultimodalMetadataGraph:
    """TiMetaGraph of a single-table dataset (see extract_metadata_graph_tables)."""
    return extract_metadata_graph_tables(
        {dataset: rows}, provider, vision_provider, enable_tools,
        text_cols={dataset: list(text_cols or [])}, image_cols={dataset: list(image_cols or [])},
        max_workers=max_workers, graph_id=f"meta_{dataset}")

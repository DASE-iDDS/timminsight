"""TiQC: interactive query clarification (paper Sec. IV, Algorithm 1).

    q <- q0
    for t <- 1 to K:
        A <- { a in DetectAmbiguities(q, G_meta, L) | sev(a) >= theta_amb }
        if A is empty: break
        a* <- argmax_{a in A} sev(a)
        O <- GenerateOptions(a*, G_meta, L)
        u <- AskUser(a*, O)
        q <- RewriteQuery(q, a*, u, L)
    return q* <- q

DetectAmbiguities takes only the current query q and the metadata graph: it
builds the metadata context of q from the graph and asks the LLM, with a
chain-of-thought prompt, for the ambiguities of the user's intent, each
classified into one of six types and scored with a severity under a fixed
rubric.  An answered ambiguity is not asked again because RewriteQuery states
the user's choice explicitly in q.  AskUser is a human in live use (``console_user``)
or an LLM user simulator conditioned on a hidden intent in batch evaluation
(``make_user_simulator``).  Without a metadata graph no metadata context is
given (the w/o-TiMetaGraph ablation); K = 0 returns q0 unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

from .llm_query_analyzer import complete_json

AMBIGUITY_TYPES = [
    "cross_modal_referential", "intra_modal_attribute", "multimodal_intent",
    "schema_structure", "value_content", "temporal_spatial",
]

# The six ambiguity types with their definitions (Sec. IV, taxonomy), as given
# to the LLM in the detection prompt.
AMBIGUITY_DEFINITIONS = {
    "cross_modal_referential":
        "a mention may resolve to different objects across modalities (e.g. 'positive "
        "reviews' may refer to the rating column of a table or to the sentiment of the "
        "review texts)",
    "intra_modal_attribute":
        "an attribute admits multiple bindings within one modality (e.g. 'best-received' "
        "as the rating column vs the number-of-votes column of the same table)",
    "multimodal_intent":
        "the analytical goal itself admits multiple readings (e.g. list the items vs "
        "compare them vs summarize a trend)",
    "schema_structure":
        "a query term matches no table or column, or several tables and columns",
    "value_content":
        "a literal matches multiple cell values or entities (e.g. 'sci-fi' matching both "
        "the 'Sci-Fi' and the 'Science Fiction' genre labels)",
    "temporal_spatial":
        "an unspecified time range or spatial scope (e.g. 'recent', 'nearby')",
}

# Chain-of-thought steps of the detection prompt (Sec. IV, step 2).
DETECTION_STEPS = (
    "Reason step by step before you answer:\n"
    "Step 1. List the terms and phrases of the query that carry its intent and look each "
    "one up in the metadata context (sources, columns, values, entities, ranges).\n"
    "Step 2. For each term, list the distinct interpretations the data can actually "
    "support; a term with a single supported interpretation is not ambiguous.\n"
    "Step 3. For each term with several interpretations, decide whether they lead to "
    "different downstream plans: different operators, different predicates, or different "
    "modalities.\n"
    "Step 4. Classify every remaining ambiguity into exactly one of the six types.\n"
    "Step 5. Score its severity with the rubric."
)

# Fixed severity rubric of the detection prompt: an ambiguity is severe to the
# extent that its readings diverge in the downstream plan.
SEVERITY_RUBRIC = (
    "Score severity sev in [0,1]: 0.0-0.3 the readings resolve to the same execution "
    "plan (cosmetic); 0.4-0.6 they change predicates or projected columns but stay in "
    "one modality; 0.7-1.0 they require DIFFERENT operators or touch DIFFERENT "
    "modalities (e.g. a value in a table column vs a fact read from an image). Higher "
    "severity = more consequential for planning."
)

# function words of queries, never matched against data values
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "these", "those", "are", "was", "were", "has",
    "have", "had", "how", "many", "much", "what", "which", "who", "whom", "whose", "when",
    "where", "why", "from", "into", "than", "then", "there", "their", "its", "all", "any",
    "each", "most", "more", "less", "least", "best", "top", "recent", "located", "find", "list",
    "show", "give", "count", "number", "average", "total", "per", "about", "over", "under",
    "between", "only", "also", "not", "does", "did", "can", "could", "would", "should",
}

AskUser = Callable[["Ambiguity", List[str]], str]


@dataclass
class Ambiguity:
    """A detected ambiguity a, classified and scored."""
    ambiguity_type: str            # one of AMBIGUITY_TYPES
    severity: float                # sev(a) in [0, 1]
    description: str
    question: str = ""
    options: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"ambiguity_type": self.ambiguity_type, "severity": self.severity,
                "description": self.description, "question": self.question,
                "options": self.options}


class TiQCError(RuntimeError):
    """The LLM output could not be parsed after a retry."""


class TiQCClarifier:
    """Metadata-grounded interactive query clarifier (Algorithm 1)."""

    def __init__(self, provider: Any, metadata_graph: Any = None,
                 theta_amb: float = 0.5, round_budget: int = 3):
        if provider is None:
            raise ValueError("TiQC needs the LLM provider")
        self.provider = provider
        self.metadata_graph = metadata_graph
        self.theta_amb = theta_amb            # severity threshold
        self.round_budget = round_budget      # K
        self._contexts: Dict[str, str] = {}   # metadata context per query

    # ---------------------------------------------------------------- helpers
    def _context(self, query: str) -> str:
        """Metadata context relevant to the query q (Sec. IV, step 1), built
        once per query; empty without a metadata graph."""
        if self.metadata_graph is None:
            return ""
        if query not in self._contexts:
            from .metadata_graph import describe_for_llm
            self._contexts[query] = describe_for_llm(self.metadata_graph, query)
        return self._contexts[query]

    def _evidence(self, query: str, ambiguity_type: str, max_items: int = 12) -> str:
        """The metadata-graph evidence that grounds a clarification of this
        ambiguity type (Sec. IV, taxonomy): CrossModalEntity mentions for
        referential and value ambiguities; ModalityMetadata summaries for schema
        ambiguities, and likewise for the two remaining types (the column
        characterizations of each modality for intra-modal attribute
        ambiguities, the per-modality summaries for multimodal-intent ones);
        and the temporal/spatial dimensions (temporal and spatial info, else the
        value ranges of numeric and date columns) for temporal/spatial
        ambiguities."""
        g = self.metadata_graph
        if g is None:
            return ""
        import re as _re
        from .metadata_graph import _norm_surface, _query_relevance
        qn = _norm_surface(query)
        qt = {_norm_surface(w) for w in _re.findall(r"\w+", query) if len(w) > 2}
        lines: List[str] = []
        if ambiguity_type in ("cross_modal_referential", "value_content"):
            # actual values of the columns that match the query's terms, so that
            # options name values exactly as they occur in the data: whole values
            # (e.g. 'Ajah, Lagos' vs 'Ajah, Lekki, Lagos') and, for multi-label
            # values, the single labels (e.g. 'Sci-fi' in 'Action, Sci-fi')
            def squash(text: str) -> str:
                return _re.sub(r"[^0-9a-z]+", "", str(text).lower())
            # words naming the data's own tables and columns ("movie", "reviews")
            # denote the entities asked about, not values to match
            schema = {squash(x) for mid, meta in g.modalities.items()
                      for x in [mid, meta.source_path] + list((meta.content_summary or {})
                                                               .get("columns") or [])}
            schema |= {x.split("table")[0].split("text")[0].split("image")[0] for x in schema}
            terms = {squash(w) for w in _re.findall(r"[\w-]+", query)
                     if len(squash(w)) > 2 and w.lower() not in _STOPWORDS}
            terms = {t for t in terms
                     if not any(x and (x == t or x.startswith(t) or t.startswith(x)) for x in schema)}

            def rank(value: str) -> int:
                v = squash(value)
                return min((0 if v == t else 1 if v.startswith(t) else 2)
                           for t in terms if t in v)

            # candidate interpretations come from the CrossModalEntity mentions:
            # the table values TiMetaGraph materialized as entities, per column
            pools: Dict[Tuple[str, str], List[str]] = {}
            known = set()
            for e in g.entities.values():
                known.add(squash(e.canonical_name))
                for m in e.mentions:
                    known.add(squash(m.mention_text))
                    col = (m.position or {}).get("column")
                    if col and terms and isinstance(m.mention_text, str):
                        pool = pools.setdefault((m.modality_id, str(col)), [])
                        if m.mention_text not in pool:
                            pool.append(m.mention_text)
            # a column value that no entity stands for (TiMetaGraph materializes
            # only entity columns and the most common categorical values) is
            # read from the column summary of its ModalityMetadata
            for mid, meta in g.modalities.items():
                stats = (meta.content_summary or {}).get("column_stats") or {}
                for col, st in stats.items():
                    for v in ((st or {}).get("distinct_values") or []) if terms else []:
                        if isinstance(v, str) and squash(v) not in known:
                            pools.setdefault((mid, str(col)), []).append(v)
            for (mid, col), vals in pools.items():
                if col.lower() == "id" or col.lower().endswith("_id"):
                    continue
                if sum(v.startswith(("http://", "https://")) for v in vals) > len(vals) / 2:
                    continue        # links, not values
                labels = sorted({x.strip() for v in vals if "," in v for x in v.split(",")
                                 if x.strip()})
                for kind, pool in (("labels", labels), ("values", vals)):
                    hits = sorted((v for v in pool if any(t in squash(v) for t in terms)),
                                  key=lambda v: (rank(v), len(v)))
                    if hits:
                        lines.append(f"{kind.capitalize()} of column '{col}' of '{mid}' "
                                     f"matching the query: "
                                     f"{', '.join(repr(v) for v in hits[:8])}.")
            ents = sorted(g.entities.values(),
                          key=lambda e: -_query_relevance(e.canonical_name, qt, qn))
            for e in ents[:max_items]:
                where = sorted({f"{m.modality_id}" + (f".{(m.position or {}).get('column')}"
                                                      if (m.position or {}).get("column") else "")
                                for m in e.mentions})
                lines.append(f"Entity '{e.canonical_name}' ({e.entity_type.value}) "
                             f"mentioned in {', '.join(where[:6])}.")
        elif ambiguity_type == "schema_structure":
            for mid, meta in g.modalities.items():
                cs = meta.content_summary or {}
                info = cs.get("column_info") or {}
                cols = [f"{c} ({(info.get(c) or {}).get('type', '?')})" for c in cs.get("columns") or []]
                lines.append(f"Source '{mid}' ({meta.modality_type.value}, {cs.get('n_rows', '?')} "
                             f"records)" + (f": columns {', '.join(cols)}" if cols else "") + ".")
        elif ambiguity_type == "intra_modal_attribute":
            # the attributes one modality offers for a binding: the column
            # characterizations of a table's ModalityMetadata summary, and the
            # attributes extractable from a text or image source
            for mid, meta in g.modalities.items():
                cs, raw = meta.content_summary or {}, meta.raw_metadata or {}
                info = cs.get("column_info") or {}
                stats = cs.get("column_stats") or {}
                attrs = []
                for c in cs.get("columns") or []:
                    ci, st = info.get(c) or {}, stats.get(c) or {}
                    parts = [str(x) for x in (ci.get("data_type") or ci.get("type"),
                                              ci.get("summary"), ci.get("value_distribution")) if x]
                    if isinstance(st, dict) and st.get("min") is not None and st.get("max") is not None:
                        parts.append(f"range {st['min']}..{st['max']}")
                    attrs.append(f"{c} ({'; '.join(parts)})" if parts else str(c))
                attrs += [f"{a} (extractable per record)"
                          for a in sorted((raw.get("doc_attributes") or {}).keys())]
                if attrs:
                    lines.append(f"Attributes of '{mid}' ({meta.modality_type.value}): "
                                 f"{', '.join(attrs)}.")
        elif ambiguity_type == "multimodal_intent":
            # what each modality holds: its ModalityMetadata summary
            for mid, meta in g.modalities.items():
                cs, raw = meta.content_summary or {}, meta.raw_metadata or {}
                size = next((cs[k] for k in ("n_rows", "n_docs", "n_images")
                             if isinstance(cs.get(k), int)), "?")
                parts = [f"{size} records"]
                cols = list(cs.get("columns") or []) + list(cs.get("text_columns") or [])
                if cs.get("image_column"):
                    cols.append(cs["image_column"])
                if cols:
                    parts.append(f"columns {', '.join(map(str, cols))}")
                for label, key in (("topics", "top_topics"), ("scenes", "top_scenes"),
                                   ("objects", "objects"), ("dominant sentiment", "dominant_sentiment")):
                    v = cs.get(key)
                    if v:
                        parts.append(f"{label} "
                                     f"{list(v)[:6] if isinstance(v, (list, tuple, set, dict)) else v}")
                attrs = sorted((raw.get("doc_attributes") or {}).keys())
                if attrs:
                    parts.append(f"attributes extractable per record: {', '.join(attrs)}")
                lines.append(f"Source '{mid}' ({meta.modality_type.value}): {'; '.join(parts)}.")
        elif ambiguity_type == "temporal_spatial":
            # the temporal and spatial dimensions of each ModalityMetadata; a
            # source without them falls back to the value ranges of its columns
            def dimension(info: Any) -> Dict[str, Any]:
                return {k: v for k, v in (getattr(info, "__dict__", None) or {}).items()
                        if v not in (None, [], "", ())}
            for mid, meta in g.modalities.items():
                temporal, spatial = dimension(meta.temporal_info), dimension(meta.spatial_info)
                if temporal:
                    lines.append(f"Temporal info of '{mid}': {temporal}.")
                if spatial:
                    lines.append(f"Spatial info of '{mid}': {spatial}.")
                if temporal or spatial:
                    continue
                stats = (meta.content_summary or {}).get("column_stats") or {}
                for col, st in stats.items():
                    if isinstance(st, dict) and st.get("min") is not None and st.get("max") is not None:
                        lines.append(f"Column '{col}' of '{mid}' ranges from {st['min']} to {st['max']}.")
        return "\n".join(lines) or self._context(query)

    def _json(self, system: str, user: str, expect: type, validate=None) -> Any:
        return complete_json(self.provider, [{"role": "system", "content": system},
                                             {"role": "user", "content": user}],
                             expect, validate, "TiQC", TiQCError)

    # ---------------------------------------------------------------- Algorithm 1
    def detect_ambiguities(self, query: str) -> List[Ambiguity]:
        """Line 4, DetectAmbiguities(q, G_meta, L): ambiguities of the user's
        intent in the current query q, with type and severity.  Step 1 builds
        the metadata context of q; step 2 is the chain-of-thought analysis."""
        ctx = self._context(query)
        ctx_block = f"Metadata context:\n{ctx}\n" if ctx else "(No metadata context.)\n"
        types = "\n".join(f"- {t}: {AMBIGUITY_DEFINITIONS[t]}" for t in AMBIGUITY_TYPES)
        system = (
            "You are TiQC, the query-clarification module of a multimodal data-analytics "
            "system. Given a natural-language query and the dataset's metadata context, "
            "DETECT every consequential ambiguity in the user's "
            "INTENT, i.e. readings of the question that lead to different answers. Never "
            "report how the system should match, join, or process the data; those are not "
            "questions for the user. Do not report an ambiguity the question already "
            "resolves (a reading the question states explicitly is settled), or one whose "
            "readings the metadata shows would give the same answer "
            "(e.g. a term that matches exactly one column value).\n\n"
            f"The six ambiguity types:\n{types}\n\n"
            f"{DETECTION_STEPS}\n\n"
            f"{SEVERITY_RUBRIC}\n\n"
            "Return JSON: {\"reasoning\": [str, ...] (one short sentence per step), "
            "\"ambiguities\": [{\"ambiguity_type\": str, \"severity\": float, "
            "\"description\": str}, ...]}; \"ambiguities\" is [] if the query is "
            "unambiguous given the metadata.")
        def parse(reply: Any) -> List[Ambiguity]:
            found = reply.get("ambiguities") if isinstance(reply, dict) else reply
            if not isinstance(found, list):
                raise ValueError("'ambiguities' must be a list")
            out: List[Ambiguity] = []
            for a in found:
                if not isinstance(a, dict) or a.get("ambiguity_type") not in AMBIGUITY_TYPES:
                    raise ValueError(f"each ambiguity needs one of the six types, got {a!r}"[:200])
                try:
                    sev = float(a["severity"])
                except (KeyError, TypeError, ValueError):
                    raise ValueError("each ambiguity needs a numeric severity")
                out.append(Ambiguity(a["ambiguity_type"], max(0.0, min(1.0, sev)),
                                     str(a.get("description", ""))))
            return out

        # the reasoning precedes the ambiguities in one JSON object (a bare
        # list of ambiguities is accepted as well)
        found = self._json(system, f"{ctx_block}Query: {query}", (dict, list), parse)
        return sorted(parse(found), key=lambda x: -x.severity)

    def generate_options(self, query: str, amb: Ambiguity) -> List[str]:
        """Line 7: the clarifying question for a* and options grounded in the
        metadata-graph evidence of its type (interpretations the data can support)."""
        ctx = self._evidence(query, amb.ambiguity_type)
        system = (
            "You are TiQC. For the given ambiguity, write ONE clarifying question and 2-3 "
            "concrete, mutually exclusive options"
            + (", drawing options STRICTLY from the metadata evidence; never invent a "
               "column or value that is not present, and write values exactly as they "
               "appear in the evidence" if ctx else "")
            + ". For image content, the system judges any visual property with a vision "
              "model, so options need not be limited to detected object labels"
            + ". Phrase the question and options in the user's domain terms (e.g. 'the "
              "movie's rating' vs 'the sentiment of its reviews'), never with internal source "
              "or table identifiers"
            + '. Return JSON: {"question": str, "options": [str, ...]}.')
        def options_ok(out: Dict[str, Any]) -> None:
            if not str(out.get("question") or "").strip():
                raise ValueError("missing 'question'")
            if not isinstance(out.get("options"), list) or len(out["options"]) < 2:
                raise ValueError("'options' must list at least two interpretations")

        out = self._json(system, (f"Metadata evidence:\n{ctx}\n" if ctx else "")
                         + f"Query: {query}\nAmbiguity ({amb.ambiguity_type}): {amb.description}",
                         dict, options_ok)
        amb.question = str(out["question"]).strip()
        amb.options = [str(o) for o in out["options"]]
        return amb.options

    def rewrite_query(self, query: str, amb: Ambiguity, answer: str) -> str:
        """Line 9: the question with the ambiguity resolved as the user answered.
        The chosen reading is written into the question itself, so that the next
        DetectAmbiguities(q, ...) finds it settled without any dialogue history."""
        system = ("You are TiQC. Rewrite the user's natural-language question into ONE "
                  "natural-language question that resolves the stated ambiguity exactly as "
                  "the user answered, and changes nothing else in the intent. State the "
                  "user's choice EXPLICITLY in the rewritten question, replacing the "
                  "ambiguous wording with the chosen interpretation (the concrete attribute, "
                  "value, time range, place, or goal, with values written exactly as in the "
                  "answer), so that the question read on its own no longer admits the other "
                  "readings. Keep every clarification the question already states. Use the "
                  "user's domain terms, never internal source or table identifiers, and do "
                  "not write SQL or code. Output ONLY the rewritten question.")
        rewritten = str(self.provider.complete([
            {"role": "system", "content": system},
            {"role": "user", "content": f"Question: {query}\nAmbiguity: {amb.description}\n"
                                        f"Clarifying question: {amb.question}\nUser answer: {answer}"},
        ])).strip().strip('"').strip()
        if not rewritten:
            raise TiQCError("empty rewritten query")
        return rewritten

    def clarify_query(self, query0: str, ask_user: AskUser) -> Dict[str, Any]:
        """Algorithm 1.  ``ask_user(a*, options) -> answer`` is the human or the
        user simulator.  Returns the clarified query and a per-round trace."""
        q = query0
        trace: List[Dict[str, Any]] = []
        for t in range(1, self.round_budget + 1):
            ambs = [a for a in self.detect_ambiguities(q) if a.severity >= self.theta_amb]
            if not ambs:
                break
            a_star = max(ambs, key=lambda a: a.severity)
            options = self.generate_options(q, a_star)
            answer = ask_user(a_star, options)
            q_new = self.rewrite_query(q, a_star, answer)
            trace.append({"round": t, "ambiguity": a_star.to_dict(), "options": options,
                          "answer": answer, "query_before": q, "query_after": q_new})
            q = q_new
        return {"clarified_query": q, "rounds": len(trace), "trace": trace,
                "original_query": query0}

    # ---------------------------------------------------------------- AskUser
    def make_user_simulator(self, hidden_intent: str) -> AskUser:
        """LLM user simulator for batch evaluation: it sees only the hidden intent
        and the current question with its options, never the gold answer."""
        def ask(amb: Ambiguity, options: List[str]) -> str:
            system = ("You simulate a human analyst answering a clarifying question. Choose "
                      "the option matching your TRUE intent, or state the intent in one short "
                      "sentence. Do not add any information beyond your intent, and never "
                      "state or guess the answer to the query.")
            return str(self.provider.complete([
                {"role": "system", "content": system},
                {"role": "user", "content": f"Your true intent: {hidden_intent}\n"
                                            f"Question: {amb.question}\nOptions: {options}"},
            ])).strip()
        return ask


def console_user(amb: Ambiguity, options: List[str]) -> str:
    """AskUser for live use: asks the analyst on the terminal."""
    print(f"\n{amb.question}")
    for i, o in enumerate(options, 1):
        print(f"  {i}. {o}")
    reply = input("Choose a number or type your answer: ").strip()
    return options[int(reply) - 1] if reply.isdigit() and 1 <= int(reply) <= len(options) else reply

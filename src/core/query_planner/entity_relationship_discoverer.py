"""Cross-modal relationship discovery (paper Sec. III-B).

Given the entity set E, relation edges of the metadata graph are discovered in
three stages:
  Stage 1 - recall-oriented statistical pre-filter.  For every entity pair the
    geometric-mean normalized co-occurrence J_w (Eq. 1) is computed over the
    per-instance prominence f(e, m), and only pairs with J_w >= theta_cooc survive.
  Stage 2 - the surviving pairs are submitted to the LLM in batches of size b,
    guided by a five-step CoT; each pair receives a relationship type and a
    confidence.
  Stage 3 - precision-oriented filter: pairs whose confidence is below
    theta_rel are discarded; each remaining pair becomes an edge of weight
    J_w * conf with the inferred type.

A modality instance m is one text document, one image, or one table row:
every mention records its instance (``position["instance"]`` for a document or
an image, ``position["row"]`` for a table row), and M(e) is the set of
instances that mention e.

Prominence f(e, m) (Eq. 2) is obtained in two steps:
  (i)  evidence mass  omega(e, m) = sum_{o in O(e,m)} c_o * g(o) over every
       extracted occurrence o of e in m, with g(o) = 1/L for a mention in a
       text of L words, 1/C for an occurrence in one cell of a row of a table
       with C columns, and s_o * a_o / A for an image detection with visual
       saliency s_o, box area a_o, and image area A;
  (ii) quantile normalization of omega within its modality type.
"""
import bisect
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import math
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .llm_query_analyzer import complete_json
from .metadata_graph import (
    CrossModalEntity, EntityMention, ModalityType,
    MultimodalMetadataGraph, CrossModalRelation, RelationType,
)

Instance = Tuple[str, Any]     # (modality id, document / image / table row)

# mention contexts given to the LLM per entity and modality in Stage 2
CONTEXTS_PER_MODALITY = 2
CONTEXT_CHARS = 160


def instance_of(m: EntityMention) -> Instance:
    """The modality instance a mention occurs in (paper Eq. 1): a text
    document, an image, or a table row."""
    pos = m.position or {}
    inst = pos.get("row") if m.modality_type == ModalityType.TABLE else pos.get("instance")
    if inst is None:
        raise ValueError(f"{m.modality_type.value} mention of {m.mention_text!r} "
                         "records no instance")
    return (m.modality_id, inst)

logger = logging.getLogger(__name__)

# LLM relation-type vocabulary -> RelationType enum (Stage 2 output space)
_REL_VOCAB = {
    "represents": RelationType.REPRESENTS, "describes": RelationType.REPRESENTS,
    "contains": RelationType.CONTAINS, "part_of": RelationType.BELONGS_TO,
    "belongs_to": RelationType.BELONGS_TO, "references": RelationType.REFERENCES,
    "similar_to": RelationType.SIMILAR_TO, "derived_from": RelationType.DERIVED_FROM,
    "temporal_related": RelationType.TEMPORAL_RELATED,
    "spatial_related": RelationType.SPATIAL_RELATED,
    "associated_with": RelationType.ASSOCIATED_WITH,
}


@dataclass
class EntityRelationConfig:
    """Relationship-discovery thresholds (paper Sec. III-B)."""
    theta_cooc: float = 0.1   # Stage-1 recall pre-filter
    theta_rel: float = 0.7    # Stage-3 precision filter
    batch_size: int = 20      # b, pairs per LLM invocation in Stage 2
    max_workers: int = 8      # concurrent Stage-2 invocations


class EntityRelationshipDiscoverer:
    """Relationship discovery DiscoverCrossModalRelations(E, L) (Sec. III-B)."""

    def __init__(self, config: Optional[EntityRelationConfig] = None):
        self.config = config or EntityRelationConfig()
        self.discovered_relations: List[CrossModalRelation] = []

    # ---------- Eq. 2 (i): evidence mass omega(e, m) ----------
    @staticmethod
    def _evidence_mass(mentions: List[EntityMention], graph: MultimodalMetadataGraph) -> float:
        """omega(e, m) = sum_o c_o * g(o) over the occurrences of e in one instance m."""
        mtype = mentions[0].modality_type
        if mtype == ModalityType.TEXT:
            # g(o) = 1/L, L = words of the document
            L = int((mentions[0].position or {}).get("words") or 0)
            if L <= 0:
                raise ValueError(f"text mention of {mentions[0].mention_text!r} records no "
                                 "document length")
            return sum(m.confidence for m in mentions) / L
        if mtype == ModalityType.IMAGE:
            # g(o) = s_o * a_o / A; ``area`` is the box area as a fraction a_o / A
            return sum(m.confidence * float(m.position["saliency"]) * float(m.position["area"])
                       for m in mentions)
        # TABLE: g(o) = 1/C for an occurrence in one cell of a row of a table
        # with C columns
        meta = graph.modalities.get(mentions[0].modality_id)
        C = len(((meta.content_summary or {}) if meta else {}).get("columns") or [])
        if C <= 0:
            raise ValueError(f"table {mentions[0].modality_id!r} has no columns")
        return sum(m.confidence for m in mentions) / C

    @staticmethod
    def _instances(e: CrossModalEntity) -> Dict[Instance, List[EntityMention]]:
        by: Dict[Instance, List[EntityMention]] = defaultdict(list)
        for m in e.mentions:
            by[instance_of(m)].append(m)
        return by

    # ---------- Eq. 2 (ii): quantile normalization within each modality type ----------
    def _prominence(self, entities: List[CrossModalEntity],
                    graph: MultimodalMetadataGraph) -> Dict[str, Dict[Instance, float]]:
        masses: Dict[Any, List[Tuple[str, Instance, float]]] = defaultdict(list)
        for e in entities:
            for inst, ms in self._instances(e).items():
                masses[ms[0].modality_type].append((e.entity_id, inst, self._evidence_mass(ms, graph)))
        f: Dict[str, Dict[Instance, float]] = defaultdict(dict)
        for triples in masses.values():
            values = sorted(w for _, _, w in triples)
            n = len(values)
            for eid, inst, w in triples:
                # |{(e',m') of this type : omega(e',m') <= omega(e,m)}| / |S_t|
                f[eid][inst] = bisect.bisect_right(values, w) / n
        return f

    # ---------- Eq. 1: geometric-mean normalized co-occurrence J_w ----------
    @staticmethod
    def _jw(f1: Dict[Instance, float], f2: Dict[Instance, float]) -> float:
        if not f1 or not f2:
            return 0.0
        num = sum(math.sqrt(f1[m] * f2[m]) for m in f1.keys() & f2.keys())
        den = sum(max(f1.get(m, 0.0), f2.get(m, 0.0)) for m in f1.keys() | f2.keys())
        return (num / den) if den > 1e-12 else 0.0

    # ---------- Stage 2: BatchInferRelationTypes(pairs, L) ----------
    def _batch_infer_types(self, pairs: List[Tuple[CrossModalEntity, CrossModalEntity, float]],
                           llm: Any) -> List[Tuple[RelationType, float]]:
        if not pairs:
            return []
        if llm is None:
            raise ValueError("relationship inference (Stage 2) needs the LLM")
        sys = (
            "You infer cross-modal relationships between entity pairs. For EACH pair follow "
            "a five-step reasoning: (1) identify each entity's semantic type and the "
            "attributes involved in the relationship; (2) analyze the contextual "
            "relationship between the entities from how each is expressed in its "
            "modalities (e1_contexts / e2_contexts: mention contexts per modality); "
            "(3) examine their co-occurrence pattern and cross-modal features (J_w: "
            "co-occurrence score in [0,1]; cooccurrence: the instances -- documents, "
            "images, table rows -- that mention both, per modality, against those "
            "mentioning each); (4) determine the MOST SPECIFIC "
            "relationship type from {represents, contains, belongs_to, references, "
            "similar_to, derived_from, temporal_related, spatial_related, associated_with}; "
            "(5) give a confidence in [0,1] (0 if the pair is unrelated / spurious). Return "
            "ONLY a JSON list [{\"id\": int, \"type\": str, \"confidence\": float}, ...] "
            "covering every id."
        )
        instances: Dict[str, Dict[Instance, List[EntityMention]]] = {}

        def inst(e: CrossModalEntity) -> Dict[Instance, List[EntityMention]]:
            if e.entity_id not in instances:
                instances[e.entity_id] = self._instances(e)
            return instances[e.entity_id]

        def contexts(e: CrossModalEntity) -> Dict[str, List[str]]:
            """A few distinct mention contexts of e per modality (truncated)."""
            out: Dict[str, List[str]] = {}
            for m in e.mentions:
                ctx = " ".join(str(m.context or "").split())
                if m.modality_type == ModalityType.TABLE and ctx:
                    ctx = f"value of column {ctx}"
                elif m.modality_type == ModalityType.IMAGE and not ctx:
                    ctx = f"detected object {m.mention_text!r}"
                got = out.setdefault(m.modality_type.value, [])
                ctx = ctx[:CONTEXT_CHARS]
                if ctx and ctx not in got and len(got) < CONTEXTS_PER_MODALITY:
                    got.append(ctx)
            return out

        def cooccurrence(e1: CrossModalEntity, e2: CrossModalEntity) -> Dict[str, Any]:
            i1, i2 = inst(e1), inst(e2)
            by_modality: Dict[str, int] = defaultdict(int)
            for m in i1.keys() & i2.keys():
                by_modality[i1[m][0].modality_type.value] += 1
            return {"shared_instances": sum(by_modality.values()),
                    "shared_by_modality": dict(by_modality),
                    "e1_instances": len(i1), "e2_instances": len(i2)}

        by_id: Dict[int, Tuple[RelationType, float]] = {}
        b = max(1, self.config.batch_size)

        def infer(start: int) -> Dict[int, Tuple[RelationType, float]]:
            chunk = pairs[start:start + b]
            ids = set(range(start, start + len(chunk)))
            items = [{"id": start + k, "e1": e1.canonical_name, "e1_type": e1.entity_type.value,
                      "e2": e2.canonical_name, "e2_type": e2.entity_type.value,
                      "e1_modalities": sorted({m.modality_type.value for m in e1.mentions}),
                      "e2_modalities": sorted({m.modality_type.value for m in e2.mentions}),
                      "e1_contexts": contexts(e1), "e2_contexts": contexts(e2),
                      "J_w": round(j, 4), "cooccurrence": cooccurrence(e1, e2)}
                     for k, (e1, e2, j) in enumerate(chunk)]

            def parse(reply: list, ids=ids) -> Dict[int, Tuple[RelationType, float]]:
                got: Dict[int, Tuple[RelationType, float]] = {}
                for o in reply:
                    if not isinstance(o, dict):
                        raise ValueError("list items must be objects")
                    try:
                        k, conf = int(o["id"]), float(o["confidence"])
                    except (KeyError, TypeError, ValueError):
                        raise ValueError("every item needs an integer id and a confidence")
                    rtype = str(o.get("type", "")).lower().strip()
                    if rtype not in _REL_VOCAB:
                        raise ValueError(f"unknown relationship type {rtype!r}")
                    got[k] = (_REL_VOCAB[rtype], max(0.0, min(1.0, conf)))
                if not ids <= got.keys():
                    raise ValueError(f"omitted ids {sorted(ids - got.keys())[:10]}")
                return got

            reply = complete_json(llm, [{"role": "system", "content": sys},
                                        {"role": "user", "content": json.dumps(items, ensure_ascii=False)}],
                                  list, parse, "relationship inference")
            return {k: v for k, v in parse(reply).items() if k in ids}

        # ceil(|pairs| / b) independent invocations, issued concurrently
        starts = list(range(0, len(pairs), b))
        with ThreadPoolExecutor(max_workers=min(self.config.max_workers, len(starts))) as pool:
            for got in pool.map(infer, starts):
                by_id.update(got)
        return [by_id[i] for i in range(len(pairs))]

    # ---------- three-stage discovery (Sec. III-B) ----------
    def discover_entity_relations(self, metadata_graph: MultimodalMetadataGraph,
                                  llm: Any = None) -> List[CrossModalRelation]:
        entities = list(metadata_graph.entities.values())
        cfg = self.config
        f = self._prominence(entities, metadata_graph)
        # Stage 1: statistical recall pre-filter.  J_w > 0 needs a shared
        # instance, so only pairs co-occurring in some instance are scored.
        by_instance: Dict[Instance, List[int]] = defaultdict(list)
        for k, e in enumerate(entities):
            for inst in f.get(e.entity_id, {}):
                by_instance[inst].append(k)
        candidates = sorted({(a, b) for ks in by_instance.values()
                             for x, a in enumerate(ks) for b in ks[x + 1:]})
        pairs: List[Tuple[CrossModalEntity, CrossModalEntity, float]] = []
        for a, b in candidates:
            e1, e2 = entities[a], entities[b]
            j = self._jw(f[e1.entity_id], f[e2.entity_id])
            if j >= cfg.theta_cooc:
                pairs.append((e1, e2, j))
        # Stage 2: LLM batch semantic typing
        typed = self._batch_infer_types(pairs, llm)

        # Stage 3: precision filter + edge construction
        relations: List[CrossModalRelation] = []
        for (e1, e2, j), (rtype, conf) in zip(pairs, typed):
            if conf < cfg.theta_rel:
                continue
            relations.append(CrossModalRelation(
                relation_id=f"rel_{uuid.uuid4().hex[:8]}",
                relation_type=rtype,
                source_modality=e1.entity_id, target_modality=e2.entity_id,
                confidence=round(j * conf, 4),        # edge weight = J_w · conf
                evidence={"J_w": round(j, 4), "type_confidence": round(conf, 4),
                          "discovery_method": "three_stage",
                          "e1": e1.canonical_name, "e2": e2.canonical_name},
                properties={"relation_category": "cross_modal_relationship"}))
        self.discovered_relations = relations
        logger.info("Relationship discovery: %d entities -> %d Stage-1 pairs -> %d edges (θ_cooc=%.2f, θ_rel=%.2f)",
                    len(entities), len(pairs), len(relations), cfg.theta_cooc, cfg.theta_rel)
        return relations

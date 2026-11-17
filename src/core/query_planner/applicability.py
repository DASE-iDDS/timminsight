"""Per-instance applicability of physical implementations (paper Sec. V-D).

I(o) is registered per operator type and gives the type-level priors of Eq. 6
(c(o), c_hat(o) are means over all of I(o)).  An implementation may however
have a precondition that holds for one operator instance and not for another,
so the physical stage assigns to an instance o only the implementations of
I(tau(o)) whose precondition holds for o: its bound source, its parent, its
inputs, and the binding environment B.

Rules (types not listed have no preconditions; every registered
implementation applies):

  SCAN               INDEX_SCAN iff the parent is a FILTER bound to the same
                     table source with an equality/IN condition on one of its
                     columns (the index answers it); TABLE_SCAN always.
  FILTER             THRESHOLD_FILTER iff B has a semantic condition (it scores
                     that condition in [0, 1] and keeps scores >= the threshold);
                     PREDICATE_FILTER always.
  CROSS_MODAL_MATCH  ENTITY_LINK_MATCH iff a metadata graph is present and a
                     document of one bound source links to a CrossModalEntity
                     mentioned in the other; LLM_MATCH always.
  CONTENT_EXTRACT    METADATA_LOOKUP iff TiMetaGraph precomputed the attribute
                     for every document of the bound (text) source; TOOL_EXTRACT
                     iff text sentiment (VADER) or an image column with a boolean
                     type or candidate values (CLIP zero-shot); LLM_EXTRACT always.
  SIMILARITY_JOIN    INTRA_MODAL_JOIN iff both inputs have the same known
                     modality; SEMANTIC_JOIN always.
  FEATURE_TRANSFORM  CLIP_IMAGE_ENCODER for an image column, CLIP_TEXT_ENCODER
                     otherwise (exactly one).
  EXPORT             CSV_EXPORT for a .csv path, JSON_EXPORT for a .json path
                     (exactly one; any other path is a binding error).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Set

from .mcts_plan_search import LogicalPlanNode

Table = List[Dict[str, Any]]

INDEXABLE_OPS = {"=", "==", "in"}
EXPORT_BY_EXTENSION = {".csv": "CSV_EXPORT", ".json": "JSON_EXPORT"}


class NoApplicableImplementation(ValueError):
    """No registered implementation's precondition holds for an instance."""


def export_implementation(path: Any) -> str:
    """The EXPORT implementation the bound path's extension determines."""
    ext = os.path.splitext(str(path or ""))[1].lower()
    if ext not in EXPORT_BY_EXTENSION:
        raise ValueError(f"EXPORT path {path!r} must end in one of {sorted(EXPORT_BY_EXTENSION)}")
    return EXPORT_BY_EXTENSION[ext]


def indexable_conditions(filter_binding: Dict[str, Any], columns: Set[str]) -> List[Dict[str, Any]]:
    """The equality/IN conditions of a FILTER binding on the given columns."""
    return [c for c in filter_binding.get("conditions") or []
            if isinstance(c, dict) and c.get("column") in columns
            and str(c.get("op", "=")).strip().lower() in INDEXABLE_OPS]


class ApplicabilityChecker:
    """Decides the applicable implementations of each operator instance."""

    def __init__(self, bindings: Optional[Dict[str, Dict[str, Any]]],
                 sources: Optional[Dict[str, Table]] = None, graph: Any = None):
        self._b = bindings or {}
        self._sources = sources or {}
        self._graph = graph

    # -- helpers ------------------------------------------------------------
    def _binding(self, op: str) -> Dict[str, Any]:
        return self._b.get(op) or {}

    def _modality(self, source: Optional[str]) -> Optional[str]:
        """'table', 'text' or 'image' for a source, None when unknown."""
        if source is None:
            return None
        if self._graph is not None and source in self._graph.modalities:
            return self._graph.modalities[source].modality_type.value
        rows = self._sources.get(source)
        if rows:
            from .plan_executor import _is_image_column
            return "image" if any(_is_image_column(rows, c) for c in rows[0]) else None
        return None

    def _table_columns(self, source: Optional[str]) -> Set[str]:
        """Structured columns of a table source (empty if not a table)."""
        if self._graph is not None and source in self._graph.modalities:
            meta = self._graph.modalities[source]
            if meta.modality_type.value != "table":
                return set()
            return set(map(str, (meta.content_summary or {}).get("columns") or []))
        rows = self._sources.get(source)
        if not rows:
            return set()
        from .plan_executor import _is_image_column
        return {c for c in rows[0] if not _is_image_column(rows, c)}

    @staticmethod
    def _leaf_source(node: LogicalPlanNode) -> Optional[str]:
        for n in node.get_all_operators():
            if not n.children and n.source_name:
                return n.source_name
        return None

    def _input_sources(self, node: LogicalPlanNode, op: str) -> List[Optional[str]]:
        bound = list(self._binding(op).get("sources") or [])
        return [bound[i] if i < len(bound) else self._leaf_source(c)
                for i, c in enumerate(node.children)]

    def _is_image(self, source: Optional[str], columns: Sequence[str]) -> bool:
        """Whether the bound columns of ``source`` hold images."""
        if self._graph is not None and source in self._graph.modalities:
            meta = self._graph.modalities[source]
            img = (meta.content_summary or {}).get("image_column")
            if meta.modality_type.value == "image" or (img and img in columns):
                return True
        rows = self._sources.get(source) or []
        if rows:
            from .plan_executor import _is_image_column
            return any(_is_image_column(rows, c) for c in columns)
        return False

    # -- rules --------------------------------------------------------------
    def applicable(self, node: LogicalPlanNode, registered: Sequence[str]) -> List[str]:
        """The registered implementations whose precondition holds for ``node``."""
        op = node.operator_type
        rule = getattr(self, f"_rule_{op.lower()}", None)
        allowed = set(rule(node)) if rule else set(registered)
        out = [i for i in registered if i in allowed]
        if not out:
            raise NoApplicableImplementation(
                f"no implementation of {op} applies to instance {node.node_id} "
                f"(registered: {list(registered)}, applicable: {sorted(allowed)})")
        return out

    def _rule_scan(self, node):
        keep = {"TABLE_SCAN"}
        parent = node.parent
        fb = self._binding("FILTER")
        if (parent is not None and parent.operator_type == "FILTER" and node.source_name
                and fb.get("source") == node.source_name
                and indexable_conditions(fb, self._table_columns(node.source_name))):
            keep.add("INDEX_SCAN")
        return keep

    def _rule_filter(self, node):
        keep = {"PREDICATE_FILTER"}
        if self._binding("FILTER").get("semantic"):
            keep.add("THRESHOLD_FILTER")
        return keep

    def _rule_cross_modal_match(self, node):
        keep = {"LLM_MATCH"}
        g = self._graph
        srcs = [s for s in self._input_sources(node, "CROSS_MODAL_MATCH") if s]
        if g is not None and len(set(srcs)) == 2:
            def mentioned_in(src):
                return {e.entity_id for e in g.entities.values()
                        if any(m.modality_id == src for m in e.mentions)}

            def linked(text_src, row_src):
                """A document of text_src links to an entity row_src mentions."""
                meta = g.modalities.get(text_src)
                docs = ((meta.raw_metadata or {}).get("doc_entities") or {}) if meta else {}
                targets = mentioned_in(row_src)
                return any(eid in targets for ids in docs.values() for eid in ids)

            a, b = srcs
            if linked(a, b) or linked(b, a):
                keep.add("ENTITY_LINK_MATCH")
        return keep

    def _rule_content_extract(self, node):
        keep = {"LLM_EXTRACT"}
        b = self._binding("CONTENT_EXTRACT")
        src, attr = b.get("source") or self._leaf_source(node), b.get("attribute")
        image = self._is_image(src, b.get("columns") or [])
        if self._graph is not None and not image:
            from .metadata_graph import precomputed_attributes
            # precomputed for every document of the bound source (or of every
            # text source when the binding names a table rather than a source)
            scope = src if src in self._graph.modalities else None
            if attr in precomputed_attributes(self._graph, scope):
                keep.add("METADATA_LOOKUP")
        if (image and (b.get("type") == "boolean" or b.get("values"))) or \
                (not image and attr == "sentiment"):
            keep.add("TOOL_EXTRACT")
        return keep

    def _rule_similarity_join(self, node):
        keep = {"SEMANTIC_JOIN"}
        mods = [self._modality(s) for s in self._input_sources(node, "SIMILARITY_JOIN")]
        if len(mods) == 2 and mods[0] is not None and mods[0] == mods[1]:
            keep.add("INTRA_MODAL_JOIN")
        return keep

    def _rule_feature_transform(self, node):
        b = self._binding("FEATURE_TRANSFORM")
        src = b.get("source") or self._leaf_source(node)
        return {"CLIP_IMAGE_ENCODER" if self._is_image(src, b.get("columns") or [])
                else "CLIP_TEXT_ENCODER"}

    def _rule_export(self, node):
        try:
            return {export_implementation(self._binding("EXPORT").get("path"))}
        except ValueError as e:
            raise NoApplicableImplementation(str(e)) from None


def annotate_applicability(plan: LogicalPlanNode, registry: Dict[str, List[Any]],
                           bindings: Optional[Dict[str, Dict[str, Any]]],
                           sources: Optional[Dict[str, Table]] = None, graph: Any = None) -> None:
    """Set ``applicable_impls`` on every operator instance of ``plan``."""
    checker = ApplicabilityChecker(bindings, sources, graph)
    for node in plan.get_all_operators():
        registered = [i.impl_type for i in registry.get(node.operator_type) or []]
        if not registered:
            raise KeyError(f"no physical implementation registered for {node.operator_type}")
        node.applicable_impls = tuple(checker.applicable(node, registered))

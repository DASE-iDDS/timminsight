"""
End-to-End Query Planner.

Orchestrates the full planning pipeline:
  NL query → QueryContext → MCTS logical search → Uncertainty propagation
  → Physical optimization → Plan execution → NL answer

Entry points:
  - ``QueryPlanner.plan(query_text) → PlanResult`` (planning only)
  - ``QueryPlanner.plan_and_execute(query_text, data) → ExecutableResult``
    (planning + execution + NL answer)

Planning strategies (``planning_strategy``):
  - "titsp"                 full TiTSP (MCTS + uncertainty propagation + Pareto/Tchebycheff)
  - "titsp-oneshot"         w/o-TiTSP ablation: the backbone writes the plan in one shot
  - "titsp-no-mcts"         one greedy OCG-guided plan instead of the MCTS search
  - "titsp-no-uncertainty"  top-Q plan without propagation-based re-ranking
  - "titsp-no-pareto"       max-confidence implementation per operator
  - "titsp-no-ocg"          search without OCG edge constraints
  - "caesura", "lotus", "thalamusdb", "palimpzest", "nirvana"
                            the baselines' official releases (src/baselines)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .applicability import annotate_applicability
from .default_impl_registry import build_default_impl_registry
from .metadata_graph import describe_for_llm
from .mcts_plan_search import (
    LogicalCostModel,
    LogicalPlanCandidate,
    LogicalPlanNode,
    MCTSLogicalConfig,
    MCTSLogicalPlanSearch,
    PhysicalImplInfo,
    ResourceConstraints,
    UserPreference,
    annotate_plan_estimates,
)
from .multi_objective_optimizer import (
    MultiObjectivePhysicalOptimizer,
    instance_implementations,
    ObjectiveEvaluator,
    ParetoCandidate,
)
from .operator_compatibility_graph import (
    OperatorCompatibilityGraph,
    QueryContext,
)
from .uncertainty_propagation import (
    OperatorUncertaintyProfile,
    UncertaintyEstimate,
    UncertaintyPrior,
    UncertaintyPropagator,
)
from .plan_executor import PlanExecutionError, PlanExecutionResult, PlanExecutor, Table
from .answer_generator import AnswerGenerator


# Baselines run their official releases end to end (src/baselines): CAESURA takes
# the NL question natively; the others receive it compiled into their input
# program by the same backbone LLM.
BASELINE_STRATEGIES = {"caesura", "lotus", "thalamusdb", "palimpzest", "nirvana"}
# "titsp-oneshot" is the w/o-TiTSP ablation of the paper: the backbone generates
# the logical plan in one shot from the same query context, and every operator
# uses its first registered implementation.
VALID_STRATEGIES = {
    "titsp", "titsp-no-mcts", "titsp-no-uncertainty", "titsp-no-pareto", "titsp-no-ocg",
    "titsp-oneshot",
} | BASELINE_STRATEGIES


@dataclass
class PlanTiming:
    """Per-stage timing breakdown (milliseconds)."""
    parse_ms: float = 0.0
    search_ms: float = 0.0
    uncertainty_ms: float = 0.0
    physical_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class PlanResult:
    """Complete output of the end-to-end planning pipeline."""
    query_context: QueryContext
    logical_candidates: List[LogicalPlanCandidate]
    best_logical_plan: LogicalPlanNode
    uncertainty_map: Dict[str, UncertaintyEstimate]
    selected_physical: ParetoCandidate
    pareto_front: List[ParetoCandidate]
    timing: Optional[PlanTiming] = None
    best_quality_score: float = 0.0


@dataclass
class ExecutableResult:
    """Full end-to-end result: planning + execution + NL answer."""
    plan: Optional[PlanResult]
    execution: PlanExecutionResult
    answer: str
    chart: Optional[Dict[str, Any]] = None   # recommended chart (paper Sec. II)
    baseline: Any = None   # BaselineResponse when a baseline system ran


def _baseline_runner():
    """``run_baseline`` of src/baselines/runner.py, whether this package is
    imported as ``src.core`` (tests) or as ``core`` with src/ on sys.path
    (src/experiments/benchmark.py)."""
    try:
        from ...baselines.runner import run_baseline
    except ImportError:
        from baselines.runner import run_baseline
    return run_baseline


def _baseline_proxy(llm):
    """Local OpenAI-compatible endpoint of src/baselines/openai_proxy.py."""
    try:
        from ...baselines.openai_proxy import endpoint_for
    except ImportError:
        from baselines.openai_proxy import endpoint_for
    return endpoint_for(llm)


class QueryPlanner:
    """End-to-end query planner: natural language → physical plan.

    Usage::

        planner = QueryPlanner()
        result = planner.plan("Plot paintings depicting Madonna by century")
        print(result.selected_physical.assignment)
        print(result.best_logical_plan.operator_type)

    With LLM-based NL parsing::

        from llm_query_analyzer import create_provider
        provider = create_provider("dashscope", model="qwen-plus")
        planner = QueryPlanner(llm_provider=provider)
        result = planner.plan("找出所有包含猫的图片")

    With a baseline system (its official release, run end to end)::

        planner = QueryPlanner(llm_provider=provider, planning_strategy="caesura")
        result = planner.plan_and_execute(question, rows)
    """

    def __init__(
        self,
        llm_provider=None,
        impl_registry: Optional[Dict[str, List[PhysicalImplInfo]]] = None,
        history_manager=None,
        constraints: Optional[ResourceConstraints] = None,
        preference: Optional[UserPreference] = None,
        mcts_config: Optional[MCTSLogicalConfig] = None,
        planning_strategy: str = "titsp",
        metadata_context: Optional[str] = None,
        custom_uncertainty_profiles: Optional[Dict[str, "OperatorUncertaintyProfile"]] = None,
        clarifier: Any = None,
        metadata_graph: Any = None,
        ask_user: Any = None,
    ):
        """Args (metadata-graph integration):
            metadata_context: Summary of the metadata graph (modality
                bindings, cross-modal entities). Injected into the Stage-0
                analyzer prompt so operator extraction is grounded in where
                attributes actually live (table column vs. image vs. text).
            custom_uncertainty_profiles: Per-operator prior overrides for
                Stage-3 propagation, e.g. instantiated from the metadata
                graph's cross-modal entity confidence scores.
        """
        if planning_strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Unknown planning_strategy '{planning_strategy}'. "
                f"Valid: {sorted(VALID_STRATEGIES)}"
            )
        self._llm_provider = llm_provider
        self._impl_registry = impl_registry or build_default_impl_registry()
        # I(o), type-level; per-instance preconditions are in applicability.py
        self._active_registry = self._impl_registry
        self._history_manager = history_manager
        self._constraints = constraints or ResourceConstraints()
        self._preference = preference or UserPreference()
        self._mcts_config = mcts_config or MCTSLogicalConfig()
        self._planning_strategy = planning_strategy
        # None: built per query from the metadata graph, ranked by relevance
        self._metadata_context = metadata_context
        # TiMetaGraph: exact source sizes n_o for Phi, per-document metadata and
        # entity links for METADATA_LOOKUP / ENTITY_LINK_MATCH at execution time.
        self._metadata_graph = metadata_graph
        self._custom_uncertainty_profiles = custom_uncertainty_profiles
        # TiQC: optional pre-Stage-0 query-clarification component (a TiQCClarifier).
        # When set, plan() first rewrites the NL query into an unambiguous form,
        # grounded by TiMetagraph. clarifier=None reproduces the no-TiQC ablation.
        if clarifier is not None and ask_user is None:
            raise ValueError("TiQC asks the user: pass ask_user (a human or an LLM user "
                             "simulator conditioned on the hidden intent)")
        self._clarifier = clarifier
        self._ask_user = ask_user

    @property
    def llm_provider(self):
        """The configured LLM provider (or None). Exposed so experiment
        harnesses can read/reset its token-usage counter."""
        return self._llm_provider

    def plan(
        self, query_text: str, sources: Optional[Dict[str, Table]] = None,
    ) -> PlanResult:
        """Plan a natural-language question.

        Args:
            query_text: natural-language question.
            sources: the source tables; without a metadata graph they give the
                source names, columns, and exact sizes n_o used by TiTSP.
        """
        # TiQC pre-Stage-0: disambiguate the NL query (grounded by TiMetagraph)
        # before operator extraction. A no-op when no clarifier is configured.
        if self._clarifier is not None:
            res = self._clarifier.clarify_query(query_text, self._ask_user)
            query_text = res.get("clarified_query") or query_text
        if self._planning_strategy in BASELINE_STRATEGIES:
            raise ValueError("baseline systems run end to end only; use plan_and_execute")

        if self._planning_strategy == "titsp-oneshot":
            return self._plan_oneshot(query_text, sources)
        return self._plan_titsp(query_text, sources)

    def _annotate_applicability(self, plan: LogicalPlanNode, context: QueryContext,
                                sources: Optional[Dict[str, Table]]) -> None:
        """Offer every operator instance of the chosen plan only the
        implementations whose precondition holds for it (applicability.py);
        the logical priors stay type-level over all of I(o) (Eq. 6)."""
        annotate_applicability(plan, self._impl_registry, context.bindings, sources,
                               self._metadata_graph)

    def _plan_oneshot(self, query_text: str, sources: Optional[Dict[str, Table]] = None) -> PlanResult:
        """w/o TiTSP: one-shot LLM plan generation over the same query context."""
        import json as _json
        import uuid as _uuid
        if self._llm_provider is None:
            raise ValueError("one-shot plan generation needs the backbone LLM")
        timing = PlanTiming()
        t0 = time.perf_counter()
        context = self._parse_query(query_text, sources)
        ops = sorted(self._active_registry)
        prompt = (
            "Write the logical query plan that answers the question as a JSON tree "
            '{"op": <operator>, "children": [...]}; leaves are SCAN. Operators: '
            f"{ops}. Required operators: {sorted(context.required_operators)}. "
            f"Bindings: {_json.dumps(context.bindings)}. Return the JSON only.")
        raw = self._llm_provider.complete([
            {"role": "system", "content": prompt},
            {"role": "user", "content": context.query_text}])
        text = raw[raw.find("{"): raw.rfind("}") + 1]

        ocg = OperatorCompatibilityGraph()

        def build(spec, parent=None):
            op = str(spec.get("op", "SCAN")).upper()
            ocg.get_spec(op)   # KeyError for an operator outside Sigma
            node = LogicalPlanNode(node_id=str(_uuid.uuid4())[:8], operator_type=op, parent=parent)
            node.children = [build(c, node) for c in spec.get("children") or []
                             if isinstance(c, dict)]
            return node

        root = build(_json.loads(text))
        annotate_plan_estimates(root, query_context=context,
                                cost_lookup=LogicalCostModel(self._active_registry).operator_cost)
        self._annotate_applicability(root, context, sources)
        assignment = {n.node_id: instance_implementations(n, self._active_registry)[0]
                      for n in root.get_all_operators()}
        uncertainty_map = UncertaintyPropagator(
            prior=UncertaintyPrior(self._custom_uncertainty_profiles, self._active_registry),
            history_manager=self._history_manager,
        ).propagate(root).uncertainty_map
        selected = ParetoCandidate(assignment, self._objectives(root, assignment))
        timing.total_ms = timing.search_ms = (time.perf_counter() - t0) * 1000
        return PlanResult(query_context=context, logical_candidates=[], best_logical_plan=root,
                          uncertainty_map=uncertainty_map, selected_physical=selected,
                          pareto_front=[selected], timing=timing)

    def _plan_titsp(
        self, query_text: str, sources: Optional[Dict[str, Table]] = None,
    ) -> PlanResult:
        """Full TiTSP pipeline with ablation support.

        Ablation variants:
          - titsp:                full pipeline
          - titsp-no-mcts:        one greedy OCG-guided plan (no MCTS search)
          - titsp-no-uncertainty: top-Q plan, no propagation-based re-ranking
          - titsp-no-pareto:      max-confidence implementation per operator
          - titsp-no-ocg:         search without OCG edge constraints
        All variants bind the SCAN leaves identically, and every selected
        physical plan is scored with the same ObjectiveEvaluator.
        """
        strategy = self._planning_strategy
        timing = PlanTiming()

        t0 = time.perf_counter()
        context = self._parse_query(query_text, sources)
        timing.parse_ms = (time.perf_counter() - t0) * 1000

        ocg = OperatorCompatibilityGraph(history_manager=self._history_manager,
                                         unconstrained=(strategy == "titsp-no-ocg"))

        # --- Stage 2: Logical Plan Search ---
        t1 = time.perf_counter()
        if strategy == "titsp-no-mcts":
            logical_candidates = MCTSLogicalPlanSearch(
                ocg=ocg, query_context=context, config=self._mcts_config,
                impl_registry=self._active_registry,
            ).greedy()
        else:
            mcts = MCTSLogicalPlanSearch(
                ocg=ocg,
                query_context=context,
                config=self._mcts_config,
                impl_registry=self._active_registry,
            )
            logical_candidates = mcts.search()
        timing.search_ms = (time.perf_counter() - t1) * 1000

        if not logical_candidates:
            raise ValueError("Planning produced no valid logical plans.")

        # --- Stage 3: Uncertainty Propagation + uncertainty-aware selection ---
        t2 = time.perf_counter()
        # Bind every SCAN leaf to its source, estimate the cardinalities
        # (Eq. 7), and attach per-record costs c(o) (Eq. 6).
        cost_model = LogicalCostModel(self._active_registry)
        for cand in logical_candidates:
            annotate_plan_estimates(
                cand.plan,
                query_context=context,
                cost_lookup=cost_model.operator_cost,
            )
        propagator = UncertaintyPropagator(
            prior=UncertaintyPrior(self._custom_uncertainty_profiles,
                                   self._active_registry),
            history_manager=self._history_manager,
        )
        if strategy == "titsp-no-uncertainty":
            # Ablation: no re-ranking by propagated confidence; the top-Q
            # plan's map is kept for inspection only.
            best = logical_candidates[0]
            uncertainty_map = propagator.propagate(best.plan).uncertainty_map
        else:
            best, uncertainty_map = self._select_by_uncertainty(
                logical_candidates, propagator,
            )
        timing.uncertainty_ms = (time.perf_counter() - t2) * 1000

        # --- Stage 4: Physical Optimization ---
        t3 = time.perf_counter()
        self._annotate_applicability(best.plan, context, sources)
        if strategy == "titsp-no-pareto":
            selected = self._greedy_physical_select(best.plan)
            pareto_front = [selected]
        else:
            optimizer = MultiObjectivePhysicalOptimizer(
                logical_plan=best.plan,
                impl_registry=self._active_registry,
                constraints=self._constraints,
                preference=self._preference,
                history_manager=self._history_manager,
            )
            selected, pareto_front = optimizer.optimize()
        timing.physical_ms = (time.perf_counter() - t3) * 1000

        timing.total_ms = (
            timing.parse_ms + timing.search_ms
            + timing.uncertainty_ms + timing.physical_ms
        )

        return PlanResult(
            query_context=context,
            logical_candidates=logical_candidates,
            best_logical_plan=best.plan,
            uncertainty_map=uncertainty_map,
            selected_physical=selected,
            pareto_front=pareto_front,
            timing=timing,
            best_quality_score=best.quality_score,
        )

    def _select_by_uncertainty(self, candidates, propagator):
        """Re-rank the cached candidates by propagated confidence (Sec. V-C).

        The full propagation is applied once to each cached complete
        candidate, and its root confidence replaces the structural
        confidence term of Q (Eq. 7).  Returns ``(best_candidate,
        uncertainty_map_of_best)``.
        """
        gamma = self._mcts_config.confidence_weight
        best_cand = None
        best_score = -float("inf")
        best_map: Dict[str, UncertaintyEstimate] = {}
        for cand in candidates:
            umap = propagator.propagate(cand.plan).uncertainty_map
            plan_conf = umap[cand.plan.node_id].confidence
            refined = cand.quality_score + gamma * (plan_conf - cand.estimated_confidence)
            if refined > best_score:
                best_score = refined
                best_cand = cand
                best_map = umap
        return best_cand, best_map

    def _objectives(self, plan: LogicalPlanNode, assignment: Dict[str, str]):
        """f_lat, f_res, f_conf of an assignment, as in the physical stage."""
        return ObjectiveEvaluator(self._active_registry, plan, self._history_manager,
                                  self._constraints).evaluate(assignment)

    def _greedy_physical_select(self, logical_plan: LogicalPlanNode) -> ParetoCandidate:
        """Ablation (w/o Pareto): the most confident implementation of every
        operator, scored with the same objectives as the physical stage."""
        assignment = {}
        for node in logical_plan.get_all_operators():
            allowed = instance_implementations(node, self._active_registry)
            impls = [i for i in self._active_registry[node.operator_type] if i.impl_type in allowed]
            assignment[node.node_id] = max(impls, key=lambda x: x.base_confidence).impl_type
        return ParetoCandidate(assignment, self._objectives(logical_plan, assignment))

    def plan_and_execute(
        self,
        query_text: str,
        sources: Dict[str, Table],
        vision_provider: Any = None,
        table_info: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> ExecutableResult:
        """Full pipeline: NL question -> plan -> execution -> answer and chart.

        Args:
            query_text: natural-language question.
            sources: tables keyed by source name; a SCAN leaf reads the source
                bound to it by the binding environment B.
            vision_provider: vision model for operators over image columns.
            table_info: optional per-table descriptions given to baselines.
        """
        if self._planning_strategy in BASELINE_STRATEGIES:
            return self._run_baseline(query_text, sources, table_info)
        if not sources:
            raise ValueError("plan_and_execute needs at least one source table")

        plan_result = self.plan(query_text, sources)
        metadata = self._execution_metadata(plan_result.query_context, sources)
        executor = PlanExecutor(
            data_source=next(iter(sources.values())),
            llm_provider=self._llm_provider,
            query_text=plan_result.query_context.query_text,
            metadata=metadata,
            vision_provider=vision_provider,
        )
        try:
            execution = executor.execute(
                plan_result.best_logical_plan,
                plan_result.selected_physical.assignment,
            )
        except PlanExecutionError as err:
            err.plan = plan_result      # lets the caller record the failure (Eq. 11)
            raise
        generator = AnswerGenerator(llm_provider=self._llm_provider)
        # the answer addresses the query that was planned (q* when TiQC ran)
        answer, chart = generator.generate(plan_result.query_context.query_text, execution,
                                           metadata.get("chart"))
        return ExecutableResult(plan=plan_result, execution=execution, answer=answer, chart=chart)

    def _run_baseline(self, query_text: str, tables: Dict[str, Table],
                      table_info: Optional[Dict[str, Dict[str, Any]]]) -> ExecutableResult:
        """Run a baseline's official release on the NL question (src/baselines),
        then phrase the answer with the same answer generator as TiMMInsight.
        Baselines do not receive TiMetaGraph."""
        run_baseline = _baseline_runner()
        llm = self._llm_provider
        if llm is None:
            raise ValueError("baselines need the backbone LLM provider")
        if all(hasattr(llm, a) for a in ("_model", "_base_url", "_api_key")):
            model, base_url, api_key, own_usage = llm._model, llm._base_url, llm._api_key, False
        elif hasattr(llm, "complete_with_usage"):
            # a backbone without an OpenAI-compatible endpoint (Claude on AWS
            # Bedrock): the baselines reach it through a local proxy over the
            # same provider, which already counts their tokens in llm.usage
            proxy = _baseline_proxy(llm)
            model, base_url, api_key, own_usage = proxy.model, proxy.base_url, "local-proxy", True
        else:
            raise ValueError(f"baselines cannot call the backbone {llm.provider_name}")
        resp = run_baseline(
            self._planning_strategy, query_text, tables,
            model=model, base_url=base_url, api_key=api_key, table_info=table_info)
        usage = getattr(llm, "usage", None)
        if usage is not None and not own_usage:
            for tok in (resp.translation_tokens, resp.execution_tokens):
                usage.add(tok.get("prompt", 0), tok.get("completion", 0), 0)
        execution = PlanExecutionResult(data=list(resp.rows), operator_results={},
                                        root_operator=self._planning_strategy)
        answer, chart = AnswerGenerator(llm_provider=llm).generate(query_text, execution)
        return ExecutableResult(plan=None, execution=execution, answer=answer, chart=chart,
                                baseline=resp)

    def _parse_query(self, query_text: str,
                     sources: Optional[Dict[str, Table]] = None) -> QueryContext:
        """Compile the query context Phi with the LLM-based analyzer.  Source
        names, columns, extractable attributes, and exact sizes come from the
        metadata graph, or from the source tables when there is none."""
        if self._llm_provider is None:
            raise ValueError("the query analyzer needs the backbone LLM provider")
        if self._metadata_graph is not None:
            from .metadata_graph import (source_attributes, source_columns, source_tables,
                                         source_values)
            sizes = self._source_sizes()
            columns = source_columns(self._metadata_graph)
            attributes = source_attributes(self._metadata_graph)
            values = source_values(self._metadata_graph)
            tables = source_tables(self._metadata_graph)
            # modality of each source, from which M of Phi is derived
            modalities = {mid: meta.modality_type.value
                          for mid, meta in self._metadata_graph.modalities.items()}
        else:
            sizes = [(name, len(rows)) for name, rows in (sources or {}).items()]
            columns = {name: set(rows[0]) if rows else set() for name, rows in (sources or {}).items()}
            attributes, values = {}, {}     # w/o TiMetaGraph: no column statistics
            tables = {}                     # each source table is its own table
            modalities = {}
        context = QueryContext.from_query_llm(
            query_text, self._llm_provider,
            metadata_context=(self._metadata_context if self._metadata_context is not None
                              else describe_for_llm(self._metadata_graph, query_text)
                              if self._metadata_graph is not None else None),
            sources=[name for name, _ in sizes],
            source_columns=columns,
            source_attributes=attributes,
            source_values=values,
            source_tables=tables,
            source_modalities=modalities,
        )
        context.source_sizes = sizes
        if self._metadata_graph is not None:
            # statistics from which the cardinalities are estimated (Eq. 7)
            from .cardinality import source_statistics
            context.statistics = source_statistics(self._metadata_graph)
        return context

    def _source_sizes(self):
        if self._metadata_graph is None:
            return []
        from .metadata_graph import source_sizes
        return source_sizes(self._metadata_graph)

    def _execution_metadata(self, context: QueryContext, sources: Optional[Dict[str, Table]]) -> Dict[str, Any]:
        meta: Dict[str, Any] = {"bindings": context.bindings, "sources": sources or {}}
        if self._metadata_graph is not None:
            from .metadata_graph import (
                document_entities, entity_ids_by_surface, precomputed_attributes,
            )
            meta["doc_attributes"] = precomputed_attributes(self._metadata_graph)
            meta["doc_entities"] = document_entities(self._metadata_graph)
            meta["entity_ids_by_surface"] = entity_ids_by_surface(self._metadata_graph)
        return meta

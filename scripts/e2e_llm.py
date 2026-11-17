"""End-to-end run of the paper's running example with a real LLM.

Runs every component of the paper on a small movie dataset: TiMetaGraph
(TiProcessors with spaCy and VADER, entity resolution, relationship discovery),
TiQC (clarification with the LLM user simulator), and TiTSP (query context,
MCTS, uncertainty propagation, physical optimization, execution).

Usage (DashScope key in DASHSCOPE_API_KEY):
    python -m scripts.e2e_llm [model]          # default: deepseek-v4.1-flash-1017
"""
import json
import sys
import time

from src.core.query_planner.entity_relationship_discoverer import EntityRelationshipDiscoverer
from src.core.query_planner.llm_query_analyzer import create_provider
from src.core.query_planner.metadata_graph import MultimodalMetadataGraph, source_sizes
from src.core.query_planner.processors import (
    TableProcessor, TextProcessor, build_tools, merge_cross_modal_entities,
)
from src.core.query_planner.query_clarifier import TiQCClarifier
from src.core.query_planner.query_planner import QueryPlanner

MOVIES = [
    {"title": "Dune: Part Two", "genre": "Sci-Fi", "year": 2024, "rating": 8.5, "votes": 610000},
    {"title": "Alien: Romulus", "genre": "Sci-Fi", "year": 2024, "rating": 7.2, "votes": 230000},
    {"title": "The Creator", "genre": "Sci-Fi", "year": 2023, "rating": 6.7, "votes": 140000},
    {"title": "Arrival", "genre": "Sci-Fi", "year": 2016, "rating": 7.9, "votes": 780000},
    {"title": "Interstellar", "genre": "Sci-Fi", "year": 2014, "rating": 8.7, "votes": 2200000},
    {"title": "Oppenheimer", "genre": "Drama", "year": 2023, "rating": 8.3, "votes": 850000},
]
REVIEWS = [
    {"review": "Dune: Part Two is breathtaking; the sound and visuals are wonderful."},
    {"review": "I loved Dune 2. Stunning, moving, and beautifully shot."},
    {"review": "Alien: Romulus was tense at first but the ending felt lazy and disappointing."},
    {"review": "Romulus is a dull retread of better Alien films."},
    {"review": "The Creator looks gorgeous but the story is shallow and forgettable."},
    {"review": "Arrival is a masterpiece, quiet and deeply emotional."},
    {"review": "Interstellar is ambitious, but far too long."},
    {"review": "Oppenheimer is a great and powerful film."},
]
HIDDEN_INTENT = ("I want the sci-fi movie released in 2020 or later whose review texts have "
                 "the highest average sentiment (not the rating column).")
QUERY = "Which recent sci-fi movie has the most positive reviews?"


def section(title):
    print(f"\n=== {title} ===")


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "deepseek-v4.1-flash-1017"
    llm = create_provider("dashscope", model)
    t0 = time.time()

    section("TiMetaGraph")
    tools = build_tools(True)
    graph = MultimodalMetadataGraph(graph_id="movies")
    table = TableProcessor(llm, tools=tools).process("Movie", MOVIES, graph)
    text = TextProcessor(llm, tools=tools).process("Review", REVIEWS, ["review"], graph)
    merged = merge_cross_modal_entities(graph, llm=llm)
    relations = EntityRelationshipDiscoverer().discover_entity_relations(graph, llm=llm)
    print("sources:", source_sizes(graph))
    print("entities:", len(graph.entities), "| merged groups:", merged, "| relation edges:", len(relations))
    print("cross-modal entities:", sorted(e.canonical_name for e in graph.entities.values()
                                          if (e.properties or {}).get("cross_modal")))

    section("TiQC")
    tiqc = TiQCClarifier(llm, metadata_graph=graph)
    clarified = tiqc.clarify_query(QUERY, tiqc.make_user_simulator(HIDDEN_INTENT))
    for r in clarified["trace"]:
        amb = r["ambiguity"]
        print(f"- round {r['round']} [{amb['ambiguity_type']}, sev={amb['severity']}] "
              f"{amb['question']} options={r['options']} -> {r['answer']}")
    q_star = clarified["clarified_query"]
    print("q*:", q_star)

    section("TiTSP")
    planner = QueryPlanner(llm_provider=llm, metadata_graph=graph)
    result = planner.plan_and_execute(q_star, {table: MOVIES, text: REVIEWS})
    ctx = result.plan.query_context
    print("Sigma_req:", sorted(ctx.required_operators), "| root:", ctx.root_operator_type)
    print("bindings:", json.dumps(ctx.bindings, ensure_ascii=False))

    def show(n, d=0):
        print("  " * d + n.operator_type, n.source_name or "", n.source_cardinality or "",
              result.plan.selected_physical.assignment.get(n.node_id))
        for c in n.children:
            show(c, d + 1)

    show(result.plan.best_logical_plan)
    root = result.plan.best_logical_plan.node_id
    print("propagated root confidence:", round(result.plan.uncertainty_map[root].confidence, 3))
    for r in result.execution.operator_results.values():
        print(f"  {r.operator_type:18s} rows={r.row_count:3d}")
    print("result rows:", json.dumps(result.execution.data, ensure_ascii=False, default=str)[:600])
    print("answer:", result.answer)
    print("chart:", result.chart)
    print(f"\nwall time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

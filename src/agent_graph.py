"""
agent_graph.py

Orchestrates the full pipeline as a LangGraph state machine:

    fetch_data -> compute_stats -> write_narrative -> evaluate -> (retry once if failed) -> END

Each node does exactly one job and passes a typed state dict to the next.
This is what makes it an "agentic pipeline" rather than a single prompt:
each stage is independently testable, inspectable, and swappable (e.g. you
could swap write_narrative for a different model/prompt without touching
the stats engine at all).
"""

from __future__ import annotations

import os
from typing import TypedDict

from langgraph.graph import StateGraph, END

from data_loader import find_match, load_events
from eval_harness import EvalResult, check_report
from narrative_agent import generate_report
from stats_engine import build_match_brief


class PipelineState(TypedDict, total=False):
    competition_id: int
    season_id: int
    team_a: str
    team_b: str
    match_id: int
    match_brief: dict
    report: str
    eval_result: EvalResult
    retries: int
    status: str


MAX_RETRIES = 1


def node_fetch_data(state: PipelineState) -> PipelineState:
    match = find_match(state["competition_id"], state["season_id"], state["team_a"], state["team_b"])
    if match is None:
        return {**state, "status": "error: match not found"}
    events = load_events(match.match_id)
    state["_events"] = events  # not part of TypedDict on purpose (internal only)
    return {**state, "match_id": match.match_id, "status": "data_fetched"}


def node_compute_stats(state: PipelineState) -> PipelineState:
    events = state["_events"]
    brief = build_match_brief(state["match_id"], events)
    return {**state, "match_brief": brief.as_dict(), "status": "stats_computed"}


def node_write_narrative(state: PipelineState) -> PipelineState:
    report = generate_report(state["match_brief"])
    return {**state, "report": report, "status": "narrative_written"}


def node_evaluate(state: PipelineState) -> PipelineState:
    result = check_report(state["report"], state["match_brief"])
    status = "eval_passed" if result.passed else "eval_failed"
    return {**state, "eval_result": result, "status": status}


def route_after_eval(state: PipelineState) -> str:
    if state["status"] == "eval_passed":
        return "end"
    if state.get("retries", 0) < MAX_RETRIES:
        return "retry"
    return "end"  # give up after MAX_RETRIES, surface the failure to the user


def node_increment_retry(state: PipelineState) -> PipelineState:
    return {**state, "retries": state.get("retries", 0) + 1}


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("fetch_data", node_fetch_data)
    graph.add_node("compute_stats", node_compute_stats)
    graph.add_node("write_narrative", node_write_narrative)
    graph.add_node("evaluate", node_evaluate)
    graph.add_node("increment_retry", node_increment_retry)

    graph.set_entry_point("fetch_data")
    graph.add_edge("fetch_data", "compute_stats")
    graph.add_edge("compute_stats", "write_narrative")
    graph.add_edge("write_narrative", "evaluate")
    graph.add_conditional_edges(
        "evaluate", route_after_eval, {"retry": "increment_retry", "end": END}
    )
    graph.add_edge("increment_retry", "write_narrative")

    return graph.compile()


def run_pipeline(competition_id: int, season_id: int, team_a: str, team_b: str) -> PipelineState:
    app = build_graph()
    result = app.invoke(
        {
            "competition_id": competition_id,
            "season_id": season_id,
            "team_a": team_a,
            "team_b": team_b,
            "retries": 0,
        }
    )
    return result


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run the full pipeline end-to-end.")
    else:
        result = run_pipeline(43, 106, "Argentina", "France")
        print(result["report"])
        print("\n--- Eval ---")
        print(result["eval_result"].summary())

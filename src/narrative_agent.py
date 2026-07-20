"""
narrative_agent.py

Takes a structured MatchBrief (from stats_engine.py) and asks an LLM to write
a tactical analysis report in natural language.

The prompt explicitly constrains the model to only use numbers it has been
given — this is what eval_harness.py then verifies, catching hallucinated
stats before a report ships.
"""

from __future__ import annotations

import json
import os

from anthropic import Anthropic

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a football tactical analyst writing a match report.

You will be given a JSON "match brief" containing verified statistics for
both teams and a list of key attacking moments, all pre-computed from raw
match event data.

Rules:
1. Only use numbers that appear in the JSON you are given. Never invent,
   round dramatically, or estimate a statistic that isn't provided.
2. When you cite a number, use it exactly as given (e.g. if possession_pct
   is 46.3, write "46.3%", not "roughly half").
3. Write like a real tactical analyst: focus on WHY the numbers happened
   (patterns, phases of play, momentum shifts), not just a stat recital.
4. Structure: a short headline sentence, then 3-4 paragraphs covering
   territory/possession, attacking threat and shot quality, and pressing/
   defensive shape, then a one-line verdict.
5. Keep it under 400 words. Plain, direct prose. No bullet points.
"""


def generate_report(match_brief: dict, api_key: str | None = None) -> str:
    """Call Claude to turn a match brief into a tactical narrative report."""
    client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    user_prompt = (
        "Here is the match brief:\n\n"
        f"{json.dumps(match_brief, indent=2)}\n\n"
        "Write the tactical report now."
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    return "".join(block.text for block in response.content if block.type == "text")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_events
    from stats_engine import build_match_brief

    events = load_events(3869685)
    brief = build_match_brief(3869685, events)
    report = generate_report(brief.as_dict())
    print(report)

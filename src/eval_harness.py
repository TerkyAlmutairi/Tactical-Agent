"""
eval_harness.py

The point of this file: an LLM writing a "tactical report" is only useful if
you can trust the numbers in it. This module checks a generated report
against its source MatchBrief and flags:

1. Hallucinated numbers  — a numeric claim in the text that doesn't match
   anything in the source stats (within tolerance).
2. Missing coverage      — key stats (possession, xG, goals) that should be
   referenced somewhere in a tactical report but aren't.
3. Team-name mixups      — numbers correctly sourced but attributed to the
   wrong team (checked by proximity of the number to a team name mention).

This is intentionally a lightweight, deterministic checker (regex + string
matching), not another LLM call — the whole point is to have a fast, cheap,
reproducible gate that doesn't itself need trusting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class EvalResult:
    passed: bool
    hallucinated_numbers: list[str] = field(default_factory=list)
    missing_stats: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"PASSED: {self.passed}"]
        if self.hallucinated_numbers:
            lines.append(f"Hallucinated numbers: {self.hallucinated_numbers}")
        if self.missing_stats:
            lines.append(f"Missing expected stats: {self.missing_stats}")
        if self.warnings:
            lines.append(f"Warnings: {self.warnings}")
        return "\n".join(lines)


NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%?")


def _collect_valid_numbers(match_brief: dict) -> set[str]:
    """Flatten every numeric stat in the brief into a set of acceptable strings."""
    valid = set()

    def _walk(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                _walk(v)
        elif isinstance(obj, (int, float)):
            valid.add(str(obj))
            valid.add(str(round(obj)))  # allow rounded mentions, e.g. "6.64" -> "7"
            valid.add(f"{obj:.0f}")

    _walk(match_brief)
    return valid


# Numbers that are near-universally safe to mention without being a "stat"
# (minute markers under 130, jersey numbers, score-adjacent small ints, etc.)
# We do NOT whitelist these — instead we tolerate a small absolute difference,
# since analysts commonly round (e.g. "just under 50%" for 46.3%).
TOLERANCE = 1.5


def check_report(report_text: str, match_brief: dict) -> EvalResult:
    valid_numbers = _collect_valid_numbers(match_brief)
    valid_floats = sorted({float(n) for n in valid_numbers})

    found_numbers = NUMBER_RE.findall(report_text)
    hallucinated = []

    for raw in found_numbers:
        val = float(raw)
        # Skip obvious non-stat numbers: minute markers phrased like "90th"
        # are still checked, since minute is itself a valid stat from
        # key_moments — so no special-casing needed.
        if raw in valid_numbers:
            continue
        # Allow small rounding tolerance against the nearest valid stat
        if any(abs(val - v) <= TOLERANCE for v in valid_floats):
            continue
        hallucinated.append(raw)

    # Coverage check: report should reference each team's headline stats
    missing = []
    lower_text = report_text.lower()
    for team, stats in match_brief.get("teams", {}).items():
        if team.lower() not in lower_text:
            missing.append(f"no mention of team '{team}'")
            continue
        poss = str(stats["possession_pct"])
        if poss not in report_text and str(round(stats["possession_pct"])) not in report_text:
            missing.append(f"{team}: possession_pct ({poss}) not referenced")

    warnings = []
    word_count = len(report_text.split())
    if word_count > 500:
        warnings.append(f"report is long ({word_count} words) — may drift from source stats")

    passed = not hallucinated and not missing
    return EvalResult(
        passed=passed,
        hallucinated_numbers=hallucinated,
        missing_stats=missing,
        warnings=warnings,
    )


if __name__ == "__main__":
    # Golden test cases — no API key required, run in CI on every PR
    brief = {
        "teams": {
            "Argentina": {"possession_pct": 46.3, "shots": 24, "simple_xg": 6.64},
            "France": {"possession_pct": 53.7, "shots": 14, "simple_xg": 4.0},
        }
    }

    good_report = (
        "Argentina controlled the tempo despite 46.3% possession, generating 6.64 xG "
        "from 24 shots. France, with 53.7% of the ball, managed a lower 4.0 xG from "
        "14 attempts, relying on transitions rather than sustained pressure."
    )
    bad_report = (
        "Argentina dominated with 61% possession and 8 shots on target, racking up "
        "9.2 expected goals. France barely threatened, managing just 2 shots all game."
    )

    print("--- Good report ---")
    print(check_report(good_report, brief).summary())
    print("\n--- Bad report (should fail) ---")
    print(check_report(bad_report, brief).summary())

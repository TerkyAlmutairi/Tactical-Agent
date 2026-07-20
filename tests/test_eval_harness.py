import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval_harness import check_report  # noqa: E402

BRIEF = {
    "teams": {
        "Argentina": {"possession_pct": 46.3, "shots": 24, "simple_xg": 6.64},
        "France": {"possession_pct": 53.7, "shots": 14, "simple_xg": 4.0},
    }
}


def test_accurate_report_passes():
    report = (
        "Argentina controlled the tempo despite 46.3% possession, generating 6.64 xG "
        "from 24 shots. France, with 53.7% of the ball, managed a lower 4.0 xG from "
        "14 attempts."
    )
    result = check_report(report, BRIEF)
    assert result.passed
    assert not result.hallucinated_numbers
    assert not result.missing_stats


def test_hallucinated_number_is_caught():
    report = (
        "Argentina dominated with 61% possession and 24 shots, racking up 6.64 xG. "
        "France, with 53.7% of the ball, managed 4.0 xG from 14 attempts."
    )
    result = check_report(report, BRIEF)
    assert not result.passed
    assert "61" in result.hallucinated_numbers


def test_missing_team_mention_is_caught():
    report = "Argentina played well with 46.3% possession and 6.64 xG from 24 shots."
    result = check_report(report, BRIEF)
    assert not result.passed
    assert any("France" in m for m in result.missing_stats)


def test_small_rounding_is_tolerated():
    # 46.3 rounded to 46 or "just under 47" should not count as hallucinated
    report = (
        "Argentina held 46% possession and created 24 shots worth 6.64 xG. "
        "France's 53.7% possession yielded 4.0 xG from 14 shots."
    )
    result = check_report(report, BRIEF)
    assert not result.hallucinated_numbers

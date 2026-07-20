import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from stats_engine import simple_shot_xg  # noqa: E402


def test_xg_center_penalty_spot_is_high():
    # Roughly the penalty spot: (108, 40) in StatsBomb coords.
    # Note: this proxy model has no explicit "is this a penalty" flag, so it
    # scores the location on distance/angle alone (~0.44) rather than the
    # ~0.76 real-world penalty conversion rate. That's a known, documented
    # limitation of the simple model -- see README "Limitations".
    xg = simple_shot_xg(108, 40, "Right Foot", is_open_play=False)
    assert xg > 0.4


def test_xg_tight_angle_near_byline_is_low():
    # Very close to the goal line but wide (tight angle)
    xg = simple_shot_xg(118, 5, "Right Foot", is_open_play=True)
    assert xg < 0.2


def test_xg_headers_penalized_vs_same_spot_footed_shot():
    x, y = 105, 40
    foot_xg = simple_shot_xg(x, y, "Right Foot", True)
    head_xg = simple_shot_xg(x, y, "Head", True)
    assert head_xg < foot_xg


def test_xg_further_shots_score_lower():
    close_xg = simple_shot_xg(110, 40, "Right Foot", True)
    far_xg = simple_shot_xg(70, 40, "Right Foot", True)
    assert far_xg < close_xg


def test_xg_bounded_between_0_and_1():
    for x, y in [(0, 0), (120, 40), (60, 0), (100, 80)]:
        xg = simple_shot_xg(x, y, "Right Foot", True)
        assert 0.0 <= xg <= 1.0

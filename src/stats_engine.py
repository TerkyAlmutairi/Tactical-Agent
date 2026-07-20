"""
stats_engine.py

Turns raw StatsBomb event data into a compact, structured "match brief":
possession, shot quality, territory, pressing, and passing network summaries
for each team.

Deliberately uses transparent, explainable formulas rather than a trained
model. This makes every number traceable back to source events, which is
exactly what the eval harness (eval_harness.py) checks against later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

# Standard pitch dimensions in StatsBomb's coordinate system (120 x 80)
PITCH_LENGTH = 120
PITCH_WIDTH = 80
GOAL_CENTER = (120, 40)


@dataclass
class TeamStats:
    team: str
    possession_pct: float
    shots: int
    shots_on_target: int
    goals: int
    simple_xg: float
    avg_shot_quality: float
    passes_completed: int
    pass_accuracy_pct: float
    final_third_entries: int
    high_press_actions: int  # pressures/recoveries in opposition's defensive third
    territory_pct: float  # share of touches in attacking third


@dataclass
class MatchBrief:
    match_id: int
    teams: dict[str, TeamStats] = field(default_factory=dict)
    key_moments: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "match_id": self.match_id,
            "teams": {k: vars(v) for k, v in self.teams.items()},
            "key_moments": self.key_moments,
        }


def _shot_distance_angle(x: float, y: float) -> tuple[float, float]:
    """Distance to goal center (in metres-equivalent units) and shot angle (deg)."""
    dx = GOAL_CENTER[0] - x
    dy = GOAL_CENTER[1] - y
    distance = math.hypot(dx, dy)
    # Angle subtended by the goal mouth (7.32m wide) from the shot location
    goal_width = 7.32 * (PITCH_LENGTH / 105)  # scale to statsbomb units roughly
    angle = math.degrees(math.atan2(goal_width / 2, distance)) * 2 if distance > 0 else 0
    return distance, angle


def simple_shot_xg(x: float, y: float, body_part: str, is_open_play: bool) -> float:
    """
    A transparent, hand-built xG proxy — NOT a trained model.

    Uses distance + angle to goal as the dominant signal (this explains the
    vast majority of variance in real xG models), with small adjustments for
    body part and set-piece context. Deliberately simple so every prediction
    is traceable to a formula rather than a black box.
    """
    distance, angle = _shot_distance_angle(x, y)
    if distance <= 0:
        return 0.0

    # Base probability decays with distance, boosted by shooting angle
    base = 1.0 / (1.0 + math.exp((distance - 12) / 5))
    angle_factor = min(angle / 40, 1.5)
    xg = base * (0.4 + 0.6 * angle_factor)

    if body_part == "Head":
        xg *= 0.75
    if not is_open_play:
        xg *= 0.9

    return round(min(max(xg, 0.01), 0.95), 3)


def compute_team_stats(events: pd.DataFrame, team_name: str) -> TeamStats:
    team_events = events[events["team.name"] == team_name]
    all_teams_events = events

    # --- Possession: share of total ball-in-play time attributed to team ---
    poss_counts = events.groupby("possession_team.name")["possession"].nunique()
    total_possessions = poss_counts.sum()
    possession_pct = (
        round(100 * poss_counts.get(team_name, 0) / total_possessions, 1)
        if total_possessions
        else 0.0
    )

    # --- Shots & xG ---
    shots = team_events[team_events["type.name"] == "Shot"]
    n_shots = len(shots)
    goals = len(shots[shots["shot.outcome.name"] == "Goal"]) if n_shots else 0
    on_target_outcomes = {"Goal", "Saved", "Saved to Post"}
    shots_on_target = (
        len(shots[shots["shot.outcome.name"].isin(on_target_outcomes)]) if n_shots else 0
    )

    xg_values = []
    for _, s in shots.iterrows():
        loc = s.get("location")
        if not isinstance(loc, list) or len(loc) < 2:
            continue
        body_part = s.get("shot.body_part.name", "")
        play_pattern = s.get("play_pattern.name", "Regular Play")
        is_open_play = play_pattern == "Regular Play"
        xg = simple_shot_xg(loc[0], loc[1], body_part, is_open_play)
        xg_values.append(xg)

    simple_xg_total = round(sum(xg_values), 2)
    avg_shot_quality = round(sum(xg_values) / len(xg_values), 3) if xg_values else 0.0

    # --- Passing ---
    passes = team_events[team_events["type.name"] == "Pass"]
    n_passes = len(passes)
    completed = len(passes[passes["pass.outcome.name"].isna()]) if n_passes else 0
    pass_accuracy = round(100 * completed / n_passes, 1) if n_passes else 0.0

    # Final third entries: passes ending with x >= 80 (of 120)
    def _end_x(row):
        end = row.get("pass.end_location")
        return end[0] if isinstance(end, list) and len(end) >= 1 else None

    if n_passes:
        end_xs = passes.apply(_end_x, axis=1)
        final_third_entries = int((end_xs >= 80).sum())
    else:
        final_third_entries = 0

    # --- Pressing: pressure events applied in opposition defensive third (x >= 80 from attacker's perspective) ---
    pressures = team_events[team_events["type.name"] == "Pressure"]

    def _loc_x(row):
        loc = row.get("location")
        return loc[0] if isinstance(loc, list) and len(loc) >= 1 else None

    if len(pressures):
        press_x = pressures.apply(_loc_x, axis=1)
        high_press_actions = int((press_x >= 80).sum())
    else:
        high_press_actions = 0

    # --- Territory: share of all touches (any event with a location) in attacking third ---
    touch_events = team_events[team_events["location"].apply(lambda l: isinstance(l, list))]
    if len(touch_events):
        touch_x = touch_events["location"].apply(lambda l: l[0])
        territory_pct = round(100 * (touch_x >= 80).sum() / len(touch_x), 1)
    else:
        territory_pct = 0.0

    return TeamStats(
        team=team_name,
        possession_pct=possession_pct,
        shots=n_shots,
        shots_on_target=shots_on_target,
        goals=goals,
        simple_xg=simple_xg_total,
        avg_shot_quality=avg_shot_quality,
        passes_completed=completed,
        pass_accuracy_pct=pass_accuracy,
        final_third_entries=final_third_entries,
        high_press_actions=high_press_actions,
        territory_pct=territory_pct,
    )


def compute_key_moments(events: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """Pull out the highest-xG shots of the match as 'key moments' for the narrative agent."""
    shots = events[events["type.name"] == "Shot"].copy()
    moments = []
    for _, s in shots.iterrows():
        loc = s.get("location")
        if not isinstance(loc, list):
            continue
        body_part = s.get("shot.body_part.name", "")
        play_pattern = s.get("play_pattern.name", "Regular Play")
        xg = simple_shot_xg(loc[0], loc[1], body_part, play_pattern == "Regular Play")
        moments.append(
            {
                "minute": int(s.get("minute", 0)),
                "team": s.get("team.name"),
                "player": s.get("player.name"),
                "outcome": s.get("shot.outcome.name"),
                "xg": xg,
            }
        )
    moments.sort(key=lambda m: m["xg"], reverse=True)
    return moments[:top_n]


def build_match_brief(match_id: int, events: pd.DataFrame) -> MatchBrief:
    teams = [t for t in events["team.name"].dropna().unique()]
    brief = MatchBrief(match_id=match_id)
    for t in teams:
        brief.teams[t] = compute_team_stats(events, t)
    brief.key_moments = compute_key_moments(events)
    return brief


if __name__ == "__main__":
    from data_loader import load_events

    events = load_events(3869685)
    brief = build_match_brief(3869685, events)
    import json

    print(json.dumps(brief.as_dict(), indent=2))

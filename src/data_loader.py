"""
data_loader.py

Fetches free, open match event data from StatsBomb's open-data repository
(https://github.com/statsbomb/open-data) and loads it into a pandas DataFrame.

No API key or scraping required — StatsBomb publish this data publicly for
research and non-commercial use.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
import requests

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


@dataclass
class MatchInfo:
    match_id: int
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    competition: str
    stage: str
    match_date: str


def list_matches(competition_id: int, season_id: int) -> list[MatchInfo]:
    """List all matches available for a given competition/season."""
    url = f"{BASE_URL}/matches/{competition_id}/{season_id}.json"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    raw = resp.json()

    matches = []
    for m in raw:
        matches.append(
            MatchInfo(
                match_id=m["match_id"],
                home_team=m["home_team"]["home_team_name"],
                away_team=m["away_team"]["away_team_name"],
                home_score=m["home_score"],
                away_score=m["away_score"],
                competition=m["competition"]["competition_name"],
                stage=m["competition_stage"]["name"],
                match_date=m["match_date"],
            )
        )
    return matches


def find_match(
    competition_id: int, season_id: int, team_a: str, team_b: str
) -> MatchInfo | None:
    """Convenience lookup: find a match between two named teams."""
    for m in list_matches(competition_id, season_id):
        teams = {m.home_team, m.away_team}
        if team_a in teams and team_b in teams:
            return m
    return None


def load_events(match_id: int) -> pd.DataFrame:
    """
    Load the full event stream for a match as a DataFrame.

    Each row is one on-ball / off-ball event (pass, shot, carry, etc).
    Nested StatsBomb JSON fields are flattened with pandas.json_normalize.
    """
    url = f"{BASE_URL}/events/{match_id}.json"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    raw = resp.json()

    df = pd.json_normalize(raw, sep=".")
    return df


def load_lineups(match_id: int) -> pd.DataFrame:
    """Load starting lineups + formations for a match."""
    url = f"{BASE_URL}/lineups/{match_id}.json"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    raw = resp.json()
    return pd.json_normalize(raw, sep=".")


if __name__ == "__main__":
    # Quick smoke test: 2022 World Cup Final, Argentina vs France
    info = find_match(43, 106, "Argentina", "France")
    print(info)
    events = load_events(info.match_id)
    print(f"{len(events)} events loaded, {events['team.name'].nunique()} teams")

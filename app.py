"""
app.py

Streamlit front end for the Tactical Analyst Agent.

Pick a competition/season and two teams, and the app runs the full
LangGraph pipeline (fetch data -> compute stats -> write narrative ->
evaluate) live, then shows the report alongside the eval result and the
raw stats it was built from.

Run locally:   streamlit run app.py
Deploy free:   https://share.streamlit.io  (see README for steps)
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import list_matches  # noqa: E402
from agent_graph import run_pipeline  # noqa: E402

st.set_page_config(page_title="Tactical Analyst Agent", page_icon="⚽", layout="centered")

st.title("⚽ Tactical Analyst Agent")
st.caption(
    "An agentic pipeline (LangGraph) that turns raw match event data into a "
    "verified tactical report. Data: StatsBomb Open Data. "
    "Every number in the report is checked against source stats before it's shown."
)

# A small curated set of competitions with known-good open data, to keep the
# demo simple. See README for how to point this at any StatsBomb competition.
COMPETITIONS = {
    "FIFA World Cup 2022": (43, 106),
    "FIFA World Cup 2018": (43, 3),
}

with st.sidebar:
    st.header("Match selection")
    comp_name = st.selectbox("Competition", list(COMPETITIONS.keys()))
    competition_id, season_id = COMPETITIONS[comp_name]

    if "matches" not in st.session_state or st.session_state.get("_comp") != comp_name:
        with st.spinner("Loading match list..."):
            st.session_state["matches"] = list_matches(competition_id, season_id)
            st.session_state["_comp"] = comp_name

    matches = st.session_state["matches"]
    options = {
        f"{m.home_team} {m.home_score}-{m.away_score} {m.away_team} ({m.stage})": m
        for m in matches
    }
    choice = st.selectbox("Match", list(options.keys()))
    selected = options[choice]

    api_key = st.text_input(
        "Anthropic API key",
        type="password",
        help="Get one at console.anthropic.com. Not stored anywhere.",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
    )

    run_button = st.button("Generate report", type="primary", use_container_width=True)

if run_button:
    if not api_key:
        st.error("Add an Anthropic API key in the sidebar to generate the narrative.")
    else:
        os.environ["ANTHROPIC_API_KEY"] = api_key
        with st.spinner("Running pipeline: fetching data → computing stats → writing report → evaluating..."):
            result = run_pipeline(
                competition_id, season_id, selected.home_team, selected.away_team
            )

        eval_result = result["eval_result"]

        if eval_result.passed:
            st.success("Report passed the eval check — every number is traceable to source stats.")
        else:
            st.warning(
                "Report flagged by the eval harness after retry. Showing it anyway, "
                "with the flagged issues below, so you can see what the check caught."
            )

        st.subheader("Tactical Report")
        st.write(result["report"])

        with st.expander("Eval harness result"):
            if eval_result.hallucinated_numbers:
                st.write("**Hallucinated numbers found:**", eval_result.hallucinated_numbers)
            if eval_result.missing_stats:
                st.write("**Missing expected stats:**", eval_result.missing_stats)
            if eval_result.warnings:
                st.write("**Warnings:**", eval_result.warnings)
            if eval_result.passed:
                st.write("No issues found.")

        with st.expander("Raw match stats (source of truth for the report)"):
            teams_df = pd.DataFrame(result["match_brief"]["teams"]).T
            st.dataframe(teams_df)

            st.write("**Key moments (highest shot quality):**")
            st.dataframe(pd.DataFrame(result["match_brief"]["key_moments"]))
else:
    st.info("Pick a match in the sidebar and click **Generate report**.")

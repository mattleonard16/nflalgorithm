"""Week-by-week Streamlit dashboard for NFL algorithm."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Add parent directory to path for imports
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import config
from materialized_value_view import materialize_week


st.set_page_config(
    page_title="NFL Algorithm Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

FRESHNESS_THRESHOLDS = {
    'odds': 2,
    'injuries': 30,
    'weather': 60
}


def get_available_seasons_weeks() -> tuple[list[int], dict[int, list[int]]]:
    with sqlite3.connect(config.database.path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT season, week FROM weekly_projections ORDER BY season DESC, week DESC"
        ).fetchall()

    if not rows:
        current_year = datetime.utcnow().year
        return [current_year], {current_year: [1]}

    seasons: dict[int, list[int]] = {}
    for season, week in rows:
        seasons.setdefault(season, []).append(int(week))

    for weeks in seasons.values():
        weeks.sort(reverse=True)

    ordered_seasons = sorted(seasons.keys(), reverse=True)
    return ordered_seasons, seasons


def load_weekly_value(season: int, week: int) -> pd.DataFrame:
    query = """
        SELECT mv.*, wp.team, wp.opponent, wp.model_version,
               ps.name AS player_name, ps.position
        FROM materialized_value_view mv
        LEFT JOIN weekly_projections wp
          ON mv.season = wp.season
         AND mv.week = wp.week
         AND mv.player_id = wp.player_id
         AND mv.market = wp.market
        LEFT JOIN player_stats_enhanced ps
          ON ps.player_id = mv.player_id
         AND ps.season = mv.season
         AND ps.week = mv.week
        WHERE mv.season = ? AND mv.week = ?
    """

    with sqlite3.connect(config.database.path) as conn:
        df = pd.read_sql_query(query, conn, params=(season, week))

    if df.empty:
        return df

    df['player_name'] = df['player_name'].fillna(df['player_id'])
    df['position'] = df['position'].fillna('FLEX')
    df['prop_type'] = df['market'].str.replace('_', ' ').str.title()
    df['edge_percentage'] = df['edge_percentage'].astype(float)
    df['expected_roi'] = df['expected_roi'].astype(float)
    df['p_win'] = df['p_win'].astype(float)
    df['kelly_fraction'] = df['kelly_fraction'].astype(float)
    df['stake'] = df['stake'].astype(float)
    df['line'] = df['line'].astype(float)
    df['price'] = df['price'].astype(int)
    df['edge_pct_display'] = df['edge_percentage'] * 100
    df['expected_roi_display'] = df['expected_roi'] * 100
    df['p_win_display'] = df['p_win'] * 100
    df['kelly_pct_display'] = df['kelly_fraction'] * 100
    return df


def load_feed_freshness(season: int, week: int) -> pd.DataFrame:
    with sqlite3.connect(config.database.path) as conn:
        df = pd.read_sql_query(
            "SELECT feed, as_of FROM feed_freshness WHERE season = ? AND week = ?",
            conn,
            params=(season, week)
        )
    if df.empty:
        return df
    df['as_of'] = pd.to_datetime(df['as_of'])
    df['age_minutes'] = (datetime.utcnow() - df['as_of']).dt.total_seconds() / 60.0
    df['threshold'] = df['feed'].map(FRESHNESS_THRESHOLDS).fillna(60)
    df['status'] = np.where(df['age_minutes'] <= df['threshold'], 'PASS', 'STALE')
    return df


def load_weekly_clv(season: int, week: int) -> pd.DataFrame:
    with sqlite3.connect(config.database.path) as conn:
        try:
            df = pd.read_sql_query(
                """
                SELECT cw.*, bw.sportsbook
                FROM clv_weekly cw
                LEFT JOIN bets_weekly bw ON cw.bet_id = bw.bet_id
                WHERE bw.season = ? AND bw.week = ?
                """,
                conn,
                params=(season, week)
            )
        except Exception:
            return pd.DataFrame()
    return df


def main() -> None:
    st.title("NFL Algorithm Professional Dashboard")
    st.markdown("**Weekly projections, value bets, and system health**")

    seasons, season_map = get_available_seasons_weeks()
    season = st.sidebar.selectbox("Season", seasons, index=0)
    week = st.sidebar.selectbox("Week", season_map.get(season, [1]), index=0)

    presentation_mode = st.sidebar.toggle("Presentation Mode", value=False)
    quick_bet_mode = st.sidebar.toggle("Quick Bet mode", value=False)
    min_edge_slider = st.sidebar.slider("Minimum Edge %", min_value=0.0, max_value=30.0, value=5.0, step=0.5)
    include_pass = st.sidebar.toggle("Include PASS rows", value=True)

    if st.sidebar.button("Refresh Data"):
        with st.spinner("Materializing weekly view..."):
            materialize_week(season, week, min_edge=min_edge_slider / 100)
        st.success("Weekly view refreshed")

    value_bets = load_weekly_value(season, week)
    edge_threshold = min_edge_slider / 100.0

    if not value_bets.empty:
        value_bets['recommendation'] = np.where(value_bets['edge_percentage'] >= edge_threshold, 'BET', 'PASS')
        if not include_pass:
            value_bets = value_bets[value_bets['recommendation'] == 'BET']

    clv_data = load_weekly_clv(season, week)

    col1, col2, col3, col4 = st.columns(4)
    active_count = int((value_bets['recommendation'] == 'BET').sum()) if not value_bets.empty else 0
    avg_roi = value_bets['expected_roi'].mean() if not value_bets.empty else 0.0
    avg_clv = ((value_bets['mu'] - value_bets['line']).mean() * 100) if not value_bets.empty else 0.0

    col1.metric("Active Value Bets", active_count)
    col2.metric("Avg Expected ROI", f"{avg_roi:.1%}")
    col3.metric("Avg CLV (bps)", f"{avg_clv:.1f}")
    col4.metric("System Status", "ACTIVE" if datetime.now().hour in range(8, 23) else "MAINTENANCE")

    tab1, tab2, tab3, tab4 = st.tabs(["Live Bets", "Performance", "CLV Analysis", "System"])

    with tab1:
        st.header(f"Season {season} • Week {week}")
        if value_bets.empty:
            st.info("No value bets for the selected filters.")
        else:
            display = value_bets.copy()
            display['model_prediction'] = display['mu']
            display['p_win_pct'] = display['p_win'] * 100
            display['edge_pct'] = display['edge_percentage'] * 100
            display['roi_pct'] = display['expected_roi'] * 100
            display['kelly_pct'] = display['kelly_fraction'] * 100
            if quick_bet_mode:
                quick_cols = ['player_name', 'prop_type', 'sportsbook', 'line', 'price', 'edge_pct', 'stake']
                st.dataframe(display[quick_cols], use_container_width=True)
            else:
                columns = [
                    'player_name', 'position', 'team', 'opponent', 'prop_type', 'sportsbook',
                    'line', 'price', 'model_prediction', 'sigma', 'p_win_pct', 'edge_pct',
                    'roi_pct', 'kelly_pct', 'stake', 'recommendation'
                ]
                st.dataframe(display[columns], use_container_width=True)

    with tab2:
        st.header("Performance Snapshot")
        if value_bets.empty:
            st.info("No data available.")
        else:
            fig = px.histogram(value_bets, x='edge_pct_display', nbins=20, title="Edge Distribution")
            st.plotly_chart(fig, use_container_width=True)
            fig2 = px.box(value_bets, x='position', y='expected_roi_display', title="Expected ROI by Position")
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.header("Closing Line Value")
        if clv_data.empty:
            st.info("No CLV records captured yet.")
        else:
            st.dataframe(clv_data, use_container_width=True)

    with tab4:
        st.header("Feed Freshness")
        freshness = load_feed_freshness(season, week)
        if freshness.empty:
            st.info("No freshness records. Run health check or update pipeline.")
        else:
            st.dataframe(freshness[['feed', 'as_of', 'age_minutes', 'status']], use_container_width=True)

    if not presentation_mode:
        st.sidebar.caption("Edge filter applies after refresh. Click Refresh Data to rebuild view.")


if __name__ == "__main__":
    main()

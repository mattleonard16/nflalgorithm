"""
NFL Algorithm Professional Dashboard
====================================
Weekly projections, value bets, and performance tracking.
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import config
from utils.db import get_connection, read_dataframe
from materialized_value_view import materialize_week
from utils.player_id_utils import canonicalize_team


st.set_page_config(
    page_title="NFL Algorithm | Value Betting",
    page_icon="football",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal styling - optimized for performance + overflow chips
st.markdown("""
<style>
    .stApp {
        background: #1e1e1e;
    }
    [data-testid="stMetric"] {
        background: #252525;
        border: 1px solid #333;
        border-radius: 4px;
        padding: 12px;
    }
    [data-testid="stMetricValue"] {
        color: #eee !important;
    }
    [data-testid="stMetricLabel"] {
        color: #999 !important;
    }
    h1, h2, h3 {
        color: #eee !important;
    }
    [data-testid="stSidebar"] {
        background: #1a1a1a;
    }
    .stTabs [data-baseweb="tab"] {
        background: #252525;
        color: #999;
    }
    .stTabs [aria-selected="true"] {
        background: #333 !important;
        color: #eee !important;
    }
    
    /* Overflow chip container */
    .chip-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        max-height: 40px;
        overflow: hidden;
        position: relative;
    }
    .chip-container.expanded {
        max-height: none;
    }
    .chip {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        background: #333;
        border-radius: 16px;
        font-size: 12px;
        color: #eee;
        white-space: nowrap;
    }
    .chip.premium { background: #4a3728; color: #fbbf24; }
    .chip.strong { background: #3b2f2f; color: #f87171; }
    .chip.standard { background: #1e3a5f; color: #60a5fa; }
    .chip-more {
        background: #444;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)


FRESHNESS_THRESHOLDS = {'odds': 2, 'injuries': 30, 'weather': 60}


def render_overflow_chips(items: list, max_visible: int = 5, chip_class: str = "chip"):
    """Render items as chips with overflow indicator."""
    if not items:
        return
    
    visible = items[:max_visible]
    hidden_count = len(items) - max_visible
    
    chips_html = ''.join([f'<span class="{chip_class}">{item}</span>' for item in visible])
    
    if hidden_count > 0:
        chips_html += f'<span class="chip chip-more">+{hidden_count} more</span>'
    
    st.markdown(f'<div class="chip-container">{chips_html}</div>', unsafe_allow_html=True)


def get_confidence_tier(edge_pct: float) -> tuple[str, str]:
    """Return (tier_name, marker) based on edge percentage."""
    if edge_pct >= 0.15:
        return "Premium", "***"
    elif edge_pct >= 0.10:
        return "Strong", "**"
    elif edge_pct >= 0.05:
        return "Standard", "*"
    else:
        return "Low", ""


def get_available_seasons_weeks() -> tuple[list[int], dict[int, list[int]]]:
    try:
        df = read_dataframe(
            "SELECT DISTINCT season, week FROM weekly_projections ORDER BY season DESC, week DESC"
        )
        rows = df.values.tolist()
    except Exception:
        rows = []

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


@st.cache_data(ttl=60)
def _fetch_weekly_value(season: int, week: int) -> pd.DataFrame:
    """Cached database query for weekly value bets."""
    query = """
        SELECT mv.*, wp.team AS team_projection, wp.opponent, wp.model_version,
               ps.name AS player_name, ps.position, ps.team AS team_stats
        FROM materialized_value_view mv
        LEFT JOIN weekly_projections wp
          ON mv.season = wp.season AND mv.week = wp.week 
          AND mv.player_id = wp.player_id AND mv.market = wp.market
        LEFT JOIN player_stats_enhanced ps
          ON ps.player_id = mv.player_id AND ps.season = mv.season AND ps.week = mv.week
        WHERE mv.season = ? AND mv.week = ?
    """
    return read_dataframe(query, params=(season, week))


def load_weekly_value(season: int, week: int) -> pd.DataFrame:
    df = _fetch_weekly_value(season, week)

    if df.empty:
        return df

    df['is_synthetic'] = df['sportsbook'].str.lower().eq('simbook')
    
    df['team_stats_canon'] = df.get('team_stats').apply(canonicalize_team) if 'team_stats' in df.columns else ''
    df['team_proj_canon'] = df.get('team_projection').apply(canonicalize_team) if 'team_projection' in df.columns else ''
    df['team_odds_canon'] = df.get('team_odds').apply(canonicalize_team) if 'team_odds' in df.columns else ''

    team_effective = df['team_stats_canon']
    mask_missing = team_effective == ''
    team_effective = team_effective.where(~mask_missing, df['team_proj_canon'])
    mask_missing = team_effective == ''
    team_effective = team_effective.where(~mask_missing, df['team_odds_canon'])

    df['team'] = team_effective
    df['team_display'] = df['team']
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
    
    df['tier'], df['tier_marker'] = zip(*df['edge_percentage'].apply(get_confidence_tier))
    df['confidence'] = df['tier']
    
    return df


@st.cache_data(ttl=120)
def load_performance_history() -> pd.DataFrame:
    """Cached query for performance history."""
    try:
        return read_dataframe("""
            SELECT season, week, total_bets, wins, losses, pushes, 
                   profit_units, roi_pct, avg_edge, best_bet, worst_bet
            FROM weekly_performance
            ORDER BY season DESC, week DESC
        """)
    except Exception:
        return pd.DataFrame()


def load_feed_freshness(season: int, week: int) -> pd.DataFrame:
    df = read_dataframe(
        "SELECT feed, as_of FROM feed_freshness WHERE season = ? AND week = ?",
        params=(season, week),
    )
    if df.empty:
        return df
    df['as_of'] = pd.to_datetime(df['as_of'])
    df['age_minutes'] = (datetime.utcnow() - df['as_of']).dt.total_seconds() / 60.0
    df['threshold'] = df['feed'].map(FRESHNESS_THRESHOLDS).fillna(60)
    df['status'] = np.where(df['age_minutes'] <= df['threshold'], 'FRESH', 'STALE')
    return df


def render_kpi_cards(value_bets: pd.DataFrame, perf_history: pd.DataFrame):
    col1, col2, col3, col4, col5 = st.columns(5)
    
    active_count = int((value_bets['edge_percentage'] >= 0.05).sum()) if not value_bets.empty else 0
    premium_count = int((value_bets['edge_percentage'] >= 0.15).sum()) if not value_bets.empty else 0
    avg_edge = value_bets['edge_percentage'].mean() * 100 if not value_bets.empty else 0.0
    
    if not perf_history.empty:
        total_wins = perf_history['wins'].sum()
        total_losses = perf_history['losses'].sum()
        season_profit = perf_history['profit_units'].sum()
        season_roi = (season_profit / perf_history['total_bets'].sum() * 100) if perf_history['total_bets'].sum() > 0 else 0
    else:
        total_wins, total_losses, season_profit, season_roi = 0, 0, 0, 0
    
    col1.metric("Active Bets", active_count)
    col2.metric("Premium Picks", premium_count)
    col3.metric("Avg Edge", f"{avg_edge:.1f}%")
    col4.metric("Season Record", f"{total_wins}-{total_losses}")
    col5.metric("Season P/L", f"{season_profit:+.1f}u", delta=f"{season_roi:+.1f}% ROI")
    
    # Quick picks overflow chips
    if not value_bets.empty:
        premium_players = value_bets[value_bets['edge_percentage'] >= 0.15]['player_name'].head(8).tolist()
        if premium_players:
            st.markdown("##### Top Picks")
            render_overflow_chips(premium_players, max_visible=5, chip_class="chip premium")


def render_live_bets_tab(value_bets: pd.DataFrame, edge_threshold: float, quick_mode: bool, best_line_only: bool):
    st.subheader("Value Opportunities")
    
    if value_bets.empty:
        st.info("No value bets for the selected filters.")
        return
    
    display = value_bets.copy()
    
    if best_line_only:
        display = display.loc[
            display.groupby(['player_id', 'market'])['edge_percentage'].idxmax()
        ].reset_index(drop=True)
    
    display['recommendation'] = np.where(display['edge_percentage'] >= edge_threshold, 'BET', 'PASS')
    display = display[display['edge_percentage'] >= edge_threshold]
    
    if display.empty:
        st.info("No bets meet the current edge threshold.")
        return
    
    display['model_prediction'] = display['mu'].round(1)
    display['edge_pct'] = (display['edge_percentage'] * 100).round(1)
    display['roi_pct'] = (display['expected_roi'] * 100).round(1)
    display['p_win_pct'] = (display['p_win'] * 100).round(1)
    display['kelly_pct'] = (display['kelly_fraction'] * 100).round(1)
    
    tier_order = {'Premium': 0, 'Strong': 1, 'Standard': 2, 'Low': 3}
    display['tier_sort'] = display['tier'].map(tier_order)
    display = display.sort_values(['tier_sort', 'edge_percentage'], ascending=[True, False])
    
    if quick_mode:
        columns = ['confidence', 'player_name', 'prop_type', 'sportsbook', 'line', 'price', 'edge_pct', 'recommendation']
    else:
        columns = [
            'confidence', 'player_name', 'position', 'team_display', 'opponent', 'prop_type',
            'sportsbook', 'line', 'price', 'model_prediction', 'edge_pct', 'p_win_pct', 
            'roi_pct', 'kelly_pct', 'stake', 'recommendation'
        ]
    
    premium = display[display['tier'] == 'Premium']
    strong = display[display['tier'] == 'Strong']
    standard = display[display['tier'] == 'Standard']
    
    if not premium.empty:
        st.markdown("### Premium Picks (15%+ Edge)")
        st.dataframe(premium[columns], use_container_width=True, hide_index=True)
    
    if not strong.empty:
        st.markdown("### Strong Picks (10-15% Edge)")
        st.dataframe(strong[columns], use_container_width=True, hide_index=True)
    
    if not standard.empty:
        st.markdown("### Standard Picks (5-10% Edge)")
        st.dataframe(standard[columns], use_container_width=True, hide_index=True)


@st.cache_data(ttl=300)
def _compute_perf_stats(perf_df_json: str) -> dict:
    """Cache performance calculations to avoid recomputation."""
    perf = pd.read_json(perf_df_json)
    if perf.empty:
        return {'empty': True}
    total_bets = int(perf['total_bets'].sum())
    total_wins = int(perf['wins'].sum())
    total_losses = int(perf['losses'].sum())
    total_profit = float(perf['profit_units'].sum())
    win_rate = (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0
    avg_roi = float(perf['roi_pct'].mean())
    return {
        'empty': False,
        'total_bets': total_bets,
        'total_wins': total_wins,
        'total_losses': total_losses,
        'total_profit': total_profit,
        'win_rate': win_rate,
        'avg_roi': avg_roi,
    }


def render_performance_tab(perf_history: pd.DataFrame):
    st.subheader("Historical Performance")
    
    if perf_history.empty:
        st.info("No performance history yet. Run `make week-grade` after games complete.")
        return
    
    stats = _compute_perf_stats(perf_history.to_json())
    if stats.get('empty'):
        return
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Bets", stats['total_bets'])
    col2.metric("Win Rate", f"{stats['win_rate']:.1f}%")
    col3.metric("Total Profit", f"{stats['total_profit']:+.1f}u")
    col4.metric("Avg ROI", f"{stats['avg_roi']:+.1f}%")
    
    # Defer chart rendering with expander to avoid blocking
    if len(perf_history) > 1:
        with st.expander("Show Profit Chart", expanded=False):
            perf_sorted = perf_history.sort_values(['season', 'week'])
            perf_sorted['cumulative_profit'] = perf_sorted['profit_units'].cumsum()
            perf_sorted['week_label'] = 'Week ' + perf_sorted['week'].astype(str)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=perf_sorted['week_label'],
                y=perf_sorted['cumulative_profit'],
                mode='lines+markers',
                name='Cumulative Profit',
                line=dict(color='#6b7280', width=2),
                marker=dict(size=6, color='#9ca3af')
            ))
            fig.update_layout(
                title='Cumulative Profit (Units)',
                xaxis_title='Week',
                yaxis_title='Profit (Units)',
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.1)',
                font=dict(color='#9ca3af'),
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Week-by-Week Breakdown")
    display_cols = ['season', 'week', 'total_bets', 'wins', 'losses', 'profit_units', 'roi_pct', 'avg_edge', 'best_bet']
    st.dataframe(perf_history[display_cols], use_container_width=True, hide_index=True)


def render_analytics_tab(value_bets: pd.DataFrame):
    st.subheader("Analytics")
    
    if value_bets.empty:
        st.info("No data for analytics.")
        return
    
    # Show summary stats first (fast)
    if 'market' in value_bets.columns:
        market_stats = value_bets.groupby('market').agg({
            'edge_percentage': 'mean',
            'player_id': 'count'
        }).reset_index()
        market_stats.columns = ['Market', 'Avg Edge', 'Count']
        market_stats['Avg Edge'] = (market_stats['Avg Edge'] * 100).round(1)
        
        st.markdown("### Edge by Market Type")
        st.dataframe(market_stats, use_container_width=True, hide_index=True)
    
    # Defer charts to expander (lazy load)
    with st.expander("Show Charts", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                value_bets, 
                x=value_bets['edge_percentage'] * 100, 
                nbins=15,
                title="Edge Distribution (%)",
                color_discrete_sequence=['#6b7280']
            )
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.1)',
                height=280
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            pos_counts = value_bets.groupby('position').size().reset_index(name='count')
            fig = px.pie(
                pos_counts, 
                values='count', 
                names='position',
                title="By Position",
                color_discrete_sequence=['#4b5563', '#6b7280', '#9ca3af', '#d1d5db']
            )
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                height=280
            )
            st.plotly_chart(fig, use_container_width=True)


def render_system_tab(season: int, week: int):
    st.subheader("System Health")
    
    freshness = load_feed_freshness(season, week)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Data Freshness")
        if freshness.empty:
            st.warning("No freshness records. Run the data pipeline.")
        else:
            st.dataframe(freshness[['feed', 'as_of', 'age_minutes', 'status']], 
                        use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("### System Status")
        current_hour = datetime.now().hour
        is_active = 8 <= current_hour <= 23
        
        st.metric("Status", "ACTIVE" if is_active else "MAINTENANCE")
        st.metric("Database", "Connected")
        st.metric("Last Update", datetime.now().strftime("%Y-%m-%d %H:%M"))


def render_sidebar():
    """Render the enhanced sidebar with navigation and controls."""
    
    # Sidebar header
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid #333; margin-bottom: 1rem;">
        <h2 style="margin: 0; color: #eee;">NFL Algorithm</h2>
        <p style="margin: 0.25rem 0 0 0; color: #888; font-size: 0.8rem;">v2.1 Pro</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation section
    st.sidebar.markdown("##### NAVIGATION")
    page = st.sidebar.radio(
        "Select View",
        ["Dashboard", "Bet Tracker", "Model Settings"],
        index=0,
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    # Data selection
    st.sidebar.markdown("##### DATA SELECTION")
    seasons, season_map = get_available_seasons_weeks()
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        season = st.selectbox("Season", seasons, index=0, label_visibility="collapsed")
    with col2:
        week = st.selectbox("Week", season_map.get(season, [1]), index=0, label_visibility="collapsed")
    
    st.sidebar.caption(f"Viewing: {season} Week {week}")
    
    st.sidebar.markdown("---")
    
    # Filters section
    st.sidebar.markdown("##### FILTERS")
    min_edge_slider = st.sidebar.slider(
        "Minimum Edge %", 
        min_value=0.0, 
        max_value=30.0, 
        value=5.0, 
        step=0.5,
        help="Only show bets with edge above this threshold"
    )
    
    best_line_only = st.sidebar.checkbox("Best Line Only", value=True, help="Show only the best sportsbook per player")
    quick_bet_mode = st.sidebar.checkbox("Quick Mode", value=False, help="Simplified view for fast decisions")
    
    st.sidebar.markdown("---")
    
    # Actions section
    st.sidebar.markdown("##### ACTIONS")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        refresh_clicked = st.button("Refresh", use_container_width=True)
    with col2:
        export_clicked = st.button("Export", use_container_width=True)
    
    st.sidebar.markdown("---")
    
    # Quick stats in sidebar
    st.sidebar.markdown("##### QUICK STATS")
    try:
        perf = load_performance_history()
        if not perf.empty:
            total_profit = perf['profit_units'].sum()
            win_rate = (perf['wins'].sum() / (perf['wins'].sum() + perf['losses'].sum()) * 100) if (perf['wins'].sum() + perf['losses'].sum()) > 0 else 0
            st.sidebar.metric("Season P/L", f"{total_profit:+.1f}u")
            st.sidebar.metric("Win Rate", f"{win_rate:.0f}%")
        else:
            st.sidebar.caption("No performance data yet")
    except Exception:
        st.sidebar.caption("Stats unavailable")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("Built for serious bettors")
    
    return season, week, min_edge_slider, best_line_only, quick_bet_mode, refresh_clicked, page


def main() -> None:
    # Render sidebar and get values
    season, week, min_edge_slider, best_line_only, quick_bet_mode, refresh_clicked, page = render_sidebar()
    
    # Handle refresh
    if refresh_clicked:
        with st.spinner("Materializing weekly view..."):
            materialize_week(season, week, min_edge=min_edge_slider / 100)
        st.success("Data refreshed")
    
    # Main content area
    if page == "Dashboard":
        st.title("NFL Algorithm")
        st.caption("Value betting projections and performance tracking")
        
        value_bets = load_weekly_value(season, week)
        perf_history = load_performance_history()
        
        render_kpi_cards(value_bets, perf_history)
        
        st.markdown("---")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Live Bets", "Performance", "Analytics", "System"])
        
        with tab1:
            st.markdown(f"### Season {season} - Week {week}")
            render_live_bets_tab(value_bets, min_edge_slider / 100, quick_bet_mode, best_line_only)
        
        with tab2:
            render_performance_tab(perf_history)
        
        with tab3:
            render_analytics_tab(value_bets)
        
        with tab4:
            render_system_tab(season, week)
    
    elif page == "Bet Tracker":
        st.title("Bet Tracker")
        st.caption("Track your placed bets and outcomes")
        
        st.info("Bet tracking coming soon. Use `make week-grade` to record outcomes after games.")
        
        # Show recent outcomes if available
        try:
            outcomes = read_dataframe("""
                SELECT player_name, market, line, actual_result, result, profit_units, confidence_tier
                FROM bet_outcomes
                WHERE season = ? AND week = ?
                ORDER BY profit_units DESC
            """, params=(season, week))
            
            if not outcomes.empty:
                st.markdown("### Recent Outcomes")
                st.dataframe(outcomes, use_container_width=True, hide_index=True)
            else:
                st.caption("No outcomes recorded for this week yet.")
        except Exception:
            st.caption("No outcome data available.")
    
    elif page == "Model Settings":
        st.title("Model Settings")
        st.caption("Configure prediction parameters")
        
        st.markdown("### Betting Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            st.number_input("Min Edge Threshold (%)", value=5.0, min_value=0.0, max_value=50.0, step=0.5)
            st.number_input("Kelly Fraction", value=0.25, min_value=0.0, max_value=1.0, step=0.05)
        
        with col2:
            st.number_input("Max Stake (%)", value=2.0, min_value=0.0, max_value=10.0, step=0.5)
            st.number_input("Bankroll ($)", value=1000.0, min_value=0.0, step=100.0)
        
        st.markdown("### Defense Adjustments")
        st.checkbox("Enable defense multipliers", value=True)
        st.slider("Multiplier weight", 0.0, 1.0, 0.5)
        
        st.markdown("---")
        st.button("Save Settings", type="primary")
    
    # Footer
    st.markdown("---")
    st.caption("NFL Algorithm v2.1 | Bet responsibly")


if __name__ == "__main__":
    main()

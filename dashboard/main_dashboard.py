"""
Main Streamlit dashboard for NFL Algorithm monitoring and management.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sqlite3
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import config

st.set_page_config(
    page_title="NFL Algorithm Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

def normalize_opportunities(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize enhanced CSV columns to dashboard's expected schema."""
    if df is None or df.empty:
        return df
    cols = set(df.columns)
    # Enhanced CSV pathway
    if {'player', 'stat', 'book'}.issubset(cols):
        df = df.copy()
        df['player_name'] = df.get('player')
        df['prop_type'] = df.get('stat')
        df['sportsbook'] = df.get('book')
        if 'risk_level' not in cols and 'value_rating' in cols:
            df['risk_level'] = df['value_rating']
        if 'edge_percentage' not in cols:
            if 'edge_pct_display' in cols:
                df['edge_percentage'] = df['edge_pct_display']
            else:
                df['edge_percentage'] = pd.NA
        if 'expected_roi' not in cols:
            df['expected_roi'] = pd.NA
    return df

def load_data():
    """Load bets from enhanced CSV if present; fallback to DB."""
    try:
        csv_path = config.reports_dir / "value_bets_enhanced.csv"
        if csv_path.exists():
            value_bets = pd.read_csv(csv_path)
            value_bets = normalize_opportunities(value_bets)
        else:
            conn = sqlite3.connect(config.database.path)
            value_bets_query = """
            SELECT * FROM enhanced_value_bets 
            WHERE date_identified >= date('now', '-7 days')
            ORDER BY date_identified DESC
            """
            value_bets = pd.read_sql_query(value_bets_query, conn)
            conn.close()

        # CLV data (optional). If table missing, return empty silently.
        try:
            conn2 = sqlite3.connect(config.database.path)
            clv_query = """
            SELECT * FROM clv_tracking 
            WHERE date_placed >= date('now', '-30 days')
            ORDER BY date_placed DESC
            """
            clv_data = pd.read_sql_query(clv_query, conn2)
            conn2.close()
        except Exception:
            clv_data = pd.DataFrame()
        return value_bets, clv_data
    except Exception as e:
        # Show empty datasets rather than an error banner
        # This avoids noisy UI when optional tables are missing
        return pd.DataFrame(), pd.DataFrame()

def main():
    """Main dashboard function."""
    
    st.title("NFL Algorithm Professional Dashboard")
    st.markdown("**Real-time monitoring of value betting opportunities and model performance**")
    
    # Sidebar
    st.sidebar.header("Controls")
    presentation_mode = st.sidebar.toggle("Presentation Mode", value=False, help="Hide internal sections for clean demos")
    quick_bet_mode = st.sidebar.toggle("Quick Bet mode", value=False, help="Compact bet cards from enhanced CSV")
    min_edge_slider = st.sidebar.slider("Minimum Edge %", min_value=0.0, max_value=50.0, value=0.0, step=0.5)
    include_no_value = st.sidebar.toggle("Include NO_VALUE rows", value=True)
    
    # Refresh button
    if st.sidebar.button("Refresh Data"):
        st.rerun()
    
    # Load data
    value_bets, clv_data = load_data()
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        high_count = 0
        if not value_bets.empty:
            if 'value_rating' in value_bets.columns:
                high_count = int((value_bets['value_rating'] == 'HIGH_VALUE').sum())
            elif 'risk_level' in value_bets.columns:
                high_count = int((value_bets['risk_level'] == 'HIGH_VALUE').sum())
        st.metric(label="Active Value Bets", value=len(value_bets), delta=f"{high_count} High Value")
    
    with col2:
        if not value_bets.empty and 'expected_roi' in value_bets.columns:
            avg_roi = value_bets['expected_roi'].mean()
            st.metric(
                label="Avg Expected ROI",
                value=f"{avg_roi:.1%}",
                delta="12% Target"
            )
        else:
            st.metric("Avg Expected ROI", "N/A")
    
    with col3:
        if not clv_data.empty and 'clv_percentage' in clv_data.columns:
            avg_clv = clv_data['clv_percentage'].mean()
            st.metric(
                label="Avg CLV",
                value=f"{avg_clv:.1f}%",
                delta="Positive is good"
            )
        else:
            st.metric("Avg CLV", "N/A")
    
    with col4:
        model_status = "ACTIVE" if datetime.now().hour in range(8, 23) else "MAINTENANCE"
        st.metric(label="System Status", value=model_status)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Live Bets", "Performance", "CLV Analysis", "System"])
    
    with tab1:
        st.header("Live Value Betting Opportunities")
        
        if not value_bets.empty:
            # Filter controls
            col1, col2 = st.columns(2)
            with col1:
                risk_col = 'risk_level' if 'risk_level' in value_bets.columns else ('value_rating' if 'value_rating' in value_bets.columns else None)
                if risk_col:
                    risk_filter = st.selectbox(
                        "Risk Level",
                        ["All"] + list(value_bets[risk_col].unique())
                    )
                else:
                    risk_filter = "All"
            with col2:
                position_filter = st.selectbox(
                    "Position",
                    ["All"] + list(value_bets['position'].unique())
                )
            
            # Apply filters
            filtered_bets = value_bets.copy()
            if risk_filter != "All" and 'risk_level' in filtered_bets.columns:
                filtered_bets = filtered_bets[filtered_bets['risk_level'] == risk_filter]
            elif risk_filter != "All" and 'value_rating' in filtered_bets.columns:
                filtered_bets = filtered_bets[filtered_bets['value_rating'] == risk_filter]
            # Edge filter (abs edge % >= slider)
            edge_col = 'edge_percentage' if 'edge_percentage' in filtered_bets.columns else ( 'edge_pct_display' if 'edge_pct_display' in filtered_bets.columns else None)
            if edge_col:
                filtered_bets = filtered_bets[filtered_bets[edge_col].abs() >= min_edge_slider]
            # Include/exclude NO_VALUE
            if not include_no_value and 'value_rating' in filtered_bets.columns:
                filtered_bets = filtered_bets[filtered_bets['value_rating'] != 'NO_VALUE']
            if position_filter != "All":
                filtered_bets = filtered_bets[filtered_bets['position'] == position_filter]
            
            # Display table or compact cards
            if not filtered_bets.empty:
                if quick_bet_mode and all(c in filtered_bets.columns for c in [
                    'player','stat','line','model_prediction','edge_pct_display','book','confidence_score'
                ]):
                    st.caption("Quick Bet mode: compact cards from enhanced CSV")
                    for _, r in filtered_bets.sort_values(
                        'confidence_score', ascending=False
                    ).head(30).iterrows():
                        st.write(
                            f"{r.get('player','')} — {r.get('stat','')} {r.get('line','')} | {float(r.get('edge_pct_display',0)):+.1f}% | {r.get('book','')} | conf {int(r.get('confidence_score',0))}"
                        )
                else:
                    display_cols = [
                        'player_name', 'position', 'team', 'prop_type', 'sportsbook',
                        'line', 'model_prediction', 'edge_yards', 'edge_percentage', 'expected_roi',
                        'risk_level', 'value_rating', 'recommendation'
                    ]
                    available_cols = [col for col in display_cols if col in filtered_bets.columns]
                    st.dataframe(
                        filtered_bets[available_cols],
                        use_container_width=True
                    )

                # Export controls
                export_col1, export_col2 = st.columns(2)
                with export_col1:
                    if st.button("Export table to CSV"):
                        reports_dir = config.reports_dir
                        reports_dir.mkdir(exist_ok=True)
                        export_path = reports_dir / 'live_bets_export.csv'
                        filtered_bets[available_cols].to_csv(export_path, index=False)
                        st.success(f"Exported to {export_path}")
                with export_col2:
                    st.caption("Download charts as PNG from Performance tab")
            else:
                st.info("No bets match the selected filters.")
        else:
            st.info("No recent value bets found.")
    
    with tab2:
        st.header("Model Performance Analytics")
        
        # Performance chart
        if not value_bets.empty and 'expected_roi' in value_bets.columns:
            fig = px.histogram(
                value_bets,
                x='expected_roi',
                title="Expected ROI Distribution",
                nbins=20
            )
            fig.add_vline(x=0.12, line_dash="dash", line_color="red", 
                         annotation_text="12% Target")
            st.plotly_chart(fig, use_container_width=True)
            
            # ROI by position
            fig2 = px.box(
                value_bets,
                x='position',
                y='expected_roi',
                title="Expected ROI by Position"
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Top-10 expected ROI by player
            if 'player_name' in value_bets.columns:
                vb = value_bets.dropna(subset=['expected_roi']).copy()
                top10 = vb.sort_values('expected_roi', ascending=False).head(10)
                fig3 = px.bar(top10, x='player_name', y='expected_roi', title='Top-10 Expected ROI by Player')
                st.plotly_chart(fig3, use_container_width=True)
                # Optional PNG download if kaleido present
                try:
                    import tempfile
                    tmp_png = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                    fig3.write_image(tmp_png.name)
                    st.download_button(
                        label="Download Top-10 ROI PNG",
                        data=open(tmp_png.name, 'rb').read(),
                        file_name='top10_expected_roi.png',
                        mime='image/png'
                    )
                except Exception:
                    pass
        else:
            st.info("No performance data available.")
    
    with tab3:
        st.header("Closing Line Value (CLV) Analysis")
        
        if not clv_data.empty:
            # CLV trend
            if 'date_placed' in clv_data.columns and 'clv_percentage' in clv_data.columns:
                clv_data['date_placed'] = pd.to_datetime(clv_data['date_placed'])
                daily_clv = clv_data.groupby('date_placed')['clv_percentage'].mean().reset_index()
                
                fig = px.line(
                    daily_clv,
                    x='date_placed',
                    y='clv_percentage',
                    title="Daily Average CLV Trend"
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)
            
            # CLV summary table
            st.subheader("Recent CLV Results")
            display_cols = ['bet_id', 'prop_type', 'bet_line', 'closing_line', 'clv_percentage']
            available_cols = [col for col in display_cols if col in clv_data.columns]
            if available_cols:
                st.dataframe(
                    clv_data[available_cols].head(20),
                    use_container_width=True
                )
        else:
            st.info("No CLV data available.")
    
    with tab4:
        st.header("System Configuration & Status")
        
        if not presentation_mode:
            # System info
            st.subheader("Configuration")
            config_data = {
                "Target MAE": f"≤ {config.model.target_mae}",
                "Min Edge Threshold": f"{config.betting.min_edge_threshold:.1%}",
                "Min Confidence": f"{config.betting.min_confidence:.1%}",
                "Max Kelly Fraction": f"{config.betting.max_kelly_fraction:.1%}",
                "Update Interval": f"{config.pipeline.update_interval_minutes} minutes"
            }
            
            for key, value in config_data.items():
                st.metric(key, value)
            
            # Log viewer
            st.subheader("Recent Logs")
            log_file = config.logs_dir / "scheduler.log"
            if log_file.exists():
                with open(log_file, 'r') as f:
                    logs = f.readlines()
                
                # Show last 20 lines
                recent_logs = logs[-20:] if len(logs) > 20 else logs
                log_text = "".join(recent_logs)
                st.text_area("System Logs", log_text, height=300)
            else:
                st.info("No log file found.")
    
    # Footer
    st.markdown("---")
    st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | NFL Algorithm v2.0*")

if __name__ == "__main__":
    main() 
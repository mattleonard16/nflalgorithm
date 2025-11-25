#!/usr/bin/env python3
"""
Integration module to connect prop line scraper with existing NFL prediction system
"""
"""Integration module to connect prop line scraper with existing NFL prediction system"""

import logging
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import pandas as pd
from datetime import datetime
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go

from config import config
from utils.db import read_dataframe, get_connection, execute, executemany, write_dataframe
from utils.player_id_utils import (
    canonicalize_team,
    normalize_name,
    normalized_name_from_player_id,
    team_from_player_id,
)

logger = logging.getLogger(__name__)


_FUZZY_THRESHOLD = 0.82  # Slightly lower to increase match rate across name variants


def _safe_nunique(series: pd.Series) -> int:
    return series.dropna().nunique()


def normalize_player_name(name: Optional[str]) -> str:
    return normalize_name(name)


def _sequence_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def join_odds_projections(season: int, week: int) -> pd.DataFrame:
    """Join weekly odds snapshots with model projections."""
    projections = read_dataframe(
        """
        SELECT season, week, player_id, team, opponent, market, mu, sigma, model_version,
               featureset_hash, generated_at
        FROM weekly_projections
        WHERE season = ? AND week = ?
        """,
        params=(season, week),
    )

    odds = read_dataframe(
        """
        SELECT event_id, season, week, player_id, market, sportsbook, line, price, as_of
        FROM weekly_odds
        WHERE season = ? AND week = ?
        """,
        params=(season, week),
    )

    players = read_dataframe(
        """
        SELECT player_id, name AS player_name, position, team AS team_stats
        FROM player_stats_enhanced
        WHERE season = ? AND week = ?
        """,
        params=(season, week),
    )

    injuries = read_dataframe(
        """
        SELECT player_id, status, practice_participation
        FROM injury_data
        WHERE season = ? AND week = ?
        """,
        params=(season, week),
    )

    try:
        player_mappings_df = read_dataframe(
            """
            SELECT player_id_canonical, player_id_odds, player_id_projections, match_type, confidence_score
            FROM player_mappings
            """,
        )
    except Exception:
        player_mappings_df = pd.DataFrame(
            columns=[
                'player_id_canonical',
                'player_id_odds',
                'player_id_projections',
                'match_type',
                'confidence_score',
            ]
        )

    if projections.empty or odds.empty:
        logger.warning(
            "join_odds_projections: insufficient data for season=%s week=%s (projections=%d, odds=%d)",
            season,
            week,
            len(projections),
            len(odds),
        )
        return pd.DataFrame()

    projections = projections.copy().reset_index(drop=True)
    odds = odds.copy().reset_index(drop=True)

    projections['__proj_idx'] = projections.index
    odds['__odds_idx'] = odds.index

    projections['team'] = projections['team'].apply(canonicalize_team)

    player_name_lookup = (
        players.set_index('player_id')['player_name'].to_dict() if not players.empty else {}
    )

    odds['player_id_original'] = odds['player_id']
    odds['__mapped_from_table'] = False
    odds['mapping_confidence'] = 1.0
    odds['mapping_type'] = 'player_id'

    if not player_mappings_df.empty:
        mapping_df = player_mappings_df.dropna(subset=['player_id_odds', 'player_id_canonical'])
        if not mapping_df.empty:
            mapping_df = mapping_df.sort_values('confidence_score', ascending=False).drop_duplicates('player_id_odds')
            odds = odds.merge(
                mapping_df[['player_id_odds', 'player_id_canonical', 'confidence_score', 'match_type']],
                left_on='player_id',
                right_on='player_id_odds',
                how='left',
            )
            odds['__mapped_from_table'] = odds['player_id_canonical'].notna()
            mapped_mask = odds['__mapped_from_table']
            odds.loc[mapped_mask, 'player_id'] = odds.loc[mapped_mask, 'player_id_canonical']
            odds.loc[mapped_mask, 'mapping_confidence'] = odds.loc[mapped_mask, 'confidence_score'].fillna(1.0)
            odds.loc[mapped_mask, 'mapping_type'] = odds.loc[mapped_mask, 'match_type'].fillna('mapping_table')
            odds.drop(columns=['player_id_canonical', 'player_id_odds', 'confidence_score', 'match_type'], inplace=True, errors='ignore')

    def _resolve_projection_name(pid: Optional[str]) -> str:
        if pid in player_name_lookup:
            resolved = normalize_player_name(player_name_lookup[pid])
            if resolved:
                return resolved
        return normalized_name_from_player_id(pid)

    projections['normalized_name'] = projections['player_id'].apply(_resolve_projection_name)
    odds['normalized_name'] = odds['player_id'].apply(normalized_name_from_player_id)

    projections['team_token'] = projections['player_id'].apply(team_from_player_id)
    odds['team_token'] = odds['player_id'].apply(team_from_player_id)

    proj_players = set(projections['player_id'].dropna())
    odds_players = set(odds['player_id'].dropna())
    odds_players_original = set(odds['player_id_original'].dropna())
    proj_markets = set(projections['market'].dropna())
    odds_markets = set(odds['market'].dropna())

    logger.info(
        "join_odds_projections: start season=%s week=%s proj_rows=%d proj_players=%d proj_markets=%d odds_rows=%d odds_players=%d odds_markets=%d",
        season,
        week,
        len(projections),
        _safe_nunique(projections['player_id']),
        len(proj_markets),
        len(odds),
        _safe_nunique(odds['player_id']),
        len(odds_markets),
    )

    # -----------------------------------------------------------------
    # Match tier logic:
    #   tier=1: exact player_id + market match
    #   tier=2: normalized_name + market + team_canon match
    #   tier=3: normalized_name + market match (team differences allowed but penalized)
    # -----------------------------------------------------------------
    id_matches = projections.merge(
        odds,
        on=['season', 'week', 'player_id', 'market'],
        how='inner',
        suffixes=('_proj', '_odds')
    )
    if not id_matches.empty:
        id_matches['match_type'] = 'player_id'
        id_matches['match_tier'] = 1
        id_matches['match_score'] = 1.0
        if '__mapped_from_table' in id_matches.columns:
            mapped_mask = id_matches['__mapped_from_table'].fillna(False)
            if mapped_mask.any():
                id_matches.loc[mapped_mask, 'match_type'] = id_matches.loc[mapped_mask, 'mapping_type'].fillna('mapping_table')
                id_matches.loc[mapped_mask, 'match_score'] = id_matches.loc[mapped_mask, 'mapping_confidence'].fillna(1.0)

    matched_proj_idx = set(id_matches['__proj_idx']) if not id_matches.empty else set()
    matched_odds_idx = set(id_matches['__odds_idx']) if not id_matches.empty else set()

    unmatched_proj = projections[~projections['__proj_idx'].isin(matched_proj_idx)]
    unmatched_odds = odds[~odds['__odds_idx'].isin(matched_odds_idx)]

    # Tier 2: normalized_name + market + team_token match
    name_team_matches = unmatched_proj.merge(
        unmatched_odds,
        on=['season', 'week', 'market', 'normalized_name', 'team_token'],
        how='inner',
        suffixes=('_proj', '_odds')
    )
    if not name_team_matches.empty:
        name_team_matches['match_type'] = 'normalized_name_team'
        name_team_matches['match_tier'] = 2
        name_team_matches['match_score'] = 0.95
        matched_proj_idx.update(name_team_matches['__proj_idx'])
        matched_odds_idx.update(name_team_matches['__odds_idx'])

    # Tier 3: normalized_name + market match (team differences allowed, lower score)
    unmatched_proj_t3 = projections[~projections['__proj_idx'].isin(matched_proj_idx)]
    unmatched_odds_t3 = odds[~odds['__odds_idx'].isin(matched_odds_idx)]
    name_matches = unmatched_proj_t3.merge(
        unmatched_odds_t3,
        on=['season', 'week', 'market', 'normalized_name'],
        how='inner',
        suffixes=('_proj', '_odds')
    )
    if not name_matches.empty:
        name_matches['match_type'] = 'normalized_name'
        name_matches['match_tier'] = 3
        name_matches['match_score'] = 0.85  # Penalized for team mismatch
        matched_proj_idx.update(name_matches['__proj_idx'])
        matched_odds_idx.update(name_matches['__odds_idx'])

    remaining_proj = projections[~projections['__proj_idx'].isin(matched_proj_idx)]
    remaining_odds = odds[~odds['__odds_idx'].isin(matched_odds_idx)]

    # Fuzzy name matching (tier 3 with lower confidence)
    fuzzy_pairs: List[Tuple[int, int, float]] = []
    used_odds_idx: set[int] = set()
    proj_candidates = list(zip(remaining_proj['__proj_idx'], remaining_proj['normalized_name']))
    odds_candidates = list(zip(remaining_odds['__odds_idx'], remaining_odds['normalized_name']))
    for proj_idx, proj_name in proj_candidates:
        best_idx: Optional[int] = None
        best_ratio: float = 0.0
        for odds_idx, odds_name in odds_candidates:
            if odds_idx in used_odds_idx:
                continue
            ratio = _sequence_ratio(proj_name, odds_name)
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = odds_idx
        if best_idx is not None and best_ratio >= _FUZZY_THRESHOLD:
            fuzzy_pairs.append((proj_idx, best_idx, best_ratio))
            used_odds_idx.add(best_idx)

    fuzzy_matches = pd.DataFrame(columns=[])  # default empty
    if fuzzy_pairs:
        pair_df = pd.DataFrame(fuzzy_pairs, columns=['__proj_idx', '__odds_idx', 'match_score'])
        fuzzy_matches = pair_df.merge(remaining_proj, on='__proj_idx').merge(
            remaining_odds, on='__odds_idx', suffixes=('_proj', '_odds')
        )
        fuzzy_matches['match_type'] = 'fuzzy_name'
        fuzzy_matches['match_tier'] = 3
        # Fuzzy score already reflects confidence from ratio
        matched_proj_idx.update(pair_df['__proj_idx'])
        matched_odds_idx.update(pair_df['__odds_idx'])

    def _prepare_match_frame(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return frame
        frame = frame.copy()
        # Ensure season and week are explicitly set (handle merge suffixes)
        if 'season_proj' in frame.columns:
            frame['season'] = frame.pop('season_proj')
        elif 'season_odds' in frame.columns:
            frame['season'] = frame.pop('season_odds')
        elif 'season' not in frame.columns:
            # Use function parameter if somehow missing
            frame['season'] = season
        
        if 'week_proj' in frame.columns:
            frame['week'] = frame.pop('week_proj')
        elif 'week_odds' in frame.columns:
            frame['week'] = frame.pop('week_odds')
        elif 'week' not in frame.columns:
            # Use function parameter if somehow missing
            frame['week'] = week
        
        # Ensure season/week are non-null integers
        frame['season'] = frame['season'].fillna(season).astype(int)
        frame['week'] = frame['week'].fillna(week).astype(int)
        
        if 'player_id_proj' in frame.columns:
            frame['player_id_projection'] = frame.pop('player_id_proj')
        else:
            frame['player_id_projection'] = frame.get('player_id')
        if 'player_id_odds' in frame.columns:
            frame['player_id_odds'] = frame['player_id_odds']
        elif 'player_id_original' in frame.columns:
            frame['player_id_odds'] = frame['player_id_original']
        else:
            frame['player_id_odds'] = frame.get('player_id')
        frame['player_id'] = frame['player_id_projection']

        if 'normalized_name_proj' in frame.columns:
            frame['normalized_name_projection'] = frame.pop('normalized_name_proj')
        else:
            frame['normalized_name_projection'] = frame.get('normalized_name')
        if 'normalized_name_odds' in frame.columns:
            frame['normalized_name_odds'] = frame.pop('normalized_name_odds')
        else:
            frame['normalized_name_odds'] = frame.get('normalized_name')
        if 'normalized_name' not in frame.columns:
            frame['normalized_name'] = frame['normalized_name_projection'].where(
                frame['normalized_name_projection'].notna() & (frame['normalized_name_projection'] != ""),
                frame['normalized_name_odds']
            )

        if 'team_token_proj' in frame.columns:
            frame['team_projection_token'] = frame.pop('team_token_proj')
        else:
            frame['team_projection_token'] = frame.get('team_token')
        if 'team_token_odds' in frame.columns:
            frame['team_odds_token'] = frame.pop('team_token_odds')
        else:
            frame['team_odds_token'] = frame.get('team_token')
        frame.drop(columns=['team_token'], inplace=True, errors='ignore')

        if 'market_proj' in frame.columns:
            frame['market'] = frame.pop('market_proj')
        elif 'market' not in frame.columns and 'market_odds' in frame.columns:
            frame['market'] = frame['market_odds']
        if 'market_odds' in frame.columns:
            frame['market_odds_source'] = frame.pop('market_odds')

        frame['team_odds'] = frame['player_id_odds'].apply(team_from_player_id)
        return frame

    candidate_frames = [
        _prepare_match_frame(df)
        for df in (id_matches, name_team_matches, name_matches, fuzzy_matches)
        if df is not None and not df.empty
    ]

    if not candidate_frames:
        logger.warning(
            "join_odds_projections: no matches found for season=%s week=%s",
            season,
            week,
        )
        return pd.DataFrame()

    merged = pd.concat(candidate_frames, ignore_index=True)
    
    # Ensure season and week are explicitly set (may be lost in merges/concat)
    merged['season'] = merged.get('season', season)
    merged['week'] = merged.get('week', week)
    merged['season'] = merged['season'].fillna(season).astype(int)
    merged['week'] = merged['week'].fillna(week).astype(int)

    match_breakdown = {
        key: int(value) for key, value in merged['match_type'].value_counts().to_dict().items()
    }

    matched_projection_players = set(merged['player_id_projection'].dropna())
    matched_odds_players = set(merged['player_id_odds'].dropna())
    unmatched_projection_players = list(proj_players - matched_projection_players)[:15]
    unmatched_odds_players = list(odds_players_original - matched_odds_players)[:15]

    merged = merged.merge(players, on='player_id', how='left', suffixes=('', '_stats'))
    merged = merged.merge(injuries, on='player_id', how='left')

    # Fill player descriptors from stats when available
    if 'player_name_stats' in merged.columns:
        merged['player_name'] = merged['player_name'].fillna(merged['player_name_stats'])
    merged['player_name'] = merged['player_name'].fillna(merged['player_id'])

    if 'position_stats' in merged.columns:
        merged['position'] = merged['position'].fillna(merged['position_stats'])
    merged['position'] = merged['position'].fillna('FLEX')
    merged['status'] = merged['status'].fillna('ACTIVE')
    merged['practice_participation'] = merged['practice_participation'].fillna('FULL')

    # Team reconciliation: stats > projections > odds
    stats_team = merged.get('team_stats')
    proj_team = merged.get('team')
    odds_team = merged.get('team_odds')

    merged['team_stats_canon'] = stats_team.apply(canonicalize_team) if stats_team is not None else ''
    merged['team_proj_canon'] = proj_team.apply(canonicalize_team) if proj_team is not None else ''
    merged['team_odds_canon'] = odds_team.apply(canonicalize_team) if odds_team is not None else ''

    merged['team_effective'] = merged['team_stats_canon']
    mask_missing = merged['team_effective'] == ''
    merged.loc[mask_missing, 'team_effective'] = merged.loc[mask_missing, 'team_proj_canon']
    mask_missing = merged['team_effective'] == ''
    merged.loc[mask_missing, 'team_effective'] = merged.loc[mask_missing, 'team_odds_canon']

    merged['team'] = merged['team_effective']

    merged['team_match_flag'] = merged['team'].fillna('').eq(merged['team_odds_canon'].fillna(''))

    mismatch_mask = ~merged['team_match_flag']
    team_mismatch_count = int(mismatch_mask.sum())
    team_mismatch_sample: List[Dict[str, str]] = []
    
    # WR-specific team mismatch tolerance (Issue 4)
    # Allow WRs to match even if team differs (e.g., traded players)
    allow_wr_mismatch = config.integration.allow_wr_team_mismatch
    
    if team_mismatch_count:
        team_mismatch_sample = (
            merged.loc[
                mismatch_mask,
                ['player_id', 'player_id_odds', 'team_stats_canon', 'team_proj_canon', 'team_odds_canon', 'season', 'week', 'position'],
            ]
            .head(5)
            .to_dict(orient='records')
        )
        
        # Build valid mask: team matches OR team_odds empty OR (WR with allowed mismatch)
        valid_mask = merged['team_match_flag'] | merged['team_odds_canon'].eq('')
        if allow_wr_mismatch:
            # Allow WRs with team mismatch but penalize their match_score
            position_upper = merged['position'].fillna('').str.upper()
            wr_mismatch_mask = (
                mismatch_mask 
                & (position_upper == 'WR')
                & merged['team_odds_canon'].ne('')
            )
            valid_mask = valid_mask | wr_mismatch_mask
            # Penalize WR team mismatches in match_score
            merged.loc[wr_mismatch_mask, 'match_score'] = merged.loc[wr_mismatch_mask, 'match_score'] * 0.9
            merged.loc[wr_mismatch_mask, 'match_tier'] = merged.loc[wr_mismatch_mask, 'match_tier'].fillna(3).astype(int)
            wr_tolerated = int(wr_mismatch_mask.sum())
            if wr_tolerated:
                logger.info(
                    "join_odds_projections: tolerating %d WR rows with team mismatch (allow_wr_team_mismatch=True)",
                    wr_tolerated,
                )
        
        dropped_mismatch = int((~valid_mask).sum())
        if dropped_mismatch:
            logger.warning(
                "join_odds_projections: dropping %d rows with team mismatch (team_stats=%s team_proj=%s team_odds=%s sample=%s)",
                dropped_mismatch,
                merged['team_stats_canon'].nunique() if 'team_stats_canon' in merged else 0,
                merged['team_proj_canon'].nunique() if 'team_proj_canon' in merged else 0,
                merged['team_odds_canon'].nunique() if 'team_odds_canon' in merged else 0,
                team_mismatch_sample,
            )
        merged = merged[valid_mask].copy()
        mismatch_mask = ~merged['team_match_flag']
        team_mismatch_count = int(mismatch_mask.sum())

    merged['price'] = merged['price'].astype(int)
    merged['line'] = merged['line'].astype(float)
    merged['mu'] = merged['mu'].astype(float)
    merged['sigma'] = merged['sigma'].astype(float)
    merged['match_confidence'] = merged['match_score']
    
    # Ensure match_tier is set (default to 3 if missing)
    if 'match_tier' not in merged.columns:
        merged['match_tier'] = 3
    merged['match_tier'] = merged['match_tier'].fillna(3).astype(int)
    
    # Apply match tier filtering based on sportsbook type
    min_tier_real = config.integration.min_match_tier_real
    min_tier_synthetic = config.integration.min_match_tier_synthetic
    
    is_simbook = merged['sportsbook'].str.upper().str.contains('SIM', na=False)
    tier_filter = (
        (is_simbook & (merged['match_tier'] <= min_tier_synthetic)) |
        (~is_simbook & (merged['match_tier'] <= min_tier_real))
    )
    filtered_count = int((~tier_filter).sum())
    if filtered_count:
        logger.info(
            "join_odds_projections: filtering %d rows by match_tier (real<=%d, synthetic<=%d)",
            filtered_count,
            min_tier_real,
            min_tier_synthetic,
        )
    merged = merged[tier_filter].copy()

    logger.info(
        "join_odds_projections: matched_rows=%d matched_players=%d match_breakdown=%s team_mismatches=%d unmatched_projection_players=%d unmatched_odds_players=%d",
        len(merged),
        len(matched_projection_players),
        match_breakdown,
        team_mismatch_count,
        max(len(proj_players - matched_projection_players), 0),
        max(len(odds_players_original - matched_odds_players), 0),
    )

    if unmatched_projection_players:
        logger.debug(
            "join_odds_projections: sample unmatched projection player_ids=%s",
            unmatched_projection_players,
        )

    if unmatched_odds_players:
        logger.debug(
            "join_odds_projections: sample unmatched odds player_ids=%s",
            unmatched_odds_players,
        )

    if team_mismatch_count:
        logger.warning(
            "join_odds_projections: team mismatches sample=%s",
            team_mismatch_sample,
        )

    merged.sort_values(['market', 'player_id', 'sportsbook', 'as_of'], inplace=True)
    merged.reset_index(drop=True, inplace=True)

    merged.drop(
        columns=[
            '__proj_idx',
            '__odds_idx',
            'team_projection_token',
            'team_odds_token',
            'player_id_original',
            '__mapped_from_table',
            'mapping_type',
            'mapping_confidence',
        ],
        inplace=True,
        errors='ignore',
    )

    return merged

class PropIntegration:
    """Integrate prop lines with existing NFL prediction system"""
    
    def __init__(self, prop_db_path: str = "nfl_prop_lines.db", 
                 prediction_db_path: str = "nfl_data.db"):
        self.prop_db_path = prop_db_path
        self.prediction_db_path = prediction_db_path
        self.projections_file = "2024_nfl_projections.csv"
        # Ensure reports directories exist
        try:
            config.reports_dir.mkdir(exist_ok=True)
            config.reports_img_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def export_opportunities_csv(self, opportunities: pd.DataFrame, path: str | Path = None) -> Path:
        """Export opportunities to CSV."""
        if opportunities is None or opportunities.empty:
            # Still create an empty CSV with headers for consistency
            opportunities = pd.DataFrame(columns=[
                'player', 'team', 'position', 'book', 'stat', 'line', 'model_prediction',
                'edge_yards', 'edge_percentage', 'over_odds', 'under_odds', 'value_rating'
            ])
        if path is None:
            path = config.reports_dir / "value_bets.csv"
        path = Path(path)
        opportunities.to_csv(path, index=False)
        return path

    def export_opportunities_json(self, opportunities: pd.DataFrame, path: str | Path = None) -> Path:
        """Export opportunities to JSON."""
        if opportunities is None or opportunities.empty:
            opportunities = pd.DataFrame(columns=[
                'player', 'team', 'position', 'book', 'stat', 'line', 'model_prediction',
                'edge_yards', 'edge_percentage', 'over_odds', 'under_odds', 'value_rating'
            ])
        if path is None:
            path = config.reports_dir / "value_bets.json"
        path = Path(path)
        opportunities.to_json(path, orient='records', indent=2)
        return path

    def generate_charts(self, opportunities: pd.DataFrame) -> Dict[str, Path]:
        """Generate and save charts for the report. Returns dict of image paths."""
        output_paths: Dict[str, Path] = {}
        try:
            img_dir = config.reports_img_dir
            img_dir.mkdir(parents=True, exist_ok=True)

            if opportunities is None or opportunities.empty:
                return output_paths

            # Normalize edge percentage if presented as percent values instead of fraction
            edges = opportunities.copy()
            if edges['edge_percentage'].max() > 1.5:
                # Assume values are in percent already
                edges['edge_pct_norm'] = edges['edge_percentage']
            else:
                edges['edge_pct_norm'] = edges['edge_percentage'] * 100.0

            # Top edges bar chart
            top = edges.reindex(edges['edge_pct_norm'].abs().sort_values(ascending=False).index).head(10)
            if not top.empty:
                top['label'] = top['player'].astype(str) + " (" + top['stat'].astype(str) + ")"
                fig = px.bar(top, x='label', y='edge_pct_norm', title='Top 10 Edge %', labels={'edge_pct_norm': 'Edge %', 'label': 'Player/Stat'})
                fig.update_layout(xaxis_tickangle=-30, margin=dict(l=10, r=10, t=40, b=120), height=500)
                out_path = img_dir / 'top_edges.png'
                try:
                    fig.write_image(str(out_path))
                    output_paths['top_edges'] = out_path
                except Exception as e:
                    # Fallback: matplotlib PNG
                    try:
                        import matplotlib.pyplot as plt
                        plt.figure(figsize=(10, 5))
                        color = ['#1f77b4' if v >= 0 else '#d62728' for v in top['edge_pct_norm']]
                        plt.bar(top['label'], top['edge_pct_norm'], color=color)
                        plt.xticks(rotation=30, ha='right')
                        plt.ylabel('Edge %')
                        plt.title('Top 10 Edge %')
                        plt.tight_layout()
                        plt.savefig(out_path, dpi=150)
                        plt.close()
                        output_paths['top_edges'] = out_path
                    except Exception:
                        # Last resort: HTML
                        html_fallback = img_dir / 'top_edges.html'
                        fig.write_html(str(html_fallback))
                        output_paths['top_edges_html'] = html_fallback

            # ROI histogram if available
            if 'expected_roi' in opportunities.columns and opportunities['expected_roi'].notna().any():
                roi = opportunities['expected_roi']
                roi_pct = roi * 100 if roi.max() <= 1.0 else roi
                fig2 = px.histogram(roi_pct.dropna(), nbins=20, title='Expected ROI Distribution', labels={'value': 'ROI %'})
                fig2.update_layout(margin=dict(l=10, r=10, t=40, b=40), height=400)
                out_path2 = img_dir / 'roi_hist.png'
                try:
                    fig2.write_image(str(out_path2))
                    output_paths['roi_hist'] = out_path2
                except Exception:
                    try:
                        import matplotlib.pyplot as plt
                        plt.figure(figsize=(8, 4))
                        plt.hist(roi_pct.dropna(), bins=20, color='#1f77b4')
                        plt.xlabel('ROI %')
                        plt.title('Expected ROI Distribution')
                        plt.tight_layout()
                        plt.savefig(out_path2, dpi=150)
                        plt.close()
                        output_paths['roi_hist'] = out_path2
                    except Exception:
                        html_fallback2 = img_dir / 'roi_hist.html'
                        fig2.write_html(str(html_fallback2))
                        output_paths['roi_hist_html'] = html_fallback2

        except Exception as e:
            logger.warning(f"Chart generation skipped: {e}")
        return output_paths

    def render_html_from_markdown(self, md_text: str, html_path: str | Path = None) -> Path:
        """Render HTML from Markdown. Prefer python-markdown, else fallback to minimal HTML wrapper."""
        if html_path is None:
            html_path = config.reports_dir / 'weekly_value_report.html'
        html_path = Path(html_path)

        html: str
        try:
            import markdown  # type: ignore

            html = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])
            html = f"""
<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <meta name='viewport' content='width=device-width, initial-scale=1'/>
  <title>NFL Weekly Value Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; padding: 24px; max-width: 1100px; margin: 0 auto; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; }}
    th {{ background: #f7f7f7; text-align: left; }}
    h1, h2, h3 {{ color: #222; }}
    code {{ background: #f3f3f3; padding: 2px 4px; border-radius: 4px; }}
  </style>
  </head>
  <body>
  {html}
  </body>
</html>
"""
        except Exception:
            # Minimal fallback
            html = f"""
<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <meta name='viewport' content='width=device-width, initial-scale=1'/>
  <title>NFL Weekly Value Report</title>
  <style>body {{ font-family: sans-serif; padding: 24px; max-width: 1100px; margin: 0 auto; white-space: pre-wrap; }}</style>
</head>
<body>{md_text}</body>
</html>
"""
        html_path.write_text(html, encoding='utf-8')
        return html_path

    def generate_markdown_report(self, opportunities: pd.DataFrame) -> str:
        """Generate a Markdown report from opportunities with summary, table, and sections."""
        lines: List[str] = []
        lines.append("# NFL Prop Line Value Opportunities\n")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append("")

        if opportunities is None or opportunities.empty:
            lines.append("No significant value opportunities found.")
            return "\n".join(lines)

        # Safe columns
        df = opportunities.copy()
        if 'edge_percentage' in df.columns and df['edge_percentage'].max() <= 1.0:
            df['edge_percentage_display'] = df['edge_percentage'] * 100.0
        else:
            df['edge_percentage_display'] = df['edge_percentage']

        total = len(df)
        high_count = int((df['value_rating'] == 'HIGH_VALUE').sum()) if 'value_rating' in df.columns else 0
        med_count = int((df['value_rating'] == 'MEDIUM_VALUE').sum()) if 'value_rating' in df.columns else 0

        lines.append("## Executive Summary\n")
        lines.append(f"- Total opportunities: {total}")
        lines.append(f"- High Value: {high_count}")
        lines.append(f"- Medium Value: {med_count}\n")

        # Top 10 table
        lines.append("## Top 10 Opportunities\n")
        display_cols = [
            'player', 'position', 'team', 'stat', 'line', 'model_prediction', 'edge_yards',
            'edge_percentage_display', 'book'
        ]
        # Recommendation derived
        if 'edge_yards' in df.columns:
            df['recommendation'] = df['edge_yards'].apply(lambda x: 'OVER' if pd.notna(x) and x > 0 else 'UNDER')
        display_cols_with_rec = display_cols + ['recommendation']
        available = [c for c in display_cols_with_rec if c in df.columns]

        top10 = df.reindex(df['edge_percentage_display'].abs().sort_values(ascending=False).index).head(10)
        # Build markdown table
        header = "| " + " | ".join([c.replace('_', ' ').title() for c in available]) + " |"
        sep = "|" + "---|" * len(available)
        lines.append(header)
        lines.append(sep)
        for _, row in top10[available].iterrows():
            values = []
            for c in available:
                val = row[c]
                if c == 'edge_percentage_display' and pd.notna(val):
                    values.append(f"{val:+.1f}%")
                elif c in ('model_prediction', 'line', 'edge_yards') and pd.notna(val):
                    values.append(f"{float(val):.1f}")
                else:
                    values.append(str(val) if pd.notna(val) else "")
            lines.append("| " + " | ".join(values) + " |")

        # Sections: High Value detailed
        if 'value_rating' in df.columns:
            high_df = df[df['value_rating'] == 'HIGH_VALUE']
            if not high_df.empty:
                lines.append("\n## High Value Opportunities\n")
                for _, r in high_df.reindex(high_df['edge_percentage_display'].abs().sort_values(ascending=False).index).head(20).iterrows():
                    direction = 'OVER' if r.get('edge_yards', 0) > 0 else 'UNDER'
                    lines.append(f"- {r.get('player','')} ({r.get('position','')}, {r.get('team','')}) – {r.get('stat','')}")
                    lines.append(f"  - Line: {r.get('line','')} | Model: {r.get('model_prediction','')}")
                    edge_pct_str = f"{r.get('edge_percentage_display', 0):+.1f}%" if pd.notna(r.get('edge_percentage_display', None)) else ""
                    lines.append(f"  - Edge: {r.get('edge_yards',''):+} ({edge_pct_str}) | Book: {r.get('book','')} | Rec: {direction}")
                    lines.append("")

            med_df = df[df['value_rating'] == 'MEDIUM_VALUE']
            if not med_df.empty:
                lines.append("\n## Medium Value Opportunities\n")
                for _, r in med_df.reindex(med_df['edge_percentage_display'].abs().sort_values(ascending=False).index).head(20).iterrows():
                    edge_pct_str = f"{r.get('edge_percentage_display', 0):+.1f}%" if pd.notna(r.get('edge_percentage_display', None)) else ""
                    direction = 'OVER' if r.get('edge_yards', 0) > 0 else 'UNDER'
                    lines.append(f"- {r.get('player','')} – {r.get('stat','')}: {r.get('line','')} ({edge_pct_str} edge, {direction})")

        return "\n".join(lines)
    
    def get_current_prop_lines(self) -> pd.DataFrame:
        """Get current prop lines from database"""
        try:
            df = read_dataframe(
                '''
                SELECT player, team, position, book, stat, line, over_odds, under_odds, last_updated
                FROM prop_lines
                WHERE season = '2025-2026'
                ORDER BY player, stat, book
                '''
            )
            return df
        except Exception as e:
            logger.error(f"Error loading prop lines: {e}")
            return pd.DataFrame()
    
    def get_player_projections(self) -> pd.DataFrame:
        """Get player projections from existing files"""
        try:
            # Load existing projections
            projections_df = pd.read_csv(self.projections_file)
            
            # Also try to load from database if available
            try:
                db_projections = read_dataframe(
                    '''
                    SELECT name, position, team, 
                           rushing_yards as proj_rush_yds, 
                           receiving_yards as proj_rec_yds,
                           (rushing_yards + receiving_yards) as proj_total_yds
                    FROM player_stats 
                    WHERE season = 2024
                    '''
                )
                
                # Merge with CSV projections if available
                if not projections_df.empty:
                    projections_df = projections_df.merge(db_projections, 
                                                        left_on='name', right_on='name', 
                                                        how='outer', suffixes=('', '_db'))
                else:
                    projections_df = db_projections
                    
            except Exception as e:
                logger.warning(f"Could not load from database: {e}")
            
            return projections_df
            
        except Exception as e:
            logger.error(f"Error loading projections: {e}")
            return pd.DataFrame()
    
    def merge_props_with_projections(self) -> pd.DataFrame:
        """Merge prop lines with player projections"""
        
        # Get data
        prop_lines = self.get_current_prop_lines()
        projections = self.get_player_projections()
        
        if prop_lines.empty or projections.empty:
            logger.warning("Missing prop lines or projections data")
            return pd.DataFrame()
        
        # Standardize player names for matching
        prop_lines['player_clean'] = prop_lines['player'].str.lower().str.strip()
        projections['name_clean'] = projections['name'].str.lower().str.strip()
        
        # Merge data
        merged = prop_lines.merge(
            projections, 
            left_on='player_clean', 
            right_on='name_clean', 
            how='left',
            suffixes=('_prop', '_proj')
        )
        
        # Calculate value metrics
        merged = self._calculate_value_metrics(merged)

        # Create canonical team/position columns for downstream selection
        # Prefer prop-side values, fall back to projections if missing
        if 'team_prop' in merged.columns or 'team_proj' in merged.columns:
            if 'team_prop' in merged.columns and 'team_proj' in merged.columns:
                merged['team'] = merged['team_prop'].where(merged['team_prop'].notna(), merged['team_proj'])
            elif 'team_prop' in merged.columns:
                merged['team'] = merged['team_prop']
            else:
                merged['team'] = merged['team_proj']
        if 'position_prop' in merged.columns or 'position_proj' in merged.columns:
            if 'position_prop' in merged.columns and 'position_proj' in merged.columns:
                merged['position'] = merged['position_prop'].where(merged['position_prop'].notna(), merged['position_proj'])
            elif 'position_prop' in merged.columns:
                merged['position'] = merged['position_prop']
            else:
                merged['position'] = merged['position_proj']
        
        return merged
    
    def _calculate_value_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate value betting metrics"""
        
        # Map prop stats to projection columns
        stat_mapping = {
            'rushing_yards': '2024_proj_rush',
            'receiving_yards': '2024_proj_rec',
            'passing_yards': 'proj_pass_yds'  # Would need to add this to projections
        }
        
        df['model_prediction'] = 0.0
        df['edge_yards'] = 0.0
        df['edge_percentage'] = 0.0
        df['value_rating'] = 'NO_VALUE'
        
        for _, row in df.iterrows():
            stat = row['stat']
            line = row['line']
            
            # Get model prediction
            if stat in stat_mapping:
                proj_col = stat_mapping[stat]
                if proj_col in df.columns:
                    model_pred = row[proj_col]
                    if pd.notna(model_pred):
                        df.loc[df.index == row.name, 'model_prediction'] = model_pred
                        
                        # Calculate edge
                        edge_yards = model_pred - line
                        edge_percentage = (edge_yards / line) * 100 if line > 0 else 0
                        
                        df.loc[df.index == row.name, 'edge_yards'] = edge_yards
                        df.loc[df.index == row.name, 'edge_percentage'] = edge_percentage
                        
                        # Value rating
                        if abs(edge_percentage) >= 15:
                            rating = 'HIGH_VALUE'
                        elif abs(edge_percentage) >= 8:
                            rating = 'MEDIUM_VALUE'
                        elif abs(edge_percentage) >= 3:
                            rating = 'LOW_VALUE'
                        else:
                            rating = 'NO_VALUE'
                        
                        df.loc[df.index == row.name, 'value_rating'] = rating
        
        return df
    
    def get_best_value_opportunities(self, min_edge_threshold: float = 8.0) -> pd.DataFrame:
        """Get the best value opportunities from merged data"""
        
        merged_df = self.merge_props_with_projections()
        
        if merged_df.empty:
            return pd.DataFrame()
        
        # Filter for significant edges
        value_opportunities = merged_df[
            (abs(merged_df['edge_percentage']) >= min_edge_threshold) &
            (merged_df['model_prediction'] > 0)
        ].copy()
        
        # Sort by absolute edge percentage
        value_opportunities = value_opportunities.sort_values(
            'edge_percentage', key=abs, ascending=False
        )
        
        return value_opportunities[['player', 'team', 'position', 'book', 'stat', 'line', 
                                  'model_prediction', 'edge_yards', 'edge_percentage', 
                                  'over_odds', 'under_odds', 'value_rating']]
    
    def update_real_time_value_finder(self):
        """Update the real-time value finder with current prop lines"""
        
        value_opportunities = self.get_best_value_opportunities()
        
        if value_opportunities.empty:
            logger.warning("No value opportunities found")
            return
        
        with get_connection() as conn:
            execute(
                '''
                CREATE TABLE IF NOT EXISTS prop_opportunities (
                    id INTEGER PRIMARY KEY,
                    player TEXT NOT NULL,
                    team TEXT NOT NULL,
                    position TEXT NOT NULL,
                    book TEXT NOT NULL,
                    stat TEXT NOT NULL,
                    line REAL NOT NULL,
                    model_prediction REAL NOT NULL,
                    edge_yards REAL NOT NULL,
                    edge_percentage REAL NOT NULL,
                    over_odds INTEGER NOT NULL,
                    under_odds INTEGER NOT NULL,
                    value_rating TEXT NOT NULL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(player, book, stat)
                )
                ''',
                conn=conn,
            )

            execute('DELETE FROM prop_opportunities', conn=conn)

            rows = [
                (
                    r['player'],
                    r['team'],
                    r['position'],
                    r['book'],
                    r['stat'],
                    r['line'],
                    r['model_prediction'],
                    r['edge_yards'],
                    r['edge_percentage'],
                    r['over_odds'],
                    r['under_odds'],
                    r['value_rating'],
                )
                for _, r in value_opportunities.iterrows()
            ]

            if rows:
                executemany(
                    '''
                    INSERT INTO prop_opportunities 
                    (player, team, position, book, stat, line, model_prediction, 
                     edge_yards, edge_percentage, over_odds, under_odds, value_rating)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    params=rows,
                    conn=conn,
                )

            conn.commit()
        
        logger.info(f"Updated {len(value_opportunities)} value opportunities")
        return value_opportunities

    # ------------------------ Weekly integration ------------------------
    def get_weekly_prop_lines(self, week: int) -> pd.DataFrame:
        try:
            df = read_dataframe(
                """
                SELECT week, season, player, team, position, book, stat, line, over_odds, under_odds, game_date
                FROM weekly_prop_lines
                WHERE week = ?
                ORDER BY player, stat, book
                """,
                params=(week,),
            )
            return df
        except Exception as e:
            logger.error(f"Error reading weekly_prop_lines: {e}")
            return pd.DataFrame()

    def get_weekly_predictions(self, week: int) -> pd.DataFrame:
        """Fast baseline weekly predictions using weighted rolling averages over last 3 games."""
        try:
            df = read_dataframe(
                """
                SELECT player_id, name as player, team, position, season, week as game_week,
                       rushing_yards, receiving_yards, passing_yards
                FROM player_stats_enhanced
                WHERE week < ?
                """,
                params=(week,),
            )
        except Exception as e:
            logger.warning(f"Falling back: could not read player_stats_enhanced: {e}")
            return pd.DataFrame()

        if df.empty:
            return df

        df = df.sort_values(['player_id','season','game_week']).groupby('player_id', as_index=False).apply(
            lambda g: g.assign(
                rush_wra = g['rushing_yards'].rolling(3).apply(lambda x: (0.6*x.iloc[-1] + 0.3*(x.iloc[-2] if len(x)>1 else 0) + 0.1*(x.iloc[-3] if len(x)>2 else 0)), raw=False),
                rec_wra  = g['receiving_yards'].rolling(3).apply(lambda x: (0.6*x.iloc[-1] + 0.3*(x.iloc[-2] if len(x)>1 else 0) + 0.1*(x.iloc[-3] if len(x)>2 else 0)), raw=False),
                pass_wra = g['passing_yards'].rolling(3).apply(lambda x: (0.6*x.iloc[-1] + 0.3*(x.iloc[-2] if len(x)>1 else 0) + 0.1*(x.iloc[-3] if len(x)>2 else 0)), raw=False)
            )
        ).reset_index(drop=True)

        latest = df.groupby('player_id', as_index=False).tail(1)
        latest = latest.rename(columns={'rush_wra':'predict_rush_yds','rec_wra':'predict_rec_yds','pass_wra':'predict_pass_yds'})
        latest['week'] = week
        return latest[['player','team','position','week','predict_rush_yds','predict_rec_yds','predict_pass_yds']]

    def merge_weekly_props_with_predictions(self, week: int) -> pd.DataFrame:
        props = self.get_weekly_prop_lines(week)
        preds = self.get_weekly_predictions(week)
        if props.empty or preds.empty:
            return pd.DataFrame()
        # Normalize names (simple)
        props['player_clean'] = props['player'].str.lower().str.strip()
        preds['player_clean'] = preds['player'].str.lower().str.strip()
        merged = props.merge(preds, on='player_clean', how='left', suffixes=('', '_pred'))
        merged['model_prediction'] = 0.0
        def pick_pred(row):
            if row['stat'] == 'rushing_yards':
                return row.get('predict_rush_yds', 0) or 0.0
            if row['stat'] == 'receiving_yards':
                return row.get('predict_rec_yds', 0) or 0.0
            if row['stat'] == 'passing_yards':
                return row.get('predict_pass_yds', 0) or 0.0
            return 0.0
        merged['model_prediction'] = merged.apply(pick_pred, axis=1)
        merged['edge_yards'] = merged['model_prediction'] - merged['line']
        merged['edge_percentage'] = merged.apply(lambda r: (r['edge_yards']/r['line']*100.0) if r['line'] else 0.0, axis=1)
        merged['recommendation'] = merged['edge_yards'].apply(lambda x: 'OVER' if x>0 else 'UNDER')
        def tier(ep):
            a = abs(ep)
            if a >= 15: return 'HIGH_VALUE'
            if a >= 8: return 'MEDIUM_VALUE'
            if a >= 3: return 'LOW_VALUE'
            return 'NO_VALUE'
        merged['value_rating'] = merged['edge_percentage'].apply(tier)
        return merged

    def export_weekly_value_bets(self, week: int) -> pd.DataFrame:
        df = self.merge_weekly_props_with_predictions(week)
        if df.empty:
            return df
        out_dir = Path('reports'); out_dir.mkdir(exist_ok=True)
        df.to_csv(out_dir / f"week_{week}_value_bets.csv", index=False)
        df.to_json(out_dir / f"week_{week}_value_bets.json", orient='records', indent=2)
        # Markdown/HTML using season-long layout
        md = self.generate_markdown_report(df)
        (out_dir / f"week_{week}_value_report.md").write_text(md, encoding='utf-8')
        self.render_html_from_markdown(md, out_dir / f"week_{week}_value_report.html")
        # Write canonical table for dashboard
        try:
            canonical = df.copy()
            canonical['player_name'] = canonical.get('player')
            canonical['prop_type'] = canonical.get('stat')
            canonical['sportsbook'] = canonical.get('book')
            canonical['risk_level'] = canonical.get('value_rating')
            canonical['expected_roi'] = None
            canonical_cols = ['player_name','position','team','prop_type','sportsbook','line','model_prediction','edge_yards','edge_percentage','risk_level','recommendation','expected_roi','week','game_date']
            for c in canonical_cols:
                if c not in canonical.columns:
                    canonical[c] = None
            write_dataframe(canonical[canonical_cols], 'weekly_enhanced_value_bets', if_exists='replace', index=False)
        except Exception as e:
            logger.warning(f"Could not write weekly_enhanced_value_bets: {e}")
        return df
    
    def generate_value_report(self) -> str:
        """Generate a Markdown report of value opportunities (backward compatible)."""
        opportunities = self.get_best_value_opportunities()
        return self.generate_markdown_report(opportunities)

def main():
    """Main function to test prop integration"""
    
    integrator = PropIntegration()
    
    # Test the integration
    print("Testing prop line integration...")
    
    # Get value opportunities
    opportunities = integrator.get_best_value_opportunities()
    
    if opportunities.empty:
        print("No value opportunities found. Make sure prop lines are scraped first.")
        return
    
    # Generate report
    report = integrator.generate_value_report()
    print(report)
    
    # Update real-time value finder
    integrator.update_real_time_value_finder()
    print(f"\n✅ Updated real-time value finder with {len(opportunities)} opportunities")

if __name__ == "__main__":
    main()

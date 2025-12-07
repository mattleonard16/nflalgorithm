from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from rich.console import Console
from rich.table import Table

from config import config
from prop_integration import PropIntegration
try:
    from scripts.filter_engine import enrich_opportunities, top_quick_cards
except ImportError:
    from filter_engine import enrich_opportunities, top_quick_cards
import sqlite3

console = Console()

def load_opportunities() -> pd.DataFrame:
    integrator = PropIntegration()
    # Lower threshold to surface more rows for visual testing
    try:
        df = integrator.get_best_value_opportunities(min_edge_threshold=0.0)
    except TypeError:
        # Older signature fallback
        df = integrator.get_best_value_opportunities()
    if df is None or df.empty:
        csv_path = config.reports_dir / "value_bets.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            df = pd.DataFrame()
    return df

def render_dashboard(df: pd.DataFrame, template_dir: Path) -> Path:
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html"])
    )
    tpl = env.get_template("enhanced_dashboard.html")

    enriched = enrich_opportunities(df)
    rows = enriched.to_dict(orient="records")
    quick_cards = top_quick_cards(enriched, n=8).to_dict(orient="records")
    data_json = json.dumps([
        {"edge_pct_display": float(r.get("edge_pct_display", 0.0))}
        for r in rows
    ])

    html = tpl.render(rows=rows, quick_cards=quick_cards, data_json=data_json)

    out = config.reports_dir / "enhanced_dashboard.html"
    out.write_text(html, encoding="utf-8")

    if not enriched.empty:
        t = Table(title="Top Opportunities (Preview)", show_lines=False)
        for col in [
            "player","stat","line","model_prediction","edge_pct_display","book",
            "recommendation","visual_tier","confidence_score"
        ]:
            t.add_column(col)
        for _, r in enriched.sort_values("edge_pct_display", key=lambda s: s.abs(), ascending=False).head(10).iterrows():
            t.add_row(
                str(r.get("player","")), str(r.get("stat","")), f"{r.get('line','')}",
                f"{float(r.get('model_prediction',0)):.1f}", f"{float(r.get('edge_pct_display',0)):+.1f}%",
                str(r.get("book","")), str(r.get("recommendation","")), str(r.get("visual_tier","")), str(r.get("confidence_score",""))
            )
        console.print(t)

    return out

def export_interactive_csv(df: pd.DataFrame) -> Path:
    enriched = enrich_opportunities(df)
    cols = [
        "player","team","position","book","stat","line","model_prediction","edge_yards","edge_percentage",
        "value_rating","recommendation","edge_pct_display",
        "confidence_score","visual_tier","correlation_group","filter_tags"
    ]
    for c in cols:
        if c not in enriched.columns:
            enriched[c] = None
    out = config.reports_dir / "value_bets_enhanced.csv"
    enriched[cols].to_csv(out, index=False)
    # Optional: write canonical view into DB for dashboard fallback
    try:
        conn = sqlite3.connect(config.database.path)
        canonical = enriched.copy()
        canonical['player_name'] = canonical.get('player')
        canonical['prop_type'] = canonical.get('stat')
        canonical['sportsbook'] = canonical.get('book')
        if 'risk_level' not in canonical.columns and 'value_rating' in canonical.columns:
            canonical['risk_level'] = canonical['value_rating']
        if 'expected_roi' not in canonical.columns:
            canonical['expected_roi'] = None
        canonical_cols = [
            'player_name','position','team','prop_type','sportsbook','line','model_prediction',
            'edge_yards','edge_percentage','risk_level','recommendation','expected_roi'
        ]
        for cc in canonical_cols:
            if cc not in canonical.columns:
                canonical[cc] = None
        canonical[canonical_cols].to_sql('enhanced_value_bets', conn, if_exists='replace', index=False)
        conn.close()
    except Exception:
        pass
    return out

def write_quick_picks(df: pd.DataFrame, n: int = 5) -> Path:
    enriched = enrich_opportunities(df)
    order = {"HIGH_VALUE":0,"MEDIUM_VALUE":1,"LOW_VALUE":2,"NO_VALUE":3}
    enriched["__o"] = enriched["value_rating"].map(order).fillna(4)
    picks = enriched.sort_values(["__o","confidence_score","edge_pct_display"], ascending=[True, False, False]).head(n)
    def flames(score: int) -> str:
        if score >= 80: return "🔥🔥🔥"
        if score >= 60: return "🔥🔥"
        if score >= 40: return "🔥"
        return ""
    lines = []
    for _, r in picks.iterrows():
        tier = "HIGH" if str(r.get("value_rating","")) == "HIGH_VALUE" else "MEDIUM" if str(r.get("value_rating","")) == "MEDIUM_VALUE" else "LOW"
        stat_label = str(r.get("stat",""))
        edge = float(r.get("edge_pct_display",0.0))
        l = f"{'🔥' if tier=='HIGH' else '📈' if tier=='MEDIUM' else '➖'} {tier}: {r.get('player','')} {r.get('recommendation','')} {r.get('line','')} {stat_label} | {edge:+.1f}% edge | {r.get('book','')} {flames(int(r.get('confidence_score',0)))}"
        lines.append(l)
    out = config.reports_dir / "quick_picks.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out

def build_all() -> dict[str, Path]:
    config.reports_dir.mkdir(exist_ok=True)
    (config.reports_dir / "img").mkdir(exist_ok=True)
    template_dir = Path("templates")
    df = load_opportunities()
    if df is None or df.empty:
        console.print("[yellow]No opportunities to render[/yellow]")
        return {}
    paths = {}
    paths["html"] = render_dashboard(df, template_dir)
    paths["csv"] = export_interactive_csv(df)
    paths["quick"] = write_quick_picks(df, n=5)
    console.print(f"[green]Saved:[/green] {paths}")
    return paths

if __name__ == "__main__":
    build_all()


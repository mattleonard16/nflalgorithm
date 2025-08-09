from __future__ import annotations
import pandas as pd

def compute_confidence_score(row: pd.Series) -> int:
    edge_pct = float(row.get("edge_percentage", 0.0))
    if abs(edge_pct) <= 1.0:
        edge_pct *= 100.0
    edge_abs = abs(edge_pct)
    roi = row.get("expected_roi", None)
    roi_pct = (float(roi) * 100.0) if roi is not None and roi <= 1.0 else float(roi) if roi is not None else None
    line = float(row.get("line", 0.0))
    model_pred = float(row.get("model_prediction", 0.0))
    vol = abs(model_pred - line)

    score = 0.0
    score += min(edge_abs, 30) * 1.6
    if roi_pct is not None:
        score += max(min(roi_pct, 25), 0) * 1.0
    score += min(vol / 50.0, 20)
    score = max(0.0, min(100.0, score))
    return int(round(score))

def assign_visual_tier(value_rating: str | None) -> str:
    v = (value_rating or "").upper()
    if v == "HIGH_VALUE":
        return "high"
    if v == "MEDIUM_VALUE":
        return "medium"
    if v == "LOW_VALUE":
        return "low"
    return "none"

def derive_correlation_group(row: pd.Series) -> str:
    team = row.get("team", "")
    stat = row.get("stat", "")
    return f"{team}:{stat}"

def derive_filter_tags(row: pd.Series) -> list[str]:
    tags: list[str] = []
    for key in ("team","position","book","stat"):
        val = row.get(key, None)
        if pd.notna(val):
            tags.append(str(val).lower())
    return tags

def enrich_opportunities(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "edge_percentage" in out.columns:
        out["edge_pct_display"] = out["edge_percentage"].apply(lambda v: v*100.0 if abs(float(v)) <= 1.0 else float(v))
    else:
        out["edge_pct_display"] = 0.0
    if "edge_yards" in out.columns:
        out["recommendation"] = out["edge_yards"].apply(lambda x: "OVER" if pd.notna(x) and float(x) > 0 else "UNDER")
    else:
        out["recommendation"] = ""
    out["confidence_score"] = out.apply(compute_confidence_score, axis=1)
    out["visual_tier"] = out["value_rating"].apply(assign_visual_tier) if "value_rating" in out.columns else "none"
    out["correlation_group"] = out.apply(derive_correlation_group, axis=1)
    out["filter_tags"] = out.apply(derive_filter_tags, axis=1)
    out["quick_text"] = out.apply(lambda r: f"{r.get('player','')} {r.get('recommendation','')} {r.get('line','')} {r.get('stat','')} | {r.get('edge_pct_display',0):+.1f}% edge | {r.get('book','')}", axis=1)
    return out

def top_quick_cards(df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    d = df.copy()
    d = d.sort_values(["visual_tier","confidence_score","edge_pct_display"], ascending=[True, False, False])
    order = {"high":0,"medium":1,"low":2,"none":3}
    d["__o"] = d["visual_tier"].map(order).fillna(4)
    d = d.sort_values(["__o","confidence_score","edge_pct_display"], ascending=[True, False, False]).drop(columns="__o")
    return d.head(n)



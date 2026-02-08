"""Market bias detection agent.

Wraps the TE market bias analysis module and checks for position-specific
market inefficiencies that represent structural mispricing opportunities.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from agents import AgentReport
from agents.base_agent import BaseAgent


def _check_te_bias(season: int) -> Dict:
    """Run TE market bias analysis and return the report dict.

    Returns an empty dict if the analysis module cannot load data.
    """
    try:
        from scripts.te_market_bias import run_analysis
        return run_analysis(seasons=[season], output_dir=None)
    except Exception:
        return {}


def _te_bias_recommendation(report: Dict) -> tuple:
    """Derive recommendation, confidence, rationale from TE bias report.

    Returns (recommendation, confidence, rationale, data).
    """
    playoff_metrics = report.get("playoff_metrics", {})
    adjustment = report.get("suggested_adjustment", {})
    adj_value = adjustment.get("value")

    sample_size = playoff_metrics.get("sample_size", 0)
    if sample_size < 5:
        return (
            "NEUTRAL",
            0.40,
            "Insufficient TE playoff data for bias detection.",
            {"te_sample_size": sample_size},
        )

    if adj_value is not None and adj_value < 0:
        return (
            "REJECT",
            min(0.85, 0.60 + sample_size * 0.005),
            f"TE receiving yards props appear overpriced in playoffs "
            f"(suggested adjustment: {adj_value:+.1f} yards, n={sample_size})",
            {
                "te_adjustment_yards": adj_value,
                "te_sample_size": sample_size,
                "playoff_over_hit_rate": playoff_metrics.get("over_hit_rate"),
            },
        )

    return (
        "APPROVE",
        0.65,
        f"No significant TE market bias detected (n={sample_size})",
        {"te_sample_size": sample_size},
    )


class MarketBiasAgent(BaseAgent):
    """Detects position-specific market inefficiencies."""

    def __init__(self) -> None:
        super().__init__("market_bias_agent")

    def analyze(
        self,
        season: int,
        week: int,
        player_id: Optional[str] = None,
    ) -> List[AgentReport]:
        reports: List[AgentReport] = []

        # TE bias analysis (applies to all TE receiving_yards plays)
        te_report = _check_te_bias(season)

        if te_report:
            rec, conf, rationale, data = _te_bias_recommendation(te_report)
            reports.append(AgentReport(
                agent_name=self.name,
                recommendation=rec,
                confidence=conf,
                rationale=rationale,
                data=data,
                market="receiving_yards",
            ))
            self.logger.info("TE bias report: %s (conf=%.2f)", rec, conf)
        else:
            reports.append(AgentReport(
                agent_name=self.name,
                recommendation="NEUTRAL",
                confidence=0.30,
                rationale="Unable to run TE market bias analysis.",
                data={},
                market="receiving_yards",
            ))

        self.logger.info(
            "MarketBiasAgent produced %d reports for season=%d week=%d",
            len(reports),
            season,
            week,
        )
        return reports

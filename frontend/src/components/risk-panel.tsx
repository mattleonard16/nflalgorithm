"use client";

import type {
  CorrelationResponse,
  RiskSummary,
} from "@/lib/types";

function ExposureBar({
  label,
  fraction,
  limit,
}: {
  label: string;
  fraction: number;
  limit: number;
}) {
  const pct = Math.min((fraction / limit) * 100, 100);
  const isOver = fraction > limit;

  return (
    <div className="flex items-center gap-2">
      <span className="text-[11px] text-slate-400 w-12 shrink-0 font-[family-name:var(--font-jetbrains)]">
        {label}
      </span>
      <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden relative">
        <div
          className={`h-full rounded-full transition-all ${
            isOver ? "bg-red-500" : "bg-slate-500"
          }`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span
        className={`text-[11px] font-[family-name:var(--font-jetbrains)] tabular-nums w-12 text-right ${
          isOver ? "text-red-400" : "text-slate-400"
        }`}
      >
        {(fraction * 100).toFixed(1)}%
      </span>
    </div>
  );
}

export function RiskPanel({
  correlations,
  riskSummary,
}: {
  correlations: CorrelationResponse | null;
  riskSummary: RiskSummary | null;
}) {
  if (!correlations && !riskSummary) return null;

  const hasCorrelations =
    correlations &&
    (correlations.correlation_groups.length > 0 || correlations.team_stacks.length > 0);
  const hasRisk = riskSummary && riskSummary.total_stake > 0;

  if (!hasCorrelations && !hasRisk) return null;

  return (
    <div className="rounded-lg border border-slate-800/60 bg-[#111827]/50 p-4 space-y-4">
      <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">
        Risk & Exposure
      </h3>

      {/* Guardrail meters */}
      {hasRisk && riskSummary && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[11px] text-slate-500">
              Total Stake:{" "}
              <span className="text-slate-300 font-[family-name:var(--font-jetbrains)] tabular-nums">
                {riskSummary.total_stake.toFixed(2)}u
              </span>
            </span>
            <span className="text-[11px] text-slate-500">
              Bankroll:{" "}
              <span className="text-slate-300 font-[family-name:var(--font-jetbrains)] tabular-nums">
                {riskSummary.bankroll.toFixed(0)}u
              </span>
            </span>
          </div>

          {/* Team exposure */}
          {riskSummary.team_exposure
            .sort((a, b) => b.fraction - a.fraction)
            .slice(0, 5)
            .map((exp) => (
              <ExposureBar
                key={exp.team}
                label={exp.team ?? "?"}
                fraction={exp.fraction}
                limit={riskSummary.guardrails.max_team_exposure}
              />
            ))}
        </div>
      )}

      {/* Correlation groups */}
      {hasCorrelations && correlations && correlations.correlation_groups.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-[11px] text-slate-500 uppercase tracking-wider">
            Correlated Bets
          </h4>
          {correlations.correlation_groups.map((group) => (
            <div
              key={group.group}
              className="border-l-2 border-slate-700/60 pl-3 py-0.5"
            >
              <div className="flex items-center gap-2 mb-0.5">
                <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
                  {group.type === "pos"
                    ? "Pos Corr"
                    : group.type === "neg"
                      ? "Neg Corr"
                      : "Same Team"}
                </span>
                <span className="text-[10px] text-slate-500 font-[family-name:var(--font-jetbrains)]">
                  {group.combined_stake.toFixed(2)}u combined
                </span>
              </div>
              <div className="text-[11px] text-slate-300 leading-relaxed">
                {group.players.map((p, i) => (
                  <span key={`${p.player_id}-${p.market}-${i}`}>
                    {i > 0 && <span className="text-slate-600">{" · "}</span>}
                    {p.player_name ?? p.player_id}{" "}
                    <span className="text-slate-500">
                      ({p.market?.replace(/_/g, " ")})
                    </span>
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Warnings */}
      {riskSummary && riskSummary.warnings.length > 0 && (
        <div className="space-y-1">
          <h4 className="text-[11px] text-slate-500 uppercase tracking-wider">
            Warnings
          </h4>
          {riskSummary.warnings.map((w, i) => (
            <div
              key={`${w.player_id}-${i}`}
              className="flex items-center gap-2 text-[11px]"
            >
              <span className="text-red-400 font-medium">
                {w.player_name ?? w.player_id}
              </span>
              <span className="text-slate-500">{w.warning}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

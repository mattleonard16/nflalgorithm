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
            isOver
              ? "bg-gradient-to-r from-red-500 to-red-400"
              : pct > 70
                ? "bg-gradient-to-r from-amber-500 to-amber-400"
                : "bg-gradient-to-r from-emerald-500 to-emerald-400"
          }`}
          style={{ width: `${pct}%` }}
        />
        {/* Limit indicator line */}
        <div
          className="absolute top-0 bottom-0 w-px bg-slate-400/50"
          style={{ left: "100%" }}
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
              className="rounded border border-slate-700/40 bg-slate-800/20 p-2"
            >
              <div className="flex items-center gap-2 mb-1">
                <span
                  className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold border ${
                    group.type === "pos"
                      ? "bg-amber-500/10 text-amber-400 border-amber-500/20"
                      : group.type === "neg"
                        ? "bg-purple-500/10 text-purple-400 border-purple-500/20"
                        : "bg-blue-500/10 text-blue-400 border-blue-500/20"
                  }`}
                >
                  {group.type === "pos"
                    ? "POS CORR"
                    : group.type === "neg"
                      ? "NEG CORR"
                      : "SAME TEAM"}
                </span>
                <span className="text-[10px] text-slate-500 font-[family-name:var(--font-jetbrains)]">
                  {group.combined_stake.toFixed(2)}u combined
                </span>
              </div>
              <div className="flex flex-wrap gap-1">
                {group.players.map((p) => (
                  <span
                    key={`${p.player_id}-${p.market}`}
                    className="text-[11px] text-slate-300 bg-slate-700/30 px-1.5 py-0.5 rounded"
                  >
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
              <span className="inline-flex items-center px-1.5 py-0.5 rounded bg-red-500/10 text-red-400 border border-red-500/20">
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

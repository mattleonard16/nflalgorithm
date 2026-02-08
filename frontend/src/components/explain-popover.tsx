"use client";

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import type { WhyPayload } from "@/lib/types";

function MiniBar({ value, max = 100, label }: { value: number | null; max?: number; label: string }) {
  const pct = value != null ? Math.min((value / max) * 100, 100) : 0;
  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] text-slate-500 w-16 shrink-0">{label}</span>
      <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full bg-gradient-to-r from-blue-500 to-cyan-400 transition-all"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-[10px] text-slate-400 font-[family-name:var(--font-jetbrains)] tabular-nums w-8 text-right">
        {value != null ? value.toFixed(0) : "--"}
      </span>
    </div>
  );
}

export function ExplainPopover({
  why,
  children,
}: {
  why: WhyPayload | undefined;
  children: React.ReactNode;
}) {
  if (!why) return <>{children}</>;

  const decision = why.agents?.decision;
  const decisionColor =
    decision === "APPROVED"
      ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/30"
      : decision === "REJECTED"
        ? "bg-red-500/20 text-red-300 border-red-500/30"
        : "bg-slate-700/20 text-slate-400 border-slate-600/30";

  return (
    <Popover>
      <PopoverTrigger asChild>{children}</PopoverTrigger>
      <PopoverContent
        side="left"
        align="start"
        className="w-72 bg-[#0d1220] border-slate-700/60 p-3 space-y-3"
      >
        {/* Model section */}
        <div>
          <h4 className="text-[10px] text-slate-500 uppercase tracking-wider mb-1.5">
            Model
          </h4>
          <div className="grid grid-cols-3 gap-2">
            <div>
              <span className="text-[10px] text-slate-500">Mu</span>
              <p className="text-[13px] text-slate-200 font-[family-name:var(--font-jetbrains)] tabular-nums">
                {why.model?.mu?.toFixed(1) ?? "--"}
              </p>
            </div>
            <div>
              <span className="text-[10px] text-slate-500">Sigma</span>
              <p className="text-[13px] text-slate-200 font-[family-name:var(--font-jetbrains)] tabular-nums">
                {why.model?.sigma?.toFixed(1) ?? "--"}
              </p>
            </div>
            <div>
              <span className="text-[10px] text-slate-500">Ctx Sens</span>
              <p className="text-[13px] text-slate-200 font-[family-name:var(--font-jetbrains)] tabular-nums">
                {why.model?.context_sensitivity?.toFixed(2) ?? "--"}
              </p>
            </div>
          </div>
        </div>

        {/* Confidence bars */}
        <div>
          <h4 className="text-[10px] text-slate-500 uppercase tracking-wider mb-1.5">
            Confidence
          </h4>
          <div className="space-y-1">
            <MiniBar label="Total" value={why.confidence?.total ? why.confidence.total * 100 : null} />
            <MiniBar label="Edge" value={why.confidence?.edge_pct ? why.confidence.edge_pct * 100 : null} />
            <MiniBar label="Win%" value={why.confidence?.p_win ? why.confidence.p_win * 100 : null} />
          </div>
        </div>

        {/* Volume / Volatility */}
        <div className="grid grid-cols-2 gap-2">
          <div>
            <span className="text-[10px] text-slate-500">Tgt Share</span>
            <p className="text-[12px] text-slate-300 font-[family-name:var(--font-jetbrains)] tabular-nums">
              {why.volume?.target_share != null
                ? `${(why.volume.target_share * 100).toFixed(1)}%`
                : "--"}
            </p>
          </div>
          <div>
            <span className="text-[10px] text-slate-500">Volatility</span>
            <p className="text-[12px] text-slate-300 font-[family-name:var(--font-jetbrains)] tabular-nums">
              {why.volatility?.score?.toFixed(2) ?? "--"}
            </p>
          </div>
        </div>

        {/* Risk flags */}
        {(why.risk?.correlation_group || why.risk?.exposure_warning) && (
          <div>
            <h4 className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">
              Risk Flags
            </h4>
            {why.risk?.correlation_group && (
              <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] bg-amber-500/10 text-amber-400 border border-amber-500/20 mr-1">
                {why.risk.correlation_group}
              </span>
            )}
            {why.risk?.exposure_warning && (
              <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] bg-red-500/10 text-red-400 border border-red-500/20">
                {why.risk.exposure_warning}
              </span>
            )}
          </div>
        )}

        {/* Agent verdict */}
        {why.agents?.decision && (
          <div>
            <h4 className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">
              Agent Verdict
            </h4>
            <div className="flex items-center gap-2">
              <span
                className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-bold border ${decisionColor}`}
              >
                {why.agents.decision}
              </span>
              {why.agents.merged_confidence != null && (
                <span className="text-[10px] text-slate-400 font-[family-name:var(--font-jetbrains)]">
                  {(why.agents.merged_confidence * 100).toFixed(0)}% conf
                </span>
              )}
            </div>
            {why.agents.top_rationale && (
              <p className="text-[10px] text-slate-500 mt-1 line-clamp-2">
                {why.agents.top_rationale}
              </p>
            )}
          </div>
        )}
      </PopoverContent>
    </Popover>
  );
}

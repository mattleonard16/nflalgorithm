"use client";

import Link from "next/link";
import type { WatchedPick } from "@/lib/slate-watchlist";
import { watchlistKey } from "@/lib/slate-watchlist";

const MONO = "font-[family-name:var(--font-jetbrains)]";

const MARKET_DOTS: Record<string, string> = {
  Pass: "#d4a84b",
  Rec: "#5b9fd4",
  Rush: "#3d9b6e",
};

function marketDot(label: string): string {
  return MARKET_DOTS[label] ?? "#64748b";
}

/**
 * Read-only view of the picks a user starred on the Slate. Editing the set
 * happens on the Slate itself, so this table never mutates storage.
 */
export function WatchedPicksTable({
  picks,
  emptyCopy = "Star picks on the Slate to track them here.",
}: {
  picks: WatchedPick[];
  emptyCopy?: string;
}) {
  const headCell =
    "border-b border-amber-400/25 px-3 py-2.5 text-left text-[11px] font-semibold uppercase tracking-wider text-slate-500";

  if (picks.length === 0) {
    return (
      <div className="rounded-lg border border-slate-800/70 bg-[#111827]/60 px-6 py-10 text-center">
        <p className="text-sm text-slate-400">{emptyCopy}</p>
        <Link
          href="/"
          className="mt-1 inline-block text-xs text-amber-400/80 transition-colors hover:text-amber-300"
        >
          Open the Slate
        </Link>
      </div>
    );
  }

  return (
    <div className="overflow-auto rounded-lg border border-slate-800/70 bg-[#111827]/80">
      <table className="w-full border-collapse text-[13px]">
        <thead>
          <tr>
            <th className={headCell}>Player</th>
            <th className={headCell}>Team</th>
            <th className={headCell}>Market</th>
            <th className={`${headCell} text-right`}>μ</th>
            <th className={`${headCell} text-right`}>σ</th>
          </tr>
        </thead>
        <tbody>
          {picks.map((pick) => (
            <tr
              key={watchlistKey(pick)}
              className="border-b border-slate-800/50 last:border-b-0 hover:bg-amber-400/5"
            >
              <td className="px-3 py-2">
                <span className="font-medium text-slate-100">{pick.player_name}</span>{" "}
                {pick.position && (
                  <span className={`${MONO} ml-1 text-[11px] text-slate-500`}>
                    {pick.position}
                  </span>
                )}
              </td>
              <td className={`${MONO} px-3 py-2 text-[12px] text-slate-400`}>
                {pick.team ?? "--"}
                {pick.opponent && (
                  <span className="text-slate-600"> @ {pick.opponent}</span>
                )}
              </td>
              <td className="px-3 py-2">
                <span className="inline-flex items-center gap-1.5 rounded border border-slate-700/60 bg-slate-500/10 px-2 py-0.5 text-[11px] uppercase tracking-wider text-slate-300">
                  <span
                    className="h-1.5 w-1.5 shrink-0 rounded-full"
                    style={{ background: marketDot(pick.market_label) }}
                    aria-hidden="true"
                  />
                  {pick.market_label}
                </span>
              </td>
              <td
                className={`${MONO} px-3 py-2 text-right font-semibold tabular-nums text-slate-100`}
              >
                {pick.mu != null ? pick.mu.toFixed(1) : "\u2014"}
              </td>
              <td className={`${MONO} px-3 py-2 text-right tabular-nums text-slate-400`}>
                {pick.sigma != null ? pick.sigma.toFixed(1) : "\u2014"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

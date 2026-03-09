"use client";

import { useEffect, useState, useCallback, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

type Sport = "nfl" | "nba";

interface BacktestSummary {
  avg_model_mae: number | null;
  avg_line_mae: number | null;
  model_beats_line_pct: number | null;
  total_bets: number;
}

interface RecentPick {
  game_date: string;
  player_name: string | null;
  market: string;
  line: number;
  actual: number;
  mu: number | null;
  model_abs_error: number | null;
  line_abs_error: number | null;
  model_beats_line: number | null;
}

interface BacktestResponse {
  summary: BacktestSummary;
  recent_picks: RecentPick[];
}

type SortKey = keyof Pick<RecentPick, "game_date" | "market" | "line" | "actual" | "model_abs_error" | "line_abs_error">;

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function fetchBacktest(sport: Sport, season?: number): Promise<BacktestResponse> {
  const params = new URLSearchParams({ sport });
  if (season != null) params.set("season", String(season));
  const res = await fetch(`${API_BASE}/api/backtest/summary?${params}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function StatCard({
  title,
  value,
  subtitle,
  colorClass,
}: {
  title: string;
  value: string;
  subtitle?: string;
  colorClass?: string;
}) {
  return (
    <Card className="bg-[#0d1220] border-slate-800/50">
      <CardContent className="pt-5 pb-4 px-5">
        <p className="text-[10px] font-medium text-slate-600 uppercase tracking-[0.15em] mb-1">
          {title}
        </p>
        <p className={cn("text-2xl font-bold font-[family-name:var(--font-jetbrains)]", colorClass ?? "text-slate-100")}>
          {value}
        </p>
        {subtitle && (
          <p className="text-xs text-slate-600 mt-0.5">{subtitle}</p>
        )}
      </CardContent>
    </Card>
  );
}

function SortIcon({ col, sortKey, sortAsc }: { col: SortKey; sortKey: SortKey; sortAsc: boolean }) {
  return sortKey === col ? (
    <span className="ml-1 text-blue-400">{sortAsc ? "↑" : "↓"}</span>
  ) : (
    <span className="ml-1 text-slate-700">↕</span>
  );
}

function WinnerBadge({ modelWon }: { modelWon: number | null }) {
  if (modelWon === null || modelWon === undefined) {
    return <span className="text-slate-600 text-xs">—</span>;
  }
  return modelWon === 1 ? (
    <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400">Model</span>
  ) : (
    <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-red-500/10 text-red-400">Line</span>
  );
}

const SPORT_TABS: { value: Sport; label: string }[] = [
  { value: "nfl", label: "NFL" },
  { value: "nba", label: "NBA" },
];

function BacktestContent() {
  const searchParams = useSearchParams();
  const initialSport = (searchParams.get("sport") as Sport | null) ?? "nfl";
  const [sport, setSport] = useState<Sport>(initialSport);
  const [data, setData] = useState<BacktestResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("game_date");
  const [sortAsc, setSortAsc] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchBacktest(sport);
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load backtest data");
    } finally {
      setLoading(false);
    }
  }, [sport]);

  useEffect(() => {
    load();
  }, [load]);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortAsc((prev) => !prev);
    } else {
      setSortKey(key);
      setSortAsc(false);
    }
  };

  const sorted = [...(data?.recent_picks ?? [])].sort((a, b) => {
    const av = a[sortKey];
    const bv = b[sortKey];
    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
    if (typeof av === "string" && typeof bv === "string") {
      return sortAsc ? av.localeCompare(bv) : bv.localeCompare(av);
    }
    return sortAsc ? (av as number) - (bv as number) : (bv as number) - (av as number);
  });

  const summary = data?.summary;
  const maeColor =
    summary?.avg_model_mae != null && summary.avg_model_mae < 3
      ? "text-emerald-400"
      : "text-amber-400";
  const beatsPctColor =
    summary?.model_beats_line_pct != null && summary.model_beats_line_pct >= 50
      ? "text-emerald-400"
      : "text-red-400";

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">Backtest Results</h1>
          <p className="text-sm text-slate-500 mt-0.5">
            Historical model vs line accuracy
          </p>
        </div>

        {/* Sport tabs */}
        <div className="flex gap-1 p-1 rounded-lg bg-slate-900/60 border border-slate-800/50">
          {SPORT_TABS.map((tab) => (
            <button
              key={tab.value}
              onClick={() => setSport(tab.value)}
              className={cn(
                "px-4 py-1.5 text-xs font-semibold rounded-md transition-all duration-150 font-[family-name:var(--font-jetbrains)]",
                sport === tab.value
                  ? "bg-blue-500/20 text-blue-300"
                  : "text-slate-500 hover:text-slate-300"
              )}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-red-800/40 bg-red-900/10 p-4 text-sm text-red-400">
          {error}
        </div>
      )}

      {/* Summary Cards */}
      {loading ? (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => (
            <Card key={i} className="bg-[#0d1220] border-slate-800/50">
              <CardContent className="pt-5 pb-4 px-5">
                <div className="h-3 w-24 bg-slate-800 rounded animate-pulse mb-2" />
                <div className="h-7 w-16 bg-slate-800 rounded animate-pulse" />
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard
            title="Model MAE"
            value={summary?.avg_model_mae != null ? summary.avg_model_mae.toFixed(2) : "—"}
            subtitle="Avg absolute error"
            colorClass={maeColor}
          />
          <StatCard
            title="Line MAE"
            value={summary?.avg_line_mae != null ? summary.avg_line_mae.toFixed(2) : "—"}
            subtitle="Sportsbook abs error"
          />
          <StatCard
            title="Model Beats Line"
            value={summary?.model_beats_line_pct != null ? `${summary.model_beats_line_pct.toFixed(1)}%` : "—"}
            subtitle="Pct of records"
            colorClass={beatsPctColor}
          />
          <StatCard
            title="Total Records"
            value={summary?.total_bets != null ? String(summary.total_bets) : "0"}
            subtitle="Graded picks"
          />
        </div>
      )}

      {/* Picks Table */}
      <Card className="bg-[#0d1220] border-slate-800/50">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium text-slate-400 uppercase tracking-wider">
            Recent Picks with Outcomes
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {loading ? (
            <div className="px-6 py-12 text-center text-slate-600 text-sm">
              Loading picks…
            </div>
          ) : sorted.length === 0 ? (
            <div className="px-6 py-12 text-center text-slate-600 text-sm">
              No graded picks found.
              <br />
              <span className="text-xs mt-1 block">
                Run{" "}
                <code className="font-mono bg-slate-800 px-1 rounded">
                  {sport === "nba" ? "make nba-backtest" : "make week-grade"}
                </code>{" "}
                to populate backtest data.
              </span>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-800/50">
                    {[
                      { key: "game_date" as SortKey, label: "Date", align: "left" },
                      { key: null, label: "Player", align: "left" },
                      { key: "market" as SortKey, label: "Market", align: "left" },
                      { key: "line" as SortKey, label: "Line", align: "right" },
                      { key: "actual" as SortKey, label: "Actual", align: "right" },
                      { key: null, label: "Model Pred", align: "right" },
                      { key: "model_abs_error" as SortKey, label: "Mdl Err", align: "right" },
                      { key: "line_abs_error" as SortKey, label: "Line Err", align: "right" },
                      { key: null, label: "Winner", align: "center" },
                    ].map(({ key, label, align }) => (
                      <th
                        key={label}
                        className={cn(
                          "px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider",
                          align === "right" ? "text-right" : align === "center" ? "text-center" : "text-left",
                          key ? "cursor-pointer hover:text-slate-400 select-none" : ""
                        )}
                        onClick={key ? () => handleSort(key) : undefined}
                      >
                        {label}
                        {key && <SortIcon col={key} sortKey={sortKey} sortAsc={sortAsc} />}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sorted.map((pick, i) => (
                    <tr
                      key={`${pick.game_date}-${pick.player_name}-${pick.market}-${i}`}
                      className="border-b border-slate-800/30 hover:bg-slate-800/20 transition-colors"
                    >
                      <td className="px-4 py-3 font-mono text-xs text-slate-500">
                        {pick.game_date}
                      </td>
                      <td className="px-4 py-3 font-medium text-slate-200">
                        {pick.player_name ?? "—"}
                      </td>
                      <td className="px-4 py-3">
                        <span className="text-xs font-mono text-blue-400 bg-blue-500/10 px-2 py-0.5 rounded uppercase">
                          {pick.market}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-slate-300">
                        {pick.line.toFixed(1)}
                      </td>
                      <td className="px-4 py-3 text-right font-mono font-bold text-slate-100">
                        {pick.actual.toFixed(1)}
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-blue-400">
                        {pick.mu != null ? pick.mu.toFixed(1) : "—"}
                      </td>
                      <td className="px-4 py-3 text-right">
                        {pick.model_abs_error != null ? (
                          <ErrorBadge value={pick.model_abs_error} />
                        ) : (
                          <span className="text-slate-600">—</span>
                        )}
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-slate-500 text-xs">
                        {pick.line_abs_error != null ? pick.line_abs_error.toFixed(2) : "—"}
                      </td>
                      <td className="px-4 py-3 text-center">
                        <WinnerBadge modelWon={pick.model_beats_line} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

export default function BacktestPage() {
  return (
    <Suspense>
      <BacktestContent />
    </Suspense>
  );
}

function ErrorBadge({ value }: { value: number }) {
  const isLow = value < 2;
  const isMid = value < 4;
  return (
    <span
      className={cn(
        "text-xs font-mono px-2 py-0.5 rounded",
        isLow && "bg-emerald-500/10 text-emerald-400",
        !isLow && isMid && "bg-amber-500/10 text-amber-400",
        !isLow && !isMid && "bg-red-500/10 text-red-400"
      )}
    >
      {value.toFixed(2)}
    </span>
  );
}

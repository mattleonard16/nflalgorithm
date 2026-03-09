"use client";

import { useEffect, useState, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { CheckCircle2, XCircle, MinusCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import { getPerformance, getOutcomes, getUserBets, getUserStats } from "@/lib/api";
import { getNbaPerformance, getNbaOutcomes } from "@/lib/nba-api";
import type { BetOutcome, WeeklyPerformance, UserBet, UserStats } from "@/lib/types";
import type { NbaBetOutcome, NbaDailyPerformance } from "@/lib/nba-types";

type Sport = "nfl" | "nba";
type ResultFilter = "all" | "WIN" | "LOSS" | "PUSH";

const SPORT_TABS: { value: Sport; label: string }[] = [
  { value: "nfl", label: "NFL" },
  { value: "nba", label: "NBA" },
];

const RESULT_FILTERS: { value: ResultFilter; label: string }[] = [
  { value: "all", label: "All" },
  { value: "WIN", label: "Wins" },
  { value: "LOSS", label: "Losses" },
  { value: "PUSH", label: "Pushes" },
];

// Normalise result strings to canonical form
function normaliseResult(r: string | null): string | null {
  if (!r) return null;
  const u = r.toUpperCase();
  if (u === "WIN" || u === "W") return "WIN";
  if (u === "LOSS" || u === "L") return "LOSS";
  if (u === "PUSH" || u === "P") return "PUSH";
  return u;
}

// ─── Shared UI pieces ────────────────────────────────────────────────────────

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
        <p
          className={cn(
            "text-2xl font-bold font-[family-name:var(--font-jetbrains)]",
            colorClass ?? "text-slate-100"
          )}
        >
          {value}
        </p>
        {subtitle && <p className="text-xs text-slate-600 mt-0.5">{subtitle}</p>}
      </CardContent>
    </Card>
  );
}

function ResultIcon({ result }: { result: string | null }) {
  if (result === "WIN") return <CheckCircle2 className="h-4 w-4 text-emerald-400" />;
  if (result === "LOSS") return <XCircle className="h-4 w-4 text-red-400" />;
  return <MinusCircle className="h-4 w-4 text-slate-600" />;
}

function TierBadge({ tier }: { tier: string }) {
  const tierMap: Record<string, string> = {
    Premium: "bg-emerald-500/10 text-emerald-400",
    Strong: "bg-blue-500/10 text-blue-400",
    Marginal: "bg-amber-500/10 text-amber-400",
    Pass: "bg-slate-700/50 text-slate-500",
  };
  return (
    <span className={cn("text-[11px] font-mono px-2 py-0.5 rounded", tierMap[tier] ?? "bg-slate-700/50 text-slate-500")}>
      {tier}
    </span>
  );
}

function ResultFilterBar({
  value,
  onChange,
}: {
  value: ResultFilter;
  onChange: (v: ResultFilter) => void;
}) {
  return (
    <div className="flex gap-1 p-0.5 rounded-md bg-slate-900/60 border border-slate-800/50">
      {RESULT_FILTERS.map((f) => (
        <button
          key={f.value}
          onClick={() => onChange(f.value)}
          className={cn(
            "px-3 py-1 text-xs rounded-sm transition-all font-medium",
            value === f.value
              ? "bg-slate-700 text-slate-200"
              : "text-slate-600 hover:text-slate-400"
          )}
        >
          {f.label}
        </button>
      ))}
    </div>
  );
}

function LoadingSkeleton() {
  return (
    <div className="space-y-6">
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
      <Card className="bg-[#0d1220] border-slate-800/50">
        <CardContent className="py-12 text-center text-slate-600 text-sm">
          Loading…
        </CardContent>
      </Card>
    </div>
  );
}

// ─── My Slip: pending bets from /api/user/bets ───────────────────────────────

function MySlipTab() {
  const [userBets, setUserBets] = useState<UserBet[]>([]);
  const [stats, setStats] = useState<UserStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [resultFilter, setResultFilter] = useState<ResultFilter>("all");

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [betsRes, statsRes] = await Promise.all([getUserBets(), getUserStats()]);
      setUserBets(betsRes.bets);
      setStats(statsRes);
    } catch (err) {
      setError(
        err instanceof Error && err.message.includes("401")
          ? "Sign in to view your bet slip."
          : (err instanceof Error ? err.message : "Failed to load bets")
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const pending = userBets.filter((b) => b.result === null || b.result === undefined || b.result === "");
  const settled = userBets.filter((b) => b.result !== null && b.result !== undefined && b.result !== "");

  const filteredSettled = settled.filter((b) => {
    if (resultFilter === "all") return true;
    return normaliseResult(b.result) === resultFilter;
  });

  if (loading) return <LoadingSkeleton />;

  if (error) {
    return (
      <div className="rounded-lg border border-amber-800/40 bg-amber-900/10 p-4 text-sm text-amber-400">
        {error}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Stats cards */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard title="Total Bets" value={String(stats.total_bets)} />
          <StatCard
            title="Win Rate"
            value={stats.total_bets > 0 ? `${((stats.wins / stats.total_bets) * 100).toFixed(1)}%` : "—"}
            colorClass={stats.wins / Math.max(stats.total_bets, 1) >= 0.5 ? "text-emerald-400" : "text-red-400"}
          />
          <StatCard
            title="Total P/L"
            value={`${stats.total_profit >= 0 ? "+" : ""}${stats.total_profit.toFixed(1)}u`}
            colorClass={stats.total_profit >= 0 ? "text-emerald-400" : "text-red-400"}
          />
          <StatCard
            title="ROI"
            value={`${stats.roi >= 0 ? "+" : ""}${stats.roi.toFixed(1)}%`}
            colorClass={stats.roi >= 0 ? "text-emerald-400" : "text-red-400"}
          />
        </div>
      )}

      {/* Pending bets */}
      <Card className="bg-[#0d1220] border-slate-800/50">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium text-slate-400 uppercase tracking-wider">
            Pending ({pending.length})
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {pending.length === 0 ? (
            <div className="px-6 py-10 text-center text-slate-600 text-sm">
              No pending bets. Use the{" "}
              <span className="text-blue-400">+</span> button on value bets to add to your slip.
            </div>
          ) : (
            <UserBetsTable bets={pending} showResult={false} />
          )}
        </CardContent>
      </Card>

      {/* Settled history */}
      <Card className="bg-[#0d1220] border-slate-800/50">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between gap-3 flex-wrap">
            <CardTitle className="text-sm font-medium text-slate-400 uppercase tracking-wider">
              History ({settled.length})
            </CardTitle>
            <ResultFilterBar value={resultFilter} onChange={setResultFilter} />
          </div>
        </CardHeader>
        <CardContent className="p-0">
          {filteredSettled.length === 0 ? (
            <div className="px-6 py-10 text-center text-slate-600 text-sm">
              No settled bets for this filter.
            </div>
          ) : (
            <UserBetsTable bets={filteredSettled} showResult />
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function UserBetsTable({ bets, showResult }: { bets: UserBet[]; showResult: boolean }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-800/50">
            {showResult && <th className="w-10 px-4 py-3" />}
            <th className="text-left px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider">Placed</th>
            <th className="text-left px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider">Market</th>
            <th className="text-left px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider">Book</th>
            <th className="text-right px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider">Line</th>
            <th className="text-right px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider">Price</th>
            <th className="text-right px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider">Stake</th>
            {showResult && (
              <>
                <th className="text-right px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider">Actual</th>
                <th className="text-right px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider">Profit</th>
              </>
            )}
          </tr>
        </thead>
        <tbody>
          {bets.map((bet) => {
            const norm = normaliseResult(bet.result);
            const dateStr = bet.placed_at ? new Date(bet.placed_at).toLocaleDateString() : "—";
            return (
              <tr
                key={bet.bet_id}
                className="border-b border-slate-800/30 hover:bg-slate-800/20 transition-colors"
              >
                {showResult && (
                  <td className="px-4 py-3">
                    <ResultIcon result={norm} />
                  </td>
                )}
                <td className="px-4 py-3 font-mono text-xs text-slate-500">{dateStr}</td>
                <td className="px-4 py-3">
                  <span className="text-xs font-mono text-blue-400 bg-blue-500/10 px-2 py-0.5 rounded uppercase">
                    {bet.market}
                  </span>
                </td>
                <td className="px-4 py-3 text-xs text-slate-500">{bet.sportsbook}</td>
                <td className="px-4 py-3 text-right font-mono text-slate-300">O {bet.line?.toFixed(1)}</td>
                <td className="px-4 py-3 text-right font-mono text-slate-400">
                  {bet.price > 0 ? `+${bet.price}` : bet.price}
                </td>
                <td className="px-4 py-3 text-right font-mono text-slate-300">{bet.stake?.toFixed(2)}u</td>
                {showResult && (
                  <>
                    <td className="px-4 py-3 text-right font-mono font-bold text-slate-100">
                      {bet.actual_result?.toFixed(0) ?? "—"}
                    </td>
                    <td
                      className={cn(
                        "px-4 py-3 text-right font-mono font-medium",
                        (bet.profit_units ?? 0) >= 0 ? "text-emerald-400" : "text-red-400"
                      )}
                    >
                      {bet.profit_units != null
                        ? `${bet.profit_units >= 0 ? "+" : ""}${bet.profit_units.toFixed(1)}u`
                        : "—"}
                    </td>
                  </>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ─── Historical outcomes from grading pipeline ───────────────────────────────

function NflHistoryTab() {
  const [weeks, setWeeks] = useState<WeeklyPerformance[]>([]);
  const [selectedWeek, setSelectedWeek] = useState<{ season: number; week: number } | null>(null);
  const [outcomes, setOutcomes] = useState<BetOutcome[]>([]);
  const [resultFilter, setResultFilter] = useState<ResultFilter>("all");
  const [loading, setLoading] = useState(true);
  const [outcomesLoading, setOutcomesLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [totalBets, setTotalBets] = useState(0);
  const [winRate, setWinRate] = useState(0);
  const [totalProfit, setTotalProfit] = useState(0);
  const [overallRoi, setOverallRoi] = useState(0);

  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        const data = await getPerformance();
        setWeeks(data.weeks);
        setTotalBets(data.total_bets);
        setWinRate(data.win_rate);
        setTotalProfit(data.total_profit);
        setOverallRoi(data.overall_roi);
        if (data.weeks.length > 0) {
          setSelectedWeek({ season: data.weeks[0].season, week: data.weeks[0].week });
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  useEffect(() => {
    if (!selectedWeek) return;
    async function loadOutcomes() {
      setOutcomesLoading(true);
      try {
        const data = await getOutcomes(selectedWeek!.season, selectedWeek!.week);
        setOutcomes(data);
      } catch {
        setOutcomes([]);
      } finally {
        setOutcomesLoading(false);
      }
    }
    loadOutcomes();
  }, [selectedWeek]);

  const filteredOutcomes = outcomes.filter((o) => {
    if (resultFilter === "all") return true;
    return normaliseResult(o.result) === resultFilter;
  });

  if (loading) return <LoadingSkeleton />;
  if (error) {
    return (
      <div className="rounded-lg border border-red-800/40 bg-red-900/10 p-4 text-sm text-red-400">
        {error}
      </div>
    );
  }
  if (weeks.length === 0) {
    return (
      <Card className="bg-[#0d1220] border-slate-800/50">
        <CardContent className="py-12 text-center text-slate-600 text-sm">
          No outcomes yet. Run{" "}
          <code className="font-mono bg-slate-800 px-1 rounded">make week-grade</code> after games.
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard title="Total Bets" value={String(totalBets)} />
        <StatCard
          title="Win Rate"
          value={`${winRate.toFixed(1)}%`}
          colorClass={winRate >= 50 ? "text-emerald-400" : "text-red-400"}
        />
        <StatCard
          title="Total P/L"
          value={`${totalProfit >= 0 ? "+" : ""}${totalProfit.toFixed(1)}u`}
          colorClass={totalProfit >= 0 ? "text-emerald-400" : "text-red-400"}
        />
        <StatCard
          title="ROI"
          value={`${overallRoi >= 0 ? "+" : ""}${overallRoi.toFixed(1)}%`}
          colorClass={overallRoi >= 0 ? "text-emerald-400" : "text-red-400"}
        />
      </div>

      <Card className="bg-[#0d1220] border-slate-800/50">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between flex-wrap gap-3">
            <CardTitle className="text-sm font-medium text-slate-400 uppercase tracking-wider">
              Bet History
            </CardTitle>
            <div className="flex items-center gap-3 flex-wrap">
              <ResultFilterBar value={resultFilter} onChange={setResultFilter} />
              <select
                value={selectedWeek ? `${selectedWeek.season}-${selectedWeek.week}` : ""}
                onChange={(e) => {
                  const [s, w] = e.target.value.split("-").map(Number);
                  setSelectedWeek({ season: s, week: w });
                }}
                className="text-xs bg-slate-900 border border-slate-700 text-slate-300 rounded px-2 py-1.5 focus:outline-none focus:border-blue-500"
              >
                {weeks.map((w) => (
                  <option key={`${w.season}-${w.week}`} value={`${w.season}-${w.week}`}>
                    Week {w.week}, {w.season}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          {outcomesLoading ? (
            <div className="px-6 py-8 text-center text-slate-600 text-sm">Loading…</div>
          ) : filteredOutcomes.length === 0 ? (
            <div className="px-6 py-10 text-center text-slate-600 text-sm">
              No records for this selection.
            </div>
          ) : (
            <GradedOutcomesTable outcomes={filteredOutcomes} />
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function NbaHistoryTab() {
  const [days, setDays] = useState<NbaDailyPerformance[]>([]);
  const [selectedDate, setSelectedDate] = useState<string | null>(null);
  const [outcomes, setOutcomes] = useState<NbaBetOutcome[]>([]);
  const [resultFilter, setResultFilter] = useState<ResultFilter>("all");
  const [loading, setLoading] = useState(true);
  const [outcomesLoading, setOutcomesLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [totalBets, setTotalBets] = useState(0);
  const [winRate, setWinRate] = useState(0);
  const [totalProfit, setTotalProfit] = useState(0);
  const [overallRoi, setOverallRoi] = useState(0);

  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        const data = await getNbaPerformance();
        setDays(data.days);
        setTotalBets(data.total_bets);
        setWinRate(data.win_rate);
        setTotalProfit(data.total_profit);
        setOverallRoi(data.overall_roi);
        if (data.days.length > 0) setSelectedDate(data.days[0].game_date);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  useEffect(() => {
    if (!selectedDate) return;
    async function loadOutcomes() {
      setOutcomesLoading(true);
      try {
        const data = await getNbaOutcomes(selectedDate!);
        setOutcomes(data);
      } catch {
        setOutcomes([]);
      } finally {
        setOutcomesLoading(false);
      }
    }
    loadOutcomes();
  }, [selectedDate]);

  const filteredOutcomes = outcomes.filter((o) => {
    if (resultFilter === "all") return true;
    return normaliseResult(o.result) === resultFilter;
  });

  if (loading) return <LoadingSkeleton />;
  if (error) {
    return (
      <div className="rounded-lg border border-red-800/40 bg-red-900/10 p-4 text-sm text-red-400">
        {error}
      </div>
    );
  }
  if (days.length === 0) {
    return (
      <Card className="bg-[#0d1220] border-slate-800/50">
        <CardContent className="py-12 text-center text-slate-600 text-sm">
          No outcomes yet. Run{" "}
          <code className="font-mono bg-slate-800 px-1 rounded">make nba-grade</code> after games.
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard title="Total Bets" value={String(totalBets)} />
        <StatCard
          title="Win Rate"
          value={`${winRate.toFixed(1)}%`}
          colorClass={winRate >= 50 ? "text-emerald-400" : "text-red-400"}
        />
        <StatCard
          title="Total P/L"
          value={`${totalProfit >= 0 ? "+" : ""}${totalProfit.toFixed(1)}u`}
          colorClass={totalProfit >= 0 ? "text-emerald-400" : "text-red-400"}
        />
        <StatCard
          title="ROI"
          value={`${overallRoi >= 0 ? "+" : ""}${overallRoi.toFixed(1)}%`}
          colorClass={overallRoi >= 0 ? "text-emerald-400" : "text-red-400"}
        />
      </div>

      <Card className="bg-[#0d1220] border-slate-800/50">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between flex-wrap gap-3">
            <CardTitle className="text-sm font-medium text-slate-400 uppercase tracking-wider">
              Bet History
            </CardTitle>
            <div className="flex items-center gap-3 flex-wrap">
              <ResultFilterBar value={resultFilter} onChange={setResultFilter} />
              <select
                value={selectedDate ?? ""}
                onChange={(e) => setSelectedDate(e.target.value)}
                className="text-xs bg-slate-900 border border-slate-700 text-slate-300 rounded px-2 py-1.5 focus:outline-none focus:border-blue-500"
              >
                {days.map((d) => (
                  <option key={d.game_date} value={d.game_date}>
                    {d.game_date}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          {outcomesLoading ? (
            <div className="px-6 py-8 text-center text-slate-600 text-sm">Loading…</div>
          ) : filteredOutcomes.length === 0 ? (
            <div className="px-6 py-10 text-center text-slate-600 text-sm">
              No records for this selection.
            </div>
          ) : (
            <NbaOutcomesTable outcomes={filteredOutcomes} />
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function GradedOutcomesTable({ outcomes }: { outcomes: BetOutcome[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-800/50">
            <th className="w-10 px-4 py-3" />
            <th className="text-left px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider">Player</th>
            <th className="text-left px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider">Market</th>
            <th className="text-right px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider">Line</th>
            <th className="text-right px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider">Actual</th>
            <th className="text-right px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider">Profit</th>
            <th className="text-left px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider">Tier</th>
          </tr>
        </thead>
        <tbody>
          {outcomes.map((bet) => {
            const norm = normaliseResult(bet.result);
            return (
              <tr
                key={bet.bet_id}
                className="border-b border-slate-800/30 hover:bg-slate-800/20 transition-colors"
              >
                <td className="px-4 py-3"><ResultIcon result={norm} /></td>
                <td className="px-4 py-3 font-medium text-slate-200">{bet.player_name ?? "—"}</td>
                <td className="px-4 py-3">
                  <span className="text-xs font-mono text-blue-400 bg-blue-500/10 px-2 py-0.5 rounded uppercase">
                    {bet.market}
                  </span>
                </td>
                <td className="px-4 py-3 text-right font-mono text-slate-300">O {bet.line}</td>
                <td className="px-4 py-3 text-right font-mono font-bold text-slate-100">
                  {bet.actual_result?.toFixed(0) ?? "—"}
                </td>
                <td className={cn("px-4 py-3 text-right font-mono font-medium",
                  (bet.profit_units ?? 0) >= 0 ? "text-emerald-400" : "text-red-400"
                )}>
                  {bet.profit_units != null
                    ? `${bet.profit_units >= 0 ? "+" : ""}${bet.profit_units.toFixed(1)}u`
                    : "—"}
                </td>
                <td className="px-4 py-3">
                  {bet.confidence_tier ? <TierBadge tier={bet.confidence_tier} /> : <span className="text-slate-700 text-xs">—</span>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function NbaOutcomesTable({ outcomes }: { outcomes: NbaBetOutcome[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-800/50">
            <th className="w-10 px-4 py-3" />
            <th className="text-left px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider">Player</th>
            <th className="text-left px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider">Market</th>
            <th className="text-right px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider">Line</th>
            <th className="text-right px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider">Actual</th>
            <th className="text-right px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider">Profit</th>
            <th className="text-left px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider">Tier</th>
          </tr>
        </thead>
        <tbody>
          {outcomes.map((bet) => {
            const norm = normaliseResult(bet.result);
            return (
              <tr
                key={bet.bet_id}
                className="border-b border-slate-800/30 hover:bg-slate-800/20 transition-colors"
              >
                <td className="px-4 py-3"><ResultIcon result={norm} /></td>
                <td className="px-4 py-3 font-medium text-slate-200">{bet.player_name ?? "—"}</td>
                <td className="px-4 py-3">
                  <span className="text-xs font-mono text-blue-400 bg-blue-500/10 px-2 py-0.5 rounded uppercase">
                    {bet.market}
                  </span>
                </td>
                <td className="px-4 py-3 text-right font-mono text-slate-300">O {bet.line}</td>
                <td className="px-4 py-3 text-right font-mono font-bold text-slate-100">
                  {bet.actual_result?.toFixed(0) ?? "—"}
                </td>
                <td className={cn("px-4 py-3 text-right font-mono font-medium",
                  (bet.profit_units ?? 0) >= 0 ? "text-emerald-400" : "text-red-400"
                )}>
                  {bet.profit_units != null
                    ? `${bet.profit_units >= 0 ? "+" : ""}${bet.profit_units.toFixed(1)}u`
                    : "—"}
                </td>
                <td className="px-4 py-3">
                  {bet.confidence_tier ? <TierBadge tier={bet.confidence_tier} /> : <span className="text-slate-700 text-xs">—</span>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ─── Page ────────────────────────────────────────────────────────────────────

export default function BetsPage() {
  const [sport, setSport] = useState<Sport>("nfl");

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">Bet Tracking</h1>
          <p className="text-sm text-slate-500 mt-0.5">Manage your slip and review historical outcomes</p>
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

      <Tabs defaultValue="my-slip" className="space-y-6">
        <TabsList className="bg-slate-900/60 border border-slate-800/50 p-1 h-auto">
          <TabsTrigger
            value="my-slip"
            className="text-xs data-[state=active]:bg-slate-700 data-[state=active]:text-slate-100 text-slate-500 px-4 py-1.5"
          >
            My Slip
          </TabsTrigger>
          <TabsTrigger
            value="history"
            className="text-xs data-[state=active]:bg-slate-700 data-[state=active]:text-slate-100 text-slate-500 px-4 py-1.5"
          >
            History
          </TabsTrigger>
        </TabsList>

        <TabsContent value="my-slip" className="mt-0">
          <MySlipTab />
        </TabsContent>

        <TabsContent value="history" className="mt-0">
          {sport === "nfl" ? <NflHistoryTab /> : <NbaHistoryTab />}
        </TabsContent>
      </Tabs>
    </div>
  );
}

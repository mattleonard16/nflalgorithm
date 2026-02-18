"use client";

import { Fragment, useEffect, useState, useCallback } from "react";
import { getNbaProjections, getNbaSchedule, fetchNbaValueBets, getNbaExplanation } from "@/lib/nba-api";
import type { NbaProjection, NbaGame, NbaValueBet, NbaWhyPayload } from "@/lib/nba-types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

type SortKey = "projected_value" | "stat_last5_avg" | "confidence";

function SortIcon({ col, sortKey, sortAsc }: { col: SortKey; sortKey: SortKey; sortAsc: boolean }) {
  return sortKey === col ? (
    <span className="ml-1 text-blue-400">{sortAsc ? "↑" : "↓"}</span>
  ) : (
    <span className="ml-1 text-slate-700">↕</span>
  );
}

const MARKETS = [
  { value: "pts", label: "Points" },
  { value: "reb", label: "Rebounds" },
  { value: "ast", label: "Assists" },
  { value: "fg3m", label: "3-Pointers" },
] as const;

type Market = (typeof MARKETS)[number]["value"];

export default function NbaDashboardPage() {
  const [projections, setProjections] = useState<NbaProjection[]>([]);
  const [games, setGames] = useState<NbaGame[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("projected_value");
  const [sortAsc, setSortAsc] = useState(false);

  // Value bets state
  const [valueBets, setValueBets] = useState<NbaValueBet[]>([]);
  const [valueBetsLoading, setValueBetsLoading] = useState(true);
  const [valueBetsError, setValueBetsError] = useState<string | null>(null);
  const [valueBetsMarket, setValueBetsMarket] = useState<Market>("pts");
  const [bestLineOnly, setBestLineOnly] = useState(false);

  // Explainability state
  const [expandedBet, setExpandedBet] = useState<string | null>(null);
  const [whyCache, setWhyCache] = useState<Record<string, NbaWhyPayload>>({});
  const [whyLoading, setWhyLoading] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const [projRes, schedRes] = await Promise.all([
          getNbaProjections(undefined, "pts", 0, 100),
          getNbaSchedule(),
        ]);
        setProjections(projRes.projections);
        setGames(schedRes.games);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load NBA data");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  useEffect(() => {
    async function loadValueBets() {
      setValueBetsLoading(true);
      setValueBetsError(null);
      try {
        const res = await fetchNbaValueBets({
          market: valueBetsMarket,
          best_line_only: bestLineOnly,
        });
        setValueBets(res.bets);
      } catch (err) {
        setValueBetsError(err instanceof Error ? err.message : "Failed to load value bets");
        setValueBets([]);
      } finally {
        setValueBetsLoading(false);
      }
    }
    loadValueBets();
  }, [valueBetsMarket, bestLineOnly]);

  const handleToggleWhy = useCallback(async (bet: NbaValueBet) => {
    const key = `${bet.player_id}:${bet.market}`;
    if (expandedBet === key) {
      setExpandedBet(null);
      return;
    }
    setExpandedBet(key);
    if (whyCache[key]) return;

    if (bet.player_id == null) return;
    setWhyLoading(key);
    try {
      const res = await getNbaExplanation(bet.player_id, bet.market);
      setWhyCache((prev) => ({ ...prev, [key]: res.why }));
    } catch (err) {
      console.error("Failed to load bet explanation:", err);
      setExpandedBet(null);
    } finally {
      setWhyLoading(null);
    }
  }, [expandedBet, whyCache]);

  const sorted = [...projections].sort((a, b) => {
    const av = a[sortKey] ?? 0;
    const bv = b[sortKey] ?? 0;
    return sortAsc ? (av as number) - (bv as number) : (bv as number) - (av as number);
  });

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortAsc(!sortAsc);
    } else {
      setSortKey(key);
      setSortAsc(false);
    }
  };

  const today = new Date().toLocaleDateString("en-US", {
    weekday: "long",
    month: "long",
    day: "numeric",
    year: "numeric",
  });

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64 text-slate-500">
        Loading NBA projections…
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">NBA Dashboard</h1>
          <p className="text-sm text-slate-500 mt-0.5">{today}</p>
        </div>
        <div className="flex items-center gap-3">
          <div className="text-right">
            <p className="text-xs text-slate-600 uppercase tracking-wider">Games Today</p>
            <p className="text-2xl font-bold text-blue-400 font-[family-name:var(--font-jetbrains)]">
              {games.length}
            </p>
          </div>
          <div className="text-right">
            <p className="text-xs text-slate-600 uppercase tracking-wider">Players Projected</p>
            <p className="text-2xl font-bold text-slate-100 font-[family-name:var(--font-jetbrains)]">
              {projections.length}
            </p>
          </div>
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-red-800/40 bg-red-900/10 p-4 text-sm text-red-400">
          {error}
          <p className="mt-1 text-xs text-red-600">
            Run <code className="font-mono bg-red-900/20 px-1 rounded">make ingest-nba</code> then{" "}
            <code className="font-mono bg-red-900/20 px-1 rounded">make nba-predict</code> to
            generate projections.
          </p>
        </div>
      )}

      {/* Today's Games */}
      {games.length > 0 && (
        <div>
          <h2 className="text-xs font-medium text-slate-600 uppercase tracking-[0.15em] mb-3">
            Today's Slate
          </h2>
          <div className="flex flex-wrap gap-2">
            {games.map((g) => (
              <div
                key={g.game_id}
                className="flex items-center gap-2 px-3 py-2 rounded-lg bg-slate-900/60 border border-slate-800/50"
              >
                <span className="text-sm font-semibold text-slate-200 font-[family-name:var(--font-jetbrains)]">
                  {g.away_team}
                </span>
                <span className="text-xs text-slate-600">@</span>
                <span className="text-sm font-semibold text-slate-200 font-[family-name:var(--font-jetbrains)]">
                  {g.home_team}
                </span>
                {g.status && (
                  <Badge
                    variant="outline"
                    className="text-[10px] border-slate-700 text-slate-500"
                  >
                    {g.status}
                  </Badge>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Projections Table */}
      <Card className="bg-[#0d1220] border-slate-800/50">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium text-slate-400 uppercase tracking-wider">
            Points Projections
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {projections.length === 0 ? (
            <div className="px-6 py-12 text-center text-slate-600 text-sm">
              No projections available for today.
              <br />
              <span className="text-xs mt-1 block">
                Run <code className="font-mono">make ingest-nba &amp;&amp; make nba-predict</code>{" "}
                to generate.
              </span>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-800/50">
                    <th className="text-left px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider">
                      Player
                    </th>
                    <th className="text-left px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider">
                      Team
                    </th>
                    <th className="text-left px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider">
                      Matchup
                    </th>
                    <th
                      className="text-right px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider cursor-pointer hover:text-slate-400 select-none"
                      onClick={() => handleSort("projected_value")}
                    >
                      Proj PTS
                      <SortIcon col="projected_value" sortKey={sortKey} sortAsc={sortAsc} />
                    </th>
                    <th
                      className="text-right px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider cursor-pointer hover:text-slate-400 select-none"
                      onClick={() => handleSort("stat_last5_avg")}
                    >
                      L5 Avg
                      <SortIcon col="stat_last5_avg" sortKey={sortKey} sortAsc={sortAsc} />
                    </th>
                    <th className="text-right px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider">
                      L10 Avg
                    </th>
                    <th
                      className="text-right px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider cursor-pointer hover:text-slate-400 select-none"
                      onClick={() => handleSort("confidence")}
                    >
                      Confidence
                      <SortIcon col="confidence" sortKey={sortKey} sortAsc={sortAsc} />
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {sorted.map((p, i) => (
                    <tr
                      key={`${p.player_id}-${i}`}
                      className="border-b border-slate-800/30 hover:bg-slate-800/20 transition-colors"
                    >
                      <td className="px-4 py-3 font-medium text-slate-200">
                        {p.player_name}
                      </td>
                      <td className="px-4 py-3">
                        <span className="text-xs font-mono text-blue-400 bg-blue-500/10 px-2 py-0.5 rounded">
                          {p.team}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-slate-500 text-xs">
                        {p.matchup ?? "—"}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <span className="text-lg font-bold text-blue-400 font-[family-name:var(--font-jetbrains)]">
                          {p.projected_value.toFixed(1)}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right text-slate-400 font-[family-name:var(--font-jetbrains)]">
                        {p.stat_last5_avg != null ? p.stat_last5_avg.toFixed(1) : "—"}
                      </td>
                      <td className="px-4 py-3 text-right text-slate-500 font-[family-name:var(--font-jetbrains)]">
                        {p.stat_last10_avg != null ? p.stat_last10_avg.toFixed(1) : "—"}
                      </td>
                      <td className="px-4 py-3 text-right">
                        {p.confidence != null ? (
                          <ConfidenceBadge value={p.confidence} />
                        ) : (
                          <span className="text-slate-600">—</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Value Bets Section */}
      <Card className="bg-[#0d1220] border-slate-800/50">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between flex-wrap gap-3">
            <CardTitle className="text-sm font-medium text-slate-400 uppercase tracking-wider">
              Value Bets
            </CardTitle>
            <div className="flex items-center gap-3">
              {/* Market selector */}
              <select
                value={valueBetsMarket}
                onChange={(e) => setValueBetsMarket(e.target.value as Market)}
                className="text-xs bg-slate-900 border border-slate-700 text-slate-300 rounded px-2 py-1.5 focus:outline-none focus:border-blue-500"
              >
                {MARKETS.map((m) => (
                  <option key={m.value} value={m.value}>
                    {m.label}
                  </option>
                ))}
              </select>
              {/* Best Line Only toggle */}
              <label className="flex items-center gap-2 cursor-pointer select-none">
                <div
                  role="checkbox"
                  aria-checked={bestLineOnly}
                  tabIndex={0}
                  onClick={() => setBestLineOnly((v) => !v)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") setBestLineOnly((v) => !v);
                  }}
                  className={cn(
                    "w-8 h-4 rounded-full transition-colors relative cursor-pointer",
                    bestLineOnly ? "bg-blue-600" : "bg-slate-700"
                  )}
                >
                  <span
                    className={cn(
                      "absolute top-0.5 w-3 h-3 rounded-full bg-white transition-transform",
                      bestLineOnly ? "translate-x-4" : "translate-x-0.5"
                    )}
                  />
                </div>
                <span className="text-xs text-slate-500">Best Line Only</span>
              </label>
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          {valueBetsLoading ? (
            <div className="px-6 py-8 text-center text-slate-600 text-sm">
              Loading value bets…
            </div>
          ) : valueBetsError ? (
            <div className="px-6 py-8 text-center text-red-500 text-sm">
              {valueBetsError}
            </div>
          ) : valueBets.length === 0 ? (
            <div className="px-6 py-12 text-center text-slate-600 text-sm">
              No value bets found for today.
              <br />
              <span className="text-xs mt-1 block">
                Run <code className="font-mono">make nba-full</code> to generate odds and value rankings.
              </span>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-800/50">
                    <th className="w-10 px-2 py-3" />
                    <th className="text-left px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider">
                      Player
                    </th>
                    <th className="text-left px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider">
                      Team
                    </th>
                    <th className="text-left px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider">
                      Market
                    </th>
                    <th className="text-right px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider">
                      Line
                    </th>
                    <th className="text-right px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider">
                      Over
                    </th>
                    <th className="text-right px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider">
                      Proj
                    </th>
                    <th className="text-right px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider">
                      Edge %
                    </th>
                    <th className="text-right px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider">
                      Kelly
                    </th>
                    <th className="text-left px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider">
                      Book
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {valueBets.map((bet) => {
                    const betKey = `${bet.player_id}:${bet.market}:${bet.sportsbook}`;
                    const isExpanded = expandedBet === betKey;
                    const why = whyCache[betKey];
                    const isLoadingWhy = whyLoading === betKey;
                    return (
                      <Fragment key={betKey}>
                        <tr
                          className={cn(
                            "border-b border-slate-800/30 hover:bg-slate-800/20 transition-colors cursor-pointer",
                            isExpanded && "bg-slate-800/30"
                          )}
                          onClick={() => handleToggleWhy(bet)}
                        >
                          <td className="px-2 py-3 text-center">
                            <button
                              className={cn(
                                "w-6 h-6 rounded-full flex items-center justify-center text-xs transition-all",
                                isExpanded
                                  ? "bg-blue-500/20 text-blue-400 ring-1 ring-blue-500/40"
                                  : "bg-slate-800/50 text-slate-600 hover:text-slate-400 hover:bg-slate-700/50"
                              )}
                              title="Why this bet?"
                              onClick={(e) => { e.stopPropagation(); handleToggleWhy(bet); }}
                            >
                              {isLoadingWhy ? (
                                <span className="animate-spin">...</span>
                              ) : isExpanded ? (
                                "−"
                              ) : (
                                "i"
                              )}
                            </button>
                          </td>
                          <td className="px-4 py-3 font-medium text-slate-200">
                            {bet.player_name}
                          </td>
                          <td className="px-4 py-3">
                            {bet.team ? (
                              <span className="text-xs font-mono text-blue-400 bg-blue-500/10 px-2 py-0.5 rounded">
                                {bet.team}
                              </span>
                            ) : (
                              <span className="text-slate-600">—</span>
                            )}
                          </td>
                          <td className="px-4 py-3">
                            <span className="text-xs font-mono text-slate-400 uppercase">
                              {bet.market}
                            </span>
                          </td>
                          <td className="px-4 py-3 text-right font-mono text-slate-300">
                            {bet.line.toFixed(1)}
                          </td>
                          <td className="px-4 py-3 text-right font-mono text-slate-400">
                            {bet.over_price > 0 ? `+${bet.over_price}` : bet.over_price}
                          </td>
                          <td className="px-4 py-3 text-right font-mono font-bold text-blue-400">
                            {bet.mu.toFixed(1)}
                          </td>
                          <td className="px-4 py-3 text-right">
                            <EdgeBadge value={bet.edge_percentage} />
                          </td>
                          <td className="px-4 py-3 text-right font-mono text-slate-400 text-xs">
                            {(bet.kelly_fraction * 100).toFixed(1)}%
                          </td>
                          <td className="px-4 py-3">
                            <span className="text-xs text-slate-500">{bet.sportsbook}</span>
                          </td>
                        </tr>
                        {isExpanded && (
                          <tr>
                            <td colSpan={10} className="p-0">
                              <WhyPanel why={why} loading={isLoadingWhy} />
                            </td>
                          </tr>
                        )}
                      </Fragment>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function WhyPanel({ why, loading }: { why?: NbaWhyPayload; loading: boolean }) {
  if (loading) {
    return (
      <div className="px-8 py-6 bg-slate-900/40 border-b border-slate-800/30 text-center text-slate-500 text-sm">
        Loading explanation…
      </div>
    );
  }

  if (!why) {
    return (
      <div className="px-8 py-6 bg-slate-900/40 border-b border-slate-800/30 text-center text-slate-600 text-sm">
        No explanation data available. Run <code className="font-mono text-xs bg-slate-800 px-1 rounded">make nba-full</code> to generate.
      </div>
    );
  }

  return (
    <div className="px-6 py-5 bg-slate-900/40 border-b border-slate-800/30">
      <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
        {/* Model */}
        <WhySection title="Model">
          <WhyRow label="Projected" value={why.model.projected_value?.toFixed(1)} />
          <WhyRow label="Sigma" value={why.model.sigma?.toFixed(2)} />
          <WhyRow label="Confidence" value={why.model.confidence != null ? `${(why.model.confidence * 100).toFixed(0)}%` : null} />
        </WhySection>

        {/* Recency */}
        <WhySection title="Recency">
          <WhyRow label="L5 Avg" value={why.recency.last5_avg?.toFixed(1)} />
          <WhyRow label="L10 Avg" value={why.recency.last10_avg?.toFixed(1)} />
          <WhyRow label="Trend" value={why.recency.trend} badge={
            why.recency.trend === "up" ? "emerald" : why.recency.trend === "down" ? "red" : "slate"
          } />
        </WhySection>

        {/* Variance */}
        <WhySection title="Variance">
          <WhyRow label="Sigma" value={why.variance.sigma?.toFixed(2)} />
          <WhyRow label="CoV" value={why.variance.cv?.toFixed(3)} />
        </WhySection>

        {/* Confidence */}
        <WhySection title="Confidence">
          <WhyRow label="P(Win)" value={why.confidence.p_win != null ? `${(why.confidence.p_win * 100).toFixed(1)}%` : null} />
          <WhyRow label="Edge" value={why.confidence.edge_percentage != null ? `${(why.confidence.edge_percentage * 100).toFixed(1)}%` : null} />
          <WhyRow label="ROI" value={why.confidence.expected_roi != null ? `${(why.confidence.expected_roi * 100).toFixed(1)}%` : null} />
          <WhyRow label="Kelly" value={why.confidence.kelly_fraction != null ? `${(why.confidence.kelly_fraction * 100).toFixed(2)}%` : null} />
        </WhySection>

        {/* Risk */}
        <WhySection title="Risk">
          <WhyRow label="Corr Group" value={why.risk.correlation_group} badge={why.risk.correlation_group ? "amber" : undefined} />
          <WhyRow label="Exposure" value={why.risk.exposure_warning} badge={why.risk.exposure_warning ? "red" : undefined} />
          <WhyRow label="Risk Kelly" value={why.risk.risk_adjusted_kelly != null ? `${(why.risk.risk_adjusted_kelly * 100).toFixed(2)}%` : null} />
        </WhySection>

        {/* Agents */}
        <WhySection title="Agent Consensus">
          <WhyRow label="Decision" value={why.agents.decision} badge={
            why.agents.decision === "APPROVED" ? "emerald" : why.agents.decision === "REJECTED" ? "red" : "slate"
          } />
          <WhyRow label="Confidence" value={why.agents.merged_confidence != null ? `${(why.agents.merged_confidence * 100).toFixed(0)}%` : null} />
          {why.agents.votes && (
            <div className="flex gap-2 mt-1">
              {Object.entries(why.agents.votes).map(([k, v]) => (
                <span key={k} className="text-[10px] font-mono text-slate-500">
                  {k}:{v}
                </span>
              ))}
            </div>
          )}
          {why.agents.top_rationale && (
            <p className="text-[10px] text-slate-600 mt-1 leading-relaxed">
              {why.agents.top_rationale}
            </p>
          )}
        </WhySection>
      </div>
    </div>
  );
}

function WhySection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <h4 className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">{title}</h4>
      <div className="space-y-1">{children}</div>
    </div>
  );
}

function WhyRow({ label, value, badge }: { label: string; value?: string | null; badge?: string }) {
  const display = value ?? "—";
  const badgeColors: Record<string, string> = {
    emerald: "bg-emerald-500/10 text-emerald-400",
    red: "bg-red-500/10 text-red-400",
    amber: "bg-amber-500/10 text-amber-400",
    slate: "bg-slate-700/50 text-slate-400",
  };
  return (
    <div className="flex items-center justify-between gap-2">
      <span className="text-[11px] text-slate-600">{label}</span>
      {badge && value ? (
        <span className={cn("text-[11px] font-mono px-1.5 py-0.5 rounded", badgeColors[badge] ?? badgeColors.slate)}>
          {display}
        </span>
      ) : (
        <span className={cn("text-[11px] font-mono", value ? "text-slate-300" : "text-slate-700")}>
          {display}
        </span>
      )}
    </div>
  );
}

function ConfidenceBadge({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const tier =
    pct >= 75 ? "high" : pct >= 50 ? "medium" : "low";
  return (
    <span
      className={cn(
        "text-xs font-mono px-2 py-0.5 rounded font-medium",
        tier === "high" && "bg-emerald-500/10 text-emerald-400",
        tier === "medium" && "bg-amber-500/10 text-amber-400",
        tier === "low" && "bg-slate-700/50 text-slate-500"
      )}
    >
      {pct}%
    </span>
  );
}

function EdgeBadge({ value }: { value: number }) {
  const pct = (value * 100).toFixed(1);
  const isHigh = value >= 0.1;
  const isMedium = value >= 0.05 && value < 0.1;
  return (
    <span
      className={cn(
        "text-xs font-mono px-2 py-0.5 rounded font-bold",
        isHigh && "bg-emerald-500/10 text-emerald-400",
        isMedium && "bg-amber-500/10 text-amber-400",
        !isHigh && !isMedium && "bg-slate-700/50 text-slate-500"
      )}
    >
      {pct}%
    </span>
  );
}

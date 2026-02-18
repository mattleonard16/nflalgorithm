"use client";

import { useEffect, useState } from "react";
import { getNbaMeta, getNbaProjections, getNbaPlayers, getNbaCorrelation, getNbaRiskSummary } from "@/lib/nba-api";
import type { NbaProjection, NbaPlayerSummary, NbaCorrelationResponse, NbaRiskSummary } from "@/lib/nba-types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Cell,
} from "recharts";

const BLUE = "#3b82f6";
const BLUE_DIM = "#1d4ed8";

function CustomTooltip({
  active,
  payload,
  market,
}: {
  active?: boolean;
  payload?: Array<{ payload: { full: string; value: number } }>;
  market: string;
}) {
  if (active && payload?.[0]) {
    const { full, value } = payload[0].payload;
    return (
      <div className="bg-[#0d1220] border border-slate-700 rounded-lg px-3 py-2 text-sm">
        <p className="text-slate-200 font-medium">{full}</p>
        <p className="text-blue-400 font-bold font-[family-name:var(--font-jetbrains)]">
          {value.toFixed(1)} {market}
        </p>
      </div>
    );
  }
  return null;
}

const MARKETS = [
  { value: "pts", label: "Points" },
  { value: "reb", label: "Rebounds" },
  { value: "ast", label: "Assists" },
  { value: "fg3m", label: "3-Pointers" },
] as const;

type Market = (typeof MARKETS)[number]["value"];

export default function NbaAnalyticsPage() {
  const [projections, setProjections] = useState<NbaProjection[]>([]);
  const [topScorers, setTopScorers] = useState<NbaPlayerSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [market, setMarket] = useState<Market>("pts");
  const [season, setSeason] = useState<number | null>(null);
  const [correlationData, setCorrelationData] = useState<NbaCorrelationResponse | null>(null);
  const [riskData, setRiskData] = useState<NbaRiskSummary | null>(null);
  const [riskLoading, setRiskLoading] = useState(true);

  // Fetch meta once to get latest season
  useEffect(() => {
    async function loadMeta() {
      try {
        const meta = await getNbaMeta();
        const latest = meta.available_seasons.length > 0
          ? meta.available_seasons[meta.available_seasons.length - 1]
          : new Date().getFullYear();
        setSeason(latest);
      } catch {
        setSeason(new Date().getFullYear());
      }
    }
    loadMeta();
  }, []);

  useEffect(() => {
    if (season === null) return;
    setLoading(true);
    setError(null);
    async function load() {
      try {
        const [projRes, playersRes] = await Promise.all([
          getNbaProjections(undefined, market, 0, 15),
          getNbaPlayers(season!, undefined, undefined, 10),
        ]);
        setProjections(projRes.projections);
        setTopScorers(playersRes.players);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load analytics");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [market, season]);

  useEffect(() => {
    setRiskLoading(true);
    async function loadRisk() {
      try {
        const [corrRes, riskRes] = await Promise.all([
          getNbaCorrelation(),
          getNbaRiskSummary(),
        ]);
        setCorrelationData(corrRes);
        setRiskData(riskRes);
      } catch (err) {
        console.error("Failed to load risk/correlation data:", err);
      } finally {
        setRiskLoading(false);
      }
    }
    loadRisk();
  }, []);

  const projChartData = projections.map((p) => ({
    name: p.player_name.split(" ").slice(-1)[0], // last name only
    value: p.projected_value,
    full: p.player_name,
  }));

  const scorersChartData = topScorers.map((p) => ({
    name: p.player_name.split(" ").slice(-1)[0],
    value: p.avg_pts,
    full: p.player_name,
  }));

  const selectedMarketLabel = MARKETS.find((m) => m.value === market)?.label ?? market.toUpperCase();

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">NBA Analytics</h1>
          <p className="text-sm text-slate-500 mt-0.5">Season leaders and today's projections</p>
        </div>
        <div className="flex flex-col items-end gap-1">
          <label className="text-xs text-slate-600 uppercase tracking-wider">Market</label>
          <Select value={market} onValueChange={(v) => setMarket(v as Market)}>
            <SelectTrigger className="w-36 bg-slate-900 border-slate-700 text-slate-200 text-sm">
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="bg-slate-900 border-slate-700">
              {MARKETS.map((m) => (
                <SelectItem
                  key={m.value}
                  value={m.value}
                  className="text-slate-200 focus:bg-slate-800 focus:text-slate-100"
                >
                  {m.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-red-800/50 bg-red-500/10 px-4 py-3 text-sm text-red-400">
          {error}
        </div>
      )}

      {loading ? (
        <div className="text-slate-500 text-sm">Loading analytics…</div>
      ) : (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          {/* Today's Projections Chart */}
          <Card className="bg-[#0d1220] border-slate-800/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-slate-400 uppercase tracking-wider">
                Today's {selectedMarketLabel} Projections (Top 15)
              </CardTitle>
            </CardHeader>
            <CardContent>
              {projChartData.length === 0 ? (
                <div className="h-72 flex items-center justify-center text-slate-600 text-sm">
                  No projections yet — run{" "}
                  <code className="mx-1 font-mono bg-slate-800 px-1 rounded">make nba-predict</code>
                </div>
              ) : (
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={projChartData} layout="vertical" margin={{ left: 8, right: 24 }}>
                    <CartesianGrid
                      strokeDasharray="3 3"
                      stroke="#1e293b"
                      horizontal={false}
                    />
                    <XAxis
                      type="number"
                      tick={{ fill: "#475569", fontSize: 11 }}
                      axisLine={{ stroke: "#1e293b" }}
                      tickLine={false}
                    />
                    <YAxis
                      type="category"
                      dataKey="name"
                      tick={{ fill: "#64748b", fontSize: 11 }}
                      axisLine={false}
                      tickLine={false}
                      width={60}
                    />
                    <Tooltip content={<CustomTooltip market={market} />} />
                    <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                      {projChartData.map((_, i) => (
                        <Cell
                          key={i}
                          fill={i === 0 ? BLUE : BLUE_DIM}
                          opacity={1 - i * 0.04}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              )}
            </CardContent>
          </Card>

          {/* Season Leaders Chart */}
          <Card className="bg-[#0d1220] border-slate-800/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-slate-400 uppercase tracking-wider">
                {season ? `${season}–${String(season + 1).slice(-2)}` : ""} Season Scoring Leaders
              </CardTitle>
            </CardHeader>
            <CardContent>
              {scorersChartData.length === 0 ? (
                <div className="h-72 flex items-center justify-center text-slate-600 text-sm">
                  No data — run{" "}
                  <code className="mx-1 font-mono bg-slate-800 px-1 rounded">make ingest-nba</code>
                </div>
              ) : (
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart data={scorersChartData} layout="vertical" margin={{ left: 8, right: 24 }}>
                    <CartesianGrid
                      strokeDasharray="3 3"
                      stroke="#1e293b"
                      horizontal={false}
                    />
                    <XAxis
                      type="number"
                      tick={{ fill: "#475569", fontSize: 11 }}
                      axisLine={{ stroke: "#1e293b" }}
                      tickLine={false}
                    />
                    <YAxis
                      type="category"
                      dataKey="name"
                      tick={{ fill: "#64748b", fontSize: 11 }}
                      axisLine={false}
                      tickLine={false}
                      width={60}
                    />
                    <Tooltip content={<CustomTooltip market={market} />} />
                    <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                      {scorersChartData.map((_, i) => (
                        <Cell
                          key={i}
                          fill={i === 0 ? BLUE : BLUE_DIM}
                          opacity={1 - i * 0.04}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              )}
            </CardContent>
          </Card>

          {/* Season Averages Table */}
          <Card className="bg-[#0d1220] border-slate-800/50 xl:col-span-2">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-slate-400 uppercase tracking-wider">
                Top Scorers — Full Stats
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-800/50">
                    {["#", "Player", "Team", "GP", "PPG", "RPG", "APG", "MPG"].map((h) => (
                      <th
                        key={h}
                        className="px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider text-right first:text-left second:text-left"
                      >
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {topScorers.map((p, i) => (
                    <tr
                      key={p.player_id}
                      className="border-b border-slate-800/30 hover:bg-slate-800/20"
                    >
                      <td className="px-4 py-2.5 text-slate-600 font-mono text-xs">{i + 1}</td>
                      <td className="px-4 py-2.5 font-medium text-slate-200">{p.player_name}</td>
                      <td className="px-4 py-2.5">
                        <span className="text-xs font-mono text-blue-400 bg-blue-500/10 px-2 py-0.5 rounded">
                          {p.team}
                        </span>
                      </td>
                      <td className="px-4 py-2.5 text-right text-slate-500 font-mono">{p.games_played}</td>
                      <td className="px-4 py-2.5 text-right text-blue-400 font-bold font-mono">
                        {p.avg_pts.toFixed(1)}
                      </td>
                      <td className="px-4 py-2.5 text-right text-slate-400 font-mono">{p.avg_reb.toFixed(1)}</td>
                      <td className="px-4 py-2.5 text-right text-slate-400 font-mono">{p.avg_ast.toFixed(1)}</td>
                      <td className="px-4 py-2.5 text-right text-slate-500 font-mono">{p.avg_min.toFixed(1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </CardContent>
          </Card>

          {/* Risk & Correlation */}
          <Card className="bg-[#0d1220] border-slate-800/50 xl:col-span-2">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-slate-400 uppercase tracking-wider">
                Risk & Correlation
              </CardTitle>
            </CardHeader>
            <CardContent>
              {riskLoading ? (
                <div className="text-sm text-slate-600">Loading risk data…</div>
              ) : !riskData || riskData.total_assessed === 0 ? (
                <div className="text-sm text-slate-600">
                  No risk data available. Run <code className="font-mono text-xs bg-slate-800 px-1 rounded">make nba-risk</code> to generate.
                </div>
              ) : (
                <div className="space-y-4">
                  {/* Summary Stats */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    <div className="bg-slate-800/30 rounded-lg px-4 py-3">
                      <p className="text-[10px] text-slate-600 uppercase tracking-wider">Assessed</p>
                      <p className="text-lg font-bold text-slate-200 font-[family-name:var(--font-jetbrains)]">
                        {riskData.total_assessed}
                      </p>
                    </div>
                    <div className="bg-slate-800/30 rounded-lg px-4 py-3">
                      <p className="text-[10px] text-slate-600 uppercase tracking-wider">Correlated</p>
                      <p className={`text-lg font-bold font-[family-name:var(--font-jetbrains)] ${riskData.correlated > 0 ? "text-amber-400" : "text-slate-200"}`}>
                        {riskData.correlated}
                      </p>
                    </div>
                    <div className="bg-slate-800/30 rounded-lg px-4 py-3">
                      <p className="text-[10px] text-slate-600 uppercase tracking-wider">Exposure Flags</p>
                      <p className={`text-lg font-bold font-[family-name:var(--font-jetbrains)] ${riskData.exposure_flagged > 0 ? "text-red-400" : "text-slate-200"}`}>
                        {riskData.exposure_flagged}
                      </p>
                    </div>
                    <div className="bg-slate-800/30 rounded-lg px-4 py-3">
                      <p className="text-[10px] text-slate-600 uppercase tracking-wider">Avg Risk Kelly</p>
                      <p className="text-lg font-bold text-slate-200 font-[family-name:var(--font-jetbrains)]">
                        {riskData.avg_risk_adjusted_kelly != null
                          ? `${(riskData.avg_risk_adjusted_kelly * 100).toFixed(2)}%`
                          : "—"}
                      </p>
                    </div>
                  </div>

                  {/* Guardrails */}
                  {riskData.guardrails.length > 0 && (
                    <div className="space-y-1.5">
                      <p className="text-[10px] text-slate-600 uppercase tracking-wider font-semibold">Guardrails</p>
                      {riskData.guardrails.map((g, i) => (
                        <div
                          key={i}
                          className="flex items-center gap-2 text-xs text-amber-400 bg-amber-500/5 border border-amber-500/20 rounded-lg px-3 py-2"
                        >
                          <span className="shrink-0 w-1.5 h-1.5 rounded-full bg-amber-400" />
                          {g}
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Team Stacks */}
                  {correlationData && Object.keys(correlationData.team_stacks).length > 0 && (
                    <div className="space-y-1.5">
                      <p className="text-[10px] text-slate-600 uppercase tracking-wider font-semibold">Team Stacks</p>
                      <div className="flex flex-wrap gap-2">
                        {Object.entries(correlationData.team_stacks).map(([team, count]) => (
                          <span
                            key={team}
                            className="text-xs font-mono px-2.5 py-1 rounded bg-slate-800/50 text-slate-300 border border-slate-700/50"
                          >
                            {team}: {count} bets
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Correlation Groups */}
                  {correlationData && Object.keys(correlationData.correlation_groups).length > 0 && (
                    <div className="space-y-1.5">
                      <p className="text-[10px] text-slate-600 uppercase tracking-wider font-semibold">Correlation Groups</p>
                      <div className="space-y-2">
                        {Object.entries(correlationData.correlation_groups).map(([group, members]) => (
                          <div key={group} className="bg-slate-800/30 rounded-lg px-3 py-2">
                            <p className="text-xs font-mono text-amber-400 mb-1">{group}</p>
                            <div className="flex flex-wrap gap-1.5">
                              {members.map((m, j) => (
                                <span
                                  key={j}
                                  className="text-[10px] font-mono text-slate-400 bg-slate-700/40 px-1.5 py-0.5 rounded"
                                >
                                  #{m.player_id} {m.market} ({m.sportsbook})
                                </span>
                              ))}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}

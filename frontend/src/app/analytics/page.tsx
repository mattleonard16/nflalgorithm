"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  getMeta,
  getProjectionWeeks,
  getEdgeDistribution,
  getAnalyticsByPosition,
  getAnalyticsByMarket,
} from "@/lib/api";
import type {
  AvailableWeek,
  MetaResponse,
  PositionStats,
  MarketStats,
  EdgeDistribution,
} from "@/lib/types";
import { filterForWeek, useWatchlist } from "@/lib/slate-watchlist";
import { WatchedPicksTable } from "@/components/watched-picks";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";

// Position colors match the dashboard's position badges
const POSITION_COLORS: Record<string, string> = {
  QB: "#d4a84b",
  RB: "#c4b08a",
  WR: "#7a8796",
  TE: "#8a9a7b",
};
const FALLBACK_COLORS = ["#d4a84b", "#c4b08a", "#8a9a7b", "#7a8796", "#b85c4a"];

export default function AnalyticsPage() {
  const [meta, setMeta] = useState<MetaResponse | null>(null);
  const [slateWeeks, setSlateWeeks] = useState<AvailableWeek[]>([]);
  const [season, setSeason] = useState(2025);
  const [week, setWeek] = useState(13);
  const [edgeDist, setEdgeDist] = useState<EdgeDistribution | null>(null);
  const [positions, setPositions] = useState<PositionStats[]>([]);
  const [markets, setMarkets] = useState<MarketStats[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const { watchlist } = useWatchlist();
  const watchedThisWeek = filterForWeek(watchlist, season, week);

  // Resolve the week to show. Published cards win; otherwise fall back to the
  // algorithm slate so the watched picks land on a week that actually exists.
  useEffect(() => {
    let cancelled = false;

    async function resolveWeek() {
      const metaData = await getMeta().catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "Failed to load metadata");
        return null;
      });
      if (cancelled) return;
      if (metaData) setMeta(metaData);
      if (metaData && metaData.available_weeks.length > 0) {
        setSeason(metaData.available_weeks[0].season);
        setWeek(metaData.available_weeks[0].week);
        return;
      }

      const slate = await getProjectionWeeks().catch(() => null);
      if (cancelled || !slate || slate.available_weeks.length === 0) return;
      setSlateWeeks(slate.available_weeks);
      setSeason(slate.available_weeks[0].season);
      setWeek(slate.available_weeks[0].week);
    }

    resolveWeek();
    return () => {
      cancelled = true;
    };
  }, []);

  // Week options: published weeks when they exist, otherwise the slate's weeks.
  const weekOptions: AvailableWeek[] =
    meta && meta.available_weeks.length > 0 ? meta.available_weeks : slateWeeks;
  const seasonOptions = [...new Set(weekOptions.map((w) => w.season))];
  const weekNumbers = weekOptions.filter((w) => w.season === season).map((w) => w.week);

  // Fetch analytics data when season/week changes
  useEffect(() => {
    async function load() {
      setLoading(true);
      setError(null);

      try {
        const [edge, pos, mkt] = await Promise.all([
          getEdgeDistribution(season, week),
          getAnalyticsByPosition(season, week),
          getAnalyticsByMarket(season, week),
        ]);
        setEdgeDist(edge);
        setPositions(pos.positions);
        setMarkets(mkt.markets);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load analytics");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [season, week]);

  // Prepare chart data
  const edgeChartData = edgeDist
    ? edgeDist.bins.map((bin, i) => ({
        edge: bin,
        count: edgeDist.counts[i] ?? 0,
      }))
    : [];

  const positionChartData = (positions ?? []).map((p) => ({
    name: p.position,
    value: p.count,
    avgEdge: p.avg_edge,
  }));

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-4xl font-bold text-slate-100 font-display uppercase tracking-tight">
          Analytics
        </h1>
        <p className="text-sm text-slate-500 mt-0.5">
          Watched picks, edge buckets, and market split for the selected week
        </p>
      </div>

      {/* Filters */}
      <Card className="bg-[#111827] border-slate-800/60">
        <CardContent className="pt-6">
          <div className="flex gap-6">
            <div className="space-y-2">
              <Label className="text-slate-400">Season</Label>
              <Select
                value={season.toString()}
                onValueChange={(v) => setSeason(parseInt(v))}
              >
                <SelectTrigger className="w-32 bg-[#0d1220] border-slate-700 text-slate-100">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-[#0d1220] border-slate-700">
                  {(seasonOptions.length > 0 ? seasonOptions : [season]).map((s) => (
                    <SelectItem key={s} value={s.toString()} className="text-slate-100">
                      {s}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label className="text-slate-400">Week</Label>
              <Select
                value={week.toString()}
                onValueChange={(v) => setWeek(parseInt(v))}
              >
                <SelectTrigger className="w-24 bg-[#0d1220] border-slate-700 text-slate-100">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-[#0d1220] border-slate-700">
                  {(weekNumbers.length > 0 ? weekNumbers : [week]).map((w) => (
                    <SelectItem key={w} value={w.toString()} className="text-slate-100">
                      {w}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardContent>
      </Card>

      {error && (
        <div className="bg-red-900/20 border border-red-800 rounded-lg p-4 text-red-300">
          {error}
        </div>
      )}

      {/* Watched slate: the model picks the user is tracking, card or no card */}
      <Card className="bg-[#111827] border-slate-800/60">
        <CardHeader>
          <div className="flex items-baseline justify-between gap-3 flex-wrap">
            <CardTitle className="text-slate-100">Watched Slate</CardTitle>
            <p className="text-xs text-slate-500">
              Algorithm picks you are tracking for Season {season} &middot; Week {week}
              {watchedThisWeek.length > 0 && (
                <>
                  {" "}
                  &middot;{" "}
                  <span className="font-[family-name:var(--font-jetbrains)] tabular-nums text-amber-400/80">
                    {watchedThisWeek.length}
                  </span>
                </>
              )}
            </p>
          </div>
        </CardHeader>
        <CardContent>
          <WatchedPicksTable picks={watchedThisWeek} />
        </CardContent>
      </Card>

      {loading ? (
        <div className="flex items-center justify-center h-64">
          <p className="text-slate-400">Loading analytics...</p>
        </div>
      ) : (
        <>
          {/* Edge Distribution */}
          <Card className="bg-[#111827] border-slate-800/60">
            <CardHeader>
              <CardTitle className="text-slate-100">Edge Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={edgeChartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="edge" stroke="#64748b" fontSize={11} interval={0} />
                    <YAxis stroke="#64748b" fontSize={12} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#0d1220",
                        border: "1px solid #1e293b",
                        borderRadius: "8px",
                      }}
                      labelStyle={{ color: "#e2e8f0" }}
                    />
                    <Bar dataKey="count" fill="#d4a84b" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* Position and Market Breakdown */}
          <div className="grid md:grid-cols-2 gap-6">
            {/* By Position */}
            <Card className="bg-[#111827] border-slate-800/60">
              <CardHeader>
                <CardTitle className="text-slate-100">Opportunities by Position</CardTitle>
              </CardHeader>
              <CardContent>
                {positions.length > 0 ? (
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={positionChartData}
                          cx="50%"
                          cy="50%"
                          innerRadius={60}
                          outerRadius={80}
                          paddingAngle={5}
                          dataKey="value"
                          label={({ name, value }) => `${name}: ${value}`}
                        >
                          {positionChartData.map((entry, index) => (
                            <Cell
                              key={`cell-${index}`}
                              fill={
                                POSITION_COLORS[entry.name] ??
                                FALLBACK_COLORS[index % FALLBACK_COLORS.length]
                              }
                            />
                          ))}
                        </Pie>
                        <Tooltip
                          contentStyle={{
                            backgroundColor: "#0d1220",
                            border: "1px solid #1e293b",
                            borderRadius: "8px",
                          }}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <p className="text-slate-400 text-center py-8">No position data available</p>
                )}
              </CardContent>
            </Card>

            {/* By Market */}
            <Card className="bg-[#111827] border-slate-800/60">
              <CardHeader>
                <CardTitle className="text-slate-100">Edge by Market Type</CardTitle>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow className="border-slate-800/60">
                      <TableHead className="text-slate-400">Market</TableHead>
                      <TableHead className="text-slate-400 text-right">Count</TableHead>
                      <TableHead className="text-slate-400 text-right">Avg Edge</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {markets.length > 0 ? (
                      markets.map((market) => (
                        <TableRow key={market.market} className="border-slate-800/60">
                          <TableCell className="text-slate-100 capitalize">
                            {market.market.replace("_", " ")}
                          </TableCell>
                          <TableCell className="text-right text-slate-300">
                            {market.count}
                          </TableCell>
                          <TableCell className="text-right text-primary">
                            {market.avg_edge.toFixed(1)}%
                          </TableCell>
                        </TableRow>
                      ))
                    ) : (
                      <TableRow>
                        <TableCell colSpan={3} className="text-center text-slate-400">
                          No market data available
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </div>
        </>
      )}
    </div>
  );
}


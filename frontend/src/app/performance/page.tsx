"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { getPerformance, getOutcomes } from "@/lib/api";
import type { PerformanceResponse, BetOutcome } from "@/lib/types";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { CheckCircle2, XCircle, MinusCircle } from "lucide-react";

function KPICard({
  title,
  value,
  subtitle,
  trend,
}: {
  title: string;
  value: string | number;
  subtitle?: string;
  trend?: "up" | "down" | "neutral";
}) {
  const trendColors = {
    up: "text-emerald-400",
    down: "text-red-400",
    neutral: "text-slate-400",
  };

  return (
    <Card className="bg-[#111827] border-slate-800/60">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium text-slate-400">
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className={`text-2xl font-bold ${trend ? trendColors[trend] : "text-slate-100"}`}>
          {value}
        </div>
        {subtitle && (
          <p className="text-xs text-slate-500 mt-1">{subtitle}</p>
        )}
      </CardContent>
    </Card>
  );
}

function ResultIcon({ result }: { result: string | null }) {
  if (result === "WIN") {
    return <CheckCircle2 className="h-5 w-5 text-green-400" />;
  } else if (result === "LOSS") {
    return <XCircle className="h-5 w-5 text-red-400" />;
  } else {
    return <MinusCircle className="h-5 w-5 text-slate-500" />;
  }
}

export default function PerformancePage() {
  const [performance, setPerformance] = useState<PerformanceResponse | null>(null);
  const [outcomes, setOutcomes] = useState<BetOutcome[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedWeek, setSelectedWeek] = useState<{ season: number; week: number } | null>(null);

  useEffect(() => {
    getPerformance()
      .then((data) => {
        setPerformance(data);
        // Auto-select the most recent week
        if (data.weeks.length > 0) {
          const latest = data.weeks[0];
          setSelectedWeek({ season: latest.season, week: latest.week });
        }
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  // Fetch outcomes when selected week changes
  useEffect(() => {
    if (selectedWeek) {
      getOutcomes(selectedWeek.season, selectedWeek.week)
        .then(setOutcomes)
        .catch(() => setOutcomes([]));
    }
  }, [selectedWeek]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-slate-400">Loading performance data...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-900/20 border border-red-800 rounded-lg p-4 text-red-300">
        {error}
      </div>
    );
  }

  if (!performance || performance.weeks.length === 0) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-4xl font-bold text-slate-100 tracking-tight font-display uppercase">
            Performance
          </h1>
          <p className="text-sm text-slate-500 mt-0.5">Graded results after kickoff</p>
        </div>
        <Card className="bg-[#111827] border-slate-800/60">
          <CardContent className="py-12 text-center text-slate-400">
            No performance data yet. Run <code className="bg-slate-800 px-2 py-1 rounded">make week-grade</code> after games complete to record outcomes.
          </CardContent>
        </Card>
      </div>
    );
  }

  // Prepare chart data - sort by week ascending
  const chartData = [...performance.weeks]
    .sort((a, b) => a.season - b.season || a.week - b.week)
    .reduce((acc: { week: string; profit: number; cumulative: number }[], week) => {
      const prev = acc.length > 0 ? acc[acc.length - 1].cumulative : 0;
      acc.push({
        week: `W${week.week}`,
        profit: week.profit_units,
        cumulative: prev + week.profit_units,
      });
      return acc;
    }, []);

  const profitTrend = performance.total_profit >= 0 ? "up" : "down";

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-4xl font-bold text-slate-100 tracking-tight font-display uppercase">
          Performance
        </h1>
        <p className="text-sm text-slate-500 mt-0.5">Graded results after kickoff</p>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KPICard title="Total Bets" value={performance.total_bets} />
        <KPICard
          title="Win Rate"
          value={`${performance.win_rate.toFixed(1)}%`}
          trend={performance.win_rate >= 50 ? "up" : "down"}
        />
        <KPICard
          title="Total Profit"
          value={`${performance.total_profit >= 0 ? "+" : ""}${performance.total_profit.toFixed(1)}u`}
          trend={profitTrend}
        />
        <KPICard
          title="Overall ROI"
          value={`${performance.overall_roi >= 0 ? "+" : ""}${performance.overall_roi.toFixed(1)}%`}
          trend={performance.overall_roi >= 0 ? "up" : "down"}
        />
      </div>

      {/* Profit Chart */}
      <Card className="bg-[#111827] border-slate-800/60">
        <CardHeader>
          <CardTitle className="text-slate-100">Cumulative Profit</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%" minWidth={0}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="week" stroke="#64748b" fontSize={12} />
                <YAxis stroke="#64748b" fontSize={12} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#0d1220",
                    border: "1px solid #1e293b",
                    borderRadius: "8px",
                  }}
                  labelStyle={{ color: "#e2e8f0" }}
                />
                <Line
                  type="monotone"
                  dataKey="cumulative"
                  stroke="#d4a84b"
                  strokeWidth={2}
                  dot={{ fill: "#d4a84b", strokeWidth: 0 }}
                  name="Cumulative Profit"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* Bet-by-Bet Details */}
      <Card className="bg-[#111827] border-slate-800/60">
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-slate-100">Bet Details</CardTitle>
          <select
            value={selectedWeek ? `${selectedWeek.season}-${selectedWeek.week}` : ""}
            onChange={(e) => {
              const [season, week] = e.target.value.split("-").map(Number);
              setSelectedWeek({ season, week });
            }}
            className="bg-[#0d1220] border border-slate-700 rounded-md px-3 py-1 text-sm text-slate-100"
          >
            {performance.weeks.map((w) => (
              <option key={`${w.season}-${w.week}`} value={`${w.season}-${w.week}`}>
                Week {w.week}, {w.season}
              </option>
            ))}
          </select>
        </CardHeader>
        <CardContent>
          {outcomes.length === 0 ? (
            <p className="text-slate-500 text-center py-4">No bet outcomes for this week</p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow className="border-slate-800/60 hover:bg-slate-800/20">
                  <TableHead className="text-slate-400 w-12"></TableHead>
                  <TableHead className="text-slate-400">Player</TableHead>
                  <TableHead className="text-slate-400">Market</TableHead>
                  <TableHead className="text-slate-400 text-right">Line</TableHead>
                  <TableHead className="text-slate-400 text-right">Actual</TableHead>
                  <TableHead className="text-slate-400 text-right">Profit</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {outcomes.map((bet) => (
                  <TableRow key={bet.bet_id} className="border-slate-800/60 hover:bg-slate-800/20">
                    <TableCell>
                      <ResultIcon result={bet.result} />
                    </TableCell>
                    <TableCell className="text-slate-100 font-medium">
                      {bet.player_name || "Unknown"}
                    </TableCell>
                    <TableCell className="text-slate-400">
                      {bet.market.replace("_", " ")}
                    </TableCell>
                    <TableCell className="text-right text-slate-300">
                      O {bet.line}
                    </TableCell>
                    <TableCell className="text-right text-slate-100 font-medium">
                      {bet.actual_result?.toFixed(0) ?? "-"}
                    </TableCell>
                    <TableCell className={`text-right font-medium ${(bet.profit_units ?? 0) >= 0 ? "text-green-400" : "text-red-400"
                      }`}>
                      {bet.profit_units !== null
                        ? `${bet.profit_units >= 0 ? "+" : ""}${bet.profit_units.toFixed(1)}u`
                        : "-"
                      }
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      {/* Weekly Breakdown Table */}
      <Card className="bg-[#111827] border-slate-800/60">
        <CardHeader>
          <CardTitle className="text-slate-100">Week-by-Week Summary</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow className="border-slate-800/60 hover:bg-slate-800/20">
                <TableHead className="text-slate-400">Season</TableHead>
                <TableHead className="text-slate-400">Week</TableHead>
                <TableHead className="text-slate-400 text-right">Bets</TableHead>
                <TableHead className="text-slate-400 text-right">Wins</TableHead>
                <TableHead className="text-slate-400 text-right">Losses</TableHead>
                <TableHead className="text-slate-400 text-right">Profit</TableHead>
                <TableHead className="text-slate-400 text-right">ROI</TableHead>
                <TableHead className="text-slate-400">Best Bet</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {performance.weeks.map((week) => (
                <TableRow key={`${week.season}-${week.week}`} className="border-slate-800/60 hover:bg-slate-800/20">
                  <TableCell className="text-slate-300">{week.season}</TableCell>
                  <TableCell className="text-slate-300">{week.week}</TableCell>
                  <TableCell className="text-right text-slate-300">{week.total_bets}</TableCell>
                  <TableCell className="text-right text-green-400">{week.wins}</TableCell>
                  <TableCell className="text-right text-red-400">{week.losses}</TableCell>
                  <TableCell className={`text-right font-medium ${week.profit_units >= 0 ? "text-green-400" : "text-red-400"}`}>
                    {week.profit_units >= 0 ? "+" : ""}{week.profit_units.toFixed(1)}u
                  </TableCell>
                  <TableCell className={`text-right ${week.roi_pct >= 0 ? "text-green-400" : "text-red-400"}`}>
                    {week.roi_pct >= 0 ? "+" : ""}{week.roi_pct.toFixed(1)}%
                  </TableCell>
                  <TableCell className="text-slate-400">{week.best_bet || "-"}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}

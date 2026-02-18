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
import { getNbaPerformance, getNbaOutcomes } from "@/lib/nba-api";
import type { NbaPerformanceResponse, NbaBetOutcome } from "@/lib/nba-types";
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
    up: "text-green-400",
    down: "text-red-400",
    neutral: "text-slate-400",
  };

  return (
    <Card className="bg-[#0d1220] border-slate-800/50">
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
  if (result === "win") {
    return <CheckCircle2 className="h-5 w-5 text-green-400" />;
  } else if (result === "loss") {
    return <XCircle className="h-5 w-5 text-red-400" />;
  } else {
    return <MinusCircle className="h-5 w-5 text-slate-500" />;
  }
}

export default function NbaPerformancePage() {
  const [performance, setPerformance] = useState<NbaPerformanceResponse | null>(null);
  const [outcomes, setOutcomes] = useState<NbaBetOutcome[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedDate, setSelectedDate] = useState<string | null>(null);

  useEffect(() => {
    getNbaPerformance()
      .then((data) => {
        setPerformance(data);
        if (data.days.length > 0) {
          setSelectedDate(data.days[0].game_date);
        }
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (selectedDate) {
      getNbaOutcomes(selectedDate)
        .then(setOutcomes)
        .catch(() => setOutcomes([]));
    }
  }, [selectedDate]);

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

  if (!performance || performance.days.length === 0) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-slate-100">NBA Performance</h1>
          <p className="text-slate-400 mt-1">Historical NBA betting performance and outcomes</p>
        </div>
        <Card className="bg-[#0d1220] border-slate-800/50">
          <CardContent className="py-12 text-center text-slate-400">
            No performance data yet. Run <code className="bg-slate-800 px-2 py-1 rounded">make nba-grade GAME_DATE=YYYY-MM-DD</code> after games complete to record outcomes.
          </CardContent>
        </Card>
      </div>
    );
  }

  const chartData = [...performance.days]
    .sort((a, b) => a.game_date.localeCompare(b.game_date))
    .reduce((acc: { date: string; profit: number; cumulative: number }[], day) => {
      const prev = acc.length > 0 ? acc[acc.length - 1].cumulative : 0;
      const label = day.game_date.slice(5);
      acc.push({
        date: label,
        profit: day.profit_units,
        cumulative: prev + day.profit_units,
      });
      return acc;
    }, []);

  const profitTrend = performance.total_profit >= 0 ? "up" : "down";

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-slate-100">NBA Performance</h1>
        <p className="text-slate-400 mt-1">Historical NBA betting performance and outcomes</p>
      </div>

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

      <Card className="bg-[#0d1220] border-slate-800/50">
        <CardHeader>
          <CardTitle className="text-slate-100">Cumulative Profit</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%" minWidth={0}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="date" stroke="#64748b" fontSize={12} />
                <YAxis stroke="#64748b" fontSize={12} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#0f172a",
                    border: "1px solid #1e293b",
                    borderRadius: "8px",
                  }}
                  labelStyle={{ color: "#f1f5f9" }}
                />
                <Line
                  type="monotone"
                  dataKey="cumulative"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  dot={{ fill: "#3b82f6", strokeWidth: 0 }}
                  name="Cumulative Profit"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      <Card className="bg-[#0d1220] border-slate-800/50">
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-slate-100">Bet Details</CardTitle>
          <select
            value={selectedDate ?? ""}
            onChange={(e) => setSelectedDate(e.target.value)}
            className="bg-slate-800 border border-slate-700 rounded-md px-3 py-1 text-sm text-slate-100"
          >
            {performance.days.map((d) => (
              <option key={d.game_date} value={d.game_date}>
                {d.game_date}
              </option>
            ))}
          </select>
        </CardHeader>
        <CardContent>
          {outcomes.length === 0 ? (
            <p className="text-slate-500 text-center py-4">No bet outcomes for this date</p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow className="border-slate-800 hover:bg-slate-900/50">
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
                  <TableRow key={bet.bet_id} className="border-slate-800 hover:bg-slate-900/50">
                    <TableCell>
                      <ResultIcon result={bet.result} />
                    </TableCell>
                    <TableCell className="text-slate-100 font-medium">
                      {bet.player_name ?? "Unknown"}
                    </TableCell>
                    <TableCell className="text-slate-400 uppercase">
                      {bet.market}
                    </TableCell>
                    <TableCell className="text-right text-slate-300">
                      O {bet.line}
                    </TableCell>
                    <TableCell className="text-right text-slate-100 font-medium">
                      {bet.actual_result?.toFixed(0) ?? "-"}
                    </TableCell>
                    <TableCell className={`text-right font-medium ${(bet.profit_units ?? 0) >= 0 ? "text-green-400" : "text-red-400"}`}>
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

      <Card className="bg-[#0d1220] border-slate-800/50">
        <CardHeader>
          <CardTitle className="text-slate-100">Day-by-Day Summary</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow className="border-slate-800 hover:bg-slate-900/50">
                <TableHead className="text-slate-400">Date</TableHead>
                <TableHead className="text-slate-400 text-right">Bets</TableHead>
                <TableHead className="text-slate-400 text-right">Wins</TableHead>
                <TableHead className="text-slate-400 text-right">Losses</TableHead>
                <TableHead className="text-slate-400 text-right">Profit</TableHead>
                <TableHead className="text-slate-400 text-right">ROI</TableHead>
                <TableHead className="text-slate-400">Best Bet</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {performance.days.map((day) => (
                <TableRow key={day.game_date} className="border-slate-800 hover:bg-slate-900/50">
                  <TableCell className="text-slate-300">{day.game_date}</TableCell>
                  <TableCell className="text-right text-slate-300">{day.total_bets}</TableCell>
                  <TableCell className="text-right text-green-400">{day.wins}</TableCell>
                  <TableCell className="text-right text-red-400">{day.losses}</TableCell>
                  <TableCell className={`text-right font-medium ${day.profit_units >= 0 ? "text-green-400" : "text-red-400"}`}>
                    {day.profit_units >= 0 ? "+" : ""}{day.profit_units.toFixed(1)}u
                  </TableCell>
                  <TableCell className={`text-right ${day.roi_pct >= 0 ? "text-green-400" : "text-red-400"}`}>
                    {day.roi_pct >= 0 ? "+" : ""}{day.roi_pct.toFixed(1)}%
                  </TableCell>
                  <TableCell className="text-slate-400">{day.best_bet ?? "-"}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}

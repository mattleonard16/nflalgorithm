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
  getEdgeDistribution,
  getAnalyticsByPosition,
  getAnalyticsByMarket,
} from "@/lib/api";
import type { MetaResponse, PositionStats, MarketStats, EdgeDistribution } from "@/lib/types";
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

const COLORS = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"];

export default function AnalyticsPage() {
  const [meta, setMeta] = useState<MetaResponse | null>(null);
  const [season, setSeason] = useState(2025);
  const [week, setWeek] = useState(13);
  const [edgeDist, setEdgeDist] = useState<EdgeDistribution | null>(null);
  const [positions, setPositions] = useState<PositionStats[]>([]);
  const [markets, setMarkets] = useState<MarketStats[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch metadata
  useEffect(() => {
    getMeta()
      .then((data) => {
        setMeta(data);
        if (data.available_weeks.length > 0) {
          setSeason(data.available_weeks[0].season);
          setWeek(data.available_weeks[0].week);
        }
      })
      .catch((err) => setError(err.message));
  }, []);

  // Fetch analytics data when season/week changes
  useEffect(() => {
    setLoading(true);
    setError(null);

    Promise.all([
      getEdgeDistribution(season, week),
      getAnalyticsByPosition(season, week),
      getAnalyticsByMarket(season, week),
    ])
      .then(([edge, pos, mkt]) => {
        setEdgeDist(edge);
        setPositions(pos.positions);
        setMarkets(mkt.markets);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [season, week]);

  // Prepare chart data
  const edgeChartData = edgeDist
    ? edgeDist.bins.map((bin, i) => ({
        edge: bin.toFixed(1),
        count: edgeDist.counts[i],
      }))
    : [];

  const positionChartData = positions.map((p) => ({
    name: p.position,
    value: p.count,
    avgEdge: p.avg_edge,
  }));

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-zinc-100">Analytics</h1>
        <p className="text-zinc-400 mt-1">Deep dive into betting patterns and edge distribution</p>
      </div>

      {/* Filters */}
      <Card className="bg-zinc-900 border-zinc-800">
        <CardContent className="pt-6">
          <div className="flex gap-6">
            <div className="space-y-2">
              <Label className="text-zinc-400">Season</Label>
              <Select
                value={season.toString()}
                onValueChange={(v) => setSeason(parseInt(v))}
              >
                <SelectTrigger className="w-32 bg-zinc-800 border-zinc-700 text-zinc-100">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-zinc-800 border-zinc-700">
                  {[...new Set(meta?.available_weeks.map((w) => w.season) || [2025])].map((s) => (
                    <SelectItem key={s} value={s.toString()} className="text-zinc-100">
                      {s}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label className="text-zinc-400">Week</Label>
              <Select
                value={week.toString()}
                onValueChange={(v) => setWeek(parseInt(v))}
              >
                <SelectTrigger className="w-24 bg-zinc-800 border-zinc-700 text-zinc-100">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-zinc-800 border-zinc-700">
                  {(meta?.available_weeks.filter((w) => w.season === season).map((w) => w.week) || [1]).map((w) => (
                    <SelectItem key={w} value={w.toString()} className="text-zinc-100">
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

      {loading ? (
        <div className="flex items-center justify-center h-64">
          <p className="text-zinc-400">Loading analytics...</p>
        </div>
      ) : (
        <>
          {/* Edge Distribution */}
          <Card className="bg-zinc-900 border-zinc-800">
            <CardHeader>
              <CardTitle className="text-zinc-100">Edge Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={edgeChartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#3f3f46" />
                    <XAxis dataKey="edge" stroke="#a1a1aa" fontSize={12} />
                    <YAxis stroke="#a1a1aa" fontSize={12} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#18181b",
                        border: "1px solid #3f3f46",
                        borderRadius: "8px",
                      }}
                      labelStyle={{ color: "#fafafa" }}
                    />
                    <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* Position and Market Breakdown */}
          <div className="grid md:grid-cols-2 gap-6">
            {/* By Position */}
            <Card className="bg-zinc-900 border-zinc-800">
              <CardHeader>
                <CardTitle className="text-zinc-100">Opportunities by Position</CardTitle>
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
                          {positionChartData.map((_, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip
                          contentStyle={{
                            backgroundColor: "#18181b",
                            border: "1px solid #3f3f46",
                            borderRadius: "8px",
                          }}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <p className="text-zinc-400 text-center py-8">No position data available</p>
                )}
              </CardContent>
            </Card>

            {/* By Market */}
            <Card className="bg-zinc-900 border-zinc-800">
              <CardHeader>
                <CardTitle className="text-zinc-100">Edge by Market Type</CardTitle>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow className="border-zinc-800">
                      <TableHead className="text-zinc-400">Market</TableHead>
                      <TableHead className="text-zinc-400 text-right">Count</TableHead>
                      <TableHead className="text-zinc-400 text-right">Avg Edge</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {markets.length > 0 ? (
                      markets.map((market) => (
                        <TableRow key={market.market} className="border-zinc-800">
                          <TableCell className="text-zinc-100 capitalize">
                            {market.market.replace("_", " ")}
                          </TableCell>
                          <TableCell className="text-right text-zinc-300">
                            {market.count}
                          </TableCell>
                          <TableCell className="text-right text-blue-400">
                            {market.avg_edge.toFixed(1)}%
                          </TableCell>
                        </TableRow>
                      ))
                    ) : (
                      <TableRow>
                        <TableCell colSpan={3} className="text-center text-zinc-400">
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


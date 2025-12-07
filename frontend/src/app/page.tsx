"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { getValueBets, getMeta, getPerformance } from "@/lib/api";
import type { ValueBet, MetaResponse, PerformanceResponse, DashboardFilters } from "@/lib/types";

function KPICard({
  title,
  value,
  subtitle,
}: {
  title: string;
  value: string | number;
  subtitle?: string;
}) {
  return (
    <Card className="bg-zinc-900 border-zinc-800">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium text-zinc-400">
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold text-zinc-100">{value}</div>
        {subtitle && (
          <p className="text-xs text-zinc-500 mt-1">{subtitle}</p>
        )}
      </CardContent>
    </Card>
  );
}

function ConfidenceBadge({ tier }: { tier: string }) {
  const variants: Record<string, string> = {
    Premium: "bg-amber-900/50 text-amber-300 border-amber-700",
    Strong: "bg-orange-900/50 text-orange-300 border-orange-700",
    Standard: "bg-blue-900/50 text-blue-300 border-blue-700",
    Low: "bg-zinc-800 text-zinc-400 border-zinc-700",
  };

  return (
    <Badge className={variants[tier] || variants.Low} variant="outline">
      {tier}
    </Badge>
  );
}

function BetsTable({ bets, tier }: { bets: ValueBet[]; tier: string }) {
  if (bets.length === 0) return null;

  return (
    <div className="space-y-3">
      <h3 className="text-lg font-semibold text-zinc-100">
        {tier} Picks ({bets.length})
      </h3>
      <div className="rounded-lg border border-zinc-800 overflow-hidden">
        <Table>
          <TableHeader>
            <TableRow className="border-zinc-800 hover:bg-zinc-900/50">
              <TableHead className="text-zinc-400">Player</TableHead>
              <TableHead className="text-zinc-400">Pos</TableHead>
              <TableHead className="text-zinc-400">Team</TableHead>
              <TableHead className="text-zinc-400">Market</TableHead>
              <TableHead className="text-zinc-400">Book</TableHead>
              <TableHead className="text-zinc-400 text-right">Line</TableHead>
              <TableHead className="text-zinc-400 text-right">Price</TableHead>
              <TableHead className="text-zinc-400 text-right">Model</TableHead>
              <TableHead className="text-zinc-400 text-right">Edge %</TableHead>
              <TableHead className="text-zinc-400 text-right">Win %</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {bets.map((bet, idx) => (
              <TableRow key={`${bet.player_id}-${bet.market}-${idx}`} className="border-zinc-800 hover:bg-zinc-900/50">
                <TableCell className="font-medium text-zinc-100">
                  {bet.player_name || bet.player_id}
                </TableCell>
                <TableCell className="text-zinc-400">{bet.position}</TableCell>
                <TableCell className="text-zinc-400">{bet.team}</TableCell>
                <TableCell className="text-zinc-400">
                  {bet.market.replace("_", " ")}
                </TableCell>
                <TableCell className="text-zinc-400">{bet.sportsbook}</TableCell>
                <TableCell className="text-right text-zinc-300">{bet.line.toFixed(1)}</TableCell>
                <TableCell className="text-right text-zinc-300">{bet.price > 0 ? `+${bet.price}` : bet.price}</TableCell>
                <TableCell className="text-right text-zinc-100 font-medium">{bet.mu.toFixed(1)}</TableCell>
                <TableCell className="text-right">
                  <span className={bet.edge_percentage >= 0.15 ? "text-amber-400" : bet.edge_percentage >= 0.10 ? "text-orange-400" : "text-blue-400"}>
                    {(bet.edge_percentage * 100).toFixed(1)}%
                  </span>
                </TableCell>
                <TableCell className="text-right text-zinc-300">{(bet.p_win * 100).toFixed(0)}%</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}

export default function DashboardPage() {
  const [meta, setMeta] = useState<MetaResponse | null>(null);
  const [bets, setBets] = useState<ValueBet[]>([]);
  const [performance, setPerformance] = useState<PerformanceResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [filters, setFilters] = useState<DashboardFilters>({
    season: 2025,
    week: 13,
    minEdge: 0.05,
    bestLineOnly: true,
  });

  // Fetch metadata on mount
  useEffect(() => {
    getMeta()
      .then((data) => {
        setMeta(data);
        if (data.available_weeks.length > 0) {
          setFilters((f) => ({
            ...f,
            season: data.available_weeks[0].season,
            week: data.available_weeks[0].week,
          }));
        }
      })
      .catch((err) => setError(err.message));
  }, []);

  // Fetch bets and performance when filters change
  useEffect(() => {
    setLoading(true);
    setError(null);

    Promise.all([getValueBets(filters), getPerformance(filters.season)])
      .then(([betsData, perfData]) => {
        setBets(betsData.bets);
        setPerformance(perfData);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [filters]);

  // Group bets by tier
  const premiumBets = bets.filter((b) => b.confidence_tier === "Premium");
  const strongBets = bets.filter((b) => b.confidence_tier === "Strong");
  const standardBets = bets.filter((b) => b.confidence_tier === "Standard");

  // Calculate KPIs
  const activeBets = bets.length;
  const premiumCount = premiumBets.length;
  const avgEdge = bets.length > 0
    ? (bets.reduce((sum, b) => sum + b.edge_percentage, 0) / bets.length) * 100
    : 0;
  const seasonRecord = performance
    ? `${performance.total_wins}-${performance.total_losses}`
    : "0-0";
  const seasonPL = performance?.total_profit || 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-zinc-100">Dashboard</h1>
        <p className="text-zinc-400 mt-1">
          Value betting projections for Season {filters.season} Week {filters.week}
        </p>
      </div>

      {/* Error state */}
      {error && (
        <div className="bg-red-900/20 border border-red-800 rounded-lg p-4 text-red-300">
          {error}
        </div>
      )}

      {/* Filters */}
      <Card className="bg-zinc-900 border-zinc-800">
        <CardHeader>
          <CardTitle className="text-sm font-medium text-zinc-400">Filters</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-6">
            <div className="space-y-2">
              <Label className="text-zinc-400">Season</Label>
              <Select
                value={filters.season.toString()}
                onValueChange={(v) => setFilters((f) => ({ ...f, season: parseInt(v) }))}
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
                value={filters.week.toString()}
                onValueChange={(v) => setFilters((f) => ({ ...f, week: parseInt(v) }))}
              >
                <SelectTrigger className="w-24 bg-zinc-800 border-zinc-700 text-zinc-100">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-zinc-800 border-zinc-700">
                  {(meta?.available_weeks.filter((w) => w.season === filters.season).map((w) => w.week) || [1]).map((w) => (
                    <SelectItem key={w} value={w.toString()} className="text-zinc-100">
                      {w}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2 w-48">
              <Label className="text-zinc-400">Min Edge: {(filters.minEdge * 100).toFixed(0)}%</Label>
              <Slider
                value={[filters.minEdge * 100]}
                onValueChange={(v) => setFilters((f) => ({ ...f, minEdge: v[0] / 100 }))}
                min={0}
                max={30}
                step={1}
                className="w-full"
              />
            </div>

            <div className="flex items-center space-x-2">
              <Switch
                id="best-line"
                checked={filters.bestLineOnly}
                onCheckedChange={(v) => setFilters((f) => ({ ...f, bestLineOnly: v }))}
              />
              <Label htmlFor="best-line" className="text-zinc-400">Best Line Only</Label>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <KPICard title="Active Bets" value={activeBets} />
        <KPICard title="Premium Picks" value={premiumCount} />
        <KPICard title="Avg Edge" value={`${avgEdge.toFixed(1)}%`} />
        <KPICard title="Season Record" value={seasonRecord} />
        <KPICard
          title="Season P/L"
          value={`${seasonPL >= 0 ? "+" : ""}${seasonPL.toFixed(1)}u`}
          subtitle={performance ? `${performance.overall_roi.toFixed(1)}% ROI` : undefined}
        />
      </div>

      {/* Top Picks Chips */}
      {premiumBets.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-zinc-400">Top Picks</h3>
          <div className="flex flex-wrap gap-2">
            {premiumBets.slice(0, 6).map((bet, idx) => (
              <Badge
                key={`chip-${idx}`}
                variant="outline"
                className="bg-amber-900/30 text-amber-300 border-amber-700 px-3 py-1"
              >
                {bet.player_name || bet.player_id}
              </Badge>
            ))}
            {premiumBets.length > 6 && (
              <Badge variant="outline" className="bg-zinc-800 text-zinc-400 border-zinc-700 px-3 py-1">
                +{premiumBets.length - 6} more
              </Badge>
            )}
          </div>
        </div>
      )}

      {/* Bets Tables */}
      {loading ? (
        <div className="text-center py-12 text-zinc-400">Loading...</div>
      ) : (
        <Tabs defaultValue="all" className="space-y-4">
          <TabsList className="bg-zinc-900 border border-zinc-800">
            <TabsTrigger value="all" className="data-[state=active]:bg-zinc-800">
              All ({bets.length})
            </TabsTrigger>
            <TabsTrigger value="premium" className="data-[state=active]:bg-zinc-800">
              Premium ({premiumBets.length})
            </TabsTrigger>
            <TabsTrigger value="strong" className="data-[state=active]:bg-zinc-800">
              Strong ({strongBets.length})
            </TabsTrigger>
            <TabsTrigger value="standard" className="data-[state=active]:bg-zinc-800">
              Standard ({standardBets.length})
            </TabsTrigger>
          </TabsList>

          <TabsContent value="all" className="space-y-6">
            <BetsTable bets={premiumBets} tier="Premium" />
            <BetsTable bets={strongBets} tier="Strong" />
            <BetsTable bets={standardBets} tier="Standard" />
          </TabsContent>
          <TabsContent value="premium">
            <BetsTable bets={premiumBets} tier="Premium" />
          </TabsContent>
          <TabsContent value="strong">
            <BetsTable bets={strongBets} tier="Strong" />
          </TabsContent>
          <TabsContent value="standard">
            <BetsTable bets={standardBets} tier="Standard" />
          </TabsContent>
        </Tabs>
      )}
    </div>
  );
}

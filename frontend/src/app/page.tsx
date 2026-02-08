"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
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
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  getValueBets,
  getMeta,
  getPerformance,
  triggerPipelineRun,
  getPipelineRun,
  getLatestRun,
  getCorrelationAnalysis,
  getRiskSummary,
  getExplainability,
  getExportCsvUrl,
  getExportBundleUrl,
  requestAgentReview,
  getAgentReviewStatus,
} from "@/lib/api";
import type {
  ValueBet,
  MetaResponse,
  PerformanceResponse,
  DashboardFilters,
  PipelineRun,
  CorrelationResponse,
  RiskSummary,
  WhyPayload,
  AgentReviewStatus,
} from "@/lib/types";
import { ExplainPopover } from "@/components/explain-popover";
import { RiskPanel } from "@/components/risk-panel";
import {
  TrendingUp,
  Trophy,
  Target,
  BarChart3,
  DollarSign,
  RefreshCw,
  Download,
  FileJson,
  ShieldCheck,
  Info,
} from "lucide-react";

/* ─── Tier color system ─── */
const TIER_CONFIG: Record<
  string,
  {
    bg: string;
    text: string;
    border: string;
    glow: string;
    row: string;
    icon: string;
  }
> = {
  Premium: {
    bg: "bg-amber-500/15",
    text: "text-amber-300",
    border: "border-amber-500/40",
    glow: "tier-premium",
    row: "hover:bg-amber-500/[0.04]",
    icon: "text-amber-400",
  },
  Strong: {
    bg: "bg-orange-500/15",
    text: "text-orange-300",
    border: "border-orange-500/40",
    glow: "tier-strong",
    row: "hover:bg-orange-500/[0.04]",
    icon: "text-orange-400",
  },
  Marginal: {
    bg: "bg-blue-500/15",
    text: "text-blue-300",
    border: "border-blue-500/40",
    glow: "tier-marginal",
    row: "hover:bg-blue-500/[0.04]",
    icon: "text-blue-400",
  },
  Pass: {
    bg: "bg-slate-700/20",
    text: "text-slate-400",
    border: "border-slate-600/40",
    glow: "tier-pass",
    row: "hover:bg-slate-800/30",
    icon: "text-slate-500",
  },
};

function getTierConfig(tier: string) {
  return TIER_CONFIG[tier] || TIER_CONFIG.Pass;
}

/* ─── Edge color gradient ─── */
function getEdgeColor(edge: number): string {
  if (edge >= 0.25) return "text-amber-300";
  if (edge >= 0.15) return "text-amber-400";
  if (edge >= 0.1) return "text-orange-400";
  if (edge >= 0.06) return "text-blue-400";
  return "text-slate-400";
}

function getEdgeBarWidth(edge: number): string {
  const pct = Math.min(edge * 200, 100);
  return `${pct}%`;
}

/* ─── KPI Card ─── */
function KPICard({
  title,
  value,
  subtitle,
  icon: Icon,
  accent = false,
}: {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: React.ComponentType<{ className?: string }>;
  accent?: boolean;
}) {
  return (
    <div className="relative group">
      <div
        className={`rounded-lg border p-4 transition-all duration-200 ${
          accent
            ? "bg-gradient-to-br from-amber-500/10 to-amber-900/5 border-amber-500/20 hover:border-amber-500/40"
            : "bg-[#111827]/80 border-slate-800/60 hover:border-slate-700/80"
        }`}
      >
        <div className="flex items-center justify-between mb-2">
          <span className="text-[11px] font-medium text-slate-500 uppercase tracking-wider">
            {title}
          </span>
          <Icon
            className={`h-3.5 w-3.5 ${accent ? "text-amber-500/50" : "text-slate-600"}`}
          />
        </div>
        <div
          className={`text-2xl font-bold font-[family-name:var(--font-jetbrains)] tabular-nums ${
            accent ? "text-amber-200" : "text-slate-100"
          }`}
        >
          {value}
        </div>
        {subtitle && (
          <p className="text-[11px] text-slate-500 mt-1 font-[family-name:var(--font-jetbrains)]">
            {subtitle}
          </p>
        )}
      </div>
    </div>
  );
}

/* ─── Confidence Badge ─── */
function ConfidenceBadge({ tier }: { tier: string }) {
  const config = getTierConfig(tier);
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-semibold tracking-wide uppercase border ${config.bg} ${config.text} ${config.border}`}
    >
      {tier}
    </span>
  );
}

/* ─── Player name formatting ─── */
function formatPlayerName(bet: ValueBet): string {
  if (bet.player_name) return bet.player_name;
  const raw = bet.player_id?.replace(/^[A-Z]{2,3}_/, "").replace(/_/g, " ") || "Unknown";
  return raw
    .split(" ")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

/* ─── Position badge ─── */
function PositionBadge({ position }: { position: string | null }) {
  if (!position) return <span className="text-slate-600">--</span>;

  const colors: Record<string, string> = {
    QB: "bg-purple-500/20 text-purple-300 border-purple-500/30",
    RB: "bg-emerald-500/20 text-emerald-300 border-emerald-500/30",
    WR: "bg-cyan-500/20 text-cyan-300 border-cyan-500/30",
    TE: "bg-rose-500/20 text-rose-300 border-rose-500/30",
  };

  return (
    <span
      className={`inline-flex items-center justify-center w-8 h-5 rounded text-[10px] font-bold border font-[family-name:var(--font-jetbrains)] ${
        colors[position] || "bg-slate-700/30 text-slate-400 border-slate-600/30"
      }`}
    >
      {position}
    </span>
  );
}

/* ─── Reason Chips ─── */
function ReasonChips({ bet }: { bet: ValueBet }) {
  const chips: { label: string; color: string }[] = [];

  // Edge strength chip
  if (bet.edge_percentage >= 0.2) {
    chips.push({ label: "High Edge", color: "bg-amber-500/15 text-amber-400 border-amber-500/20" });
  } else if (bet.edge_percentage >= 0.12) {
    chips.push({ label: "Good Edge", color: "bg-orange-500/15 text-orange-400 border-orange-500/20" });
  }

  // Win probability chip
  if (bet.p_win >= 0.65) {
    chips.push({ label: `${(bet.p_win * 100).toFixed(0)}% Win`, color: "bg-emerald-500/15 text-emerald-400 border-emerald-500/20" });
  }

  // Model delta chip (model projection vs line)
  const delta = bet.mu - bet.line;
  if (Math.abs(delta) > 5) {
    chips.push({
      label: `+${delta.toFixed(0)} vs line`,
      color: "bg-cyan-500/15 text-cyan-400 border-cyan-500/20",
    });
  }

  if (chips.length === 0) return null;

  return (
    <div className="flex gap-1 mt-0.5">
      {chips.slice(0, 3).map((chip) => (
        <span
          key={chip.label}
          className={`inline-flex items-center px-1.5 py-0 rounded text-[9px] font-medium border ${chip.color}`}
        >
          {chip.label}
        </span>
      ))}
    </div>
  );
}

/* ─── Lazy Why Button ─── */
function WhyButton({
  bet,
  season,
  week,
}: {
  bet: ValueBet;
  season: number;
  week: number;
}) {
  const [why, setWhy] = useState<WhyPayload | undefined>(bet.why);
  const [loading, setLoading] = useState(false);

  const handleClick = async () => {
    if (why) return; // Already loaded
    setLoading(true);
    try {
      const result = await getExplainability(bet.player_id, bet.market, season, week);
      setWhy(result.why);
    } catch {
      // Silently fail - popover just won't show
    } finally {
      setLoading(false);
    }
  };

  if (why) {
    return (
      <ExplainPopover why={why}>
        <div className="flex items-center justify-end gap-2 cursor-pointer">
          <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${
                bet.edge_percentage >= 0.2
                  ? "bg-gradient-to-r from-amber-500 to-amber-400"
                  : bet.edge_percentage >= 0.1
                    ? "bg-gradient-to-r from-orange-500 to-orange-400"
                    : "bg-gradient-to-r from-blue-500 to-blue-400"
              }`}
              style={{ width: getEdgeBarWidth(bet.edge_percentage) }}
            />
          </div>
          <span
            className={`text-[13px] font-[family-name:var(--font-jetbrains)] tabular-nums font-semibold ${getEdgeColor(bet.edge_percentage)}`}
          >
            {(bet.edge_percentage * 100).toFixed(1)}%
          </span>
        </div>
      </ExplainPopover>
    );
  }

  return (
    <div
      className="flex items-center justify-end gap-2 cursor-pointer group"
      onClick={handleClick}
    >
      <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${
            bet.edge_percentage >= 0.2
              ? "bg-gradient-to-r from-amber-500 to-amber-400"
              : bet.edge_percentage >= 0.1
                ? "bg-gradient-to-r from-orange-500 to-orange-400"
                : "bg-gradient-to-r from-blue-500 to-blue-400"
          }`}
          style={{ width: getEdgeBarWidth(bet.edge_percentage) }}
        />
      </div>
      <span
        className={`text-[13px] font-[family-name:var(--font-jetbrains)] tabular-nums font-semibold ${getEdgeColor(bet.edge_percentage)}`}
      >
        {(bet.edge_percentage * 100).toFixed(1)}%
      </span>
      {loading ? (
        <div className="w-3 h-3 border border-slate-500 border-t-slate-300 rounded-full animate-spin" />
      ) : (
        <Info className="w-3 h-3 text-slate-600 opacity-0 group-hover:opacity-100 transition-opacity" />
      )}
    </div>
  );
}

/* ─── Data Health Badge ─── */
function DataHealthBadge({ overall }: { overall: string }) {
  const config: Record<string, { bg: string; text: string; label: string }> = {
    pass: { bg: "bg-emerald-500/10 border-emerald-500/20", text: "text-emerald-400", label: "Healthy" },
    warn: { bg: "bg-amber-500/10 border-amber-500/20", text: "text-amber-400", label: "Warning" },
    fail: { bg: "bg-red-500/10 border-red-500/20", text: "text-red-400", label: "Issues" },
  };
  const c = config[overall] || config.warn;

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <span className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-semibold border ${c.bg} ${c.text}`}>
            <ShieldCheck className="w-3 h-3" />
            {c.label}
          </span>
        </TooltipTrigger>
        <TooltipContent className="bg-[#0d1220] border-slate-700/60 text-slate-300 text-[11px] max-w-xs">
          Data quality checks: missing names, duplicates, null projections
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

/* ─── Bets Table ─── */
function BetsTable({ bets, tier, season, week }: { bets: ValueBet[]; tier: string; season: number; week: number }) {
  if (bets.length === 0) return null;

  const config = getTierConfig(tier);

  return (
    <div
      className="animate-fade-in"
      style={{ animationDelay: `${tier === "Premium" ? 0 : tier === "Strong" ? 80 : tier === "Marginal" ? 160 : 240}ms` }}
    >
      <div className="flex items-center gap-2 mb-3">
        <div className={`w-1 h-4 rounded-full ${config.bg} ${config.border} border`} />
        <h3 className={`text-sm font-semibold ${config.text}`}>
          {tier}
        </h3>
        <span className="text-[11px] text-slate-600 font-[family-name:var(--font-jetbrains)]">
          {bets.length} {bets.length === 1 ? "pick" : "picks"}
        </span>
      </div>

      <div className="rounded-lg border border-slate-800/60 bg-[#111827]/50 overflow-hidden">
        <Table>
          <TableHeader>
            <TableRow className="border-slate-800/40 hover:bg-transparent">
              <TableHead className="text-[11px] text-slate-500 uppercase tracking-wider font-medium h-9">
                Player
              </TableHead>
              <TableHead className="text-[11px] text-slate-500 uppercase tracking-wider font-medium h-9 w-12 text-center">
                Pos
              </TableHead>
              <TableHead className="text-[11px] text-slate-500 uppercase tracking-wider font-medium h-9 w-14">
                Team
              </TableHead>
              <TableHead className="text-[11px] text-slate-500 uppercase tracking-wider font-medium h-9">
                Market
              </TableHead>
              <TableHead className="text-[11px] text-slate-500 uppercase tracking-wider font-medium h-9">
                Book
              </TableHead>
              <TableHead className="text-[11px] text-slate-500 uppercase tracking-wider font-medium h-9 text-right">
                Line
              </TableHead>
              <TableHead className="text-[11px] text-slate-500 uppercase tracking-wider font-medium h-9 text-right">
                Price
              </TableHead>
              <TableHead className="text-[11px] text-slate-500 uppercase tracking-wider font-medium h-9 text-right">
                Model
              </TableHead>
              <TableHead className="text-[11px] text-slate-500 uppercase tracking-wider font-medium h-9 text-right w-32">
                Edge
              </TableHead>
              <TableHead className="text-[11px] text-slate-500 uppercase tracking-wider font-medium h-9 text-right">
                Win%
              </TableHead>
              <TableHead className="text-[11px] text-slate-500 uppercase tracking-wider font-medium h-9 text-center w-20">
                Tier
              </TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {bets.map((bet, idx) => (
              <TableRow
                key={`${bet.player_id}-${bet.market}-${bet.sportsbook}-${idx}`}
                className={`border-slate-800/30 ${config.row} transition-colors duration-100`}
              >
                <TableCell className="font-medium text-slate-100 text-[13px]">
                  <div>
                    {formatPlayerName(bet)}
                    <ReasonChips bet={bet} />
                  </div>
                </TableCell>
                <TableCell className="text-center">
                  <PositionBadge position={bet.position} />
                </TableCell>
                <TableCell className="text-slate-400 text-[13px] font-[family-name:var(--font-jetbrains)]">
                  {bet.team}
                </TableCell>
                <TableCell className="text-slate-400 text-[13px] capitalize">
                  {bet.market?.replace(/_/g, " ")}
                </TableCell>
                <TableCell className="text-slate-500 text-[12px]">
                  {bet.sportsbook}
                </TableCell>
                <TableCell className="text-right text-slate-300 text-[13px] font-[family-name:var(--font-jetbrains)] tabular-nums">
                  {bet.line?.toFixed(1)}
                </TableCell>
                <TableCell className="text-right text-slate-400 text-[13px] font-[family-name:var(--font-jetbrains)] tabular-nums">
                  {bet.price > 0 ? `+${bet.price}` : bet.price}
                </TableCell>
                <TableCell className="text-right text-slate-100 text-[13px] font-[family-name:var(--font-jetbrains)] tabular-nums font-medium">
                  {bet.mu?.toFixed(1)}
                </TableCell>
                <TableCell className="text-right">
                  <WhyButton bet={bet} season={season} week={week} />
                </TableCell>
                <TableCell className="text-right text-slate-300 text-[13px] font-[family-name:var(--font-jetbrains)] tabular-nums">
                  {(bet.p_win * 100).toFixed(0)}%
                </TableCell>
                <TableCell className="text-center">
                  <ConfidenceBadge tier={bet.confidence_tier || "Pass"} />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}

/* ─── Main Dashboard ─── */
export default function DashboardPage() {
  const [meta, setMeta] = useState<MetaResponse | null>(null);
  const [bets, setBets] = useState<ValueBet[]>([]);
  const [performance, setPerformance] = useState<PerformanceResponse | null>(
    null
  );
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Refresh workflow state
  const [pipelineRun, setPipelineRun] = useState<PipelineRun | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Risk & correlation state
  const [correlations, setCorrelations] = useState<CorrelationResponse | null>(null);
  const [riskSummary, setRiskSummary] = useState<RiskSummary | null>(null);

  // Agent review state
  const [reviewStatus, setReviewStatus] = useState<AgentReviewStatus | null>(null);
  const [reviewRequesting, setReviewRequesting] = useState(false);

  const [filters, setFilters] = useState<DashboardFilters>({
    season: 2025,
    week: 13,
    minEdge: 0.05,
    bestLineOnly: true,
  });

  // Stop polling on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  // Poll pipeline run status
  const startPolling = useCallback((runId: string) => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const run = await getPipelineRun(runId);
        setPipelineRun(run);
        if (run.status !== "running") {
          if (pollRef.current) clearInterval(pollRef.current);
          pollRef.current = null;
          setRefreshing(false);
          // Re-fetch bets after pipeline completes
          const betsData = await getValueBets(filters, false);
          setBets(betsData.bets);
        }
      } catch {
        if (pollRef.current) clearInterval(pollRef.current);
        pollRef.current = null;
        setRefreshing(false);
      }
    }, 2000);
  }, [filters]);

  // Trigger refresh
  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      const run = await triggerPipelineRun(filters.season, filters.week, true, true);
      setPipelineRun(run);
      startPolling(run.run_id);
    } catch (err) {
      setRefreshing(false);
      setError(err instanceof Error ? err.message : "Refresh failed");
    }
  };

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

  // Fetch bets, performance, risk, and correlations when filters change
  useEffect(() => {
    setLoading(true);
    setError(null);

    Promise.all([
      getValueBets(filters, false),
      getPerformance(filters.season),
      getCorrelationAnalysis(filters.season, filters.week).catch(() => null),
      getRiskSummary(filters.season, filters.week).catch(() => null),
      getLatestRun(filters.season, filters.week).catch(() => null),
    ])
      .then(async ([betsData, perfData, corrData, riskData, latestRun]) => {
        setBets(betsData.bets);
        setPerformance(perfData);
        setCorrelations(corrData);
        setRiskSummary(riskData);
        if (latestRun) {
          setPipelineRun(latestRun);
          // Check agent review status
          try {
            const review = await getAgentReviewStatus(latestRun.run_id, filters.season, filters.week);
            setReviewStatus(review);
          } catch {
            setReviewStatus(null);
          }
        }
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [filters]);

  // Group bets by tier
  const premiumBets = bets.filter((b) => b.confidence_tier === "Premium");
  const strongBets = bets.filter((b) => b.confidence_tier === "Strong");
  const marginalBets = bets.filter(
    (b) => b.confidence_tier === "Marginal"
  );
  const passBets = bets.filter(
    (b) => b.confidence_tier === "Pass" || !b.confidence_tier
  );

  // Calculate KPIs
  const avgEdge =
    bets.length > 0
      ? (bets.reduce((sum, b) => sum + b.edge_percentage, 0) / bets.length) *
        100
      : 0;
  const seasonRecord = performance
    ? `${performance.total_wins}-${performance.total_losses}`
    : "0-0";
  const seasonPL = performance?.total_profit || 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-end justify-between">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-[11px] text-emerald-400/80 font-[family-name:var(--font-jetbrains)] uppercase tracking-widest">
              Live
            </span>
          </div>
          <h1 className="text-2xl font-bold text-slate-100 tracking-tight">
            Value Dashboard
          </h1>
          <p className="text-sm text-slate-500 mt-0.5">
            Season {filters.season} &middot; Week {filters.week} &middot;{" "}
            <span className="font-[family-name:var(--font-jetbrains)] tabular-nums">
              {bets.length}
            </span>{" "}
            projections
          </p>
        </div>
        <div className="text-right">
          <p className="text-[10px] text-slate-600 uppercase tracking-wider mb-0.5">
            Model Version
          </p>
          <p className="text-xs text-slate-400 font-[family-name:var(--font-jetbrains)]">
            v2.1-rc4
          </p>
        </div>
      </div>

      {/* Error state */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-4 text-red-300 text-sm">
          {error}
        </div>
      )}

      {/* Filters */}
      <div className="rounded-lg border border-slate-800/60 bg-[#111827]/50 p-4">
        <div className="flex flex-wrap items-end gap-5">
          <div className="space-y-1.5">
            <Label className="text-[11px] text-slate-500 uppercase tracking-wider">
              Season
            </Label>
            <Select
              value={filters.season.toString()}
              onValueChange={(v) =>
                setFilters((f) => ({ ...f, season: parseInt(v) }))
              }
            >
              <SelectTrigger className="w-28 h-9 bg-[#0d1220] border-slate-700/60 text-slate-200 text-sm font-[family-name:var(--font-jetbrains)]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-[#111827] border-slate-700">
                {[
                  ...new Set(
                    meta?.available_weeks.map((w) => w.season) || [2025]
                  ),
                ].map((s) => (
                  <SelectItem
                    key={s}
                    value={s.toString()}
                    className="text-slate-200 focus:bg-slate-800 focus:text-slate-100"
                  >
                    {s}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-1.5">
            <Label className="text-[11px] text-slate-500 uppercase tracking-wider">
              Week
            </Label>
            <Select
              value={filters.week.toString()}
              onValueChange={(v) =>
                setFilters((f) => ({ ...f, week: parseInt(v) }))
              }
            >
              <SelectTrigger className="w-20 h-9 bg-[#0d1220] border-slate-700/60 text-slate-200 text-sm font-[family-name:var(--font-jetbrains)]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-[#111827] border-slate-700">
                {(
                  meta?.available_weeks
                    .filter((w) => w.season === filters.season)
                    .map((w) => w.week) || [1]
                ).map((w) => (
                  <SelectItem
                    key={w}
                    value={w.toString()}
                    className="text-slate-200 focus:bg-slate-800 focus:text-slate-100"
                  >
                    {w}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-1.5 w-44">
            <Label className="text-[11px] text-slate-500 uppercase tracking-wider">
              Min Edge{" "}
              <span className="text-amber-500/70 font-[family-name:var(--font-jetbrains)]">
                {(filters.minEdge * 100).toFixed(0)}%
              </span>
            </Label>
            <Slider
              value={[filters.minEdge * 100]}
              onValueChange={(v) =>
                setFilters((f) => ({ ...f, minEdge: v[0] / 100 }))
              }
              min={0}
              max={30}
              step={1}
              className="w-full"
            />
          </div>

          <div className="flex items-center gap-2 pb-0.5">
            <Switch
              id="best-line"
              checked={filters.bestLineOnly}
              onCheckedChange={(v) =>
                setFilters((f) => ({ ...f, bestLineOnly: v }))
              }
            />
            <Label
              htmlFor="best-line"
              className="text-sm text-slate-400 cursor-pointer"
            >
              Best Line
            </Label>
          </div>

          <div className="ml-auto pb-0.5 flex items-center gap-2">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <a
                    href={getExportCsvUrl(filters.season, filters.week, filters.minEdge)}
                    download
                    className="inline-flex items-center justify-center h-9 w-9 rounded-md border bg-[#0d1220] border-slate-700/60 text-slate-400 hover:text-slate-200 hover:border-slate-600 transition-colors"
                  >
                    <Download className="h-3.5 w-3.5" />
                  </a>
                </TooltipTrigger>
                <TooltipContent className="bg-[#0d1220] border-slate-700/60 text-slate-300 text-[11px]">
                  Export CSV
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>

            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <a
                    href={getExportBundleUrl(filters.season, filters.week)}
                    download
                    className="inline-flex items-center justify-center h-9 w-9 rounded-md border bg-[#0d1220] border-slate-700/60 text-slate-400 hover:text-slate-200 hover:border-slate-600 transition-colors"
                  >
                    <FileJson className="h-3.5 w-3.5" />
                  </a>
                </TooltipTrigger>
                <TooltipContent className="bg-[#0d1220] border-slate-700/60 text-slate-300 text-[11px]">
                  Export JSON Bundle
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>

            {pipelineRun && pipelineRun.status === "completed" && (
              <Button
                variant="outline"
                size="sm"
                onClick={async () => {
                  if (!pipelineRun) return;
                  setReviewRequesting(true);
                  try {
                    await requestAgentReview(pipelineRun.run_id, filters.season, filters.week);
                    // Poll for review completion
                    const checkReview = async () => {
                      const status = await getAgentReviewStatus(pipelineRun.run_id, filters.season, filters.week);
                      setReviewStatus(status);
                      if (!status.reviewed) {
                        setTimeout(checkReview, 3000);
                      } else {
                        setReviewRequesting(false);
                      }
                    };
                    setTimeout(checkReview, 2000);
                  } catch {
                    setReviewRequesting(false);
                  }
                }}
                disabled={reviewRequesting || reviewStatus?.reviewed === true}
                className="h-9 bg-[#0d1220] border-slate-700/60 text-slate-300 hover:text-slate-100 hover:border-slate-600"
              >
                <ShieldCheck className={`h-3.5 w-3.5 mr-1.5 ${reviewRequesting ? "animate-pulse" : ""}`} />
                {reviewStatus?.reviewed ? "Reviewed" : reviewRequesting ? "Reviewing..." : "Agent Review"}
              </Button>
            )}

            <Button
              variant="outline"
              size="sm"
              onClick={handleRefresh}
              disabled={refreshing}
              className="h-9 bg-[#0d1220] border-slate-700/60 text-slate-300 hover:text-slate-100 hover:border-slate-600"
            >
              <RefreshCw
                className={`h-3.5 w-3.5 mr-1.5 ${refreshing ? "animate-spin" : ""}`}
              />
              {refreshing ? "Running..." : "Refresh"}
            </Button>
          </div>
        </div>

        {/* Run metadata strip */}
        {pipelineRun && (
          <div className="flex items-center gap-3 mt-3 pt-3 border-t border-slate-800/40 flex-wrap">
            <span className="text-[10px] text-slate-500 uppercase tracking-wider">
              Last Run
            </span>
            <span
              className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold border ${
                pipelineRun.status === "completed"
                  ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                  : pipelineRun.status === "running"
                    ? "bg-blue-500/10 text-blue-400 border-blue-500/20"
                    : "bg-red-500/10 text-red-400 border-red-500/20"
              }`}
            >
              {pipelineRun.status}
            </span>
            <span className="text-[10px] text-slate-500 font-[family-name:var(--font-jetbrains)]">
              {pipelineRun.run_id.slice(0, 8)}
            </span>
            <span className="text-[10px] text-slate-500 font-[family-name:var(--font-jetbrains)]">
              {new Date(pipelineRun.started_at).toLocaleTimeString()}
            </span>
            {pipelineRun.status === "running" && (
              <span className="text-[10px] text-blue-400 font-[family-name:var(--font-jetbrains)]">
                {pipelineRun.stages_completed}/{pipelineRun.stages_requested} stages
              </span>
            )}
            {pipelineRun.error_message && (
              <span className="text-[10px] text-red-400 truncate max-w-xs">
                {pipelineRun.error_message}
              </span>
            )}
            {/* Data Health badge */}
            {pipelineRun.data_health && (
              <DataHealthBadge overall={pipelineRun.data_health.overall} />
            )}
            {/* Agent review stamp */}
            {reviewStatus?.reviewed && (
              <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-semibold border bg-purple-500/10 text-purple-400 border-purple-500/20">
                <ShieldCheck className="w-3 h-3" />
                Reviewed {reviewStatus.decision_count} bets
                {reviewStatus.reviewed_at && (
                  <span className="text-slate-500 ml-1">
                    {new Date(reviewStatus.reviewed_at).toLocaleTimeString()}
                  </span>
                )}
              </span>
            )}
            {/* Version/provenance */}
            {pipelineRun.report_json && (
              <>
                {pipelineRun.report_json.model_version && (
                  <span className="text-[10px] text-slate-500 font-[family-name:var(--font-jetbrains)]">
                    model:{String(pipelineRun.report_json.model_version)}
                  </span>
                )}
              </>
            )}
          </div>
        )}
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <KPICard
          title="Active Bets"
          value={bets.length}
          icon={BarChart3}
        />
        <KPICard
          title="Premium"
          value={premiumBets.length}
          icon={Trophy}
          accent={premiumBets.length > 0}
        />
        <KPICard
          title="Avg Edge"
          value={`${avgEdge.toFixed(1)}%`}
          icon={Target}
        />
        <KPICard
          title="Record"
          value={seasonRecord}
          icon={TrendingUp}
        />
        <KPICard
          title="P/L"
          value={`${seasonPL >= 0 ? "+" : ""}${seasonPL.toFixed(1)}u`}
          subtitle={
            performance
              ? `${performance.overall_roi.toFixed(1)}% ROI`
              : undefined
          }
          icon={DollarSign}
          accent={seasonPL > 0}
        />
      </div>

      {/* Top Picks Strip */}
      {premiumBets.length > 0 && (
        <div className="flex items-center gap-3 px-1">
          <span className="text-[11px] text-amber-500/60 uppercase tracking-wider font-medium shrink-0">
            Top Picks
          </span>
          <div className="h-px flex-1 bg-gradient-to-r from-amber-500/20 to-transparent" />
          <div className="flex flex-wrap gap-2">
            {premiumBets.slice(0, 6).map((bet, idx) => (
              <span
                key={`chip-${idx}`}
                className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-amber-500/10 border border-amber-500/20 text-[12px] text-amber-300 font-medium"
              >
                <span className="w-1 h-1 rounded-full bg-amber-400" />
                {formatPlayerName(bet)}
              </span>
            ))}
            {premiumBets.length > 6 && (
              <span className="inline-flex items-center px-2.5 py-1 rounded-md bg-slate-800/50 border border-slate-700/40 text-[12px] text-slate-500">
                +{premiumBets.length - 6} more
              </span>
            )}
          </div>
        </div>
      )}

      {/* Bets Tables */}
      {loading ? (
        <div className="flex flex-col items-center justify-center py-16 gap-3">
          <div className="w-6 h-6 border-2 border-amber-500/30 border-t-amber-500 rounded-full animate-spin" />
          <span className="text-sm text-slate-500">
            Loading projections...
          </span>
        </div>
      ) : (
        <Tabs defaultValue="all" className="space-y-4">
          <TabsList className="bg-[#111827]/80 border border-slate-800/60 p-0.5 h-auto">
            <TabsTrigger
              value="all"
              className="text-[12px] data-[state=active]:bg-slate-800 data-[state=active]:text-slate-100 text-slate-500 px-3 py-1.5 rounded-md"
            >
              All{" "}
              <span className="ml-1 font-[family-name:var(--font-jetbrains)] tabular-nums">
                {bets.length}
              </span>
            </TabsTrigger>
            <TabsTrigger
              value="premium"
              className="text-[12px] data-[state=active]:bg-amber-500/15 data-[state=active]:text-amber-300 text-slate-500 px-3 py-1.5 rounded-md"
            >
              Premium{" "}
              <span className="ml-1 font-[family-name:var(--font-jetbrains)] tabular-nums">
                {premiumBets.length}
              </span>
            </TabsTrigger>
            <TabsTrigger
              value="strong"
              className="text-[12px] data-[state=active]:bg-orange-500/15 data-[state=active]:text-orange-300 text-slate-500 px-3 py-1.5 rounded-md"
            >
              Strong{" "}
              <span className="ml-1 font-[family-name:var(--font-jetbrains)] tabular-nums">
                {strongBets.length}
              </span>
            </TabsTrigger>
            <TabsTrigger
              value="marginal"
              className="text-[12px] data-[state=active]:bg-blue-500/15 data-[state=active]:text-blue-300 text-slate-500 px-3 py-1.5 rounded-md"
            >
              Marginal{" "}
              <span className="ml-1 font-[family-name:var(--font-jetbrains)] tabular-nums">
                {marginalBets.length}
              </span>
            </TabsTrigger>
            <TabsTrigger
              value="pass"
              className="text-[12px] data-[state=active]:bg-slate-700/30 data-[state=active]:text-slate-300 text-slate-500 px-3 py-1.5 rounded-md"
            >
              Pass{" "}
              <span className="ml-1 font-[family-name:var(--font-jetbrains)] tabular-nums">
                {passBets.length}
              </span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="all" className="space-y-5">
            <BetsTable bets={premiumBets} tier="Premium" season={filters.season} week={filters.week} />
            <BetsTable bets={strongBets} tier="Strong" season={filters.season} week={filters.week} />
            <BetsTable bets={marginalBets} tier="Marginal" season={filters.season} week={filters.week} />
            <BetsTable bets={passBets} tier="Pass" season={filters.season} week={filters.week} />
          </TabsContent>
          <TabsContent value="premium">
            <BetsTable bets={premiumBets} tier="Premium" season={filters.season} week={filters.week} />
          </TabsContent>
          <TabsContent value="strong">
            <BetsTable bets={strongBets} tier="Strong" season={filters.season} week={filters.week} />
          </TabsContent>
          <TabsContent value="marginal">
            <BetsTable bets={marginalBets} tier="Marginal" season={filters.season} week={filters.week} />
          </TabsContent>
          <TabsContent value="pass">
            <BetsTable bets={passBets} tier="Pass" season={filters.season} week={filters.week} />
          </TabsContent>
        </Tabs>
      )}

      {/* Risk & Exposure Panel */}
      {!loading && (
        <RiskPanel correlations={correlations} riskSummary={riskSummary} />
      )}
    </div>
  );
}

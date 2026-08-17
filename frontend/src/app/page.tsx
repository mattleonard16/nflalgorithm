"use client";

import { useEffect, useState, useRef, useCallback, useMemo } from "react";
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
  AvailableWeek,
} from "@/lib/types";
import { useAuth } from "@/lib/auth-context";
import { ExplainPopover } from "@/components/explain-popover";
import { RiskPanel } from "@/components/risk-panel";
import { AddToSlipModal } from "@/components/add-to-slip-modal";
import {
  RefreshCw,
  Download,
  FileJson,
  ShieldCheck,
  Info,
  PlusCircle,
} from "lucide-react";

/* ─── Tier accents ───
 * One neutral ramp; gold is reserved for Premium. Tier identity comes from
 * the section grouping and the word itself, not from a per-tier palette. */
const TIER_CONFIG: Record<string, { text: string; marker: string }> = {
  Premium: { text: "text-primary", marker: "bg-primary" },
  Strong: { text: "text-slate-200", marker: "bg-slate-400" },
  Marginal: { text: "text-slate-400", marker: "bg-slate-600" },
  Pass: { text: "text-slate-500", marker: "bg-slate-700" },
};

function getTierConfig(tier: string) {
  return TIER_CONFIG[tier] || TIER_CONFIG.Pass;
}

/* ─── Edge emphasis: gold past the strong-edge mark, neutral otherwise ─── */
function getEdgeColor(edge: number): string {
  return edge >= 0.15 ? "text-primary" : "text-slate-300";
}

/* ─── KPI Card ─── */
function KPICard({
  title,
  value,
  subtitle,
  valueClassName = "text-slate-100",
}: {
  title: string;
  value: string | number;
  subtitle?: string;
  valueClassName?: string;
}) {
  return (
    <div className="rounded-lg border border-slate-800/60 bg-[#111827]/60 p-4">
      <div className="text-[11px] font-medium text-slate-500 uppercase tracking-wider mb-2">
        {title}
      </div>
      <div
        className={`text-3xl font-bold font-display tabular-nums leading-none ${valueClassName}`}
      >
        {value}
      </div>
      {subtitle && (
        <p className="text-[11px] text-slate-500 mt-1.5 font-[family-name:var(--font-jetbrains)]">
          {subtitle}
        </p>
      )}
    </div>
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

/* ─── Position: plain mono text, no colored box ─── */
function PositionBadge({ position }: { position: string | null }) {
  return (
    <span className="text-[12px] text-slate-400 font-[family-name:var(--font-jetbrains)]">
      {position ?? "--"}
    </span>
  );
}

/* ─── Over/Under: a single letter ahead of the line number ─── */
function SideBadge({ side }: { side?: string | null }) {
  const isUnder = side === "under";
  return (
    <span
      className="text-[10px] text-slate-500 font-[family-name:var(--font-jetbrains)]"
      title={isUnder ? "Under" : "Over"}
    >
      {isUnder ? "U" : "O"}
    </span>
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

  const edgeLabel = (
    <span
      className={`text-[13px] font-[family-name:var(--font-jetbrains)] tabular-nums font-semibold ${getEdgeColor(bet.edge_percentage)}`}
    >
      {(bet.edge_percentage * 100).toFixed(1)}%
    </span>
  );

  if (why) {
    return (
      <ExplainPopover why={why}>
        <div className="flex items-center justify-end cursor-pointer">
          {edgeLabel}
        </div>
      </ExplainPopover>
    );
  }

  return (
    <div
      className="flex items-center justify-end gap-1.5 cursor-pointer group"
      onClick={handleClick}
    >
      {edgeLabel}
      {loading ? (
        <div className="w-3 h-3 border border-slate-500 border-t-slate-300 rounded-full animate-spin" />
      ) : (
        <Info className="w-3 h-3 text-slate-600 opacity-0 group-hover:opacity-100 transition-opacity" />
      )}
    </div>
  );
}

/* ─── Data health: quiet when healthy, colored only on warn/fail ─── */
function DataHealthBadge({ overall }: { overall: string }) {
  const config: Record<string, { text: string; label: string }> = {
    pass: { text: "text-slate-500", label: "data ok" },
    warn: { text: "text-primary", label: "data warning" },
    fail: { text: "text-red-400", label: "data issues" },
  };
  const c = config[overall] || config.warn;

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <span className={`inline-flex items-center gap-1 text-[10px] font-medium uppercase tracking-wider ${c.text}`}>
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
  const [slipBet, setSlipBet] = useState<ValueBet | null>(null);

  if (bets.length === 0) return null;

  const config = getTierConfig(tier);

  return (
    <div>
      <div className="flex items-center gap-2 mb-3">
        <div className={`w-1 h-4 ${config.marker}`} />
        <h3 className={`text-lg font-bold font-display uppercase tracking-wide ${config.text}`}>
          {tier}
        </h3>
        <span className="text-[11px] text-slate-600 font-[family-name:var(--font-jetbrains)]">
          {bets.length} {bets.length === 1 ? "pick" : "picks"}
        </span>
        <div className="h-px flex-1 bg-slate-800/60" />
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
              <TableHead className="text-[11px] text-slate-500 uppercase tracking-wider font-medium h-9 w-10" />
            </TableRow>
          </TableHeader>
          <TableBody>
            {bets.map((bet, idx) => (
              <TableRow
                key={`${bet.player_id}-${bet.market}-${bet.sportsbook}-${idx}`}
                className="border-slate-800/30 hover:bg-slate-800/30 transition-colors duration-100"
              >
                <TableCell className="font-medium text-slate-100 text-[13px]">
                  {formatPlayerName(bet)}
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
                  <span className="inline-flex items-center gap-1.5 justify-end">
                    <SideBadge side={bet.side} />
                    {bet.line?.toFixed(1)}
                  </span>
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
                  <button
                    onClick={() => setSlipBet(bet)}
                    title="Add to bet slip"
                    className="text-slate-600 hover:text-primary transition-colors"
                  >
                    <PlusCircle className="h-4 w-4" />
                  </button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      {slipBet && (
        <AddToSlipModal
          bet={slipBet}
          season={season}
          week={week}
          onClose={() => setSlipBet(null)}
        />
      )}
    </div>
  );
}

/* ─── Main Dashboard ─── */
export default function DashboardPage() {
  const { user } = useAuth();
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
  const reviewTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reviewGenerationRef = useRef(0);

  const [selectedWeek, setSelectedWeek] = useState<AvailableWeek | null>(null);
  const [filterOptions, setFilterOptions] = useState({
    minEdge: 0.05,
    bestLineOnly: true,
  });
  const filters = useMemo<DashboardFilters | null>(
    () =>
      selectedWeek
        ? {
            season: selectedWeek.season,
            week: selectedWeek.week,
            ...filterOptions,
          }
        : null,
    [selectedWeek, filterOptions]
  );

  // Stop polling on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
      if (reviewTimeoutRef.current) clearTimeout(reviewTimeoutRef.current);
      reviewGenerationRef.current += 1;
    };
  }, []);

  // Poll pipeline run status
  const startPolling = useCallback((runId: string) => {
    if (!filters) return;
    const activeFilters = filters;
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const run = await getPipelineRun(runId);
        setPipelineRun(run);
        if (["completed", "failed", "cancelled"].includes(run.status)) {
          if (pollRef.current) clearInterval(pollRef.current);
          pollRef.current = null;
          setRefreshing(false);
          if (run.status === "completed") {
            const betsData = await getValueBets(activeFilters, false);
            setBets(betsData.bets);
          }
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
    if (!filters || !user) return;
    setRefreshing(true);
    try {
      const run = await triggerPipelineRun(filters.season, filters.week, true, false);
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
        setSelectedWeek(data.available_weeks[0] ?? null);
        if (data.available_weeks.length === 0) setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  // Fetch public data only after metadata resolves a real season/week.
  useEffect(() => {
    if (!filters) return;
    const activeFilters = filters;
    let cancelled = false;

    async function loadData() {
      setLoading(true);
      setError(null);

      try {
        const [betsData, perfData, corrData, riskData] = await Promise.all([
          getValueBets(activeFilters, false),
          getPerformance(activeFilters.season),
          getCorrelationAnalysis(activeFilters.season, activeFilters.week).catch(() => null),
          getRiskSummary(activeFilters.season, activeFilters.week).catch(() => null),
        ]);
        if (cancelled) return;
        setBets(betsData.bets);
        setPerformance(perfData);
        setCorrelations(corrData);
        setRiskSummary(riskData);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load data");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    loadData();
    return () => {
      cancelled = true;
    };
  }, [filters]);

  // Operational diagnostics require an authenticated reader.
  useEffect(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    setPipelineRun(null);
    setReviewStatus(null);
    setRefreshing(false);
    setReviewRequesting(false);
    reviewGenerationRef.current += 1;
    if (reviewTimeoutRef.current) {
      clearTimeout(reviewTimeoutRef.current);
      reviewTimeoutRef.current = null;
    }

    if (!selectedWeek || !user) return;
    const activeWeek = selectedWeek;

    let cancelled = false;
    async function loadRunStatus() {
      const latestRun = await getLatestRun(
        activeWeek.season,
        activeWeek.week
      ).catch(() => null);
      if (cancelled || !latestRun) return;
      setPipelineRun(latestRun);
      const review = await getAgentReviewStatus(
        latestRun.run_id,
        activeWeek.season,
        activeWeek.week
      ).catch(() => null);
      if (!cancelled) setReviewStatus(review);
    }
    loadRunStatus();
    return () => {
      cancelled = true;
    };
  }, [selectedWeek, user]);

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
          <h1 className="text-4xl font-bold text-slate-100 tracking-tight font-display uppercase">
            Value Dashboard
          </h1>
          <p className="text-sm text-slate-500 mt-0.5">
            {selectedWeek ? (
              <>
                Season {selectedWeek.season} &middot; Week {selectedWeek.week} &middot;{" "}
                <span className="font-[family-name:var(--font-jetbrains)] tabular-nums">
                  {bets.length}
                </span>{" "}
                projections
              </>
            ) : error ? (
              "Published weeks unavailable"
            ) : meta ? (
              "No published NFL card is available yet"
            ) : (
              "Loading published weeks..."
            )}
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
              value={selectedWeek?.season.toString() ?? ""}
              disabled={!selectedWeek}
              onValueChange={(value) => {
                const season = parseInt(value);
                const firstWeek = meta?.available_weeks.find((item) => item.season === season);
                if (firstWeek) setSelectedWeek(firstWeek);
              }}
            >
              <SelectTrigger className="w-28 h-9 bg-[#0d1220] border-slate-700/60 text-slate-200 text-sm font-[family-name:var(--font-jetbrains)]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-[#111827] border-slate-700">
                {[
                  ...new Set(
                    meta?.available_weeks.map((w) => w.season) || []
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
              value={selectedWeek?.week.toString() ?? ""}
              disabled={!selectedWeek}
              onValueChange={(value) => {
                const week = parseInt(value);
                const selection = meta?.available_weeks.find(
                  (item) => item.season === selectedWeek?.season && item.week === week
                );
                if (selection) setSelectedWeek(selection);
              }}
            >
              <SelectTrigger className="w-20 h-9 bg-[#0d1220] border-slate-700/60 text-slate-200 text-sm font-[family-name:var(--font-jetbrains)]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-[#111827] border-slate-700">
                {(
                  meta?.available_weeks
                    .filter((w) => w.season === selectedWeek?.season)
                    .map((w) => w.week) || []
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
              <span className="text-primary/80 font-[family-name:var(--font-jetbrains)]">
                {(filterOptions.minEdge * 100).toFixed(0)}%
              </span>
            </Label>
            <Slider
              value={[filterOptions.minEdge * 100]}
              onValueChange={(v) =>
                setFilterOptions((current) => ({ ...current, minEdge: v[0] / 100 }))
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
              checked={filterOptions.bestLineOnly}
              onCheckedChange={(v) =>
                setFilterOptions((current) => ({ ...current, bestLineOnly: v }))
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
            {filters && (
              <>
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
              </>
            )}

            {user && filters && pipelineRun && pipelineRun.status === "completed" && (
              <Button
                variant="outline"
                size="sm"
                onClick={async () => {
                  if (!pipelineRun) return;
                  const activeRunId = pipelineRun.run_id;
                  const activeSeason = filters.season;
                  const activeWeek = filters.week;
                  const generation = ++reviewGenerationRef.current;
                  setReviewRequesting(true);
                  try {
                    await requestAgentReview(activeRunId, activeSeason, activeWeek);
                    if (generation !== reviewGenerationRef.current) return;
                    // Poll for review completion
                    const checkReview = async () => {
                      const status = await getAgentReviewStatus(
                        activeRunId,
                        activeSeason,
                        activeWeek
                      );
                      if (generation !== reviewGenerationRef.current) return;
                      setReviewStatus(status);
                      if (!status.reviewed) {
                        reviewTimeoutRef.current = setTimeout(checkReview, 3000);
                      } else {
                        setReviewRequesting(false);
                        reviewTimeoutRef.current = null;
                      }
                    };
                    reviewTimeoutRef.current = setTimeout(checkReview, 2000);
                  } catch {
                    if (generation === reviewGenerationRef.current) {
                      setReviewRequesting(false);
                    }
                  }
                }}
                disabled={reviewRequesting || reviewStatus?.reviewed === true}
                className="h-9 bg-[#0d1220] border-slate-700/60 text-slate-300 hover:text-slate-100 hover:border-slate-600"
              >
                <ShieldCheck className={`h-3.5 w-3.5 mr-1.5 ${reviewRequesting ? "animate-pulse" : ""}`} />
                {reviewStatus?.reviewed ? "Reviewed" : reviewRequesting ? "Reviewing..." : "Agent Review"}
              </Button>
            )}

            {user && filters && (
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
            )}
          </div>
        </div>

        {/* Run metadata strip */}
        {pipelineRun && (
          <div className="flex items-center gap-3 mt-3 pt-3 border-t border-slate-800/40 flex-wrap">
            <span className="text-[10px] text-slate-500 uppercase tracking-wider">
              Last Run
            </span>
            <span
              className={`text-[10px] font-semibold uppercase tracking-wider ${
                pipelineRun.status === "completed"
                  ? "text-slate-400"
                  : pipelineRun.status === "running"
                    ? "text-primary"
                    : "text-red-400"
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
              <span className="text-[10px] text-slate-400 font-[family-name:var(--font-jetbrains)]">
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
              <span className="inline-flex items-center gap-1 text-[10px] font-medium text-slate-400">
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
        <KPICard title="Active Bets" value={bets.length} />
        <KPICard
          title="Premium"
          value={premiumBets.length}
          valueClassName={premiumBets.length > 0 ? "text-primary" : "text-slate-100"}
        />
        <KPICard title="Avg Edge" value={`${avgEdge.toFixed(1)}%`} />
        <KPICard title="Record" value={seasonRecord} />
        <KPICard
          title="P/L"
          value={`${seasonPL >= 0 ? "+" : ""}${seasonPL.toFixed(1)}u`}
          subtitle={
            performance
              ? `${performance.overall_roi.toFixed(1)}% ROI`
              : undefined
          }
          valueClassName={
            seasonPL > 0
              ? "text-emerald-400"
              : seasonPL < 0
                ? "text-red-400"
                : "text-slate-100"
          }
        />
      </div>

      {/* Top Picks Strip */}
      {premiumBets.length > 0 && (
        <div className="flex items-baseline gap-3 px-1 text-[12px]">
          <span className="text-[11px] text-primary uppercase tracking-wider font-medium shrink-0">
            Top Picks
          </span>
          <span className="text-slate-300">
            {premiumBets.slice(0, 6).map(formatPlayerName).join("  ·  ")}
            {premiumBets.length > 6 && (
              <span className="text-slate-500">
                {"  ·  "}+{premiumBets.length - 6} more
              </span>
            )}
          </span>
        </div>
      )}

      {/* Bets Tables */}
      {loading ? (
        <div className="flex flex-col items-center justify-center py-16 gap-3">
          <div className="w-6 h-6 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
          <span className="text-sm text-slate-500">
            Loading projections...
          </span>
        </div>
      ) : !filters && meta ? (
        <div className="rounded-lg border border-slate-800/60 bg-[#111827]/50 px-6 py-14 text-center">
          <p className="text-sm font-medium text-slate-300">
            No published NFL card is available yet
          </p>
          <p className="mt-1 text-xs text-slate-500">
            The dashboard will populate after a validated weekly run publishes a card.
          </p>
        </div>
      ) : !filters ? null : (
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
              className="text-[12px] data-[state=active]:bg-slate-800 data-[state=active]:text-slate-100 text-slate-500 px-3 py-1.5 rounded-md"
            >
              Premium{" "}
              <span className="ml-1 font-[family-name:var(--font-jetbrains)] tabular-nums">
                {premiumBets.length}
              </span>
            </TabsTrigger>
            <TabsTrigger
              value="strong"
              className="text-[12px] data-[state=active]:bg-slate-800 data-[state=active]:text-slate-100 text-slate-500 px-3 py-1.5 rounded-md"
            >
              Strong{" "}
              <span className="ml-1 font-[family-name:var(--font-jetbrains)] tabular-nums">
                {strongBets.length}
              </span>
            </TabsTrigger>
            <TabsTrigger
              value="marginal"
              className="text-[12px] data-[state=active]:bg-slate-800 data-[state=active]:text-slate-100 text-slate-500 px-3 py-1.5 rounded-md"
            >
              Marginal{" "}
              <span className="ml-1 font-[family-name:var(--font-jetbrains)] tabular-nums">
                {marginalBets.length}
              </span>
            </TabsTrigger>
            <TabsTrigger
              value="pass"
              className="text-[12px] data-[state=active]:bg-slate-800 data-[state=active]:text-slate-100 text-slate-500 px-3 py-1.5 rounded-md"
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

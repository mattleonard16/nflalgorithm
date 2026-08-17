"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import {
  Activity,
  Archive,
  CheckCircle2,
  CircleAlert,
  Clock3,
  Database,
  Lock,
  RefreshCw,
  RotateCcw,
  Server,
  Square,
  Workflow,
} from "lucide-react";

import { PipelineArchitecture } from "@/components/pipeline-architecture";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  cancelPipelineRun,
  getArchitectureStatus,
  getHealth,
  retryPipelineRun,
} from "@/lib/api";
import { useAuth } from "@/lib/auth-context";
import type { ArchitectureStatus, HealthResponse, PipelineRun } from "@/lib/types";

const statusStyles: Record<string, string> = {
  queued: "border-amber-400/25 bg-amber-400/10 text-amber-300",
  running: "border-sky-400/25 bg-sky-400/10 text-sky-300",
  cancelling: "border-orange-400/25 bg-orange-400/10 text-orange-300",
  completed: "border-emerald-400/25 bg-emerald-400/10 text-emerald-300",
  failed: "border-red-400/25 bg-red-400/10 text-red-300",
  cancelled: "border-slate-500/30 bg-slate-500/10 text-slate-400",
};

function formatNumber(value: number) {
  return new Intl.NumberFormat("en-US", { notation: value > 9999 ? "compact" : "standard" }).format(value);
}

function timeAgo(value: string | null) {
  if (!value) return "Never";
  const diff = Math.max(0, Date.now() - new Date(value).getTime());
  const minutes = Math.floor(diff / 60_000);
  if (minutes < 1) return "Just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

function Metric({
  label,
  value,
  detail,
  icon: Icon,
  state = "neutral",
}: {
  label: string;
  value: string;
  detail: string;
  icon: React.ComponentType<{ className?: string }>;
  state?: "good" | "warn" | "neutral";
}) {
  const accent = state === "good" ? "text-emerald-400" : state === "warn" ? "text-amber-400" : "text-sky-400";
  return (
    <div className="group border-l border-slate-700/50 px-4 first:border-l-0 first:pl-0">
      <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-500">
        <Icon className={`h-3.5 w-3.5 ${accent}`} />
        {label}
      </div>
      <p className="mt-2 font-display text-3xl font-bold tracking-wide text-slate-100 tabular-nums">{value}</p>
      <p className="mt-0.5 text-[11px] text-slate-500">{detail}</p>
    </div>
  );
}

function RunRow({
  run,
  busy,
  canControl,
  onCancel,
  onRetry,
}: {
  run: PipelineRun;
  busy: string | null;
  canControl: boolean;
  onCancel: (runId: string) => void;
  onRetry: (runId: string) => void;
}) {
  const progress = run.stages_requested > 0 ? Math.min(100, (run.stages_completed / run.stages_requested) * 100) : 0;
  const canCancel = canControl && ["queued", "running", "cancelling"].includes(run.status);
  const canRetry = canControl && run.status === "failed";

  return (
    <div className="grid gap-3 border-t border-slate-800/60 px-4 py-3.5 first:border-t-0 lg:grid-cols-[1.2fr_.7fr_.6fr_1fr_auto] lg:items-center">
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <span className="font-[family-name:var(--font-jetbrains)] text-xs text-slate-200">
            {run.run_id.slice(0, 8)}
          </span>
          <Badge variant="outline" className={`h-5 rounded px-1.5 text-[9px] uppercase tracking-wider ${statusStyles[run.status] ?? statusStyles.cancelled}`}>
            {run.status}
          </Badge>
        </div>
        <p className="mt-1 truncate text-[11px] text-slate-500">
          {run.source ?? "legacy"} · {run.worker_id ? `worker ${run.worker_id}` : "awaiting worker"}
        </p>
      </div>
      <div>
        <p className="text-xs font-semibold text-slate-200">{run.season} · Week {run.week}</p>
        <p className="mt-1 text-[10px] text-slate-500">attempt {run.attempts}/{run.max_attempts || "—"}</p>
      </div>
      <div>
        <p className="text-xs text-slate-300">{run.stages_completed}/{run.stages_requested} stages</p>
        <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-slate-800">
          <div className="h-full rounded-full bg-sky-400 transition-all duration-500" style={{ width: `${progress}%` }} />
        </div>
      </div>
      <div className="min-w-0">
        <p className="text-xs text-slate-300">{timeAgo(run.started_at)}</p>
        <p className="mt-1 truncate text-[10px] text-red-400/80">{run.error_message ?? "No execution errors"}</p>
      </div>
      <div className="flex justify-end">
        {canCancel && (
          <Button
            size="sm"
            variant="outline"
            disabled={busy === run.run_id || run.status === "cancelling"}
            onClick={() => onCancel(run.run_id)}
            className="h-7 border-slate-700 bg-transparent px-2 text-[10px] text-slate-400 hover:border-red-400/40 hover:bg-red-400/5 hover:text-red-300"
          >
            <Square className="mr-1 h-3 w-3" /> Cancel
          </Button>
        )}
        {canRetry && (
          <Button
            size="sm"
            variant="outline"
            disabled={busy === run.run_id}
            onClick={() => onRetry(run.run_id)}
            className="h-7 border-slate-700 bg-transparent px-2 text-[10px] text-slate-400 hover:border-amber-400/40 hover:bg-amber-400/5 hover:text-amber-300"
          >
            <RotateCcw className="mr-1 h-3 w-3" /> Retry
          </Button>
        )}
      </div>
    </div>
  );
}

function OperatorGate({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col items-center gap-3 px-4 py-12 text-center">
      <span className="grid h-9 w-9 place-items-center rounded-xl border border-amber-400/25 bg-amber-400/[0.06] text-amber-300">
        <Lock className="h-4 w-4" />
      </span>
      <p className="text-sm text-slate-400">{title}</p>
      <p className="max-w-md text-[11px] leading-relaxed text-slate-600">{children}</p>
      <Link
        href="/login"
        className="mt-1 rounded-lg border border-slate-700/70 bg-[#0d1624] px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wider text-slate-300 transition-colors hover:border-amber-400/40 hover:text-amber-300"
      >
        Sign in
      </Link>
    </div>
  );
}

export default function SystemPage() {
  const { user, loading: authLoading } = useAuth();
  const [architecture, setArchitecture] = useState<ArchitectureStatus | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [busyRun, setBusyRun] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const refreshInFlight = useRef(false);

  const signedIn = Boolean(user);

  const refresh = useCallback(
    async (showLoading = true) => {
      if (refreshInFlight.current) return;
      refreshInFlight.current = true;
      if (showLoading) setLoading(true);
      try {
        // Health is public and architecture is operator-gated. They are fetched
        // independently so a 401 on the control plane cannot erase the feed
        // freshness a signed-out visitor is entitled to see.
        const healthResult = await getHealth().catch(() => null);
        setHealth(healthResult);

        if (!signedIn) {
          setArchitecture(null);
          setError(null);
          return;
        }
        try {
          setArchitecture(await getArchitectureStatus());
          setError(null);
        } catch (err) {
          setArchitecture(null);
          const message = err instanceof Error ? err.message : "Unable to load pipeline state";
          setError(
            message.includes("401")
              ? "Your operator session expired. Sign in again to view live jobs."
              : message
          );
        }
      } finally {
        if (showLoading) setLoading(false);
        refreshInFlight.current = false;
      }
    },
    [signedIn]
  );

  useEffect(() => {
    if (authLoading) return;
    let cancelled = false;
    let timeout: number | undefined;
    const poll = async (showLoading: boolean) => {
      await refresh(showLoading);
      if (!cancelled) timeout = window.setTimeout(() => void poll(false), 10_000);
    };
    void poll(true);
    return () => {
      cancelled = true;
      if (timeout) window.clearTimeout(timeout);
    };
  }, [authLoading, refresh]);

  const activeRun = useMemo(
    () => architecture?.recent_runs.find((run) => ["queued", "running", "cancelling"].includes(run.status)) ?? architecture?.recent_runs[0] ?? null,
    [architecture]
  );

  const mutateRun = async (runId: string, action: "cancel" | "retry") => {
    setBusyRun(runId);
    setError(null);
    try {
      if (action === "cancel") await cancelPipelineRun(runId);
      else await retryPipelineRun(runId);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : `Unable to ${action} run`);
    } finally {
      setBusyRun(null);
    }
  };

  const queueDepth = (architecture?.queue.queued ?? 0) + (architecture?.queue.retry_scheduled ?? 0);
  const dbName = architecture?.database_backend === "sqlite" ? "SQLite / WAL" : architecture?.database_backend ?? "Unknown";
  const controlPlaneDetail = signedIn ? "awaiting pipeline state" : "operator sign-in required";
  const controlMetric = (value: number) => (architecture ? formatNumber(value) : loading ? "…" : "—");

  return (
    <div className="space-y-5 pb-10">
      <header className="flex flex-col gap-4 border-b border-slate-800/60 pb-5 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <div className="mb-2 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.22em] text-amber-400/70">
            <Workflow className="h-3.5 w-3.5" />
            Pipeline operations
          </div>
          <h1 className="font-display text-4xl font-extrabold uppercase tracking-tight text-slate-100 sm:text-5xl">
            System control room
          </h1>
          <p className="mt-1 max-w-2xl text-sm text-slate-500">
            Durable jobs, fail-closed NFL execution, persisted decisions, and materialized dashboard state.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span className="font-[family-name:var(--font-jetbrains)] text-[10px] uppercase tracking-wider text-slate-600">
            Auto-refresh 10s
          </span>
          <Button
            size="sm"
            variant="outline"
            onClick={() => void refresh()}
            disabled={loading}
            className="h-9 border-slate-700/70 bg-[#0d1624] text-slate-300 hover:border-slate-600 hover:bg-slate-800/60"
          >
            <RefreshCw className={`mr-2 h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
            Refresh state
          </Button>
        </div>
      </header>

      {error && (
        <div role="alert" className="flex items-center gap-3 rounded-xl border border-red-400/25 bg-red-400/[0.06] px-4 py-3 text-sm text-red-300">
          <CircleAlert className="h-4 w-4 shrink-0" />
          {error}
        </div>
      )}

      <section aria-label="System metrics" className="grid grid-cols-2 gap-y-5 rounded-2xl border border-slate-800/70 bg-[#0b1421]/70 px-5 py-4 lg:grid-cols-6">
        <Metric
          label="API"
          value={health ? "Online" : loading ? "…" : "Offline"}
          detail={health ? (health.status === "ACTIVE" ? "feeds healthy" : "read path available") : "health check failed"}
          icon={Server}
          state={health ? "good" : "warn"}
        />
        <Metric
          label="Queue depth"
          value={controlMetric(queueDepth)}
          detail={architecture ? (queueDepth === 1 ? "job waiting" : "jobs waiting") : controlPlaneDetail}
          icon={Clock3}
          state={architecture ? (queueDepth > 0 ? "warn" : "good") : "neutral"}
        />
        <Metric
          label="Workers"
          value={controlMetric(architecture?.workers_active ?? 0)}
          detail={architecture ? "active claims" : controlPlaneDetail}
          icon={Activity}
          state={(architecture?.workers_active ?? 0) > 0 ? "good" : "neutral"}
        />
        <Metric
          label="Decisions"
          value={controlMetric(architecture?.decision_count ?? 0)}
          detail={architecture ? "persisted reviews" : controlPlaneDetail}
          icon={CheckCircle2}
        />
        <Metric
          label="Read models"
          value={controlMetric(architecture?.read_model_rows ?? 0)}
          detail={architecture ? "dashboard rows" : controlPlaneDetail}
          icon={Database}
        />
        <Metric
          label="Artifacts"
          value={controlMetric(architecture?.artifact_count ?? 0)}
          detail={architecture ? dbName : controlPlaneDetail}
          icon={Archive}
        />
      </section>

      {architecture ? (
        <PipelineArchitecture levels={architecture.levels} activeRun={activeRun} />
      ) : (
        <div className="grid min-h-[460px] place-items-center rounded-2xl border border-slate-800/60 bg-[#07111d]/80 text-sm text-slate-500">
          {loading || authLoading ? (
            "Loading architecture state…"
          ) : signedIn ? (
            "Architecture state unavailable"
          ) : (
            <OperatorGate title="Sign in with an operator account to view live jobs.">
              Live topology reads the durable job queue, worker leases, and stage state. Those are
              authenticated reads — the pipeline never exposes run diagnostics publicly. Feed
              freshness below is public and stays visible either way.
            </OperatorGate>
          )}
        </div>
      )}

      <section aria-labelledby="run-queue-heading" className="overflow-hidden rounded-2xl border border-slate-800/70 bg-[#0b1421]/70">
        <div className="flex items-center justify-between px-4 py-4">
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-500">Execution ledger</p>
            <h2 id="run-queue-heading" className="mt-0.5 font-display text-xl font-bold uppercase tracking-wide text-slate-100">
              Recent pipeline jobs
            </h2>
          </div>
          <span className="font-[family-name:var(--font-jetbrains)] text-[10px] text-slate-600">
            {architecture ? `${architecture.recent_runs.length} visible` : "locked"}
          </span>
        </div>
        {architecture?.recent_runs.length ? (
          architecture.recent_runs.map((run) => (
            <RunRow
              key={run.run_id}
              run={run}
              busy={busyRun}
              canControl={signedIn}
              onCancel={(runId) => void mutateRun(runId, "cancel")}
              onRetry={(runId) => void mutateRun(runId, "retry")}
            />
          ))
        ) : architecture ? (
          <div className="border-t border-slate-800/60 px-4 py-10 text-center text-sm text-slate-500">
            No durable runs yet. Publish a card with Refresh on{" "}
            <Link href="/bets" className="text-sky-400 hover:text-sky-300">
              Bets
            </Link>
            , or queue one from the CLI with{" "}
            <code className="font-[family-name:var(--font-jetbrains)] text-[11px] text-slate-400">
              make production-run
            </code>{" "}
            — the pipeline worker executes it. Projections alone live on the{" "}
            <Link href="/" className="text-sky-400 hover:text-sky-300">
              Slate
            </Link>
            .
          </div>
        ) : (
          <div className="border-t border-slate-800/60">
            {loading || authLoading ? (
              <p className="px-4 py-10 text-center text-sm text-slate-500">Loading execution ledger…</p>
            ) : signedIn ? (
              <p className="px-4 py-10 text-center text-sm text-slate-500">
                Execution ledger unavailable.
              </p>
            ) : (
              <OperatorGate title="Sign in with an operator account to view live jobs.">
                Durable runs, worker ownership, and cancel/retry controls require an operator
                session. Runs are queued from Refresh on Bets or{" "}
                <code className="font-[family-name:var(--font-jetbrains)] text-[11px] text-slate-500">
                  make production-run
                </code>
                .
              </OperatorGate>
            )}
          </div>
        )}
      </section>

      <section aria-labelledby="freshness-heading" className="rounded-2xl border border-slate-800/70 bg-[#0b1421]/70 p-4">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-500">Point-in-time inputs</p>
            <h2 id="freshness-heading" className="mt-0.5 font-display text-xl font-bold uppercase tracking-wide text-slate-100">
              Feed freshness
            </h2>
          </div>
          <span className="text-[11px] text-slate-500">last update {timeAgo(health?.last_update ?? null)}</span>
        </div>
        <div className="grid gap-px overflow-hidden rounded-xl border border-slate-800/60 bg-slate-800/60 sm:grid-cols-2 xl:grid-cols-4">
          {health?.feeds.length ? (
            health.feeds.map((feed) => (
              <div key={feed.feed} className="bg-[#0b1421] p-3.5">
                <div className="flex items-center justify-between gap-3">
                  <span className="text-xs font-semibold capitalize text-slate-200">{feed.feed}</span>
                  <span className={`h-2 w-2 rounded-full ${feed.status === "FRESH" ? "bg-emerald-400" : "bg-red-400"}`} />
                </div>
                <p className="mt-2 font-[family-name:var(--font-jetbrains)] text-[10px] uppercase tracking-wider text-slate-600">
                  {feed.status} · {timeAgo(feed.as_of)}
                </p>
              </div>
            ))
          ) : (
            <div className="col-span-full bg-[#0b1421] px-4 py-8 text-center text-sm text-slate-500">
              No freshness snapshots have been persisted.
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

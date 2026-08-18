"use client";

import type { ArchitectureLevel, PipelineRun, PipelineStageRun } from "@/lib/types";
import {
  Activity,
  Bot,
  Box,
  BrainCircuit,
  CircleGauge,
  CloudCog,
  Command,
  Database,
  GitBranch,
  LayoutDashboard,
  RadioTower,
  ServerCog,
  ShieldCheck,
  TimerReset,
} from "lucide-react";

// Tone ids come from the API and must keep their names. Only the CSS behind
// them changes: a steel/gold/olive/khaki ramp instead of a rainbow.
const toneClasses = {
  blue: {
    frame: "border-[#7a8796]/25 bg-[#7a8796]/[0.05]",
    node: "border-[#7a8796]/50 bg-[#7a8796]/[0.10] text-slate-200",
    glow: "shadow-[0_0_26px_rgba(122,135,150,0.12)]",
  },
  amber: {
    frame: "border-amber-400/25 bg-amber-400/[0.035]",
    node: "border-amber-400/45 bg-amber-400/[0.08] text-amber-100",
    glow: "shadow-[0_0_26px_rgba(212,168,75,0.10)]",
  },
  green: {
    frame: "border-[#8a9a7b]/25 bg-[#8a9a7b]/[0.05]",
    node: "border-[#8a9a7b]/50 bg-[#8a9a7b]/[0.10] text-[#c9d7ba]",
    glow: "shadow-[0_0_26px_rgba(138,154,123,0.12)]",
  },
  purple: {
    frame: "border-[#c4b08a]/25 bg-[#c4b08a]/[0.05]",
    node: "border-[#c4b08a]/50 bg-[#c4b08a]/[0.10] text-[#e6dac2]",
    glow: "shadow-[0_0_26px_rgba(196,176,138,0.12)]",
  },
};

const nodeIcons: Record<string, React.ComponentType<{ className?: string }>> = {
  "Next.js Dashboard": LayoutDashboard,
  CLI: Command,
  Scheduler: TimerReset,
  FastAPI: RadioTower,
  "Pipeline Job Service": ServerCog,
  "Job Queue": GitBranch,
  "NFL Worker": CloudCog,
  "Shared Orchestrator": CircleGauge,
  "Prepare Data": Database,
  "Validate Pregame": ShieldCheck,
  "Generate Projections": BrainCircuit,
  "Fetch Live Odds": Activity,
  "Value Engine": CircleGauge,
  "Confidence + Risk": ShieldCheck,
  "Specialist Agents": Bot,
  "Final Betting Card": Box,
  "Operational Database": Database,
  "Artifact Storage": Box,
  "API Read Models": LayoutDashboard,
};

const stageByNode: Record<string, string> = {
  "Prepare Data": "prepare_week",
  "Validate Pregame": "prepare_week",
  "Generate Projections": "prepare_week",
  "Fetch Live Odds": "odds",
  "Value Engine": "value_ranking",
  "Confidence + Risk": "risk_assessment",
  "Specialist Agents": "agents",
  "Final Betting Card": "materialize",
};

/**
 * The API returns one stage row per attempt, oldest first. Only the newest
 * attempt describes the current state — a stage that failed on attempt 1 and
 * succeeded on the retry must not render as failed.
 */
function latestAttempt(stages: PipelineStageRun[], stageName: string): PipelineStageRun | null {
  let latest: PipelineStageRun | null = null;
  for (const stage of stages) {
    if (stage.name !== stageName) continue;
    if (!latest || stage.attempt >= latest.attempt) latest = stage;
  }
  return latest;
}

function nodeState(node: string, run: PipelineRun | null) {
  if (!run) return "idle";
  if (node === "Job Queue" && run.status === "queued") return "active";
  if (node === "NFL Worker" && ["running", "cancelling"].includes(run.status)) return "active";
  const stageName = stageByNode[node];
  if (!stageName) return "idle";
  const stage = latestAttempt(run.stages ?? [], stageName);
  if (!stage) return "idle";
  if (stage.status === "running") return "active";
  if (stage.status === "failed") return "failed";
  return "complete";
}

function ArchitectureNode({ node, level, run }: { node: string; level: ArchitectureLevel; run: PipelineRun | null }) {
  const Icon = nodeIcons[node] ?? ServerCog;
  const colors = toneClasses[level.tone];
  const state = nodeState(node, run);

  return (
    <div
      tabIndex={0}
      className={`group relative rounded-xl border px-3 py-3 transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-400/70 ${colors.node} ${
        state === "active" ? `${colors.glow} pipeline-node-active` : "hover:-translate-y-0.5 hover:border-current/70"
      } ${state === "failed" ? "!border-red-400/60 !bg-red-500/10" : ""}`}
      aria-label={`${node}: ${state}`}
    >
      <div className="flex items-center gap-2.5">
        <span className="grid h-7 w-7 shrink-0 place-items-center rounded-lg border border-white/10 bg-black/20">
          <Icon className="h-3.5 w-3.5" />
        </span>
        <span className="text-[12px] font-semibold leading-tight">{node}</span>
      </div>
      {state !== "idle" && (
        <span
          className={`absolute right-2 top-2 h-1.5 w-1.5 rounded-full ${
            state === "failed" ? "bg-[#b85c4a]" : state === "complete" ? "bg-[#8a9a7b]" : "bg-[#d4a84b]"
          }`}
        />
      )}
    </div>
  );
}

export function PipelineArchitecture({ levels, activeRun }: { levels: ArchitectureLevel[]; activeRun: PipelineRun | null }) {
  return (
    <section aria-labelledby="architecture-heading" className="overflow-hidden rounded-2xl border border-slate-700/40 bg-[#07111d]/90">
      <div className="flex items-end justify-between gap-4 border-b border-slate-700/40 px-5 py-4">
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-[0.22em] text-amber-400/70">Live topology</p>
          <h2 id="architecture-heading" className="font-display text-2xl font-bold uppercase tracking-wide text-slate-100">
            NFL scalable architecture
          </h2>
        </div>
        <div className="hidden items-center gap-2 text-[10px] uppercase tracking-widest text-slate-500 sm:flex">
          <span className="h-1.5 w-1.5 rounded-full bg-[#d4a84b] pipeline-node-active" />
          Live state
        </div>
      </div>

      <div className="architecture-scroll overflow-x-auto p-4 sm:p-5">
        <div className="grid min-w-[1160px] grid-cols-6 gap-3">
          {levels.map((level, index) => {
            const colors = toneClasses[level.tone];
            return (
              <article
                key={level.id}
                className={`architecture-level relative min-h-[420px] rounded-2xl border p-3 ${colors.frame}`}
              >
                {index < levels.length - 1 && (
                  <span className="architecture-flow" aria-hidden="true">
                    <span />
                  </span>
                )}
                <header className="mb-4 border-b border-white/[0.06] pb-3">
                  <p className="font-[family-name:var(--font-jetbrains)] text-[9px] font-semibold uppercase tracking-[0.22em] text-slate-500">
                    Level {level.level.toString().padStart(2, "0")}
                  </p>
                  <h3 className="mt-1 font-display text-lg font-bold uppercase tracking-wide text-slate-100">{level.title}</h3>
                </header>
                <div className="space-y-3">
                  {level.nodes.map((node) => (
                    <ArchitectureNode key={node} node={node} level={level} run={activeRun} />
                  ))}
                </div>
                {level.id === "execution" && (
                  <p className="absolute bottom-3 left-3 right-3 rounded-lg border border-amber-400/30 bg-amber-400/[0.05] px-2 py-1.5 text-center text-[9px] font-semibold uppercase tracking-[0.18em] text-amber-300/80">
                    Fail-closed execution
                  </p>
                )}
                {level.id === "pipeline" && (
                  <p className="absolute bottom-3 left-3 right-3 rounded-lg border border-[#8a9a7b]/35 bg-[#8a9a7b]/[0.07] px-2 py-1.5 text-center text-[9px] font-semibold uppercase tracking-[0.18em] text-[#a9bb98]">
                    Point-in-time inputs
                  </p>
                )}
              </article>
            );
          })}
        </div>
      </div>
    </section>
  );
}

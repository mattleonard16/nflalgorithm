"use client";

import { useEffect, useState } from "react";
import { getNbaSchedule } from "@/lib/nba-api";
import type { NbaGame } from "@/lib/nba-types";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

export default function NbaSchedulePage() {
  const [games, setGames] = useState<NbaGame[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const res = await getNbaSchedule();
        setGames(res.games);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load schedule");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const today = new Date().toLocaleDateString("en-US", {
    weekday: "long",
    month: "long",
    day: "numeric",
  });

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-100">Today's Schedule</h1>
        <p className="text-sm text-slate-500 mt-0.5">{today}</p>
      </div>

      {error && (
        <div className="rounded-lg border border-red-800/50 bg-red-500/10 px-4 py-3 text-sm text-red-400">
          {error}
        </div>
      )}

      {loading ? (
        <div className="text-slate-500 text-sm">Loading schedule…</div>
      ) : games.length === 0 ? (
        <div className="rounded-lg border border-slate-800/50 bg-slate-900/30 p-12 text-center text-slate-600 text-sm">
          No games scheduled today, or live data unavailable.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {games.map((g) => (
            <GameCard key={g.game_id} game={g} />
          ))}
        </div>
      )}
    </div>
  );
}

function GameCard({ game }: { game: NbaGame }) {
  const isLive = game.status?.toLowerCase().includes("q") || game.status === "Half";
  const isFinal = game.status?.toLowerCase().includes("final");
  const hasScore = game.home_score != null && game.away_score != null;

  return (
    <div className="rounded-xl border border-slate-800/50 bg-[#0d1220] p-5">
      {/* Status */}
      <div className="flex items-center justify-between mb-4">
        <Badge
          variant="outline"
          className={cn(
            "text-[10px] border",
            isLive && "border-blue-700/50 text-blue-400 bg-blue-500/10",
            isFinal && "border-slate-700/50 text-slate-500",
            !isLive && !isFinal && "border-slate-700/50 text-slate-500"
          )}
        >
          {game.status ?? "Scheduled"}
        </Badge>
        {isLive && (
          <span className="inline-flex items-center gap-1 text-[10px] text-blue-400">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
            LIVE
          </span>
        )}
      </div>

      {/* Teams */}
      <div className="space-y-3">
        <TeamRow
          team={game.away_team}
          score={game.away_score}
          isWinning={hasScore && (game.away_score ?? 0) > (game.home_score ?? 0)}
        />
        <div className="text-xs text-slate-700 pl-1">@</div>
        <TeamRow
          team={game.home_team}
          score={game.home_score}
          isWinning={hasScore && (game.home_score ?? 0) > (game.away_score ?? 0)}
        />
      </div>
    </div>
  );
}

function TeamRow({
  team,
  score,
  isWinning,
}: {
  team: string;
  score: number | null;
  isWinning: boolean;
}) {
  return (
    <div className="flex items-center justify-between">
      <span
        className={cn(
          "text-base font-bold font-[family-name:var(--font-jetbrains)]",
          isWinning ? "text-blue-300" : "text-slate-400"
        )}
      >
        {team}
      </span>
      {score != null && (
        <span
          className={cn(
            "text-xl font-bold font-[family-name:var(--font-jetbrains)]",
            isWinning ? "text-slate-100" : "text-slate-500"
          )}
        >
          {score}
        </span>
      )}
    </div>
  );
}

"use client";

import { useEffect, useState } from "react";
import { getNbaMeta, getNbaPlayers } from "@/lib/nba-api";
import type { NbaPlayerSummary } from "@/lib/nba-types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";

export default function NbaPlayersPage() {
  const [players, setPlayers] = useState<NbaPlayerSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [season, setSeason] = useState<number | null>(null);
  const [seasonLabel, setSeasonLabel] = useState("Loading…");

  // Fetch meta once to get latest season
  useEffect(() => {
    async function loadMeta() {
      try {
        const meta = await getNbaMeta();
        const latest = meta.available_seasons.length > 0
          ? meta.available_seasons[meta.available_seasons.length - 1]
          : new Date().getFullYear();
        setSeason(latest);
        setSeasonLabel(`${latest}–${String(latest + 1).slice(-2)} season averages`);
      } catch {
        setSeason(new Date().getFullYear());
        setSeasonLabel("Season averages");
      }
    }
    loadMeta();
  }, []);

  // Debounce search
  useEffect(() => {
    const t = setTimeout(() => setDebouncedSearch(search), 300);
    return () => clearTimeout(t);
  }, [search]);

  useEffect(() => {
    if (season === null) return;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const res = await getNbaPlayers(season!, undefined, debouncedSearch || undefined, 200);
        setPlayers(res.players);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load players");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [debouncedSearch, season]);

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">Players</h1>
          <p className="text-sm text-slate-500 mt-0.5">{seasonLabel}</p>
        </div>
        <div className="w-64">
          <Input
            placeholder="Search player name…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="bg-slate-900/60 border-slate-700 text-slate-200 placeholder:text-slate-600"
          />
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-red-800/50 bg-red-500/10 px-4 py-3 text-sm text-red-400">
          {error}
        </div>
      )}

      <Card className="bg-[#0d1220] border-slate-800/50">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium text-slate-400 uppercase tracking-wider">
            {loading ? "Loading…" : `${players.length} players`}
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-800/50">
                  {["Player", "Team", "GP", "PPG", "RPG", "APG", "MPG"].map((h) => (
                    <th
                      key={h}
                      className="px-4 py-3 text-xs font-medium text-slate-600 uppercase tracking-wider text-right first:text-left"
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {players.map((p) => (
                  <tr
                    key={p.player_id}
                    className="border-b border-slate-800/30 hover:bg-slate-800/20 transition-colors"
                  >
                    <td className="px-4 py-2.5 font-medium text-slate-200">{p.player_name}</td>
                    <td className="px-4 py-2.5 text-right">
                      <span className="text-xs font-mono text-blue-400 bg-blue-500/10 px-2 py-0.5 rounded">
                        {p.team}
                      </span>
                    </td>
                    <td className="px-4 py-2.5 text-right text-slate-500 font-[family-name:var(--font-jetbrains)]">
                      {p.games_played}
                    </td>
                    <td className="px-4 py-2.5 text-right text-blue-400 font-bold font-[family-name:var(--font-jetbrains)]">
                      {p.avg_pts.toFixed(1)}
                    </td>
                    <td className="px-4 py-2.5 text-right text-slate-400 font-[family-name:var(--font-jetbrains)]">
                      {p.avg_reb.toFixed(1)}
                    </td>
                    <td className="px-4 py-2.5 text-right text-slate-400 font-[family-name:var(--font-jetbrains)]">
                      {p.avg_ast.toFixed(1)}
                    </td>
                    <td className="px-4 py-2.5 text-right text-slate-500 font-[family-name:var(--font-jetbrains)]">
                      {p.avg_min.toFixed(1)}
                    </td>
                  </tr>
                ))}
                {!loading && players.length === 0 && !error && (
                  <tr>
                    <td colSpan={7} className="px-4 py-12 text-center text-slate-600 text-sm">
                      No players found.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

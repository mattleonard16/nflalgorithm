"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { getProjectionWeeks, getProjections } from "@/lib/api";
import type { AvailableWeek, ProjectionPick } from "@/lib/types";
import { PicksSlate } from "@/components/picks-slate";
import { filterForWeek, useWatchlist } from "@/lib/slate-watchlist";

export default function BoardPage() {
  const [weeks, setWeeks] = useState<AvailableWeek[] | null>(null);
  const [weeksError, setWeeksError] = useState<string | null>(null);
  const [selectedWeek, setSelectedWeek] = useState<AvailableWeek | null>(null);
  const [picks, setPicks] = useState<ProjectionPick[]>([]);
  const [valuesVisible, setValuesVisible] = useState(true);
  const [likely, setLikely] = useState(true);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getProjectionWeeks()
      .then((data) => {
        setWeeks(data.available_weeks);
        setSelectedWeek(data.available_weeks[0] ?? null);
        if (data.available_weeks.length === 0) setLoading(false);
      })
      .catch((err) => {
        setWeeksError(err instanceof Error ? err.message : "Failed to load projection weeks");
        setWeeks([]);
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    if (!selectedWeek) return;
    const activeWeek = selectedWeek;
    let cancelled = false;

    async function loadPicks() {
      setLoading(true);
      setError(null);
      try {
        const data = await getProjections(activeWeek.season, activeWeek.week, { likely });
        if (cancelled) return;
        setPicks(data.picks);
        setValuesVisible(data.values_visible ?? false);
      } catch (err) {
        if (!cancelled) {
          setPicks([]);
          setError(err instanceof Error ? err.message : "Failed to load projections");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    loadPicks();
    return () => {
      cancelled = true;
    };
  }, [selectedWeek, likely]);

  const seasons = useMemo(
    () => [...new Set((weeks ?? []).map((item) => item.season))],
    [weeks]
  );
  const seasonWeeks = useMemo(
    () => (weeks ?? []).filter((item) => item.season === selectedWeek?.season),
    [weeks, selectedWeek]
  );

  const { watchlist } = useWatchlist();
  const watchedCount = selectedWeek
    ? filterForWeek(watchlist, selectedWeek.season, selectedWeek.week).length
    : 0;
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-4xl font-bold text-slate-100 tracking-tight font-display uppercase">
            Slate
          </h1>
          <p className="text-sm text-slate-500 mt-0.5">
            {selectedWeek ? (
              <>
                Season {selectedWeek.season} &middot; Week {selectedWeek.week} &middot;{" "}
                <span className="font-[family-name:var(--font-jetbrains)] tabular-nums">
                  {picks.length}
                </span>{" "}
                {likely ? "projections" : "projections \u00b7 everyone on the roster"}
              </>
            ) : weeksError ? (
              "Projection weeks unavailable"
            ) : weeks ? (
              "No algorithm slate is available yet"
            ) : (
              "Loading projection weeks..."
            )}
          </p>
        </div>
      </div>

      {(weeksError || error) && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-4 text-red-300 text-sm">
          {weeksError ?? error}
        </div>
      )}

      {/* Week picker */}
      {selectedWeek && (
        <div className="rounded-lg border border-slate-800/60 bg-[#111827]/50 p-4">
          <div className="flex flex-wrap items-end gap-6">
            <div className="space-y-1.5">
              <Label className="text-[11px] text-slate-500 uppercase tracking-wider">
                Season
              </Label>
              <Select
                value={selectedWeek.season.toString()}
                onValueChange={(value) => {
                  const season = parseInt(value);
                  const firstWeek = weeks?.find((item) => item.season === season);
                  if (firstWeek) setSelectedWeek(firstWeek);
                }}
              >
                <SelectTrigger className="w-28 h-9 bg-[#0d1220] border-slate-700/60 text-slate-200 text-sm font-[family-name:var(--font-jetbrains)]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-[#111827] border-slate-700">
                  {seasons.map((season) => (
                    <SelectItem
                      key={season}
                      value={season.toString()}
                      className="text-slate-200 focus:bg-slate-800 focus:text-slate-100"
                    >
                      {season}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-1.5">
              <Label className="text-[11px] text-slate-500 uppercase tracking-wider">Week</Label>
              <Select
                value={selectedWeek.week.toString()}
                onValueChange={(value) => {
                  const week = parseInt(value);
                  const selection = seasonWeeks.find((item) => item.week === week);
                  if (selection) setSelectedWeek(selection);
                }}
              >
                <SelectTrigger className="w-20 h-9 bg-[#0d1220] border-slate-700/60 text-slate-200 text-sm font-[family-name:var(--font-jetbrains)]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-[#111827] border-slate-700">
                  {seasonWeeks.map((item) => (
                    <SelectItem
                      key={item.week}
                      value={item.week.toString()}
                      className="text-slate-200 focus:bg-slate-800 focus:text-slate-100"
                    >
                      {item.week}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-center gap-2.5 pb-2">
              <Switch
                id="likely-to-play"
                checked={likely}
                onCheckedChange={setLikely}
                className="data-[state=checked]:bg-amber-400/80"
              />
              <Label
                htmlFor="likely-to-play"
                className="text-[11px] text-slate-400 uppercase tracking-wider cursor-pointer"
              >
                Likely to play
              </Label>
            </div>
          </div>
        </div>
      )}

      {!loading && !valuesVisible && picks.length > 0 && (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-amber-400/30 bg-amber-400/5 px-4 py-3 text-sm">
          <span className="text-slate-300">
            Projected numbers are hidden. Sign in to see each player&apos;s projection.
          </span>
          <Link
            href="/login"
            className="rounded-md bg-amber-400 px-3 py-1.5 text-xs font-semibold uppercase tracking-wider text-[#0d1220] transition-colors hover:bg-amber-300"
          >
            Sign in
          </Link>
        </div>
      )}

      {/* Slate */}
      {loading ? (
        <div className="flex flex-col items-center justify-center py-16 gap-3">
          <div className="w-6 h-6 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
          <span className="text-sm text-slate-500">Loading projections...</span>
        </div>
      ) : !selectedWeek ? (
        weeksError ? null : (
          <div className="rounded-lg border border-slate-800/60 bg-[#111827]/50 px-6 py-14 text-center">
            <p className="text-sm font-medium text-slate-300">
              No algorithm slate is available yet
            </p>
            <p className="mt-1 text-xs text-slate-500">
              Projections appear here after a weekly predict run. Value bets stay under Bets after
              a published card.
            </p>
          </div>
        )
      ) : (
        <PicksSlate picks={picks} season={selectedWeek.season} week={selectedWeek.week} />
      )}

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-600">
        <Link href="/bets" className="hover:text-slate-400 transition-colors">
          Value card lives under Bets after a published run.
        </Link>
        {watchedCount > 0 && (
          <span>
            Tracking{" "}
            <span className="font-[family-name:var(--font-jetbrains)] tabular-nums text-amber-400/80">
              {watchedCount}
            </span>{" "}
            watched{" "}
            {watchedCount === 1 ? "pick" : "picks"} &middot;{" "}
            <Link href="/analytics" className="hover:text-slate-400 transition-colors">
              Analytics
            </Link>{" "}
            &middot;{" "}
            <Link href="/bets" className="hover:text-slate-400 transition-colors">
              Bets
            </Link>
          </span>
        )}
      </div>
    </div>
  );
}

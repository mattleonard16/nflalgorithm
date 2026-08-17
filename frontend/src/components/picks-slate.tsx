"use client";

import { useMemo, useState } from "react";
import { Check, ChevronDown, ChevronUp, Minus } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { ProjectionPick } from "@/lib/types";
import {
  isWatched,
  loadWatchlist,
  saveWatchlist,
  useWatchlist,
  watchlistKey,
  type WatchedPick,
} from "@/lib/slate-watchlist";

const MARKET_DOTS: Record<string, string> = {
  Pass: "#d4a84b",
  Rec: "#5b9fd4",
  Rush: "#3d9b6e",
};

const MARKET_ORDER = ["Pass", "Rec", "Rush"];
const POSITION_ORDER = ["QB", "RB", "WR", "TE"];

const MONO = "font-[family-name:var(--font-jetbrains)]";

type SortKey = "name" | "mu" | "sigma";
type SortDirection = "asc" | "desc";

function marketDot(label: string): string {
  return MARKET_DOTS[label] ?? "#64748b";
}

function roleLabel(pick: ProjectionPick): string {
  const position = pick.position ?? "";
  if (pick.depth_rank != null && position) return `${position}${pick.depth_rank}`;
  if (pick.depth_rank != null) return `#${pick.depth_rank}`;
  if (pick.is_starter) return "Starter";
  return "--";
}

function Chip({
  label,
  count,
  active,
  dot,
  onClick,
}: {
  label: string;
  count: number;
  active: boolean;
  dot?: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={`inline-flex h-7 shrink-0 items-center gap-1.5 rounded-full border px-2.5 text-[11px] font-medium uppercase tracking-wider transition-colors ${
        active
          ? "border-amber-400/70 bg-[#111827] text-amber-300"
          : "border-slate-700/70 text-slate-500 hover:text-slate-300"
      }`}
    >
      {dot && (
        <span
          className="h-1.5 w-1.5 shrink-0 rounded-full"
          style={{ background: dot }}
          aria-hidden="true"
        />
      )}
      <span>{label}</span>
      <span className={`${MONO} pl-1.5 tabular-nums ${active ? "text-amber-400/70" : "text-slate-600"}`}>
        {count}
      </span>
    </button>
  );
}

function MatchupCell({ pick }: { pick: ProjectionPick }) {
  const home = pick.home_team;
  const away = pick.away_team;
  if (!home || !away) {
    return (
      <>
        {pick.team ?? "--"} <span className="text-slate-600">vs</span> {pick.opponent ?? "--"}
      </>
    );
  }
  const side = (code: string) => (code === pick.team ? "text-slate-200" : "text-slate-500");
  return (
    <>
      <span className={side(away)}>{away}</span>
      <span className="px-1 text-slate-600">@</span>
      <span className={side(home)}>{home}</span>
    </>
  );
}

function WatchCheckbox({
  checked,
  indeterminate = false,
  label,
  onToggle,
}: {
  checked: boolean;
  indeterminate?: boolean;
  label: string;
  onToggle: () => void;
}) {
  const filled = checked || indeterminate;
  return (
    <button
      type="button"
      role="checkbox"
      aria-checked={indeterminate ? "mixed" : checked}
      aria-label={label}
      title={label}
      onClick={onToggle}
      className={`flex h-4 w-4 shrink-0 items-center justify-center rounded-[3px] border transition-colors ${
        filled
          ? "border-amber-400 bg-amber-400 text-[#0d1220]"
          : "border-slate-600 hover:border-amber-400/70"
      }`}
    >
      {indeterminate ? (
        <Minus className="h-3 w-3" strokeWidth={3} />
      ) : checked ? (
        <Check className="h-3 w-3" strokeWidth={3} />
      ) : null}
    </button>
  );
}

function SortHeader({
  label,
  sortKey,
  activeKey,
  direction,
  align = "left",
  onSort,
}: {
  label: string;
  sortKey: SortKey;
  activeKey: SortKey;
  direction: SortDirection;
  align?: "left" | "right";
  onSort: (key: SortKey) => void;
}) {
  const active = activeKey === sortKey;
  return (
    <button
      type="button"
      onClick={() => onSort(sortKey)}
      className={`inline-flex w-full items-center gap-1 uppercase tracking-wider transition-colors hover:text-slate-300 ${
        align === "right" ? "justify-end" : "justify-start"
      } ${active ? "text-amber-400" : "text-slate-500"}`}
    >
      <span>{label}</span>
      {active &&
        (direction === "asc" ? (
          <ChevronUp className="h-3 w-3" />
        ) : (
          <ChevronDown className="h-3 w-3" />
        ))}
    </button>
  );
}

export function PicksSlate({
  picks,
  season,
  week,
}: {
  picks: ProjectionPick[];
  season?: number;
  week?: number;
}) {
  const [market, setMarket] = useState<string>("all");
  const [position, setPosition] = useState<string>("all");
  const [team, setTeam] = useState<string>("all");
  const [sortKey, setSortKey] = useState<SortKey>("mu");
  const [direction, setDirection] = useState<SortDirection>("desc");

  const marketCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const pick of picks) {
      counts[pick.market_label] = (counts[pick.market_label] ?? 0) + 1;
    }
    return counts;
  }, [picks]);

  const positionCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const pick of picks) {
      const key = pick.position ?? "--";
      counts[key] = (counts[key] ?? 0) + 1;
    }
    return counts;
  }, [picks]);

  const teamOptions = useMemo(() => {
    const counts = new Map<string, number>();
    for (const pick of picks) {
      if (!pick.team) continue;
      counts.set(pick.team, (counts.get(pick.team) ?? 0) + 1);
    }
    return [...counts.entries()]
      .sort((a, b) => a[0].localeCompare(b[0]))
      .map(([code, count]) => ({ code, count }));
  }, [picks]);

  const visible = useMemo(() => {
    const filtered = picks.filter(
      (pick) =>
        (market === "all" || pick.market_label === market) &&
        (position === "all" || pick.position === position) &&
        (team === "all" || pick.team === team)
    );
    const sign = direction === "asc" ? 1 : -1;
    const numeric = (pick: ProjectionPick) =>
      (sortKey === "mu" ? pick.mu : pick.sigma) ?? Number.NEGATIVE_INFINITY;
    return [...filtered].sort((a, b) => {
      if (sortKey === "name") return sign * a.player_name.localeCompare(b.player_name);
      return sign * (numeric(a) - numeric(b));
    });
  }, [picks, market, position, team, sortKey, direction]);

  const valued = visible.filter((pick) => pick.mu != null);
  const avgMu =
    valued.length > 0
      ? valued.reduce((sum, pick) => sum + (pick.mu ?? 0), 0) / valued.length
      : null;

  const handleSort = (key: SortKey) => {
    if (key === sortKey) {
      setDirection((current) => (current === "asc" ? "desc" : "asc"));
      return;
    }
    setSortKey(key);
    setDirection(key === "name" ? "asc" : "desc");
  };

  const { watchlist, toggle } = useWatchlist();
  const selectable = typeof season === "number" && typeof week === "number";

  const keyParts = (pick: ProjectionPick) => ({
    season: season as number,
    week: week as number,
    player_id: pick.player_id,
    market: pick.market,
  });

  const toWatched = (pick: ProjectionPick): WatchedPick => ({
    ...keyParts(pick),
    player_name: pick.player_name,
    team: pick.team ?? null,
    opponent: pick.opponent ?? null,
    market_label: pick.market_label,
    mu: pick.mu,
    sigma: pick.sigma,
    position: pick.position ?? null,
    added_at: new Date().toISOString(),
  });

  const watchedVisible = selectable
    ? visible.filter((pick) => isWatched(watchlist, keyParts(pick))).length
    : 0;
  const allVisibleWatched = visible.length > 0 && watchedVisible === visible.length;

  const toggleVisible = () => {
    if (!selectable) return;
    const current = loadWatchlist();
    const visibleKeys = new Set(visible.map((pick) => watchlistKey(keyParts(pick))));
    if (allVisibleWatched) {
      saveWatchlist(current.filter((item) => !visibleKeys.has(watchlistKey(item))));
      return;
    }
    const existing = new Set(current.map(watchlistKey));
    const additions = visible
      .map(toWatched)
      .filter((item) => !existing.has(watchlistKey(item)));
    saveWatchlist([...current, ...additions]);
  };

  const headCell =
    "border-b border-amber-400/25 px-3 py-2.5 text-left text-[11px] font-semibold uppercase tracking-wider text-slate-500";

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <div className="flex flex-wrap items-center gap-2">
          <span className="mr-1 text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-600">
            Market
          </span>
          <Chip
            label="All"
            count={picks.length}
            active={market === "all"}
            onClick={() => setMarket("all")}
          />
          {MARKET_ORDER.filter((label) => marketCounts[label]).map((label) => (
            <Chip
              key={label}
              label={label}
              count={marketCounts[label] ?? 0}
              dot={marketDot(label)}
              active={market === label}
              onClick={() => setMarket(label)}
            />
          ))}
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <span className="mr-1 text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-600">
            Position
          </span>
          <Chip
            label="All"
            count={picks.length}
            active={position === "all"}
            onClick={() => setPosition("all")}
          />
          {POSITION_ORDER.filter((label) => positionCounts[label]).map((label) => (
            <Chip
              key={label}
              label={label}
              count={positionCounts[label] ?? 0}
              active={position === label}
              onClick={() => setPosition(label)}
            />
          ))}
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <span className="mr-1 text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-600">
            Team
          </span>
          <Select value={team} onValueChange={setTeam}>
            <SelectTrigger
              aria-label="Filter by team"
              className={`${MONO} h-7 w-28 rounded-full border-slate-700/70 bg-[#0d1220] px-2.5 text-[11px] uppercase tracking-wider ${
                team === "all" ? "text-slate-500" : "text-amber-300"
              }`}
            >
              <SelectValue>{team === "all" ? "All" : team}</SelectValue>
            </SelectTrigger>
            <SelectContent className="max-h-72 border-slate-700 bg-[#111827]">
              <SelectItem
                value="all"
                className={`${MONO} text-[12px] text-slate-200 focus:bg-slate-800 focus:text-slate-100`}
              >
                All
                <span className="pl-2 tabular-nums text-slate-500">{picks.length}</span>
              </SelectItem>
              {teamOptions.map((option) => (
                <SelectItem
                  key={option.code}
                  value={option.code}
                  className={`${MONO} text-[12px] text-slate-200 focus:bg-slate-800 focus:text-slate-100`}
                >
                  {option.code}
                  <span className="pl-2 tabular-nums text-slate-500">{option.count}</span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="max-h-[min(70vh,840px)] overflow-auto rounded-lg border border-slate-800/70 bg-[#111827]/80">
        <table className="w-full table-fixed border-collapse text-[13px]">
          <colgroup>
            <col style={{ width: "32%" }} />
            <col style={{ width: "18%" }} />
            <col style={{ width: "16%" }} />
            <col style={{ width: "12%" }} />
            <col style={{ width: "12%" }} />
            <col style={{ width: "10%" }} />
          </colgroup>
          <thead>
            <tr>
              <th className={`${headCell} sticky left-0 top-0 z-30 bg-[#0d1220]`}>
                <div className="flex items-center gap-2.5">
                  {selectable && (
                    <WatchCheckbox
                      checked={allVisibleWatched}
                      indeterminate={!allVisibleWatched && watchedVisible > 0}
                      label={
                        allVisibleWatched
                          ? "Unwatch all visible picks"
                          : "Watch all visible picks"
                      }
                      onToggle={toggleVisible}
                    />
                  )}
                  <SortHeader
                    label="Player"
                    sortKey="name"
                    activeKey={sortKey}
                    direction={direction}
                    onSort={handleSort}
                  />
                </div>
              </th>
              <th className={`${headCell} sticky top-0 z-20 bg-[#0d1220]`}>Matchup</th>
              <th className={`${headCell} sticky top-0 z-20 bg-[#0d1220]`}>Market</th>
              <th className={`${headCell} sticky top-0 z-20 bg-[#0d1220] text-right`}>
                <SortHeader
                  label="μ"
                  sortKey="mu"
                  activeKey={sortKey}
                  direction={direction}
                  align="right"
                  onSort={handleSort}
                />
              </th>
              <th className={`${headCell} sticky top-0 z-20 bg-[#0d1220] text-right`}>
                <SortHeader
                  label="σ"
                  sortKey="sigma"
                  activeKey={sortKey}
                  direction={direction}
                  align="right"
                  onSort={handleSort}
                />
              </th>
              <th className={`${headCell} sticky top-0 z-20 bg-[#0d1220]`}>Role</th>
            </tr>
          </thead>
          <tbody>
            {visible.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-3 py-10 text-center text-[13px] text-slate-500">
                  No picks for this cut.
                </td>
              </tr>
            ) : (
              visible.map((pick) => (
                <tr
                  key={`${pick.player_id}-${pick.market}`}
                  className="group border-b border-slate-800/50 last:border-b-0 hover:bg-amber-400/5"
                >
                  <td className="sticky left-0 z-10 bg-[#111827] px-3 py-2 group-hover:bg-[#161f31]">
                    <div className="flex items-center gap-2.5">
                      {selectable && (
                        <WatchCheckbox
                          checked={isWatched(watchlist, keyParts(pick))}
                          label={`Watch ${pick.player_name} ${pick.market_label}`}
                          onToggle={() => toggle(toWatched(pick))}
                        />
                      )}
                      <span className="truncate font-medium text-slate-100">
                        {pick.player_name}
                      </span>
                      {(pick.injury_status ?? "").trim() !== "" && (
                        <span
                          title={pick.injury_status ?? undefined}
                          className="inline-flex h-4 shrink-0 items-center rounded border border-amber-400/40 px-1 text-[9px] font-semibold uppercase leading-none text-amber-400/80"
                        >
                          {(pick.injury_status ?? "").trim().charAt(0)}
                        </span>
                      )}
                    </div>
                  </td>
                  <td className={`${MONO} px-3 py-2 text-[12px] text-slate-400`}>
                    <MatchupCell pick={pick} />
                  </td>
                  <td className="px-3 py-2">
                    <span className="inline-flex items-center gap-1.5 rounded border border-slate-700/60 bg-slate-500/10 px-2 py-0.5 text-[11px] uppercase tracking-wider text-slate-300">
                      <span
                        className="h-1.5 w-1.5 shrink-0 rounded-full"
                        style={{ background: marketDot(pick.market_label) }}
                        aria-hidden="true"
                      />
                      {pick.market_label}
                    </span>
                  </td>
                  <td className={`${MONO} px-3 py-2 text-right font-semibold tabular-nums text-slate-100`}>
                    {pick.mu != null ? pick.mu.toFixed(1) : "\u2014"}
                  </td>
                  <td className={`${MONO} px-3 py-2 text-right tabular-nums text-slate-400`}>
                    {pick.sigma != null ? pick.sigma.toFixed(1) : "\u2014"}
                  </td>
                  <td className={`${MONO} px-3 py-2 text-[12px] text-slate-500`}>
                    {roleLabel(pick)}
                  </td>
                </tr>
              ))
            )}
          </tbody>
          <tfoot>
            <tr className="sticky bottom-0 z-10 bg-[#0d1220]">
              <td className="sticky left-0 z-20 border-t border-amber-400/25 bg-[#0d1220] px-3 py-2 text-[12px] text-slate-400">
                <span className="text-[10px] uppercase tracking-[0.1em] text-slate-600">
                  Showing
                </span>{" "}
                <span className={`${MONO} tabular-nums text-slate-300`}>{visible.length}</span>
              </td>
              <td className="border-t border-amber-400/25 px-3 py-2" colSpan={2} />
              <td
                className={`${MONO} border-t border-amber-400/25 px-3 py-2 text-right tabular-nums text-slate-400`}
              >
                <span className="text-[10px] uppercase tracking-[0.1em] text-slate-600">avg </span>
                {avgMu != null ? avgMu.toFixed(1) : "\u2014"}
              </td>
              <td className="border-t border-amber-400/25 px-3 py-2" colSpan={2} />
            </tr>
          </tfoot>
        </table>
      </div>
    </div>
  );
}

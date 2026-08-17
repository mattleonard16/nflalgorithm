"use client";

/**
 * Client-side watchlist of algorithm picks selected on the Slate.
 *
 * These are model projections the user wants to track, not placed bets. They
 * live in localStorage so a signed-out visitor keeps a working slate, and they
 * never touch `user_bets`.
 */

import { useCallback, useSyncExternalStore } from "react";

export const WATCHLIST_STORAGE_KEY = "nflalgorithm.slate-watchlist.v1";
export const WATCHLIST_CHANGE_EVENT = "nflalgorithm-watchlist-changed";

export type WatchedPick = {
  season: number;
  week: number;
  player_id: string;
  player_name: string;
  team: string | null;
  opponent: string | null;
  market: string;
  market_label: string;
  /** Null when the pick was watched while signed out — values are gated. */
  mu: number | null;
  sigma: number | null;
  position: string | null;
  added_at: string;
};

export type WatchlistKeyParts = Pick<
  WatchedPick,
  "season" | "week" | "player_id" | "market"
>;

export function watchlistKey(pick: WatchlistKeyParts): string {
  return `${pick.season}|${pick.week}|${pick.player_id}|${pick.market}`;
}

function isWatchedPick(value: unknown): value is WatchedPick {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate.season === "number" &&
    typeof candidate.week === "number" &&
    typeof candidate.player_id === "string" &&
    typeof candidate.market === "string" &&
    (typeof candidate.mu === "number" || candidate.mu === null) &&
    (typeof candidate.sigma === "number" || candidate.sigma === null)
  );
}

export function loadWatchlist(): WatchedPick[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(WATCHLIST_STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isWatchedPick);
  } catch {
    return [];
  }
}

export function saveWatchlist(picks: WatchedPick[]): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(WATCHLIST_STORAGE_KEY, JSON.stringify(picks));
  } catch {
    // Quota or private-mode failures must not break the slate.
    return;
  }
  window.dispatchEvent(new CustomEvent(WATCHLIST_CHANGE_EVENT));
}

export function upsertWatched(pick: WatchedPick): WatchedPick[] {
  const key = watchlistKey(pick);
  const existing = loadWatchlist();
  const next = existing.filter((item) => watchlistKey(item) !== key);
  next.push(pick);
  saveWatchlist(next);
  return next;
}

export function removeWatched(key: WatchlistKeyParts): WatchedPick[] {
  const target = watchlistKey(key);
  const next = loadWatchlist().filter((item) => watchlistKey(item) !== target);
  saveWatchlist(next);
  return next;
}

export function isWatched(picks: WatchedPick[], key: WatchlistKeyParts): boolean {
  const target = watchlistKey(key);
  return picks.some((item) => watchlistKey(item) === target);
}

export function watchlistForWeek(season: number, week: number): WatchedPick[] {
  return loadWatchlist().filter(
    (item) => item.season === season && item.week === week
  );
}

export function filterForWeek(
  picks: WatchedPick[],
  season: number,
  week: number
): WatchedPick[] {
  return picks.filter((item) => item.season === season && item.week === week);
}

/** Most recent season/week present in the list, for pages with no week picker. */
export function latestWatchedWeek(
  picks: WatchedPick[]
): { season: number; week: number } | null {
  let latest: { season: number; week: number } | null = null;
  for (const pick of picks) {
    if (
      !latest ||
      pick.season > latest.season ||
      (pick.season === latest.season && pick.week > latest.week)
    ) {
      latest = { season: pick.season, week: pick.week };
    }
  }
  return latest;
}

const EMPTY_WATCHLIST: WatchedPick[] = [];

/**
 * Snapshot cache. `useSyncExternalStore` compares snapshots by reference, so a
 * fresh array on every read would loop forever. Reparse only when the stored
 * string actually changed.
 */
let snapshotRaw: string | null = null;
let snapshotValue: WatchedPick[] = EMPTY_WATCHLIST;

function watchlistSnapshot(): WatchedPick[] {
  let raw: string | null;
  try {
    raw = window.localStorage.getItem(WATCHLIST_STORAGE_KEY);
  } catch {
    return snapshotValue;
  }
  if (raw === snapshotRaw) return snapshotValue;
  snapshotRaw = raw;
  snapshotValue = loadWatchlist();
  return snapshotValue;
}

function serverWatchlistSnapshot(): WatchedPick[] {
  return EMPTY_WATCHLIST;
}

function subscribeToWatchlist(onChange: () => void): () => void {
  const onStorage = (event: StorageEvent) => {
    if (event.key === null || event.key === WATCHLIST_STORAGE_KEY) onChange();
  };
  window.addEventListener(WATCHLIST_CHANGE_EVENT, onChange);
  window.addEventListener("storage", onStorage);
  return () => {
    window.removeEventListener(WATCHLIST_CHANGE_EVENT, onChange);
    window.removeEventListener("storage", onStorage);
  };
}

/**
 * Subscribe to the watchlist.
 *
 * Renders empty on the server and during hydration, then reads storage. The
 * custom event keeps Slate/Analytics/Bets in sync inside one tab; the storage
 * event covers other tabs.
 */
export function useWatchlist(): {
  watchlist: WatchedPick[];
  toggle: (pick: WatchedPick) => void;
  add: (pick: WatchedPick) => void;
  remove: (key: WatchlistKeyParts) => void;
} {
  const watchlist = useSyncExternalStore(
    subscribeToWatchlist,
    watchlistSnapshot,
    serverWatchlistSnapshot
  );

  const add = useCallback((pick: WatchedPick) => {
    upsertWatched(pick);
  }, []);

  const remove = useCallback((key: WatchlistKeyParts) => {
    removeWatched(key);
  }, []);

  const toggle = useCallback((pick: WatchedPick) => {
    if (isWatched(loadWatchlist(), pick)) {
      removeWatched(pick);
    } else {
      upsertWatched(pick);
    }
  }, []);

  return { watchlist, toggle, add, remove };
}

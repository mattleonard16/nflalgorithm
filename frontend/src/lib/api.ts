/**
 * API client for NFL Algorithm FastAPI backend
 */

import type {
  ValueBetsResponse,
  PerformanceResponse,
  BetOutcome,
  HealthResponse,
  MetaResponse,
  EdgeDistribution,
  PositionStats,
  MarketStats,
  DashboardFilters,
  AuthResponse,
  User,
  UserPreferencesData,
  UserBet,
  UserStats,
  WeeklySummaryResponse,
} from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/**
 * Generic fetch wrapper with error handling
 */
async function fetchAPI<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;

  const response = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`API Error (${response.status}): ${errorText}`);
  }

  return response.json();
}

/**
 * Get metadata (available weeks, sportsbooks, markets)
 */
export async function getMeta(): Promise<MetaResponse> {
  return fetchAPI<MetaResponse>("/api/meta");
}

/**
 * Get value bets for a specific week with filters
 */
export async function getValueBets(filters: DashboardFilters): Promise<ValueBetsResponse> {
  const params = new URLSearchParams({
    season: filters.season.toString(),
    week: filters.week.toString(),
    min_edge: filters.minEdge.toString(),
    best_line_only: filters.bestLineOnly.toString(),
  });

  if (filters.sportsbook) params.append("sportsbook", filters.sportsbook);
  if (filters.market) params.append("market", filters.market);
  if (filters.position) params.append("position", filters.position);

  return fetchAPI<ValueBetsResponse>(`/api/value-bets?${params.toString()}`);
}

/**
 * Get historical performance data
 */
export async function getPerformance(season?: number): Promise<PerformanceResponse> {
  const params = season ? `?season=${season}` : "";
  return fetchAPI<PerformanceResponse>(`/api/performance${params}`);
}

/**
 * Get bet outcomes for a specific week
 */
export async function getOutcomes(season: number, week: number): Promise<BetOutcome[]> {
  return fetchAPI<BetOutcome[]>(`/api/outcomes?season=${season}&week=${week}`);
}

/**
 * Get system health and feed freshness
 */
export async function getHealth(season?: number, week?: number): Promise<HealthResponse> {
  const params = new URLSearchParams();
  if (season) params.append("season", season.toString());
  if (week) params.append("week", week.toString());

  const queryString = params.toString();
  return fetchAPI<HealthResponse>(`/api/health${queryString ? `?${queryString}` : ""}`);
}

/**
 * Get edge distribution data for charts
 */
export async function getEdgeDistribution(
  season: number,
  week: number,
  bins: number = 20
): Promise<EdgeDistribution> {
  return fetchAPI<EdgeDistribution>(
    `/api/analytics/edge-distribution?season=${season}&week=${week}&bins=${bins}`
  );
}

/**
 * Get analytics by position
 */
export async function getAnalyticsByPosition(
  season: number,
  week: number
): Promise<{ positions: PositionStats[] }> {
  return fetchAPI<{ positions: PositionStats[] }>(
    `/api/analytics/by-position?season=${season}&week=${week}`
  );
}

/**
 * Get analytics by market
 */
export async function getAnalyticsByMarket(
  season: number,
  week: number
): Promise<{ markets: MarketStats[] }> {
  return fetchAPI<{ markets: MarketStats[] }>(
    `/api/analytics/by-market?season=${season}&week=${week}`
  );
}

/**
 * Get weekly summary for sidebar widget
 */
export async function getWeeklySummary(weeks: number = 4): Promise<WeeklySummaryResponse> {
  return fetchAPI<WeeklySummaryResponse>(`/api/weekly-summary?weeks=${weeks}`);
}

/**
 * Health check - simple ping
 */
export async function ping(): Promise<boolean> {
  try {
    await fetchAPI("/");
    return true;
  } catch {
    return false;
  }
}

// ============================================================================
// Authentication API
// ============================================================================

/**
 * Get stored session token
 */
function getSessionToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("session_id");
}

/**
 * Store session token
 */
function setSessionToken(token: string): void {
  if (typeof window !== "undefined") {
    localStorage.setItem("session_id", token);
  }
}

/**
 * Clear session token
 */
function clearSessionToken(): void {
  if (typeof window !== "undefined") {
    localStorage.removeItem("session_id");
  }
}

/**
 * Get auth headers
 */
function getAuthHeaders(): HeadersInit {
  const token = getSessionToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

/**
 * Authenticated fetch wrapper
 */
async function fetchAPIAuth<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;

  const response = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...getAuthHeaders(),
      ...options?.headers,
    },
  });

  if (!response.ok) {
    if (response.status === 401) {
      clearSessionToken();
    }
    const errorText = await response.text();
    throw new Error(`API Error (${response.status}): ${errorText}`);
  }

  return response.json();
}

/**
 * Register a new user
 */
export async function register(
  email: string,
  password: string,
  name?: string
): Promise<AuthResponse> {
  const result = await fetchAPI<AuthResponse>("/api/auth/register", {
    method: "POST",
    body: JSON.stringify({ email, password, name }),
  });
  setSessionToken(result.session_id);
  return result;
}

/**
 * Login with email and password
 */
export async function login(email: string, password: string): Promise<AuthResponse> {
  const result = await fetchAPI<AuthResponse>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
  setSessionToken(result.session_id);
  return result;
}

/**
 * Logout current user
 */
export async function logout(): Promise<void> {
  try {
    await fetchAPIAuth("/api/auth/logout", { method: "POST" });
  } finally {
    clearSessionToken();
  }
}

/**
 * Get current user
 */
export async function getCurrentUser(): Promise<User | null> {
  const token = getSessionToken();
  if (!token) return null;

  try {
    return await fetchAPIAuth<User>("/api/auth/me");
  } catch {
    clearSessionToken();
    return null;
  }
}

/**
 * Check if user is logged in
 */
export function isLoggedIn(): boolean {
  return !!getSessionToken();
}

// ============================================================================
// User Preferences API
// ============================================================================

/**
 * Get user preferences
 */
export async function getUserPreferences(): Promise<UserPreferencesData> {
  return fetchAPIAuth<UserPreferencesData>("/api/user/preferences");
}

/**
 * Update user preferences
 */
export async function updateUserPreferences(
  prefs: Partial<UserPreferencesData>
): Promise<UserPreferencesData> {
  return fetchAPIAuth<UserPreferencesData>("/api/user/preferences", {
    method: "PUT",
    body: JSON.stringify(prefs),
  });
}

/**
 * Update bankroll
 */
export async function updateBankroll(bankroll: number): Promise<{ bankroll: number }> {
  return fetchAPIAuth<{ bankroll: number }>(`/api/user/bankroll?bankroll=${bankroll}`, {
    method: "PUT",
  });
}

// ============================================================================
// User Bet Tracking API
// ============================================================================

/**
 * Place a bet (record in user's history)
 */
export async function placeBet(bet: {
  season: number;
  week: number;
  player_id: string;
  player_name?: string;
  market: string;
  sportsbook: string;
  side: string;
  line: number;
  price: number;
  stake_units: number;
  stake_dollars?: number;
  model_edge?: number;
  confidence_tier?: string;
}): Promise<UserBet> {
  return fetchAPIAuth<UserBet>("/api/user/bets", {
    method: "POST",
    body: JSON.stringify(bet),
  });
}

/**
 * Get user's placed bets
 */
export async function getUserBets(season?: number, week?: number): Promise<UserBet[]> {
  const params = new URLSearchParams();
  if (season) params.append("season", season.toString());
  if (week) params.append("week", week.toString());
  const query = params.toString() ? `?${params.toString()}` : "";
  return fetchAPIAuth<UserBet[]>(`/api/user/bets${query}`);
}

/**
 * Get user betting stats
 */
export async function getUserStats(): Promise<UserStats> {
  return fetchAPIAuth<UserStats>("/api/user/stats");
}


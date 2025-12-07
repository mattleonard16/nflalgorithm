/**
 * TypeScript types for NFL Algorithm API responses
 */

export interface ValueBet {
  player_id: string;
  player_name: string | null;
  position: string | null;
  team: string | null;
  opponent: string | null;
  market: string;
  sportsbook: string;
  line: number;
  price: number;
  mu: number;
  sigma: number;
  p_win: number;
  edge_percentage: number;
  expected_roi: number;
  kelly_fraction: number;
  stake: number;
  confidence_tier: "Premium" | "Strong" | "Standard" | "Low";
  recommendation: "BET" | "PASS";
}

export interface ValueBetsResponse {
  season: number;
  week: number;
  total_count: number;
  filtered_count: number;
  bets: ValueBet[];
}

export interface WeeklyPerformance {
  season: number;
  week: number;
  total_bets: number;
  wins: number;
  losses: number;
  pushes: number;
  profit_units: number;
  roi_pct: number;
  avg_edge: number;
  best_bet: string | null;
  worst_bet: string | null;
}

export interface PerformanceResponse {
  total_bets: number;
  total_wins: number;
  total_losses: number;
  total_profit: number;
  overall_roi: number;
  win_rate: number;
  weeks: WeeklyPerformance[];
}

export interface BetOutcome {
  bet_id: string;
  player_name: string | null;
  market: string;
  line: number;
  actual_result: number | null;
  result: string | null;
  profit_units: number | null;
  confidence_tier: string | null;
}

export interface FeedFreshness {
  feed: string;
  as_of: string | null;
  age_minutes: number | null;
  status: "FRESH" | "STALE";
}

export interface HealthResponse {
  status: "ACTIVE" | "MAINTENANCE";
  database: string;
  last_update: string;
  feeds: FeedFreshness[];
}

export interface AvailableWeek {
  season: number;
  week: number;
}

export interface MetaResponse {
  available_weeks: AvailableWeek[];
  sportsbooks: string[];
  markets: string[];
}

export interface EdgeDistribution {
  bins: number[];
  counts: number[];
}

export interface PositionStats {
  position: string;
  count: number;
  avg_edge: number;
}

export interface MarketStats {
  market: string;
  count: number;
  avg_edge: number;
}

// Filter state for dashboard
export interface DashboardFilters {
  season: number;
  week: number;
  minEdge: number;
  bestLineOnly: boolean;
  sportsbook?: string;
  market?: string;
  position?: string;
}

// Authentication types
export interface User {
  id: string;
  email: string;
  name: string | null;
  subscription_tier: string;
  bankroll: number;
  created_at: string;
}

export interface AuthResponse {
  session_id: string;
  expires_at: string;
  user: User;
}

export interface UserPreferencesData {
  default_min_edge: number;
  default_kelly_fraction: number;
  default_max_stake: number;
  best_line_only: boolean;
  show_synthetic_odds: boolean;
  defense_multipliers: boolean;
  weather_adjustments: boolean;
  injury_weighting: boolean;
  preferred_sportsbooks: string | null;
  preferred_markets: string | null;
}

export interface UserBet {
  id: string;
  season: number;
  week: number;
  player_id: string;
  player_name: string | null;
  market: string;
  sportsbook: string;
  side: string;
  line: number;
  price: number;
  stake_units: number;
  stake_dollars: number | null;
  model_edge: number | null;
  confidence_tier: string | null;
  outcome: string | null;
  profit_units: number | null;
  placed_at: string;
}

export interface UserStats {
  total_bets: number;
  wins: number;
  losses: number;
  pushes: number;
  pending: number;
  total_profit: number;
  roi_pct: number;
  win_rate: number;
}

// Weekly summary types for sidebar widget
export interface EdgeTierPerformance {
  tier: string;
  min_edge: number;
  max_edge: number;
  total: number;
  wins: number;
  losses: number;
  win_rate: number;
}

export interface WeeklySummaryItem {
  season: number;
  week: number;
  total_picks: number;
  wins: number;
  losses: number;
  pushes: number;
  win_rate: number;
  roi_pct: number;
  by_edge_tier: EdgeTierPerformance[];
}

export interface WeeklySummaryResponse {
  weeks: WeeklySummaryItem[];
}

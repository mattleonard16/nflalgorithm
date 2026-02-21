/**
 * TypeScript types for NFL Algorithm API responses
 */

// Explainability payload
export interface WhyPayload {
  model: {
    mu: number | null;
    sigma: number | null;
    context_sensitivity: number | null;
  };
  volume: {
    target_share: number | null;
  };
  volatility: {
    score: number | null;
  };
  confidence: {
    total: number | null;
    tier?: string | null;
    edge_pct?: number | null;
    p_win?: number | null;
  };
  risk: {
    correlation_group: string | null;
    exposure_warning: string | null;
    risk_adjusted_kelly: number | null;
  };
  agents: {
    decision: string | null;
    merged_confidence: number | null;
    votes: Record<string, string> | null;
    top_rationale: string | null;
  };
}

// Data health check result
export interface DataHealthCheck {
  check: string;
  status: "pass" | "warn" | "fail";
  [key: string]: unknown;
}

export interface DataHealth {
  overall: "pass" | "warn" | "fail";
  checks: DataHealthCheck[];
  season: number;
  week: number;
}

// Agent review status
export interface AgentReviewStatus {
  run_id: string;
  reviewed: boolean;
  reviewed_at: string | null;
  decision_count: number;
}

// Pipeline run
export interface PipelineRun {
  run_id: string;
  season: number;
  week: number;
  status: "running" | "completed" | "failed";
  stages_requested: number;
  stages_completed: number;
  error_message: string | null;
  started_at: string;
  finished_at: string | null;
  report_json: Record<string, unknown> | null;
  data_health: DataHealth | null;
}

// Risk & correlation types
export interface CorrelationPlayer {
  player_id: string;
  player_name: string | null;
  market: string;
  team: string | null;
}

export interface CorrelationGroup {
  group: string;
  type: string;
  players: CorrelationPlayer[];
  combined_stake: number;
}

export interface TeamStack {
  team: string;
  count: number;
  player_ids: string[];
}

export interface CorrelationResponse {
  correlation_groups: CorrelationGroup[];
  team_stacks: TeamStack[];
}

export interface Guardrails {
  max_team_exposure: number;
  max_game_exposure: number;
  max_player_exposure: number;
}

export interface ExposureItem {
  team?: string;
  game?: string;
  stake: number;
  fraction: number;
}

export interface RiskWarning {
  player_id: string;
  player_name: string | null;
  warning: string;
}

export interface RiskSummary {
  total_stake: number;
  bankroll: number;
  team_exposure: ExposureItem[];
  game_exposure: ExposureItem[];
  guardrails: Guardrails;
  warnings: RiskWarning[];
}

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
  confidence_tier: "Premium" | "Strong" | "Marginal" | "Pass";
  confidence_score?: number;
  event_id?: string;
  generated_at?: string;
  recommendation?: "BET" | "PASS";
  why?: WhyPayload;
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

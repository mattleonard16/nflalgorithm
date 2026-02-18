export interface NbaProjection {
  player_id: number
  player_name: string
  team: string
  game_date: string
  matchup: string | null
  market: string
  projected_value: number
  confidence: number | null
  stat_last5_avg: number | null
  stat_last10_avg: number | null
  fga_last5_avg: number | null
  min_last5_avg: number | null
}

export interface NbaProjectionsResponse {
  projections: NbaProjection[]
  game_date: string
  games_count: number
  total_players: number
}

export interface NbaGame {
  game_id: string
  game_date: string
  home_team: string
  away_team: string
  status: string | null
  home_score: number | null
  away_score: number | null
}

export interface NbaScheduleResponse {
  games: NbaGame[]
  game_date: string
}

export interface NbaPlayerSummary {
  player_id: number
  player_name: string
  team: string
  games_played: number
  avg_pts: number
  avg_reb: number
  avg_ast: number
  avg_min: number
}

export interface NbaPlayersResponse {
  players: NbaPlayerSummary[]
  season: number
  total: number
}

export interface NbaMeta {
  available_seasons: number[]
  latest_game_date: string | null
  total_players: number
  total_games: number
}

export interface NbaWhyModel {
  projected_value: number | null
  sigma: number | null
  confidence: number | null
}

export interface NbaWhyRecency {
  last5_avg: number | null
  last10_avg: number | null
  trend: "up" | "down" | "stable" | null
}

export interface NbaWhyVariance {
  sigma: number | null
  cv: number | null
}

export interface NbaWhyConfidence {
  p_win: number | null
  edge_percentage: number | null
  expected_roi: number | null
  kelly_fraction: number | null
}

export interface NbaWhyRisk {
  correlation_group: string | null
  exposure_warning: string | null
  risk_adjusted_kelly: number | null
}

export interface NbaWhyAgents {
  decision: string | null
  merged_confidence: number | null
  votes: Record<string, number> | null
  top_rationale: string | null
}

export interface NbaWhyPayload {
  model: NbaWhyModel
  recency: NbaWhyRecency
  variance: NbaWhyVariance
  confidence: NbaWhyConfidence
  risk: NbaWhyRisk
  agents: NbaWhyAgents
}

export interface NbaValueBet {
  player_id: number | null
  player_name: string
  team: string | null
  event_id: string
  market: string
  sportsbook: string
  line: number
  over_price: number
  under_price: number | null
  mu: number
  sigma: number
  p_win: number
  edge_percentage: number
  expected_roi: number
  kelly_fraction: number
  confidence: number | null
  generated_at: string | null
  why?: NbaWhyPayload | null
}

export interface NbaValueBetsResponse {
  bets: NbaValueBet[]
  total: number
  game_date: string
  filters: Record<string, unknown>
}

export interface NbaBetOutcome {
  bet_id: string
  player_name: string | null
  market: string
  line: number
  actual_result: number | null
  result: string | null
  profit_units: number | null
  confidence_tier: string | null
}

export interface NbaDailyPerformance {
  season: number
  game_date: string
  total_bets: number
  wins: number
  losses: number
  pushes: number
  profit_units: number
  roi_pct: number
  avg_edge: number
  best_bet: string | null
  worst_bet: string | null
}

export interface NbaPerformanceResponse {
  total_bets: number
  total_wins: number
  total_losses: number
  total_profit: number
  overall_roi: number
  win_rate: number
  days: NbaDailyPerformance[]
}

export interface NbaCorrelationMember {
  player_id: number
  market: string
  sportsbook: string
}

export interface NbaCorrelationResponse {
  game_date: string
  correlation_groups: Record<string, NbaCorrelationMember[]>
  team_stacks: Record<string, number>
}

export interface NbaRiskSummary {
  game_date: string
  total_assessed: number
  correlated: number
  exposure_flagged: number
  avg_risk_adjusted_kelly: number | null
  avg_drawdown: number | null
  guardrails: string[]
}

export interface NbaExplainResponse {
  player_id: number
  market: string
  game_date: string
  why: NbaWhyPayload
}

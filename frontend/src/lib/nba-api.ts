import type {
  NbaMeta,
  NbaPlayersResponse,
  NbaProjectionsResponse,
  NbaScheduleResponse,
  NbaValueBetsResponse,
  NbaPerformanceResponse,
  NbaBetOutcome,
  NbaCorrelationResponse,
  NbaRiskSummary,
  NbaExplainResponse,
} from "@/lib/nba-types"

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

async function fetchJson<T>(path: string, params?: Record<string, string | number>): Promise<T> {
  const url = new URL(`${BASE}${path}`)
  if (params) {
    Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, String(v)))
  }
  const res = await fetch(url.toString(), { cache: "no-store" })
  if (!res.ok) {
    throw new Error(`NBA API error ${res.status}: ${await res.text()}`)
  }
  return res.json() as Promise<T>
}

export async function getNbaMeta(): Promise<NbaMeta> {
  return fetchJson<NbaMeta>("/api/nba/meta")
}

export async function getNbaSchedule(): Promise<NbaScheduleResponse> {
  return fetchJson<NbaScheduleResponse>("/api/nba/schedule")
}

export async function getNbaProjections(
  gameDate?: string,
  market = "pts",
  minConfidence = 0,
  limit = 50
): Promise<NbaProjectionsResponse> {
  const params: Record<string, string | number> = { market, min_confidence: minConfidence, limit }
  if (gameDate) params.game_date = gameDate
  return fetchJson<NbaProjectionsResponse>("/api/nba/projections", params)
}

export async function getNbaPlayers(
  season = 2025,
  team?: string,
  search?: string,
  limit = 100
): Promise<NbaPlayersResponse> {
  const params: Record<string, string | number> = { season, limit }
  if (team) params.team = team
  if (search) params.search = search
  return fetchJson<NbaPlayersResponse>("/api/nba/players", params)
}

export async function fetchNbaValueBets(params?: {
  game_date?: string
  market?: string
  min_edge?: number
  best_line_only?: boolean
  sportsbook?: string
  include_why?: boolean
}): Promise<NbaValueBetsResponse> {
  const searchParams: Record<string, string | number> = {}
  if (params?.game_date) searchParams.game_date = params.game_date
  if (params?.market) searchParams.market = params.market
  if (params?.min_edge !== undefined) searchParams.min_edge = params.min_edge
  if (params?.best_line_only) searchParams.best_line_only = "true"
  if (params?.sportsbook) searchParams.sportsbook = params.sportsbook
  if (params?.include_why) searchParams.include_why = "true"
  return fetchJson<NbaValueBetsResponse>("/api/nba/value-bets", searchParams)
}

export async function getNbaPerformance(
  season?: number
): Promise<NbaPerformanceResponse> {
  const params: Record<string, string | number> = {}
  if (season !== undefined) params.season = season
  return fetchJson<NbaPerformanceResponse>("/api/nba/performance", params)
}

export async function getNbaOutcomes(
  gameDate: string
): Promise<NbaBetOutcome[]> {
  return fetchJson<NbaBetOutcome[]>("/api/nba/outcomes", { game_date: gameDate })
}

export async function getNbaExplanation(
  playerId: number,
  market: string,
  gameDate?: string
): Promise<NbaExplainResponse> {
  const params: Record<string, string | number> = {}
  if (gameDate) params.game_date = gameDate
  return fetchJson<NbaExplainResponse>(`/api/nba/explain/${playerId}/${market}`, params)
}

export async function getNbaCorrelation(
  gameDate?: string
): Promise<NbaCorrelationResponse> {
  const params: Record<string, string | number> = {}
  if (gameDate) params.game_date = gameDate
  return fetchJson<NbaCorrelationResponse>("/api/nba/analytics/correlation", params)
}

export async function getNbaRiskSummary(
  gameDate?: string
): Promise<NbaRiskSummary> {
  const params: Record<string, string | number> = {}
  if (gameDate) params.game_date = gameDate
  return fetchJson<NbaRiskSummary>("/api/nba/analytics/risk-summary", params)
}

import { test, expect, type Page, type Route } from "@playwright/test";

const emptyPerformance = {
  total_bets: 0,
  total_wins: 0,
  total_losses: 0,
  total_profit: 0,
  overall_roi: 0,
  win_rate: 0,
  weeks: [],
};

const samplePick = {
  player_id: "KC_xavier_worthy",
  player_name: "Xavier Worthy",
  position: "WR",
  team: "KC",
  opponent: "LAC",
  market: "receiving_yards",
  market_label: "Rec",
  // Signed-out contract: projected values are gated and arrive null.
  mu: null,
  sigma: null,
  model_version: "causal_asof_v1",
  generated_at: "2026-08-15T00:00:00Z",
  depth_rank: 1,
  is_starter: true,
  injury_status: null,
  roster_status: "ACT",
};

interface MockOptions {
  projectionWeeks?: { season: number; week: number }[];
  picks?: unknown[];
  metaWeeks?: { season: number; week: number }[];
  projectionWeeksStatus?: number;
  metaStatus?: number;
  operationalAuth?: string[];
}

async function mockDashboardApi(page: Page, requests: string[], options: MockOptions = {}) {
  await page.route("**/api/**", async (route: Route) => {
    const url = new URL(route.request().url());
    requests.push(`${url.pathname}${url.search}`);

    if (url.pathname.startsWith("/api/run/")) {
      options.operationalAuth?.push(route.request().headers()["authorization"] ?? "");
    }

    let body: unknown = {};
    if (url.pathname === "/api/projections/weeks") {
      if (options.projectionWeeksStatus && options.projectionWeeksStatus !== 200) {
        await route.fulfill({
          status: options.projectionWeeksStatus,
          body: "slate lookup failed",
        });
        return;
      }
      body = { available_weeks: options.projectionWeeks ?? [] };
    } else if (url.pathname === "/api/projections") {
      const picks = options.picks ?? [];
      body = { season: 2026, week: 1, total_count: picks.length, values_visible: false, picks };
    } else if (url.pathname === "/api/meta") {
      if (options.metaStatus && options.metaStatus !== 200) {
        await route.fulfill({ status: options.metaStatus, body: "metadata unavailable" });
        return;
      }
      body = { available_weeks: options.metaWeeks ?? [], sportsbooks: [], markets: [] };
    } else if (url.pathname === "/api/value-bets") {
      body = { season: 2026, week: 1, total_count: 0, filtered_count: 0, bets: [] };
    } else if (url.pathname === "/api/performance") {
      body = emptyPerformance;
    } else if (url.pathname === "/api/analytics/correlation") {
      body = { correlation_groups: [], team_stacks: [] };
    } else if (url.pathname === "/api/analytics/risk-summary") {
      body = {
        total_stake: 0,
        bankroll: 1000,
        team_exposure: [],
        game_exposure: [],
        guardrails: {
          max_team_exposure: 0.3,
          max_game_exposure: 0.4,
          max_player_exposure: 0.15,
        },
        warnings: [],
      };
    } else if (url.pathname === "/api/weekly-summary") {
      body = { weeks: [] };
    } else if (url.pathname === "/api/auth/me") {
      body = {
        id: "operator-1",
        email: "operator@example.com",
        name: "Operator",
        subscription_tier: "operator",
        bankroll: 1000,
        created_at: "2026-08-15T00:00:00Z",
      };
    } else if (url.pathname === "/api/run/latest") {
      body = null;
    } else {
      throw new Error(`Unhandled dashboard API request: ${url.pathname}${url.search}`);
    }

    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
  });
}

/**
 * Visibility smoke tests for the Board/Bets split.
 *
 * Catches the worst class of regression: page renders blank, throws on mount,
 * or strips the headline component. Does NOT exercise data integrity — the
 * backend may return empty rows when SQLite is fresh, and the test must still
 * pass in that case.
 */
test.describe("Board slate", () => {
  test("renders header and key sections without console errors", async ({ page }) => {
    const consoleErrors: string[] = [];
    await mockDashboardApi(page, [], {
      projectionWeeks: [{ season: 2026, week: 1 }],
      picks: [samplePick],
    });
    page.on("console", (msg) => {
      if (msg.type() === "error") consoleErrors.push(msg.text());
    });
    page.on("pageerror", (err) => consoleErrors.push(err.message));

    await page.goto("/", { waitUntil: "domcontentloaded" });

    await expect(page.getByRole("heading", { name: "Slate", level: 1 })).toBeVisible();

    // Target the header subtitle specifically — "Week N" also appears in the
    // sidebar performance widget, which would trip strict mode.
    await expect(page.getByText(/Season \d{4} · Week \d+/)).toBeVisible();

    await expect(page.locator("body")).toBeVisible();

    const fatal = consoleErrors.filter(
      (e) => !/Failed to load resource|favicon|hydration|404/i.test(e),
    );
    expect(fatal, `unexpected console errors:\n${fatal.join("\n")}`).toHaveLength(0);
  });

  test("uses the projection week and skips betting requests when signed out", async ({ page }) => {
    const requests: string[] = [];
    await mockDashboardApi(page, requests, {
      projectionWeeks: [{ season: 2026, week: 1 }],
      picks: [samplePick],
    });

    await page.goto("/", { waitUntil: "domcontentloaded" });
    await expect(page.getByText("Season 2026 · Week 1 ·")).toBeVisible();
    await expect
      .poll(() => requests.some((request) => request.startsWith("/api/projections?")))
      .toBe(true);

    const weekRequests = requests.filter((request) => request.startsWith("/api/projections?"));
    expect(weekRequests).not.toHaveLength(0);
    expect(
      weekRequests.every(
        (request) => request.includes("season=2026") && request.includes("week=1")
      )
    ).toBe(true);
    expect(requests.some((request) => request.startsWith("/api/value-bets"))).toBe(false);
    expect(requests.some((request) => request.startsWith("/api/run/latest"))).toBe(false);
    await expect(page.getByRole("button", { name: "Refresh" })).toHaveCount(0);
  });

  test("shows an empty slate state without week-scoped requests", async ({ page }) => {
    const requests: string[] = [];
    await mockDashboardApi(page, requests, { projectionWeeks: [] });

    await page.goto("/", { waitUntil: "domcontentloaded" });
    await expect(page.getByText("No algorithm slate is available yet").first()).toBeVisible();
    await expect(page.getByText("No published NFL card is available yet")).toHaveCount(0);

    expect(requests.some((request) => request.startsWith("/api/projections?"))).toBe(false);
    expect(
      requests.some((request) =>
        /\/api\/(value-bets|performance|analytics\/correlation|analytics\/risk-summary|run\/latest)/.test(
          request
        )
      )
    ).toBe(false);
  });

  test("shows projection-week failure without claiming there is no published card", async ({
    page,
  }) => {
    const requests: string[] = [];
    await mockDashboardApi(page, requests, { projectionWeeksStatus: 500 });

    await page.goto("/", { waitUntil: "domcontentloaded" });

    await expect(page.getByText("Projection weeks unavailable")).toBeVisible();
    await expect(page.getByText(/API Error \(500\)/)).toBeVisible();
    await expect(page.getByText("No published NFL card is available yet")).toHaveCount(0);
    await expect(page.getByText("No algorithm slate is available yet")).toHaveCount(0);
    expect(requests.some((request) => request.startsWith("/api/projections?"))).toBe(false);
    expect(requests.some((request) => request.startsWith("/api/value-bets"))).toBe(false);
  });
});

const sampleHealth = {
  feeds: [
    { feed: "odds", season: 2026, week: 1, as_of: "2026-08-15T00:00:00Z" },
    { feed: "projections", season: 2026, week: 1, as_of: "2026-08-15T00:00:00Z" },
  ],
  overall_status: "ACTIVE",
};

const sampleArchitecture = {
  database_backend: "sqlite",
  queue: { queued: 1, running: 1, retry_scheduled: 0 },
  workers_active: 1,
  artifact_count: 4,
  decision_count: 2,
  read_model_rows: 120,
  recent_runs: [
    {
      run_id: "run-abcdef123456",
      job_id: "job-1",
      season: 2026,
      week: 1,
      status: "running",
      stages_requested: 6,
      stages_completed: 2,
      error_message: null,
      started_at: "2026-08-15T00:00:00Z",
      finished_at: null,
      report_json: null,
      data_health: null,
      source: "api",
      attempts: 1,
      max_attempts: 3,
      worker_id: "worker-1",
      cancel_requested: false,
      available_at: null,
      stages: [
        {
          name: "prepare_week",
          ordinal: 0,
          status: "completed",
          attempt: 1,
          started_at: "2026-08-15T00:00:00Z",
          finished_at: "2026-08-15T00:01:00Z",
          result: null,
          error_message: null,
        },
        {
          name: "odds",
          ordinal: 1,
          status: "running",
          attempt: 1,
          started_at: "2026-08-15T00:01:00Z",
          finished_at: null,
          result: null,
          error_message: null,
        },
      ],
    },
  ],
  levels: [
    { id: "entry", level: 1, title: "Entry Points", tone: "blue", nodes: ["Next.js Dashboard", "CLI", "Scheduler"] },
    { id: "control", level: 2, title: "API + Job Control", tone: "blue", nodes: ["FastAPI", "Pipeline Job Service"] },
    { id: "execution", level: 3, title: "Execution", tone: "amber", nodes: ["Job Queue", "NFL Worker", "Shared Orchestrator"] },
    {
      id: "pipeline",
      level: 4,
      title: "NFL Pipeline",
      tone: "green",
      nodes: ["Prepare Data", "Validate Pregame", "Generate Projections", "Fetch Live Odds"],
    },
    {
      id: "decision",
      level: 5,
      title: "Betting Decision",
      tone: "purple",
      nodes: ["Value Engine", "Confidence + Risk", "Specialist Agents", "Final Betting Card"],
    },
    {
      id: "persistence",
      level: 6,
      title: "Persistence",
      tone: "amber",
      nodes: ["Operational Database", "Artifact Storage", "API Read Models"],
    },
  ],
};

async function mockSystemApi(page: Page, { authenticated }: { authenticated: boolean }) {
  await page.route("**/api/**", async (route: Route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/api/health") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(sampleHealth),
      });
      return;
    }
    if (url.pathname === "/api/system/architecture") {
      if (!authenticated) {
        await route.fulfill({ status: 401, body: "Pipeline control requires authentication" });
        return;
      }
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(sampleArchitecture),
      });
      return;
    }
    if (url.pathname === "/api/weekly-summary") {
      // Rendered by the sidebar on every route; unmocked it throws on mount.
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ weeks: [] }),
      });
      return;
    }
    if (url.pathname === "/api/auth/me") {
      if (!authenticated) {
        await route.fulfill({ status: 401, body: "unauthenticated" });
        return;
      }
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          id: "operator-1",
          email: "operator@example.com",
          name: "Operator",
          subscription_tier: "operator",
          bankroll: 1000,
          created_at: "2026-08-15T00:00:00Z",
        }),
      });
      return;
    }
    await route.fulfill({ status: 200, contentType: "application/json", body: "{}" });
  });
}

test.describe("System control room", () => {
  test("signed-out visitors get a sign-in state, not a false outage", async ({ page }) => {
    await mockSystemApi(page, { authenticated: false });

    await page.goto("/system", { waitUntil: "domcontentloaded" });

    await expect(page.getByText("Sign in with an operator account to view live jobs.").first()).toBeVisible();

    // The public health endpoint still answers, so the API must not read Offline
    // and feed freshness must survive the architecture 401.
    await expect(page.getByText("Online")).toBeVisible();
    await expect(page.getByText("Offline")).toHaveCount(0);
    await expect(page.getByText("odds")).toBeVisible();
    await expect(page.getByText("No freshness snapshots have been persisted.")).toHaveCount(0);

    // No red banner spamming the raw 401, and no claim the control plane is broken.
    // Scoped to <main> so the Next.js dev overlay doesn't count as an alert.
    await expect(page.locator("main").getByRole("alert")).toHaveCount(0);
    await expect(page.getByText(/API Error \(401\)/)).toHaveCount(0);
    await expect(page.getByText("Architecture state unavailable")).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Cancel" })).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Retry" })).toHaveCount(0);
  });

  test("operators see the live topology, running stage, and run controls", async ({ page }) => {
    await page.addInitScript(() => localStorage.setItem("session_id", "test-session"));
    await mockSystemApi(page, { authenticated: true });

    await page.goto("/system", { waitUntil: "domcontentloaded" });

    await expect(page.getByRole("heading", { name: "NFL scalable architecture" })).toBeVisible();
    await expect(page.getByLabel("Fetch Live Odds: active")).toBeVisible();
    await expect(page.getByLabel("Prepare Data: complete")).toBeVisible();
    await expect(page.getByText("run-abcd")).toBeVisible();
    await expect(page.getByRole("button", { name: "Cancel" })).toBeVisible();
    await expect(page.getByText("Sign in with an operator account to view live jobs.")).toHaveCount(0);
  });
});

test.describe("Bets published card", () => {
  test("uses the metadata week and skips operational requests when signed out", async ({
    page,
  }) => {
    const requests: string[] = [];
    await mockDashboardApi(page, requests, { metaWeeks: [{ season: 2026, week: 1 }] });

    await page.goto("/bets", { waitUntil: "domcontentloaded" });
    await expect(page.getByRole("heading", { name: "Value Card", level: 1 })).toBeVisible();
    await expect(page.getByText("Season 2026 · Week 1 ·")).toBeVisible();
    await expect
      .poll(() => requests.some((request) => request.startsWith("/api/value-bets")))
      .toBe(true);

    const weekRequests = requests.filter((request) =>
      /\/api\/(value-bets|analytics\/correlation|analytics\/risk-summary)/.test(request)
    );
    expect(weekRequests).not.toHaveLength(0);
    expect(
      weekRequests.every(
        (request) => request.includes("season=2026") && request.includes("week=1")
      )
    ).toBe(true);
    expect(requests.some((request) => request.startsWith("/api/run/latest"))).toBe(false);
    await expect(page.getByRole("button", { name: "Refresh" })).toHaveCount(0);
  });

  test("loads run status after metadata for an authenticated user", async ({ page }) => {
    const requests: string[] = [];
    const operationalAuth: string[] = [];
    await page.addInitScript(() => localStorage.setItem("session_id", "test-session"));
    await mockDashboardApi(page, requests, {
      metaWeeks: [{ season: 2026, week: 1 }],
      operationalAuth,
    });

    await page.goto("/bets", { waitUntil: "domcontentloaded" });
    await expect(page.getByRole("button", { name: "Refresh" })).toBeVisible();
    await expect.poll(() => requests.find((request) => request.startsWith("/api/run/latest"))).toBe(
      "/api/run/latest?season=2026&week=1"
    );
    expect(operationalAuth).toContain("Bearer test-session");
  });

  test("shows an empty card state without week-scoped requests", async ({ page }) => {
    const requests: string[] = [];
    await mockDashboardApi(page, requests, { metaWeeks: [] });

    await page.goto("/bets", { waitUntil: "domcontentloaded" });
    await expect(page.getByText("No published NFL card is available yet").first()).toBeVisible();

    expect(
      requests.some((request) =>
        /\/api\/(value-bets|performance|analytics\/correlation|analytics\/risk-summary|run\/latest)/.test(
          request
        )
      )
    ).toBe(false);
  });

  test("shows metadata failure without claiming there is no published card", async ({ page }) => {
    const requests: string[] = [];
    await mockDashboardApi(page, requests, { metaStatus: 500 });

    await page.goto("/bets", { waitUntil: "domcontentloaded" });

    await expect(page.getByText("Published weeks unavailable")).toBeVisible();
    await expect(page.getByText(/API Error \(500\)/)).toBeVisible();
    await expect(page.getByText("No published NFL card is available yet")).toHaveCount(0);
    expect(
      requests.some((request) =>
        /\/api\/(value-bets|performance|analytics\/correlation|analytics\/risk-summary|run\/latest)/.test(
          request
        )
      )
    ).toBe(false);
  });
});

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

async function mockDashboardApi(
  page: Page,
  availableWeeks: { season: number; week: number }[],
  requests: string[],
  options: { metaStatus?: number; operationalAuth?: string[] } = {},
) {
  await page.route("**/api/**", async (route: Route) => {
    const url = new URL(route.request().url());
    requests.push(`${url.pathname}${url.search}`);

    if (url.pathname.startsWith("/api/run/")) {
      options.operationalAuth?.push(route.request().headers()["authorization"] ?? "");
    }

    let body: unknown = {};
    if (url.pathname === "/api/meta") {
      if (options.metaStatus && options.metaStatus !== 200) {
        await route.fulfill({ status: options.metaStatus, body: "metadata unavailable" });
        return;
      }
      body = { available_weeks: availableWeeks, sportsbooks: [], markets: [] };
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
 * Visibility smoke test for the value-bet dashboard.
 *
 * Catches the worst class of regression: page renders blank, throws on mount,
 * or strips the headline component. Does NOT exercise data integrity — the
 * backend may return empty rows when SQLite is fresh, and the test must still
 * pass in that case.
 */
test.describe("Value Dashboard", () => {
  test("renders header and key sections without console errors", async ({ page }) => {
    const consoleErrors: string[] = [];
    await mockDashboardApi(page, [{ season: 2026, week: 1 }], []);
    page.on("console", (msg) => {
      if (msg.type() === "error") consoleErrors.push(msg.text());
    });
    page.on("pageerror", (err) => consoleErrors.push(err.message));

    await page.goto("/", { waitUntil: "domcontentloaded" });

    await expect(page.getByRole("heading", { name: "Value Dashboard", level: 1 })).toBeVisible();

    // Target the header subtitle specifically — "Week N" also appears in the
    // sidebar performance widget, which would trip strict mode.
    await expect(page.getByText(/Season \d{4} · Week \d+/)).toBeVisible();

    await expect(page.locator("body")).toBeVisible();

    const fatal = consoleErrors.filter(
      (e) => !/Failed to load resource|favicon|hydration|404/i.test(e),
    );
    expect(fatal, `unexpected console errors:\n${fatal.join("\n")}`).toHaveLength(0);
  });

  test("uses the metadata week and skips operational requests when signed out", async ({ page }) => {
    const requests: string[] = [];
    await mockDashboardApi(page, [{ season: 2026, week: 1 }], requests);

    await page.goto("/", { waitUntil: "domcontentloaded" });
    await expect(page.getByText("Season 2026 · Week 1 ·")).toBeVisible();
    await expect.poll(() => requests.some((request) => request.startsWith("/api/value-bets"))).toBe(true);

    const weekRequests = requests.filter((request) =>
      /\/api\/(value-bets|analytics\/correlation|analytics\/risk-summary)/.test(request)
    );
    expect(weekRequests).not.toHaveLength(0);
    expect(weekRequests.every((request) => request.includes("season=2026") && request.includes("week=1"))).toBe(true);
    expect(requests.some((request) => request.includes("season=2025") || request.includes("week=13"))).toBe(false);
    expect(requests.some((request) => request.startsWith("/api/run/latest"))).toBe(false);
    await expect(page.getByRole("button", { name: "Refresh" })).toHaveCount(0);
  });

  test("loads run status after metadata for an authenticated user", async ({ page }) => {
    const requests: string[] = [];
    const operationalAuth: string[] = [];
    await page.addInitScript(() => localStorage.setItem("session_id", "test-session"));
    await mockDashboardApi(page, [{ season: 2026, week: 1 }], requests, {
      operationalAuth,
    });

    await page.goto("/", { waitUntil: "domcontentloaded" });
    await expect(page.getByRole("button", { name: "Refresh" })).toBeVisible();
    await expect.poll(() => requests.find((request) => request.startsWith("/api/run/latest"))).toBe(
      "/api/run/latest?season=2026&week=1"
    );
    expect(operationalAuth).toContain("Bearer test-session");
  });

  test("shows an empty state without week-scoped requests", async ({ page }) => {
    const requests: string[] = [];
    await mockDashboardApi(page, [], requests);

    await page.goto("/", { waitUntil: "domcontentloaded" });
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
    await mockDashboardApi(page, [], requests, { metaStatus: 500 });

    await page.goto("/", { waitUntil: "domcontentloaded" });

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

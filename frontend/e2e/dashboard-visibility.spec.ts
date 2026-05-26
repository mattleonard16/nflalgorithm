import { test, expect } from "@playwright/test";

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
    page.on("console", (msg) => {
      if (msg.type() === "error") consoleErrors.push(msg.text());
    });
    page.on("pageerror", (err) => consoleErrors.push(err.message));

    await page.goto("/", { waitUntil: "domcontentloaded" });

    await expect(page.getByRole("heading", { name: "Value Dashboard", level: 1 })).toBeVisible();

    await expect(page.getByText(/Season \d{4}/)).toBeVisible();
    await expect(page.getByText(/Week \d+/)).toBeVisible();

    await expect(page.locator("body")).toBeVisible();

    const fatal = consoleErrors.filter(
      (e) => !/Failed to load resource|favicon|hydration|404/i.test(e),
    );
    expect(fatal, `unexpected console errors:\n${fatal.join("\n")}`).toHaveLength(0);
  });
});

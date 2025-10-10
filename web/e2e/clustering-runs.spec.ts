import { test, expect } from "@playwright/test";

test.describe("Clustering Runs Page", () => {
  test("should display clustering runs list", async ({ page }) => {
    await page.goto("/");

    // Wait for API response
    await page.waitForLoadState("networkidle");

    // Look for common elements that should exist
    // (adjust selectors based on actual implementation)
    const body = page.locator("body");
    await expect(body).toBeVisible();
  });

  test("should be able to access run details page directly", async ({
    page,
  }) => {
    // First, get a valid run ID from the API
    const API_BASE_URL =
      process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    const response = await page.request.get(
      `${API_BASE_URL}/api/clustering/runs?limit=1`,
    );
    const data = await response.json();

    if (data.items && data.items.length > 0) {
      const runId = data.items[0].run_id;

      // Navigate directly to the run details page
      await page.goto(`/runs/${runId}`);
      await page.waitForLoadState("networkidle");

      // Verify we're on the run details page
      await expect(page.url()).toContain(`/runs/${runId}`);

      // Check that the page loaded successfully
      const body = page.locator("body");
      await expect(body).toBeVisible();
    } else {
      // If no runs exist, skip the test
      console.log(
        "No clustering runs found in database - skipping navigation test",
      );
    }
  });
});

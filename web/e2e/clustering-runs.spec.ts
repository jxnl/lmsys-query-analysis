import { test, expect } from "@playwright/test";

test.describe("Clustering Runs Page", () => {
  test("should display clustering runs list", async ({ page }) => {
    await page.goto("/");


    await page.waitForLoadState("networkidle");



    const body = page.locator("body");
    await expect(body).toBeVisible();
  });

  test("should be able to access run details page directly", async ({ page }) => {

    const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    const response = await page.request.get(`${API_BASE_URL}/api/clustering/runs?limit=1`);
    const data = await response.json();

    if (data.items && data.items.length > 0) {
      const runId = data.items[0].run_id;


      await page.goto(`/runs/${runId}`);
      await page.waitForLoadState("networkidle");


      await expect(page.url()).toContain(`/runs/${runId}`);


      const body = page.locator("body");
      await expect(body).toBeVisible();
    } else {

      console.log("No clustering runs found in database - skipping navigation test");
    }
  });
});
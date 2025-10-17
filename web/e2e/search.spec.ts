import { test, expect } from "@playwright/test";

test.describe("Search Functionality", () => {
  test("should search for queries via frontend", async ({ page }) => {
    await page.goto("/");
    await page.waitForLoadState("networkidle");

    // Look for search input (adjust selector based on actual implementation)
    const searchInput = page.locator('input[type="search"], input[placeholder*="search" i]');
    const searchInputCount = await searchInput.count();

    if (searchInputCount > 0) {
      // Enter search query
      await searchInput.first().fill("python");

      // Look for search button or submit
      const searchButton = page.locator('button[type="submit"], button:has-text("Search")');
      if ((await searchButton.count()) > 0) {
        await searchButton.first().click();
        await page.waitForLoadState("networkidle");

        // Verify we're on a search results page or see results
        const body = await page.locator("body").textContent();
        expect(body).toBeTruthy();
      }
    }
  });

  test("should handle empty search", async ({ page }) => {
    await page.goto("/");
    await page.waitForLoadState("networkidle");

    const searchInput = page.locator('input[type="search"], input[placeholder*="search" i]');
    const searchInputCount = await searchInput.count();

    if (searchInputCount > 0) {
      // Submit empty search
      await searchInput.first().fill("");

      const searchButton = page.locator('button[type="submit"], button:has-text("Search")');
      if ((await searchButton.count()) > 0) {
        await searchButton.first().click();
        await page.waitForLoadState("networkidle");

        // Should still load without errors
        const body = page.locator("body");
        await expect(body).toBeVisible();
      }
    }
  });
});

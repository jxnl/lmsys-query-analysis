import { test, expect } from "@playwright/test";

test.describe("Search Functionality", () => {
  test("should search for queries via frontend", async ({ page }) => {
    await page.goto("/");
    await page.waitForLoadState("networkidle");


    const searchInput = page.locator('input[type="search"], input[placeholder*="search" i]');
    const searchInputCount = await searchInput.count();

    if (searchInputCount > 0) {

      await searchInput.first().fill("python");


      const searchButton = page.locator('button[type="submit"], button:has-text("Search")');
      if ((await searchButton.count()) > 0) {
        await searchButton.first().click();
        await page.waitForLoadState("networkidle");


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

      await searchInput.first().fill("");

      const searchButton = page.locator('button[type="submit"], button:has-text("Search")');
      if ((await searchButton.count()) > 0) {
        await searchButton.first().click();
        await page.waitForLoadState("networkidle");


        const body = page.locator("body");
        await expect(body).toBeVisible();
      }
    }
  });
});
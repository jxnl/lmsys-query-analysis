import { test, expect } from "@playwright/test";

test.describe("Homepage", () => {
  test("should load homepage successfully", async ({ page }) => {
    await page.goto("/");


    await expect(page).toHaveTitle(/LMSYS Query Analysis/);


    await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  });

  test("should display navigation elements", async ({ page }) => {
    await page.goto("/");


    const main = page.locator("main");
    await expect(main).toBeVisible();
  });

  test("should be responsive", async ({ page }) => {

    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto("/");
    await expect(page.locator("main")).toBeVisible();


    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.goto("/");
    await expect(page.locator("main")).toBeVisible();
  });
});
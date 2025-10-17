import { test, expect } from "@playwright/test";

test.describe("API Integration", () => {
  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  test("should connect to FastAPI backend", async ({ request }) => {
    const response = await request.get(`${API_BASE_URL}/api/health`);
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    expect(data.status).toBe("ok");
    expect(data.service).toBe("lmsys-query-analysis");
  });

  test("should fetch clustering runs", async ({ request }) => {
    const response = await request.get(`${API_BASE_URL}/api/clustering/runs?limit=10`);
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    expect(data).toHaveProperty("items");
    expect(data).toHaveProperty("total");
    expect(data).toHaveProperty("page");
    expect(Array.isArray(data.items)).toBeTruthy();
  });

  test("should fetch queries", async ({ request }) => {
    const response = await request.get(`${API_BASE_URL}/api/queries?limit=10`);
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    expect(data).toHaveProperty("items");
    expect(data).toHaveProperty("total");
    expect(Array.isArray(data.items)).toBeTruthy();
  });

  test("should perform fulltext search on queries", async ({ request }) => {
    const response = await request.get(
      `${API_BASE_URL}/api/search/queries?text=python&mode=fulltext&limit=5`
    );
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    expect(data).toHaveProperty("items");
    expect(Array.isArray(data.items)).toBeTruthy();
  });

  test("should handle 404 for non-existent run", async ({ request }) => {
    const response = await request.get(`${API_BASE_URL}/api/clustering/runs/nonexistent-run-id`);
    expect(response.status()).toBe(404);

    const data = await response.json();
    expect(data).toHaveProperty("detail");
    expect(data.detail).toHaveProperty("error");
    expect(data.detail.error.type).toBe("NotFound");
  });

  test("should return 501 for unimplemented POST endpoints", async ({ request }) => {
    const response = await request.post(`${API_BASE_URL}/api/clustering/kmeans`);
    expect(response.status()).toBe(501);

    const data = await response.json();
    expect(data).toHaveProperty("detail");
    expect(data.detail).toHaveProperty("error");
    expect(data.detail.error.type).toBe("NotImplemented");
  });
});

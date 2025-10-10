/**
 * FastAPI client for Next.js Server Actions
 *
 * This client wraps all calls to the FastAPI backend running on port 8000.
 * It replaces direct database access with HTTP calls to the API.
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

/**
 * Convert snake_case keys to camelCase recursively
 */
function toCamelCase(obj: any): any {
  if (Array.isArray(obj)) {
    return obj.map(toCamelCase);
  } else if (obj !== null && typeof obj === 'object') {
    return Object.keys(obj).reduce((result, key) => {
      // Convert snake_case to camelCase
      const camelKey = key.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());
      result[camelKey] = toCamelCase(obj[key]);
      return result;
    }, {} as any);
  }
  return obj;
}

/**
 * Generic API fetch wrapper with error handling and Next.js caching
 *
 * @param endpoint - API endpoint path
 * @param options - Fetch options with Next.js extensions
 */
async function apiFetch<T>(endpoint: string, options?: RequestInit & {
  next?: {
    revalidate?: number | false;
    tags?: string[];
  };
}): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;

  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
      // Default caching: cache for 60 seconds (can be overridden per endpoint)
      next: options?.next ?? { revalidate: 60 },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(
        errorData?.detail?.error?.message ||
        `API request failed: ${response.status} ${response.statusText}`
      );
    }

    const data = await response.json();
    // Convert snake_case to camelCase to match frontend types
    return toCamelCase(data) as T;
  } catch (error) {
    console.error(`API fetch error for ${endpoint}:`, error);
    throw error;
  }
}

/**
 * Clustering Runs API
 */
export const clusteringApi = {
  /**
   * Get all clustering runs
   * Cached for 5 minutes - clustering runs change infrequently
   */
  async listRuns(params?: { limit?: number; page?: number }) {
    const queryParams = new URLSearchParams();
    if (params?.limit) queryParams.set('limit', String(params.limit));
    if (params?.page) queryParams.set('page', String(params.page));

    const query = queryParams.toString();
    return apiFetch<{
      items: any[];
      total: number;
      page: number;
      pages: number;
    }>(`/api/clustering/runs${query ? `?${query}` : ''}`, {
      next: { revalidate: 300, tags: ['clustering-runs'] },
    });
  },

  /**
   * Get a single clustering run
   * Cached for 10 minutes - run details don't change
   */
  async getRun(runId: string) {
    return apiFetch<any>(`/api/clustering/runs/${runId}`, {
      next: { revalidate: 600, tags: ['clustering-runs', `run-${runId}`] },
    });
  },

  /**
   * Get clusters for a run
   */
  async listClusters(runId: string, params?: {
    include_counts?: boolean;
    include_percentages?: boolean;
  }) {
    const queryParams = new URLSearchParams();
    if (params?.include_counts !== undefined) {
      queryParams.set('include_counts', String(params.include_counts));
    }
    if (params?.include_percentages !== undefined) {
      queryParams.set('include_percentages', String(params.include_percentages));
    }

    const query = queryParams.toString();
    return apiFetch<{
      items: any[];
      total: number;
    }>(`/api/clustering/runs/${runId}/clusters${query ? `?${query}` : ''}`);
  },
};

/**
 * Queries API
 */
export const queriesApi = {
  /**
   * List queries with optional filtering
   */
  async listQueries(params?: {
    run_id?: string;
    cluster_id?: number;
    model?: string;
    page?: number;
    limit?: number;
  }) {
    const queryParams = new URLSearchParams();
    if (params?.run_id) queryParams.set('run_id', params.run_id);
    if (params?.cluster_id !== undefined) queryParams.set('cluster_id', String(params.cluster_id));
    if (params?.model) queryParams.set('model', params.model);
    if (params?.page) queryParams.set('page', String(params.page));
    if (params?.limit) queryParams.set('limit', String(params.limit));

    const query = queryParams.toString();
    return apiFetch<{
      items: any[];
      total: number;
      page: number;
      pages: number;
      limit: number;
    }>(`/api/queries${query ? `?${query}` : ''}`);
  },

  /**
   * Get cluster detail with queries
   */
  async getClusterDetail(runId: string, clusterId: number, params?: {
    page?: number;
    limit?: number;
  }) {
    const queryParams = new URLSearchParams();
    if (params?.page) queryParams.set('page', String(params.page));
    if (params?.limit) queryParams.set('limit', String(params.limit));

    const query = queryParams.toString();
    return apiFetch<{
      cluster: any;
      queries: {
        items: any[];
        total: number;
        page: number;
        pages: number;
        limit: number;
      };
    }>(`/api/clustering/runs/${runId}/clusters/${clusterId}${query ? `?${query}` : ''}`);
  },
};

/**
 * Search API
 */
export const searchApi = {
  /**
   * Search queries
   */
  async searchQueries(params: {
    text: string;
    mode?: 'semantic' | 'fulltext';
    run_id?: string;
    limit?: number;
    page?: number;
  }) {
    const queryParams = new URLSearchParams();
    queryParams.set('text', params.text);
    if (params.mode) queryParams.set('mode', params.mode);
    if (params.run_id) queryParams.set('run_id', params.run_id);
    if (params.limit) queryParams.set('limit', String(params.limit));
    if (params.page) queryParams.set('page', String(params.page));

    return apiFetch<{
      items: any[];
      total: number;
      page?: number;
      pages?: number;
    }>(`/api/search/queries?${queryParams.toString()}`);
  },

  /**
   * Search clusters
   */
  async searchClusters(params: {
    text: string;
    mode?: 'semantic' | 'fulltext';
    run_id?: string;
    limit?: number;
  }) {
    const queryParams = new URLSearchParams();
    queryParams.set('text', params.text);
    if (params.mode) queryParams.set('mode', params.mode);
    if (params.run_id) queryParams.set('run_id', params.run_id);
    if (params.limit) queryParams.set('limit', String(params.limit));

    return apiFetch<{
      items: any[];
      total: number;
    }>(`/api/search/clusters?${queryParams.toString()}`);
  },
};

/**
 * Hierarchy API
 */
export const hierarchyApi = {
  /**
   * List all hierarchies
   */
  async listHierarchies(params?: { run_id?: string }) {
    const queryParams = new URLSearchParams();
    if (params?.run_id) queryParams.set('run_id', params.run_id);

    const query = queryParams.toString();
    return apiFetch<{
      items: any[];
      total: number;
    }>(`/api/hierarchy/${query ? `?${query}` : ''}`);
  },

  /**
   * Get hierarchy tree
   * No caching during development to ensure fresh data
   */
  async getHierarchyTree(hierarchyRunId: string) {
    return apiFetch<{
      nodes: any[];
      hierarchy_run_id: string;
      run_id: string;
      total_queries: number;
    }>(`/api/hierarchy/${hierarchyRunId}`, {
      next: { revalidate: 0 }, // Disable caching for development
    });
  },
};

/**
 * Summaries API
 */
export const summariesApi = {
  /**
   * List summary runs
   */
  async listSummaries(params?: { run_id?: string }) {
    const queryParams = new URLSearchParams();
    if (params?.run_id) queryParams.set('run_id', params.run_id);

    const query = queryParams.toString();
    return apiFetch<{
      items: any[];
      total: number;
    }>(`/api/summaries/${query ? `?${query}` : ''}`);
  },

  /**
   * Get cluster summary
   */
  async getClusterSummary(summaryRunId: string, clusterId: number) {
    return apiFetch<any>(`/api/summaries/${summaryRunId}/clusters/${clusterId}`);
  },
};

/**
 * Curation API
 */
export const curationApi = {
  /**
   * Get cluster metadata
   */
  async getClusterMetadata(runId: string, clusterId: number) {
    return apiFetch<any>(`/api/curation/clusters/${clusterId}/metadata?run_id=${runId}`);
  },

  /**
   * Get cluster edit history
   */
  async getClusterHistory(runId: string, clusterId: number) {
    return apiFetch<{
      items: any[];
      total: number;
    }>(`/api/curation/clusters/${clusterId}/history?run_id=${runId}`);
  },

  /**
   * Get audit log
   */
  async getAuditLog(runId: string) {
    return apiFetch<{
      items: any[];
      total: number;
    }>(`/api/curation/runs/${runId}/audit`);
  },

  /**
   * Get orphaned queries
   */
  async getOrphanedQueries(runId: string) {
    return apiFetch<{
      items: any[];
      total: number;
    }>(`/api/curation/runs/${runId}/orphaned`);
  },
};

/**
 * Health check
 */
export async function healthCheck() {
  return apiFetch<{ status: string; service: string }>('/api/health');
}

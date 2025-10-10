'use server'

/**
 * Server Actions that use the FastAPI backend instead of direct database access.
 *
 * These actions replace the old actions.ts file that used Drizzle ORM directly.
 * All data now flows through the FastAPI backend on port 8000.
 */

import {
  clusteringApi,
  queriesApi,
  searchApi,
  hierarchyApi,
  summariesApi,
  curationApi,
} from '@/lib/api/client';
import type {
  ClusteringRun,
  ClusterHierarchy,
  Query,
  PaginatedQueries,
  ClusterSummary,
} from '@/lib/types';

/**
 * Get all clustering runs, ordered by creation date (newest first)
 */
export async function getRuns(): Promise<ClusteringRun[]> {
  try {
    const response = await clusteringApi.listRuns({ limit: 100 });
    return response.items as ClusteringRun[];
  } catch (error) {
    console.error('Error fetching runs:', error);
    return [];
  }
}

/**
 * Get a single clustering run by ID
 */
export async function getRun(runId: string): Promise<ClusteringRun | null> {
  try {
    const run = await clusteringApi.getRun(runId);
    return run as ClusteringRun;
  } catch (error) {
    console.error('Error fetching run:', error);
    return null;
  }
}

/**
 * Get all hierarchies for a given run
 * Note: This requires filtering on the client side since the API doesn't have this exact endpoint
 */
export async function getHierarchiesForRun(runId: string) {
  try {
    const response = await hierarchyApi.listHierarchies();
    // Filter hierarchies for this run
    const hierarchies = response.items.filter((h: any) => h.run_id === runId);

    // Group by hierarchy_run_id to get unique hierarchies
    const uniqueHierarchies = new Map();
    for (const h of hierarchies) {
      if (!uniqueHierarchies.has(h.hierarchy_run_id)) {
        uniqueHierarchies.set(h.hierarchy_run_id, {
          hierarchyRunId: h.hierarchy_run_id,
          createdAt: h.created_at,
        });
      }
    }

    return Array.from(uniqueHierarchies.values());
  } catch (error) {
    console.error('Error fetching hierarchies for run:', error);
    return [];
  }
}

/**
 * Get full hierarchy tree for a hierarchy run
 */
export async function getHierarchyTree(hierarchyRunId: string): Promise<ClusterHierarchy[]> {
  try {
    const response = await hierarchyApi.getHierarchyTree(hierarchyRunId);
    return response.nodes as ClusterHierarchy[];
  } catch (error) {
    console.error('Error fetching hierarchy tree:', error);
    return [];
  }
}

/**
 * Get cluster summary for a specific cluster
 */
export async function getClusterSummary(
  runId: string,
  clusterId: number,
  alias?: string
): Promise<ClusterSummary | null> {
  try {
    // Use the cluster detail endpoint which includes summary
    const response = await queriesApi.getClusterDetail(runId, clusterId);
    return response.cluster as ClusterSummary;
  } catch (error) {
    console.error('Error fetching cluster summary:', error);
    return null;
  }
}

/**
 * Get paginated queries for a cluster
 */
export async function getClusterQueries(
  runId: string,
  clusterId: number,
  page: number = 1,
  limit: number = 50
): Promise<PaginatedQueries> {
  console.log('[getClusterQueries] runId:', runId, 'clusterId:', clusterId, 'page:', page);

  try {
    const response = await queriesApi.getClusterDetail(runId, clusterId, { page, limit });

    console.log('[getClusterQueries] Found', response.queries.items.length, 'queries');
    console.log('[getClusterQueries] Total:', response.queries.total, 'Pages:', response.queries.pages);

    return {
      queries: response.queries.items as Query[],
      total: response.queries.total,
      page: response.queries.page,
      pages: response.queries.pages,
      limit: response.queries.limit,
    };
  } catch (error) {
    console.error('Error fetching cluster queries:', error);
    return {
      queries: [],
      total: 0,
      page: 1,
      pages: 0,
      limit,
    };
  }
}

/**
 * Search clusters using full-text search on titles and descriptions
 */
export async function searchClusters(query: string, runId: string, nResults: number = 20) {
  try {
    const response = await searchApi.searchClusters({
      text: query,
      mode: 'fulltext',
      run_id: runId,
      limit: nResults,
    });

    return response.items;
  } catch (error) {
    console.error('Error searching clusters:', error);
    return [];
  }
}

/**
 * Get all summary aliases for a run
 */
export async function getSummaryAliases(runId: string) {
  try {
    const response = await summariesApi.listSummaries({ run_id: runId });

    // Group by alias
    const aliasMap = new Map();
    for (const summary of response.items) {
      if (!aliasMap.has(summary.alias)) {
        aliasMap.set(summary.alias, {
          alias: summary.alias,
          summaryRunId: summary.summary_run_id,
          model: summary.model,
        });
      }
    }

    return Array.from(aliasMap.values());
  } catch (error) {
    console.error('Error fetching summary aliases:', error);
    return [];
  }
}

/**
 * Get query counts for all clusters in a run
 */
export async function getClusterQueryCounts(runId: string): Promise<Record<number, number>> {
  try {
    const response = await clusteringApi.listClusters(runId, {
      include_counts: true,
      include_percentages: false,
    });

    const counts: Record<number, number> = {};
    for (const cluster of response.items) {
      if (cluster.query_count !== undefined) {
        counts[cluster.cluster_id] = cluster.query_count;
      }
    }

    return counts;
  } catch (error) {
    console.error('Error fetching cluster query counts:', error);
    return {};
  }
}

/**
 * Search queries within a specific cluster using full-text search
 */
export async function searchQueriesInCluster(
  searchText: string,
  runId: string,
  clusterId: number,
  page: number = 1,
  limit: number = 50
): Promise<PaginatedQueries> {
  console.log('[searchQueriesInCluster] runId:', runId, 'clusterId:', clusterId, 'searchText:', searchText);

  try {
    // Use the queries API with cluster_id filter and search
    // Note: The current API doesn't support searching within a cluster directly,
    // so we'll need to fetch all queries for the cluster and filter client-side
    const response = await queriesApi.getClusterDetail(runId, clusterId, { page: 1, limit: 10000 });

    // Filter queries by search text
    const filteredQueries = response.queries.items.filter((q: any) =>
      q.query_text?.toLowerCase().includes(searchText.toLowerCase())
    );

    // Paginate results
    const total = filteredQueries.length;
    const pages = Math.ceil(total / limit);
    const offset = (page - 1) * limit;
    const paginatedQueries = filteredQueries.slice(offset, offset + limit);

    console.log('[searchQueriesInCluster] Found', paginatedQueries.length, 'queries on page', page);
    console.log('[searchQueriesInCluster] Total:', total, 'Pages:', pages);

    return {
      queries: paginatedQueries as Query[],
      total,
      page,
      pages,
      limit,
    };
  } catch (error) {
    console.error('Error searching queries in cluster:', error);
    return {
      queries: [],
      total: 0,
      page: 1,
      pages: 0,
      limit,
    };
  }
}

/**
 * Search queries using full-text search
 */
export async function searchQueries(
  searchText: string,
  runId?: string,
  page: number = 1,
  limit: number = 50
) {
  console.log('[searchQueries] searchText:', searchText, 'runId:', runId, 'page:', page);

  try {
    const response = await searchApi.searchQueries({
      text: searchText,
      mode: 'fulltext',
      run_id: runId,
      page,
      limit,
    });

    // Transform the response to match the expected format
    // The API returns items with query and optional cluster info
    const queriesWithClusters = response.items.map((item: any) => ({
      ...item.query,
      clusters: item.cluster_id !== undefined ? [{
        clusterId: item.cluster_id,
        runId: item.run_id,
        title: item.cluster_title,
        confidenceScore: item.confidence_score,
      }] : [],
    }));

    return {
      queries: queriesWithClusters,
      total: response.total,
      page: response.page || page,
      pages: response.pages || Math.ceil(response.total / limit),
      limit,
    };
  } catch (error) {
    console.error('Error searching queries:', error);
    return {
      queries: [],
      total: 0,
      page: 1,
      pages: 0,
      limit,
    };
  }
}

/**
 * Get cluster metadata (quality, coherence, flags, notes)
 */
export async function getClusterMetadata(runId: string, clusterId: number) {
  try {
    return await curationApi.getClusterMetadata(runId, clusterId);
  } catch (error) {
    console.error('Error fetching cluster metadata:', error);
    return null;
  }
}

/**
 * Get edit history for a cluster
 */
export async function getClusterEditHistory(runId: string, clusterId?: number) {
  try {
    if (clusterId !== undefined) {
      const response = await curationApi.getClusterHistory(runId, clusterId);
      return response.items;
    } else {
      // If no cluster ID, get full audit log
      const response = await curationApi.getAuditLog(runId);
      return response.items;
    }
  } catch (error) {
    console.error('Error fetching cluster edit history:', error);
    return [];
  }
}

/**
 * Get orphaned queries for a run
 */
export async function getOrphanedQueries(runId: string) {
  try {
    const response = await curationApi.getOrphanedQueries(runId);
    return response.items;
  } catch (error) {
    console.error('Error fetching orphaned queries:', error);
    return [];
  }
}

'use server'

import { db } from '@/lib/db/client';
import {
  clusteringRuns,
  clusterHierarchies,
  queries,
  queryClusters,
  clusterSummaries,
} from '@/lib/db/schema';
import { eq, and, desc, sql, count } from 'drizzle-orm';
import { getCollection } from '@/lib/chroma/client';
import type {
  ClusteringRun,
  ClusterHierarchy,
  Query,
  PaginatedQueries,
  ClusterSummary,
} from '@/lib/types/schemas';

/**
 * Get all clustering runs, ordered by creation date (newest first)
 */
export async function getRuns(): Promise<ClusteringRun[]> {
  try {
    const runs = await db
      .select()
      .from(clusteringRuns)
      .orderBy(desc(clusteringRuns.createdAt));

    return runs as ClusteringRun[];
  } catch (error) {
    console.error('Error fetching runs:', error);
    return [];
  }
}

/**
 * Get a single clustering run by ID
 */
export async function getRun(runId: string): Promise<ClusteringRun | null> {
  const [run] = await db
    .select()
    .from(clusteringRuns)
    .where(eq(clusteringRuns.runId, runId));

  if (!run) return null;
  return run as ClusteringRun;
}

/**
 * Get all hierarchies for a given run
 */
export async function getHierarchiesForRun(runId: string) {
  const hierarchies = await db
    .select({
      hierarchyRunId: clusterHierarchies.hierarchyRunId,
      createdAt: clusterHierarchies.createdAt,
    })
    .from(clusterHierarchies)
    .where(eq(clusterHierarchies.runId, runId))
    .groupBy(clusterHierarchies.hierarchyRunId)
    .orderBy(desc(clusterHierarchies.createdAt));

  return hierarchies;
}

/**
 * Get full hierarchy tree for a hierarchy run
 */
export async function getHierarchyTree(hierarchyRunId: string): Promise<ClusterHierarchy[]> {
  const nodes = await db
    .select()
    .from(clusterHierarchies)
    .where(eq(clusterHierarchies.hierarchyRunId, hierarchyRunId))
    .orderBy(clusterHierarchies.level);

  return nodes as ClusterHierarchy[];
}

/**
 * Get cluster summary for a specific cluster
 */
export async function getClusterSummary(runId: string, clusterId: number, alias?: string): Promise<ClusterSummary | null> {
  let query = db
    .select()
    .from(clusterSummaries)
    .where(
      and(
        eq(clusterSummaries.runId, runId),
        eq(clusterSummaries.clusterId, clusterId)
      )
    );

  // If alias is provided, filter by it
  if (alias) {
    query = query.where(eq(clusterSummaries.alias, alias)) as any;
  }

  const [summary] = await query.orderBy(desc(clusterSummaries.generatedAt));

  if (!summary) return null;
  return summary as ClusterSummary;
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
  const offset = (page - 1) * limit;

  // Get paginated queries
  const results = await db
    .select({
      query: queries,
    })
    .from(queries)
    .innerJoin(queryClusters, eq(queries.id, queryClusters.queryId))
    .where(
      and(eq(queryClusters.runId, runId), eq(queryClusters.clusterId, clusterId))
    )
    .limit(limit)
    .offset(offset);

  console.log('[getClusterQueries] Found', results.length, 'queries');

  // Get total count
  const [countResult] = await db
    .select({ count: count() })
    .from(queryClusters)
    .where(
      and(eq(queryClusters.runId, runId), eq(queryClusters.clusterId, clusterId))
    );

  const total = countResult.count;
  const pages = Math.ceil(total / limit);

  console.log('[getClusterQueries] Total:', total, 'Pages:', pages);

  return {
    queries: results.map((r) => r.query as Query),
    total,
    page,
    pages,
    limit,
  };
}

/**
 * Search clusters using ChromaDB semantic search
 */
export async function searchClusters(query: string, runId: string, nResults: number = 20) {
  // Get embedding config from run
  const [run] = await db
    .select()
    .from(clusteringRuns)
    .where(eq(clusteringRuns.runId, runId));

  if (!run || !run.parameters) {
    throw new Error('Run not found or missing parameters');
  }

  const params = run.parameters as Record<string, any>;
  const provider = params.embedding_provider;
  const model = params.embedding_model;

  if (!provider || !model) {
    throw new Error('Run missing embedding provider or model in parameters');
  }

  // Query ChromaDB
  const collection = await getCollection(provider, model, 'cluster_summaries');

  const results = await collection.query({
    queryTexts: [query],
    nResults,
    where: { run_id: runId },
  });

  return results;
}

/**
 * Get all summary aliases for a run
 */
export async function getSummaryAliases(runId: string) {
  const aliases = await db
    .select({
      alias: clusterSummaries.alias,
      summaryRunId: clusterSummaries.summaryRunId,
      model: clusterSummaries.model,
    })
    .from(clusterSummaries)
    .where(eq(clusterSummaries.runId, runId))
    .groupBy(clusterSummaries.alias, clusterSummaries.summaryRunId, clusterSummaries.model);

  return aliases;
}

/**
 * Get query counts for all clusters in a run
 */
export async function getClusterQueryCounts(runId: string): Promise<Record<number, number>> {
  const counts = await db
    .select({
      clusterId: queryClusters.clusterId,
      count: count(),
    })
    .from(queryClusters)
    .where(eq(queryClusters.runId, runId))
    .groupBy(queryClusters.clusterId);

  return Object.fromEntries(counts.map(c => [c.clusterId, c.count]));
}

/**
 * Search queries using ChromaDB semantic search
 */
export async function searchQueries(
  searchText: string,
  runId?: string,
  page: number = 1,
  limit: number = 50
) {
  // Get embedding config from run or use defaults
  let provider = 'cohere';
  let model = 'embed-v4_0'; // Note: ChromaDB sanitizes dots to underscores
  
  if (runId) {
    const [run] = await db
      .select()
      .from(clusteringRuns)
      .where(eq(clusteringRuns.runId, runId));

    if (run && run.parameters) {
      const params = run.parameters as Record<string, any>;
      provider = params.embedding_provider || provider;
      // Sanitize model name to match ChromaDB collection naming
      model = (params.embedding_model || 'embed-v4.0').replace(/[^a-zA-Z0-9_-]/g, '_');
    }
  } else {
    // If no runId, try to get config from most recent run
    const [latestRun] = await db
      .select()
      .from(clusteringRuns)
      .orderBy(desc(clusteringRuns.createdAt))
      .limit(1);
    
    if (latestRun && latestRun.parameters) {
      const params = latestRun.parameters as Record<string, any>;
      provider = params.embedding_provider || provider;
      model = (params.embedding_model || 'embed-v4.0').replace(/[^a-zA-Z0-9_-]/g, '_');
    }
  }

  // Query ChromaDB for semantic search
  const collection = await getCollection(provider, model, 'queries');
  
  // Calculate offset for pagination
  const offset = (page - 1) * limit;
  
  // ChromaDB doesn't support offset directly, so we fetch more and slice
  // This is a limitation - for production you'd want to implement cursor-based pagination
  const fetchLimit = offset + limit;
  
  const chromaResults = await collection.query({
    queryTexts: [searchText],
    nResults: Math.min(fetchLimit, 1000), // Cap at 1000 for performance
  });

  // Extract query IDs from ChromaDB results
  const allQueryIds = chromaResults.ids[0]
    .map(id => parseInt(id.replace('query_', '')))
    .slice(offset, offset + limit);

  if (allQueryIds.length === 0) {
    return {
      queries: [],
      total: 0,
      page,
      pages: 0,
      limit,
    };
  }

  // Fetch full query data from SQLite with cluster associations
  const results = await db
    .select({
      query: queries,
      clusterId: queryClusters.clusterId,
      clusterRunId: queryClusters.runId,
      clusterTitle: clusterSummaries.title,
      confidenceScore: queryClusters.confidenceScore,
    })
    .from(queries)
    .leftJoin(queryClusters, eq(queries.id, queryClusters.queryId))
    .leftJoin(
      clusterSummaries,
      and(
        eq(queryClusters.runId, clusterSummaries.runId),
        eq(queryClusters.clusterId, clusterSummaries.clusterId)
      )
    )
    .where(sql`${queries.id} IN ${sql.raw(`(${allQueryIds.join(',')})`)}`);

  // Filter by run if provided
  let filteredResults = results;
  if (runId) {
    filteredResults = results.filter(r => r.clusterRunId === runId || r.clusterRunId === null);
  }

  // Group results by query
  const queryMap = new Map();
  for (const row of filteredResults) {
    const queryId = row.query.id;
    if (!queryMap.has(queryId)) {
      queryMap.set(queryId, {
        ...row.query,
        clusters: [],
      });
    }
    
    if (row.clusterId !== null) {
      queryMap.get(queryId).clusters.push({
        clusterId: row.clusterId,
        runId: row.clusterRunId,
        title: row.clusterTitle,
        confidenceScore: row.confidenceScore,
      });
    }
  }

  const queriesWithClusters = Array.from(queryMap.values());

  // Estimate total (ChromaDB doesn't give us exact count easily)
  const total = chromaResults.ids[0].length;
  const pages = Math.ceil(total / limit);

  return {
    queries: queriesWithClusters,
    total,
    page,
    pages,
    limit,
  };
}

'use server'

import { db } from '@/lib/db/client';
import {
  clusteringRuns,
  clusterHierarchies,
  queries,
  queryClusters,
  clusterSummaries,
} from '@/lib/db/schema';
import { eq, and, desc, sql, count, like, or } from 'drizzle-orm';
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
  // Build where conditions
  const conditions = [
    eq(clusterSummaries.runId, runId),
    eq(clusterSummaries.clusterId, clusterId)
  ];

  // If alias is provided, add it to conditions
  if (alias) {
    conditions.push(eq(clusterSummaries.alias, alias));
  }

  const [summary] = await db
    .select()
    .from(clusterSummaries)
    .where(and(...conditions))
    .orderBy(desc(clusterSummaries.generatedAt));

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
 * Search clusters using full-text search on titles and descriptions
 */
export async function searchClusters(query: string, runId: string, nResults: number = 20) {
  const searchPattern = `%${query}%`;
  
  const results = await db
    .select({
      clusterId: clusterSummaries.clusterId,
      title: clusterSummaries.title,
      description: clusterSummaries.description,
      summary: clusterSummaries.summary,
      numQueries: clusterSummaries.numQueries,
    })
    .from(clusterSummaries)
    .where(
      and(
        eq(clusterSummaries.runId, runId),
        or(
          like(clusterSummaries.title, searchPattern),
          like(clusterSummaries.description, searchPattern),
          like(clusterSummaries.summary, searchPattern)
        )
      )
    )
    .limit(nResults);

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

  const offset = (page - 1) * limit;
  const searchPattern = `%${searchText}%`;

  // Get paginated queries matching search text in this cluster
  const results = await db
    .select({
      query: queries,
    })
    .from(queries)
    .innerJoin(queryClusters, eq(queries.id, queryClusters.queryId))
    .where(
      and(
        eq(queryClusters.runId, runId),
        eq(queryClusters.clusterId, clusterId),
        like(queries.queryText, searchPattern)
      )
    )
    .limit(limit)
    .offset(offset);

  console.log('[searchQueriesInCluster] Found', results.length, 'queries on page', page);

  // Get total count
  const [countResult] = await db
    .select({ count: count() })
    .from(queryClusters)
    .innerJoin(queries, eq(queryClusters.queryId, queries.id))
    .where(
      and(
        eq(queryClusters.runId, runId),
        eq(queryClusters.clusterId, clusterId),
        like(queries.queryText, searchPattern)
      )
    );

  const total = countResult.count;
  const pages = Math.ceil(total / limit);

  console.log('[searchQueriesInCluster] Total:', total, 'Pages:', pages);

  return {
    queries: results.map((r) => r.query as Query),
    total,
    page,
    pages,
    limit,
  };
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

  const offset = (page - 1) * limit;
  const searchPattern = `%${searchText}%`;

  // Build where conditions
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const whereConditions: any[] = [like(queries.queryText, searchPattern)];
  if (runId) {
    whereConditions.push(
      or(
        eq(queryClusters.runId, runId),
        sql`${queryClusters.runId} IS NULL`
      )
    );
  }

  // Execute query with pagination
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
    .where(and(...whereConditions))
    .limit(limit)
    .offset(offset);

  // Group results by query to handle multiple cluster assignments
  const queryMap = new Map();
  for (const row of results) {
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

  // Get total count
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const countWhereConditions: any[] = [like(queries.queryText, searchPattern)];
  if (runId) {
    countWhereConditions.push(
      or(
        eq(queryClusters.runId, runId),
        sql`${queryClusters.runId} IS NULL`
      )
    );
  }

  const [countResult] = await db
    .select({ count: count() })
    .from(queries)
    .leftJoin(queryClusters, eq(queries.id, queryClusters.queryId))
    .where(and(...countWhereConditions));

  const total = countResult.count;
  const pages = Math.ceil(total / limit);

  return {
    queries: queriesWithClusters,
    total,
    page,
    pages,
    limit,
  };
}

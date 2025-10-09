import { z } from 'zod';

// Zod schema for ClusteringRun (for validation in Server Actions)
export const ClusteringRunSchema = z.object({
  runId: z.string(),
  algorithm: z.string(),
  parameters: z.record(z.any()).nullable(),
  description: z.string().nullable(),
  createdAt: z.string(),
  numClusters: z.number().nullable(),
});

// Zod schema for Query
export const QuerySchema = z.object({
  id: z.number(),
  conversationId: z.string(),
  model: z.string(),
  queryText: z.string(),
  language: z.string().nullable(),
  timestamp: z.string().nullable(),
  extraMetadata: z.record(z.any()).nullable(),
  createdAt: z.string(),
});

// Zod schema for ClusterSummary
export const ClusterSummarySchema = z.object({
  id: z.number(),
  runId: z.string(),
  clusterId: z.number(),
  summaryRunId: z.string(),
  alias: z.string().nullable(),
  title: z.string().nullable(),
  description: z.string().nullable(),
  summary: z.string().nullable(),
  numQueries: z.number().nullable(),
  representativeQueries: z.array(z.any()).nullable(),
  model: z.string().nullable(),
  parameters: z.record(z.any()).nullable(),
  generatedAt: z.string(),
});

// Zod schema for ClusterHierarchy
export const ClusterHierarchySchema = z.object({
  id: z.number(),
  runId: z.string(),
  hierarchyRunId: z.string(),
  clusterId: z.number(),
  parentClusterId: z.number().nullable(),
  level: z.number(),
  childrenIds: z.array(z.number()).nullable(),
  title: z.string().nullable(),
  description: z.string().nullable(),
  createdAt: z.string(),
});

// Zod schema for QueryCluster
export const QueryClusterSchema = z.object({
  id: z.number(),
  runId: z.string(),
  queryId: z.number(),
  clusterId: z.number(),
  confidenceScore: z.number().nullable(),
  createdAt: z.string(),
});

// Paginated response schema
export const PaginatedQueriesSchema = z.object({
  queries: z.array(QuerySchema),
  total: z.number(),
  page: z.number(),
  pages: z.number(),
  limit: z.number(),
});

// Type exports (inferred from Zod schemas)
export type ClusteringRun = z.infer<typeof ClusteringRunSchema>;
export type Query = z.infer<typeof QuerySchema>;
export type ClusterSummary = z.infer<typeof ClusterSummarySchema>;
export type ClusterHierarchy = z.infer<typeof ClusterHierarchySchema>;
export type QueryCluster = z.infer<typeof QueryClusterSchema>;
export type PaginatedQueries = z.infer<typeof PaginatedQueriesSchema>;

import { sqliteTable, text, integer, real } from 'drizzle-orm/sqlite-core';

// Mirror of Python Query model (src/lmsys_query_analysis/db/models.py)
export const queries = sqliteTable('queries', {
  id: integer('id').primaryKey(),
  conversationId: text('conversation_id').notNull().unique(),
  model: text('model').notNull(),
  queryText: text('query_text').notNull(),
  language: text('language'),
  timestamp: text('timestamp'), // ISO string
  extraMetadata: text('extra_metadata', { mode: 'json' }),
  createdAt: text('created_at').notNull(),
});

// Mirror of Python ClusteringRun model
export const clusteringRuns = sqliteTable('clustering_runs', {
  runId: text('run_id').primaryKey(),
  algorithm: text('algorithm').notNull(),
  parameters: text('parameters', { mode: 'json' }),
  description: text('description'),
  createdAt: text('created_at').notNull(),
  numClusters: integer('num_clusters'),
});

// Mirror of Python QueryCluster model
export const queryClusters = sqliteTable('query_clusters', {
  id: integer('id').primaryKey(),
  runId: text('run_id').notNull(),
  queryId: integer('query_id').notNull(),
  clusterId: integer('cluster_id').notNull(),
  confidenceScore: real('confidence_score'),
  createdAt: text('created_at').notNull(),
});

// Mirror of Python ClusterSummary model
export const clusterSummaries = sqliteTable('cluster_summaries', {
  id: integer('id').primaryKey(),
  runId: text('run_id').notNull(),
  clusterId: integer('cluster_id').notNull(),
  summaryRunId: text('summary_run_id').notNull(),
  alias: text('alias'),
  title: text('title'),
  description: text('description'),
  summary: text('summary'),
  numQueries: integer('num_queries'),
  representativeQueries: text('representative_queries', { mode: 'json' }),
  model: text('model'),
  parameters: text('parameters', { mode: 'json' }),
  generatedAt: text('generated_at').notNull(),
});

// Mirror of Python ClusterHierarchy model
export const clusterHierarchies = sqliteTable('cluster_hierarchies', {
  id: integer('id').primaryKey(),
  runId: text('run_id').notNull(),
  hierarchyRunId: text('hierarchy_run_id').notNull(),
  clusterId: integer('cluster_id').notNull(),
  parentClusterId: integer('parent_cluster_id'),
  level: integer('level').notNull(),
  childrenIds: text('children_ids', { mode: 'json' }),
  title: text('title'),
  description: text('description'),
  createdAt: text('created_at').notNull(),
});

// Type exports for use in components
export type Query = typeof queries.$inferSelect;
export type ClusteringRun = typeof clusteringRuns.$inferSelect;
export type QueryCluster = typeof queryClusters.$inferSelect;
export type ClusterSummary = typeof clusterSummaries.$inferSelect;
export type ClusterHierarchy = typeof clusterHierarchies.$inferSelect;

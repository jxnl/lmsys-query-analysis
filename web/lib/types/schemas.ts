// Import types from the Drizzle schema
import type {
  Query,
  ClusteringRun,
  QueryCluster,
  ClusterSummary,
  ClusterHierarchy,
  ClusterEdit,
  ClusterMetadata,
  OrphanedQuery,
} from '../db/schema';

// Re-export types
export type {
  Query,
  ClusteringRun,
  QueryCluster,
  ClusterSummary,
  ClusterHierarchy,
  ClusterEdit,
  ClusterMetadata,
  OrphanedQuery,
};

// Additional utility types
export type PaginatedQueries = {
  queries: Query[];
  total: number;
  page: number;
  pages: number;
  limit: number;
};

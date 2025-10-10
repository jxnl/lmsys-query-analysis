/**
 * Type exports from generated OpenAPI schema
 *
 * Re-exports commonly used types with more convenient names.
 * The source types are auto-generated from the FastAPI OpenAPI spec.
 */

import type { components } from "../api/types";

// Response models
export type Query = components["schemas"]["QueryResponse"];
export type ClusteringRun = components["schemas"]["ClusteringRunSummary"];
export type ClusteringRunDetail = components["schemas"]["ClusteringRunDetail"];
export type ClusterSummary = components["schemas"]["ClusterSummaryResponse"];
export type ClusterHierarchy = components["schemas"]["HierarchyNode"];
export type ClusterMetadata = components["schemas"]["ClusterMetadata"];
export type ClusterEdit = components["schemas"]["EditHistoryRecord"];

// Paginated responses
export type PaginatedQueriesResponse =
  components["schemas"]["PaginatedQueriesResponse"];
export type ClusteringRunListResponse =
  components["schemas"]["ClusteringRunListResponse"];
export type ClusterListResponse = components["schemas"]["ClusterListResponse"];
export type HierarchyTreeResponse =
  components["schemas"]["HierarchyTreeResponse"];
export type SearchQueriesResponse =
  components["schemas"]["SearchQueriesResponse"];
export type SearchClustersResponse =
  components["schemas"]["SearchClustersResponse"];

// Additional utility types
export type PaginatedQueries = {
  queries: Query[];
  total: number;
  page: number;
  pages: number;
  limit: number;
};

// Re-export the full components namespace for advanced usage
export type { components };

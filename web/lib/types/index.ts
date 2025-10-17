

import type { components } from "../api/types";


export type Query = components["schemas"]["QueryResponse"];
export type ClusteringRun = components["schemas"]["ClusteringRunSummary"];
export type ClusteringRunDetail = components["schemas"]["ClusteringRunDetail"];
export type ClusterSummary = components["schemas"]["ClusterSummaryResponse"];
export type ClusterHierarchy = components["schemas"]["HierarchyNode"];
export type ClusterMetadata = components["schemas"]["ClusterMetadata"];
export type ClusterEdit = components["schemas"]["EditHistoryRecord"];


export type PaginatedQueriesResponse = components["schemas"]["PaginatedQueriesResponse"];
export type ClusteringRunListResponse = components["schemas"]["ClusteringRunListResponse"];
export type ClusterListResponse = components["schemas"]["ClusterListResponse"];
export type HierarchyTreeResponse = components["schemas"]["HierarchyTreeResponse"];
export type SearchQueriesResponse = components["schemas"]["SearchQueriesResponse"];
export type SearchClustersResponse = components["schemas"]["SearchClustersResponse"];


export type PaginatedQueries = {
  queries: Query[];
  total: number;
  page: number;
  pages: number;
  limit: number;
};


export type { components };


export interface paths {
  "/api/health": {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    
    get: operations["health_check_api_health_get"];
    put?: never;
    post?: never;
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/api/clustering/runs": {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    
    get: operations["list_runs_api_clustering_runs_get"];
    put?: never;
    post?: never;
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/api/clustering/runs/{run_id}": {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    
    get: operations["get_run_api_clustering_runs__run_id__get"];
    put?: never;
    post?: never;
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/api/clustering/runs/{run_id}/status": {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    
    get: operations["get_run_status_api_clustering_runs__run_id__status_get"];
    put?: never;
    post?: never;
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/api/clustering/runs/{run_id}/clusters": {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    
    get: operations["list_clusters_api_clustering_runs__run_id__clusters_get"];
    put?: never;
    post?: never;
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/api/clustering/kmeans": {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    get?: never;
    put?: never;
    
    post: operations["create_kmeans_run_api_clustering_kmeans_post"];
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/api/clustering/hdbscan": {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    get?: never;
    put?: never;
    
    post: operations["create_hdbscan_run_api_clustering_hdbscan_post"];
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/api/queries": {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    
    get: operations["list_queries_api_queries_get"];
    put?: never;
    post?: never;
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/api/clustering/runs/{run_id}/clusters/{cluster_id}": {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    
    get: operations["get_cluster_detail_api_clustering_runs__run_id__clusters__cluster_id__get"];
    put?: never;
    post?: never;
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/api/hierarchy/": {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    
    get: operations["list_hierarchies_api_hierarchy__get"];
    put?: never;
    
    post: operations["create_hierarchy_api_hierarchy__post"];
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/api/hierarchy/{hierarchy_run_id}": {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    
    get: operations["get_hierarchy_tree_api_hierarchy__hierarchy_run_id__get"];
    put?: never;
    post?: never;
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/api/summaries/": {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    
    get: operations["list_summaries_api_summaries__get"];
    put?: never;
    
    post: operations["create_summary_api_summaries__post"];
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/api/summaries/{summary_run_id}/clusters/{cluster_id}": {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    
    get: operations["get_cluster_summary_api_summaries__summary_run_id__clusters__cluster_id__get"];
    put?: never;
    post?: never;
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/api/search/queries": {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    
    get: operations["search_queries_api_search_queries_get"];
    put?: never;
    post?: never;
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/api/search/clusters": {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    
    get: operations["search_clusters_api_search_clusters_get"];
    put?: never;
    post?: never;
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/api/curation/clusters/{cluster_id}/metadata": {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    
    get: operations["get_cluster_metadata_api_curation_clusters__cluster_id__metadata_get"];
    put?: never;
    post?: never;
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/api/curation/clusters/{cluster_id}/history": {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    
    get: operations["get_cluster_history_api_curation_clusters__cluster_id__history_get"];
    put?: never;
    post?: never;
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/api/curation/runs/{run_id}/audit": {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    
    get: operations["get_run_audit_api_curation_runs__run_id__audit_get"];
    put?: never;
    post?: never;
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/api/curation/runs/{run_id}/orphaned": {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    
    get: operations["get_orphaned_queries_api_curation_runs__run_id__orphaned_get"];
    put?: never;
    post?: never;
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/api/curation/queries/{query_id}/move": {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    get?: never;
    put?: never;
    
    post: operations["move_query_api_curation_queries__query_id__move_post"];
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/api/curation/clusters/{cluster_id}/rename": {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    get?: never;
    put?: never;
    
    post: operations["rename_cluster_api_curation_clusters__cluster_id__rename_post"];
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
  "/": {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    
    get: operations["root__get"];
    put?: never;
    post?: never;
    delete?: never;
    options?: never;
    head?: never;
    patch?: never;
    trace?: never;
  };
}
export type webhooks = Record<string, never>;
export interface components {
  schemas: {
    
    ClusterDetailResponse: {
      cluster: components["schemas"]["ClusterSummaryResponse"];
      queries: components["schemas"]["PaginatedQueriesResponse"];
    };
    
    ClusterInfo: {
      
      run_id: string;
      
      cluster_id: number;
      
      title?: string | null;
      
      confidence_score?: number | null;
    };
    
    ClusterListResponse: {
      
      items: components["schemas"]["ClusterSummaryResponse"][];
      
      total: number;
      
      page: number;
      
      pages: number;
      
      limit: number;
      
      total_queries?: number | null;
    };
    
    ClusterMetadata: {
      
      coherence_score?: number | null;
      
      quality?: ("high" | "medium" | "low") | null;
      
      notes?: string | null;
      
      flags?: string[] | null;
      
      last_edited?: string | null;
    };
    
    ClusterSearchResult: {
      
      run_id: string;
      
      cluster_id: number;
      
      title?: string | null;
      
      description?: string | null;
      
      summary?: string | null;
      
      num_queries?: number | null;
      
      distance?: number | null;
    };
    
    ClusterSummaryResponse: {
      
      run_id: string;
      
      cluster_id: number;
      
      title?: string | null;
      
      description?: string | null;
      
      summary?: string | null;
      
      num_queries?: number | null;
      
      representative_queries?: string[] | null;
      
      summary_run_id?: string | null;
      
      alias?: string | null;
      
      query_count?: number | null;
      
      percentage?: number | null;
    };
    
    ClusteringRunDetail: {
      
      run_id: string;
      
      algorithm: string;
      
      num_clusters?: number | null;
      
      description?: string | null;
      
      parameters?: {
        [key: string]: unknown;
      } | null;
      
      created_at: string;
      
      status: "pending" | "running" | "completed" | "failed";
      
      metrics?: {
        [key: string]: unknown;
      } | null;
      
      latest_errors?: string[] | null;
    };
    
    ClusteringRunListResponse: {
      
      items: components["schemas"]["ClusteringRunSummary"][];
      
      total: number;
      
      page: number;
      
      pages: number;
      
      limit: number;
    };
    
    ClusteringRunStatusResponse: {
      
      run_id: string;
      
      status: string;
      
      processed?: number | null;
    };
    
    ClusteringRunSummary: {
      
      run_id: string;
      
      algorithm: string;
      
      num_clusters?: number | null;
      
      description?: string | null;
      
      parameters?: {
        [key: string]: unknown;
      } | null;
      
      created_at: string;
      
      status: "pending" | "running" | "completed" | "failed";
    };
    
    EditHistoryRecord: {
      
      timestamp: string;
      
      cluster_id?: number | null;
      
      edit_type: string;
      
      editor: string;
      
      reason?: string | null;
      
      old_value?: {
        [key: string]: unknown;
      } | null;
      
      new_value?: {
        [key: string]: unknown;
      } | null;
    };
    
    EditHistoryResponse: {
      
      items: components["schemas"]["EditHistoryRecord"][];
      
      total: number;
      
      page: number;
      
      pages: number;
      
      limit: number;
    };
    
    ErrorDetail: {
      
      type: string;
      
      message: string;
    };
    
    ErrorResponse: {
      error: components["schemas"]["ErrorDetail"];
    };
    
    FacetBucket: {
      
      key: unknown;
      
      count: number;
      
      percentage?: number | null;
      
      meta?: {
        [key: string]: unknown;
      } | null;
    };
    
    HTTPValidationError: {
      
      detail?: components["schemas"]["ValidationError"][];
    };
    
    HierarchyListResponse: {
      
      items: components["schemas"]["HierarchyRunInfo"][];
      
      total: number;
      
      page: number;
      
      pages: number;
      
      limit: number;
    };
    
    HierarchyNode: {
      
      hierarchy_run_id: string;
      
      run_id: string;
      
      cluster_id: number;
      
      parent_cluster_id?: number | null;
      
      level: number;
      
      children_ids: number[];
      
      title?: string | null;
      
      description?: string | null;
      
      query_count?: number | null;
      
      percentage?: number | null;
    };
    
    HierarchyRunInfo: {
      
      hierarchy_run_id: string;
      
      run_id: string;
      
      created_at: string;
    };
    
    HierarchyTreeResponse: {
      
      nodes: components["schemas"]["HierarchyNode"][];
      
      total_queries?: number | null;
    };
    
    OrphanInfo: {
      
      orphan: {
        [key: string]: unknown;
      };
      query: components["schemas"]["QueryResponse"];
    };
    
    OrphanedQueriesResponse: {
      
      items: components["schemas"]["OrphanInfo"][];
      
      total: number;
      
      page: number;
      
      pages: number;
      
      limit: number;
    };
    
    PaginatedQueriesResponse: {
      
      items: components["schemas"]["QueryResponse"][];
      
      total: number;
      
      page: number;
      
      pages: number;
      
      limit: number;
    };
    
    QueryResponse: {
      
      id: number;
      
      conversation_id: string;
      
      model: string;
      
      query_text: string;
      
      language?: string | null;
      
      timestamp?: string | null;
      
      created_at: string;
    };
    
    QuerySearchResult: {
      query: components["schemas"]["QueryResponse"];
      
      clusters: components["schemas"]["ClusterInfo"][];
      
      distance?: number | null;
    };
    
    SearchClustersResponse: {
      
      items: components["schemas"]["ClusterSearchResult"][];
      
      total: number;
      
      page: number;
      
      pages: number;
      
      limit: number;
    };
    
    SearchFacets: {
      
      clusters?: components["schemas"]["FacetBucket"][] | null;
      
      language?: components["schemas"]["FacetBucket"][] | null;
      
      model?: components["schemas"]["FacetBucket"][] | null;
    };
    
    SearchQueriesResponse: {
      
      items: components["schemas"]["QuerySearchResult"][];
      
      total: number;
      
      page: number;
      
      pages: number;
      
      limit: number;
      facets?: components["schemas"]["SearchFacets"] | null;
      
      applied_clusters?: components["schemas"]["ClusterSearchResult"][] | null;
    };
    
    SummaryRunListResponse: {
      
      items: components["schemas"]["SummaryRunSummary"][];
      
      total: number;
      
      page: number;
      
      pages: number;
      
      limit: number;
    };
    
    SummaryRunSummary: {
      
      summary_run_id: string;
      
      run_id: string;
      
      alias?: string | null;
      
      model: string;
      
      generated_at: string;
      
      status: "pending" | "running" | "completed" | "failed";
    };
    
    ValidationError: {
      
      loc: (string | number)[];
      
      msg: string;
      
      type: string;
    };
  };
  responses: never;
  parameters: never;
  requestBodies: never;
  headers: never;
  pathItems: never;
}
export type $defs = Record<string, never>;
export interface operations {
  health_check_api_health_get: {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    requestBody?: never;
    responses: {
      
      200: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": unknown;
        };
      };
    };
  };
  list_runs_api_clustering_runs_get: {
    parameters: {
      query?: {
        
        algorithm?: string | null;
        
        page?: number;
        
        limit?: number;
      };
      header?: never;
      path?: never;
      cookie?: never;
    };
    requestBody?: never;
    responses: {
      
      200: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["ClusteringRunListResponse"];
        };
      };
      
      422: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["HTTPValidationError"];
        };
      };
    };
  };
  get_run_api_clustering_runs__run_id__get: {
    parameters: {
      query?: never;
      header?: never;
      path: {
        run_id: string;
      };
      cookie?: never;
    };
    requestBody?: never;
    responses: {
      
      200: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["ClusteringRunDetail"];
        };
      };
      
      404: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["ErrorResponse"];
        };
      };
      
      422: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["HTTPValidationError"];
        };
      };
    };
  };
  get_run_status_api_clustering_runs__run_id__status_get: {
    parameters: {
      query?: never;
      header?: never;
      path: {
        run_id: string;
      };
      cookie?: never;
    };
    requestBody?: never;
    responses: {
      
      200: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["ClusteringRunStatusResponse"];
        };
      };
      
      422: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["HTTPValidationError"];
        };
      };
    };
  };
  list_clusters_api_clustering_runs__run_id__clusters_get: {
    parameters: {
      query?: {
        
        include_counts?: boolean;
        
        include_percentages?: boolean;
        
        summary_run_id?: string | null;
        
        alias?: string | null;
        
        limit?: number | null;
        
        page?: number;
        
        page_limit?: number;
      };
      header?: never;
      path: {
        run_id: string;
      };
      cookie?: never;
    };
    requestBody?: never;
    responses: {
      
      200: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["ClusterListResponse"];
        };
      };
      
      422: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["HTTPValidationError"];
        };
      };
    };
  };
  create_kmeans_run_api_clustering_kmeans_post: {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    requestBody?: never;
    responses: {
      
      501: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": unknown;
        };
      };
    };
  };
  create_hdbscan_run_api_clustering_hdbscan_post: {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    requestBody?: never;
    responses: {
      
      501: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": unknown;
        };
      };
    };
  };
  list_queries_api_queries_get: {
    parameters: {
      query?: {
        
        run_id?: string | null;
        
        cluster_id?: number | null;
        
        model?: string | null;
        
        page?: number;
        
        limit?: number;
      };
      header?: never;
      path?: never;
      cookie?: never;
    };
    requestBody?: never;
    responses: {
      
      200: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["PaginatedQueriesResponse"];
        };
      };
      
      422: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["HTTPValidationError"];
        };
      };
    };
  };
  get_cluster_detail_api_clustering_runs__run_id__clusters__cluster_id__get: {
    parameters: {
      query?: {
        
        page?: number;
        
        limit?: number;
      };
      header?: never;
      path: {
        run_id: string;
        cluster_id: number;
      };
      cookie?: never;
    };
    requestBody?: never;
    responses: {
      
      200: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["ClusterDetailResponse"];
        };
      };
      
      404: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["ErrorResponse"];
        };
      };
      
      422: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["HTTPValidationError"];
        };
      };
    };
  };
  list_hierarchies_api_hierarchy__get: {
    parameters: {
      query?: {
        
        run_id?: string | null;
        
        page?: number;
        
        limit?: number;
      };
      header?: never;
      path?: never;
      cookie?: never;
    };
    requestBody?: never;
    responses: {
      
      200: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["HierarchyListResponse"];
        };
      };
      
      422: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["HTTPValidationError"];
        };
      };
    };
  };
  create_hierarchy_api_hierarchy__post: {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    requestBody?: never;
    responses: {
      
      501: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": unknown;
        };
      };
    };
  };
  get_hierarchy_tree_api_hierarchy__hierarchy_run_id__get: {
    parameters: {
      query?: {
        
        include_percentages?: boolean;
      };
      header?: never;
      path: {
        hierarchy_run_id: string;
      };
      cookie?: never;
    };
    requestBody?: never;
    responses: {
      
      200: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["HierarchyTreeResponse"];
        };
      };
      
      404: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": {
            [key: string]: unknown;
          };
        };
      };
      
      422: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["HTTPValidationError"];
        };
      };
    };
  };
  list_summaries_api_summaries__get: {
    parameters: {
      query?: {
        
        run_id?: string | null;
        
        page?: number;
        
        limit?: number;
      };
      header?: never;
      path?: never;
      cookie?: never;
    };
    requestBody?: never;
    responses: {
      
      200: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["SummaryRunListResponse"];
        };
      };
      
      422: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["HTTPValidationError"];
        };
      };
    };
  };
  create_summary_api_summaries__post: {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    requestBody?: never;
    responses: {
      
      501: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": unknown;
        };
      };
    };
  };
  get_cluster_summary_api_summaries__summary_run_id__clusters__cluster_id__get: {
    parameters: {
      query?: {
        
        run_id?: string | null;
      };
      header?: never;
      path: {
        summary_run_id: string;
        cluster_id: number;
      };
      cookie?: never;
    };
    requestBody?: never;
    responses: {
      
      200: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["ClusterSummaryResponse"];
        };
      };
      
      404: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": {
            [key: string]: unknown;
          };
        };
      };
      
      422: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["HTTPValidationError"];
        };
      };
    };
  };
  search_queries_api_search_queries_get: {
    parameters: {
      query: {
        
        text: string;
        
        mode?: "semantic" | "fulltext";
        
        run_id?: string | null;
        
        cluster_ids?: string | null;
        
        within_clusters?: string | null;
        
        top_clusters?: number;
        
        page?: number;
        
        limit?: number;
      };
      header?: never;
      path?: never;
      cookie?: never;
    };
    requestBody?: never;
    responses: {
      
      200: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["SearchQueriesResponse"];
        };
      };
      
      422: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["HTTPValidationError"];
        };
      };
    };
  };
  search_clusters_api_search_clusters_get: {
    parameters: {
      query: {
        
        text: string;
        
        mode?: "semantic" | "fulltext";
        
        run_id?: string | null;
        
        n_results?: number;
        
        page?: number;
      };
      header?: never;
      path?: never;
      cookie?: never;
    };
    requestBody?: never;
    responses: {
      
      200: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["SearchClustersResponse"];
        };
      };
      
      422: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["HTTPValidationError"];
        };
      };
    };
  };
  get_cluster_metadata_api_curation_clusters__cluster_id__metadata_get: {
    parameters: {
      query: {
        
        run_id: string;
      };
      header?: never;
      path: {
        cluster_id: number;
      };
      cookie?: never;
    };
    requestBody?: never;
    responses: {
      
      200: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["ClusterMetadata"];
        };
      };
      
      404: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": {
            [key: string]: unknown;
          };
        };
      };
      
      422: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["HTTPValidationError"];
        };
      };
    };
  };
  get_cluster_history_api_curation_clusters__cluster_id__history_get: {
    parameters: {
      query: {
        
        run_id: string;
        
        page?: number;
        
        limit?: number;
      };
      header?: never;
      path: {
        cluster_id: number;
      };
      cookie?: never;
    };
    requestBody?: never;
    responses: {
      
      200: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["EditHistoryResponse"];
        };
      };
      
      422: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["HTTPValidationError"];
        };
      };
    };
  };
  get_run_audit_api_curation_runs__run_id__audit_get: {
    parameters: {
      query?: {
        
        page?: number;
        
        limit?: number;
      };
      header?: never;
      path: {
        run_id: string;
      };
      cookie?: never;
    };
    requestBody?: never;
    responses: {
      
      200: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["EditHistoryResponse"];
        };
      };
      
      422: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["HTTPValidationError"];
        };
      };
    };
  };
  get_orphaned_queries_api_curation_runs__run_id__orphaned_get: {
    parameters: {
      query?: {
        
        page?: number;
        
        limit?: number;
      };
      header?: never;
      path: {
        run_id: string;
      };
      cookie?: never;
    };
    requestBody?: never;
    responses: {
      
      200: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["OrphanedQueriesResponse"];
        };
      };
      
      422: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": components["schemas"]["HTTPValidationError"];
        };
      };
    };
  };
  move_query_api_curation_queries__query_id__move_post: {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    requestBody?: never;
    responses: {
      
      501: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": unknown;
        };
      };
    };
  };
  rename_cluster_api_curation_clusters__cluster_id__rename_post: {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    requestBody?: never;
    responses: {
      
      501: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": unknown;
        };
      };
    };
  };
  root__get: {
    parameters: {
      query?: never;
      header?: never;
      path?: never;
      cookie?: never;
    };
    requestBody?: never;
    responses: {
      
      200: {
        headers: {
          [name: string]: unknown;
        };
        content: {
          "application/json": unknown;
        };
      };
    };
  };
}
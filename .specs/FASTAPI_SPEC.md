# FastAPI Backend Specification

## 1. Objectives
- Expose the complete CLI surface area (data loading, clustering, search, hierarchy, summarization, editing, classification) over HTTP so external clients and the web UI can operate without shell access.
- Provide a migration path away from Next.js server actions: all data mutations/read operations should be reachable via REST endpoints served by FastAPI.
- Retain parity with classification features (template sync, run orchestration, single predictions) while adding low-latency “classify single” APIs suitable for future dynamic endpoints.
- Serve as the authoritative backend for the Next.js frontend—each page flow should map cleanly onto one or more REST endpoints.

## 2. Scope & CLI Mapping
| CLI Group | Commands | FastAPI Module |
|---|---|---|
| Data ingestion | `lmsys load`, `lmsys clear`, `lmsys backfill-chroma` | `datasets` router |
| Clustering | `lmsys cluster kmeans`, `lmsys cluster hdbscan` | `clustering` router |
| Summaries | `lmsys summarize` | `summaries` router |
| Hierarchy | `lmsys merge-clusters`, `lmsys show-hierarchy` | `hierarchy` router |
| Analysis/Search | `lmsys list`, `lmsys runs`, `lmsys list-clusters`, `lmsys inspect`, `lmsys export`, `lmsys search`, `lmsys search-cluster` | `analysis` & `search` routers |
| Edit/Curation | `lmsys edit ...` | `curation` router |
| Verification | `lmsys verify ...` (optional) | `verify` router (low priority) |
| Classification | `lmsys classify ...` (templates, runs, single) | `classification` router |

The API must cover each group so the CLI can eventually become a thin client. Priority order: datasets → clustering → analysis/search → summaries/hierarchy → classification → edit/verify.

## 3. Service Architecture
- **Application**: FastAPI app (`src/lmsys_query_analysis/api/app.py`) with routers by concern (templates, runs, predictions, reports).
- **Dependencies**:
  - `Database` dependency providing SQLModel session per request (reuse existing `Database` class).
  - `TemplateRepository`, `ClassificationService`, `RunReporter` shared with CLI to avoid duplicated logic.
  - Background task queue (Starlette `BackgroundTasks` or asyncio TaskGroup) for non-blocking run execution.
- **Authentication**: Initially optional; stub middleware to add auth later (API key header or bearer token).
- **Error Handling**: Consistent error envelopes `{ "error": { "type": str, "message": str } }` with HTTP status codes.

## 4. Data Models (Pydantic Schemas)
Define request/response models under `src/lmsys_query_analysis/api/schemas.py`. Field names use snake_case to match the existing SQLModel layer.

### 4.1 Base Classes & Common Patterns

```python
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union, Generic, TypeVar
from pydantic import BaseModel, Field

T = TypeVar('T')

# Generic pagination wrapper
class PaginatedResponse(BaseModel, Generic[T]):
    items: List[T]
    total: int
    page: int
    pages: int
    limit: int

# Common status patterns
class JobStatus(BaseModel):
    status: Literal["pending", "running", "completed", "failed"]
    created_at: datetime
    completed_at: Optional[datetime] = None

class LLMConfig(BaseModel):
    provider: str
    model: str
    concurrency: Optional[int] = None
    rpm: Optional[int] = None

class EmbeddingConfig(BaseModel):
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None
    embed_batch_size: Optional[int] = None
    use_chroma: Optional[bool] = None

# Generic operation response
class OperationResponse(BaseModel):
    status: str
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

# Common curation request base
class CurationRequest(BaseModel):
    reason: Optional[str] = None
```

### 4.2 Templates

```python
class TemplateSummary(BaseModel):
    template_id: str
    name: str
    version: Optional[str] = None
    description: Optional[str] = None
    source: Literal["db", "yaml"]
    updated_at: datetime
    template_hash: str

class TemplateDetail(TemplateSummary):
    yaml_content: str
    labels: List[str]
    examples: Optional[List[Dict[str, Any]]] = None

class TemplateCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    version: Optional[str] = None
    labels: List[str]
    yaml_content: Optional[str] = None
    prompt: Optional[str] = None
    schema_module: Optional[str] = None
    examples: Optional[List[Dict[str, Any]]] = None

class TemplateUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    version: Optional[str] = None
    labels: Optional[List[str]] = None
    yaml_content: Optional[str] = None
    prompt: Optional[str] = None
    schema_module: Optional[str] = None
    examples: Optional[List[Dict[str, Any]]] = None
    template_hash: str  # optimistic concurrency token

# Use generic pagination and operation responses
TemplateListResponse = PaginatedResponse[TemplateSummary]
TemplateSyncResponse = OperationResponse  # templates in data field
TemplateDeleteResponse = OperationResponse
```

### 4.3 Dataset Operations

```python
class DatasetLoadRequest(EmbeddingConfig):
    limit: Optional[int] = None
    skip_existing: Optional[bool] = None
    use_streaming: Optional[bool] = None

class DatasetUploadMetadata(BaseModel):
    format: Optional[Literal["jsonl", "csv"]] = None
    description: Optional[str] = None

class DatasetJobStatus(JobStatus):
    job_id: str
    job_type: Literal["load", "upload", "backfill"]
    processed: Optional[int] = None
    errors: Optional[List[str]] = None

class DatasetStatusResponse(BaseModel):
    active_jobs: List[DatasetJobStatus]
    last_completed_job: Optional[DatasetJobStatus] = None
    total_records: Optional[int] = None
    last_updated_at: Optional[datetime] = None

# Use operation response
DatasetDeleteResponse = OperationResponse
```

### 4.4 Clustering

```python
class ClusteringRunBase(JobStatus):
    run_id: str
    algorithm: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

class KMeansRequest(EmbeddingConfig):
    n_clusters: int
    description: Optional[str] = None
    chunk_size: Optional[int] = None
    mb_batch_size: Optional[int] = None

class HDBSCANRequest(EmbeddingConfig):
    description: Optional[str] = None
    chunk_size: Optional[int] = None
    min_cluster_size: Optional[int] = None
    min_samples: Optional[int] = None
    cluster_selection_epsilon: Optional[float] = None
    metric: Optional[str] = None

class SegmentationRequest(BaseModel):
    algorithm: str
    parameters: Dict[str, Any]

class ClusteringRunSummary(ClusteringRunBase):
    num_clusters: Optional[int] = None

class ClusteringRunDetail(ClusteringRunSummary):
    metrics: Optional[Dict[str, Any]] = None
    latest_errors: Optional[List[str]] = None

class ClusteringRunStatusResponse(BaseModel):
    run_id: str
    status: str
    processed: Optional[int] = None

# Use generic pagination
ClusteringRunListResponse = PaginatedResponse[ClusteringRunSummary]
```

### 4.5 Summaries

```python
class SummaryRequest(LLMConfig):
    run_id: str
    alias: Optional[str] = None
    max_queries: Optional[int] = None

class SummaryRunSummary(JobStatus):
    summary_run_id: str
    run_id: str
    alias: Optional[str] = None
    model: str
    generated_at: datetime

class ClusterSummaryResponse(BaseModel):
    run_id: str
    cluster_id: int
    title: Optional[str] = None
    description: Optional[str] = None
    summary_run_id: Optional[str] = None
    alias: Optional[str] = None
    num_queries: Optional[int] = None
    representative_queries: Optional[List[str]] = None

# Use generic pagination
SummaryRunListResponse = PaginatedResponse[SummaryRunSummary]
```

### 4.6 Hierarchy

```python
class HierarchyRequest(LLMConfig):
    run_id: str
    neighborhood_size: Optional[int] = None

class HierarchyNode(BaseModel):
    hierarchy_run_id: str
    run_id: str
    cluster_id: int
    parent_cluster_id: Optional[int] = None
    level: int
    children_ids: List[int]
    title: Optional[str] = None
    description: Optional[str] = None

class HierarchyRunInfo(BaseModel):
    hierarchy_run_id: str
    run_id: str
    created_at: datetime

class HierarchyTreeResponse(BaseModel):
    nodes: List[HierarchyNode]

class HierarchyStatusResponse(BaseModel):
    hierarchy_run_id: str
    status: Literal["pending", "running", "completed", "failed"]

# Use generic pagination  
HierarchyListResponse = PaginatedResponse[HierarchyRunInfo]
```

### 4.7 Analysis & Search

```python
class QueryResponse(BaseModel):
    id: int
    conversation_id: str
    model: str
    query_text: str
    language: Optional[str] = None
    timestamp: Optional[datetime] = None
    created_at: datetime

class ClusterSearchResult(BaseModel):
    run_id: str
    cluster_id: int
    title: Optional[str] = None
    description: Optional[str] = None
    summary: Optional[str] = None
    num_queries: Optional[int] = None

class ClusterInfo(BaseModel):
    run_id: str
    cluster_id: int
    title: Optional[str] = None
    confidence_score: Optional[float] = None

class QuerySearchResult(BaseModel):
    query: QueryResponse
    clusters: List[ClusterInfo]

class ExportResponse(BaseModel):
    download_url: str
    format: Literal["csv", "json"]
    row_count: int

class ClusterDetailResponse(BaseModel):
    cluster: ClusterSummaryResponse
    queries: PaginatedResponse[QueryResponse]

class QueryDetailResponse(BaseModel):
    query: QueryResponse
    clusters: List[ClusterInfo]

# Use generic pagination for consistent responses
PaginatedQueriesResponse = PaginatedResponse[QueryResponse]
SearchQueriesResponse = PaginatedResponse[QuerySearchResult]
SearchClustersResponse = PaginatedResponse[ClusterSearchResult]
ClusterListResponse = PaginatedResponse[ClusterSummaryResponse]

# Backward compatibility aliases
QuerySearchResponse = SearchQueriesResponse
ClustersSearchResponse = SearchClustersResponse
```

### 4.8 Curation & Verification

```python
class MoveQueryRequest(CurationRequest):
    to_cluster_id: int

class MoveQueriesRequest(CurationRequest):
    query_ids: List[int]
    to_cluster_id: int

class MoveQueryResponse(BaseModel):
    from_cluster_id: int
    to_cluster_id: int

class MoveQueryError(BaseModel):
    query_id: int
    error: str

class MoveQueriesResponse(BaseModel):
    moved: int
    failed: int
    errors: List[MoveQueryError]

class RenameClusterRequest(CurationRequest):
    title: Optional[str] = None
    description: Optional[str] = None

class RenameClusterResponse(BaseModel):
    old_title: Optional[str] = None
    new_title: Optional[str] = None
    old_description: Optional[str] = None
    new_description: Optional[str] = None

class MergeClustersRequest(CurationRequest):
    source_cluster_ids: List[int]
    target_cluster_id: int
    new_title: Optional[str] = None
    new_description: Optional[str] = None

class MergeClustersResponse(BaseModel):
    source_cluster_ids: List[int]
    target_cluster_id: int
    queries_moved: int
    new_title: Optional[str] = None

class SplitClusterRequest(CurationRequest):
    query_ids: List[int]
    new_title: str
    new_description: str

class SplitClusterResponse(BaseModel):
    original_cluster_id: int
    new_cluster_id: int
    queries_moved: int

class DeleteClusterRequest(CurationRequest):
    orphan: bool
    move_to_cluster_id: Optional[int] = None

class DeleteClusterResponse(BaseModel):
    query_count: int
    status: Literal["orphaned", "moved"]
    moved_to: Optional[int] = None

class ClusterMetadataRequest(BaseModel):
    coherence_score: Optional[int] = None
    quality: Optional[Literal["high", "medium", "low"]] = None
    notes: Optional[str] = None

class ClusterFlagsRequest(BaseModel):
    flags: List[str]

class ClusterMetadata(ClusterMetadataRequest):
    flags: Optional[List[str]] = None

class VerificationRequest(BaseModel):
    run_id: str
    chroma_path: Optional[str] = None

class VerificationResponse(BaseModel):
    run_id: str
    status: Literal["ok", "mismatch"]
    issues: List[str]
    sqlite_summary_count: Optional[int] = None
    chroma_summary_count: Optional[int] = None
    embedding_space: Dict[str, Any]

class EditHistoryRecord(BaseModel):
    timestamp: datetime
    cluster_id: Optional[int] = None
    edit_type: str
    editor: str
    reason: Optional[str] = None

class OrphanInfo(BaseModel):
    orphan: Dict[str, Any]
    query: QueryResponse

# Use generic and operation responses
EditHistoryResponse = PaginatedResponse[EditHistoryRecord]
OrphanedQueriesResponse = PaginatedResponse[OrphanInfo]
ProblematicClustersResponse = PaginatedResponse[ClusterSummaryResponse]
ClusterMetadataResponse = OperationResponse  # metadata in data field
ClusterFlagsResponse = OperationResponse     # flags in data field
```

### 4.9 Classification Runs

```python
class RunFilters(BaseModel):
    run_id: Optional[str] = None
    cluster_ids: Optional[List[int]] = None
    query_ids: Optional[List[int]] = None
    sql_filter: Optional[str] = None

class RunExecution(BaseModel):
    mode: Optional[Literal["sync", "batch"]] = None
    concurrency: Optional[int] = None
    rpm: Optional[int] = None

class ClassificationRunBase(JobStatus):
    template_id: str
    template_hash: str
    alias: Optional[str] = None
    provider: str
    model: str

class ClassificationRunSummary(ClassificationRunBase):
    classification_run_id: str
    total_queries: Optional[int] = None
    processed_queries: Optional[int] = None
    error_count: Optional[int] = None

class ClassificationRunDetail(ClassificationRunSummary):
    parameters: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    latest_errors: Optional[List[str]] = None

class RunCreateRequest(LLMConfig):
    template_id: str
    alias: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    filters: RunFilters
    execution: Optional[RunExecution] = None

class RunResumeRequest(BaseModel):
    execution: Optional[RunExecution] = None

class RunStats(BaseModel):
    label_distribution: Dict[str, int]
    cluster_distribution: Dict[int, Dict[str, int]]
    query_count: int
    template_hash: str

class ClassificationResult(BaseModel):
    query: QueryResponse
    label: str
    reasoning: Optional[str] = None
    created_at: datetime

class ClassificationRunStatusResponse(BaseModel):
    run_id: str
    status: str
    processed: Optional[int] = None

class BatchIngestRequest(BaseModel):
    output_path: str

# Use generic pagination and operation responses
ClassificationRunListResponse = PaginatedResponse[ClassificationRunSummary]
ClassificationResultPage = PaginatedResponse[ClassificationResult]
BatchIngestResponse = OperationResponse
CancelResponse = OperationResponse
StatusResponse = OperationResponse
```

### 4.10 Predictions

```python

class SinglePredictionRequest(BaseModel):
    template_id: Optional[str] = None
    template_yaml: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    text: str

class SinglePredictionResponse(BaseModel):
    label: str
    reasoning: Optional[str] = None
    template_hash: str

class BatchPredictionItem(BaseModel):
    id: str
    text: str

class BatchPredictionRequest(BaseModel):
    template_id: Optional[str] = None
    template_yaml: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    items: List[BatchPredictionItem]

class BatchPredictionResultItem(BaseModel):
    id: str
    label: str
    reasoning: Optional[str] = None
    error: Optional[str] = None
```

### 4.11 Reports & Pivoting

```python
class ClusterToLabelCount(BaseModel):
    cluster_id: int
    counts: Dict[str, int]

class LabelToClusterCount(BaseModel):
    label: str
    counts: Dict[int, int]

class LabelPivotResponse(BaseModel):
    clusters_to_labels: List[ClusterToLabelCount]
    labels_to_clusters: List[LabelToClusterCount]
```

## 5. API Endpoints

### 5.1 Dataset Operations

| Method | Path | Request Model | Response Model | Notes |
|---|---|---|---|---|
| POST | `/api/datasets/load` | `DatasetLoadRequest` | `DatasetJobStatus` | 202 Accepted for async job |
| POST | `/api/datasets/upload` | multipart file + `DatasetUploadMetadata` | `DatasetJobStatus` | Multipart file upload |
| POST | `/api/datasets/backfill-chroma` | — | `DatasetJobStatus` | Optional provider/model query params |
| GET | `/api/datasets/status` | — | `DatasetStatusResponse` | `job_id` query param filters by job |
| DELETE | `/api/datasets` | — | `DatasetDeleteResponse` | Requires `confirm_token` query param |

### 5.2 Clustering & Segmentation Operations

| Method | Path | Request Model | Response Model | Notes |
|---|---|---|---|---|
| POST | `/api/clustering/kmeans` | `KMeansRequest` | `ClusteringRunStatusResponse` | Mirrors `lmsys cluster kmeans` |
| POST | `/api/clustering/hdbscan` | `HDBSCANRequest` | `ClusteringRunStatusResponse` | Mirrors `lmsys cluster hdbscan` |
| POST | `/api/clustering/segment` | `SegmentationRequest` | `ClusteringRunStatusResponse` | Placeholder for segmentation CLI |
| GET | `/api/clustering/runs` | — | `ClusteringRunListResponse` | Filters: `algorithm`, `status`, pagination |
| GET | `/api/clustering/runs/{run_id}` | — | `ClusteringRunDetail` | Fetch run metadata |
| GET | `/api/clustering/runs/{run_id}/status` | — | `ClusteringRunStatusResponse` | Poll clustering job |

### 5.3 Summaries (LLM Titles/Descriptions)

| Method | Path | Request Model | Response Model | Notes |
|---|---|---|---|---|
| POST | `/api/summaries` | `SummaryRequest` | `StatusResponse` | Launch summarization job |
| GET | `/api/summaries` | — | `SummaryRunListResponse` | Supports filters |
| GET | `/api/summaries/{summary_run_id}` | — | `SummaryRunSummary` | — |
| GET | `/api/summaries/{summary_run_id}/clusters/{cluster_id}` | — | `ClusterSummaryResponse` | — |

### 5.4 Hierarchy (Cluster Merging)

| Method | Path | Request Model | Response Model | Notes |
|---|---|---|---|---|
| POST | `/api/hierarchy` | `HierarchyRequest` | `HierarchyStatusResponse` | Initiates merge job |
| GET | `/api/hierarchy` | — | `HierarchyListResponse` | List hierarchy runs |
| GET | `/api/hierarchy/{hierarchy_run_id}` | — | `HierarchyTreeResponse` | Fetch hierarchy tree |

### 5.5 Analysis & Search

| Method | Path | Request Model | Response Model | Notes |
|---|---|---|---|---|
| GET | `/api/queries` | — | `PaginatedQueriesResponse` | Filters via query params |
| GET | `/api/clustering/runs/{run_id}/clusters` | — | `ClusterListResponse` | Supports `limit`, `include_examples` |
| GET | `/api/clustering/runs/{run_id}/clusters/{cluster_id}` | — | `ClusterDetailResponse` | Includes paginated queries |
| GET | `/api/search` | — | `SearchQueriesResponse` | Requires `text` query param |
| GET | `/api/search/clusters` | — | `SearchClustersResponse` | Cluster summary search |
| GET | `/api/export` | — | `ExportResponse` | Returns download metadata |

### 5.6 Edit, Curation & Verification

| Method | Path | Request Model | Response Model | Notes |
|---|---|---|---|---|
| GET | `/api/curation/queries/{query_id}` | — | `QueryDetailResponse` | Mirrors `lmsys edit view-query` |
| POST | `/api/curation/queries/{query_id}/move` | `MoveQueryRequest` | `MoveQueryResponse` | Requires `run_id` query param |
| POST | `/api/curation/queries/move` | `MoveQueriesRequest` | `MoveQueriesResponse` | Batch move queries |
| POST | `/api/curation/clusters/{cluster_id}/rename` | `RenameClusterRequest` | `RenameClusterResponse` | `run_id` query param |
| POST | `/api/curation/clusters/{cluster_id}/merge` | `MergeClustersRequest` | `MergeClustersResponse` | Target cluster via path |
| POST | `/api/curation/clusters/{cluster_id}/split` | `SplitClusterRequest` | `SplitClusterResponse` | Creates new cluster |
| POST | `/api/curation/clusters/{cluster_id}/delete` | `DeleteClusterRequest` | `DeleteClusterResponse` | Supports orphaning or reassignment |
| POST | `/api/curation/clusters/{cluster_id}/metadata` | `ClusterMetadataRequest` | `ClusterMetadataResponse` | Upsert metadata |
| POST | `/api/curation/clusters/{cluster_id}/flags` | `ClusterFlagsRequest` | `ClusterFlagsResponse` | Add/remove flags |
| GET | `/api/curation/clusters/{cluster_id}/history` | — | `EditHistoryResponse` | Requires `run_id` query param |
| GET | `/api/curation/runs/{run_id}/audit` | — | `EditHistoryResponse` | Optional `since` query param |
| GET | `/api/curation/runs/{run_id}/orphaned` | — | `OrphanedQueriesResponse` | Orphaned query list |
| GET | `/api/curation/runs/{run_id}/select-bad-clusters` | — | `ProblematicClustersResponse` | Quality heuristics |
| POST | `/api/verify/runs/{run_id}` | `VerificationRequest` | `VerificationResponse` | Mirrors `lmsys verify sync` |

### 5.7 Template Management (Classification)

| Method | Path | Request Model | Response Model | Notes |
|---|---|---|---|---|
| GET | `/api/templates` | — | `TemplateListResponse` | Filters via query params |
| POST | `/api/templates` | `TemplateCreateRequest` | `TemplateDetail` | Create or overwrite template |
| GET | `/api/templates/{template_id}` | — | `TemplateDetail` | — |
| PUT | `/api/templates/{template_id}` | `TemplateUpdateRequest` | `TemplateDetail` | Requires template hash |
| DELETE | `/api/templates/{template_id}` | — | `TemplateDeleteResponse` | Soft delete optional |
| POST | `/api/templates/sync` | multipart YAML files | `TemplateSyncResponse` | Bulk import/export |
| GET | `/api/templates/{template_id}/export` | — | YAML stream | Download template |

### 5.8 Classification Runs

| Method | Path | Request Model | Response Model | Notes |
|---|---|---|---|---|
| GET | `/api/classification/runs` | — | `ClassificationRunListResponse` | Filters via query params |
| POST | `/api/classification/runs` | `RunCreateRequest` | `ClassificationRunSummary` | Launch new run |
| GET | `/api/classification/runs/{run_id}` | — | `ClassificationRunDetail` | — |
| POST | `/api/classification/runs/{run_id}/resume` | `RunResumeRequest` | `ClassificationRunSummary` | Resume failed/paused |
| POST | `/api/classification/runs/{run_id}/cancel` | — | `CancelResponse` | Graceful cancel |
| GET | `/api/classification/runs/{run_id}/stats` | — | `RunStats` | Aggregated metrics |
| GET | `/api/classification/runs/{run_id}/results` | — | `ClassificationResultPage` | Query params: `label`, `query`, pagination |
| GET | `/api/classification/runs/{run_id}/status` | — | `ClassificationRunStatusResponse` | Poll status |
| POST | `/api/classification/runs/{run_id}/ingest` | `BatchIngestRequest` | `BatchIngestResponse` | Provider batch ingest |

### 5.9 Predictions

| Method | Path | Request Model | Response Model | Notes |
|---|---|---|---|---|
| POST | `/api/predict` | `SinglePredictionRequest` | `SinglePredictionResponse` | No persistence |
| POST | `/api/predict/batch` | `BatchPredictionRequest` | Stream of `BatchPredictionResultItem` | NDJSON streaming |

### 5.10 Reports & Pivoting

| Method | Path | Request Model | Response Model | Notes |
|---|---|---|---|---|
| GET | `/api/classification/runs/{run_id}/clusters/{cluster_id}` | — | `ClassificationResultPage` | Supports `page`, `limit`, `search` |
| GET | `/api/classification/runs/{run_id}/labels/{label}` | — | `ClassificationResultPage` | Filter by label |
| GET | `/api/classification/runs/{run_id}/pivot` | — | `LabelPivotResponse` | Cluster ↔ label distributions |

## 6. Execution Model
- **Dataset & Clustering Jobs**: Long-running dataset loads and clustering runs execute via background tasks; clients poll `/api/datasets/status` or `/api/clustering/runs/{run_id}/status` for progress.
- **Synchronous Runs**: API triggers a background task that reuses the existing service layer; clients poll `/api/classification/runs/{run_id}` (or `/api/clustering/runs/{run_id}` for clustering jobs) or subscribe to WebSocket/Server-Sent Events (future enhancement) for status updates.
- **Batch Mode**: When `execution.mode="batch"`, service submits jobs to provider (OpenAI Batch, etc.) and exposes `/api/classification/runs/{run_id}/status` plus `/api/classification/runs/{run_id}/ingest` endpoints for result ingestion.
- **Single Prediction**: Directly calls ProviderClient via `InstructorFactory`; no DB writes.

## 7. Integration with Web UI
- Replace existing Next.js server actions with REST calls:
  - **Home / Runs overview** (`web/app/page.tsx`, jobs table) → `GET /api/clustering/runs`, `GET /api/classification/runs`.
  - **Clustering run detail** (`web/app/runs/[runId]/page.tsx`) → `GET /api/clustering/runs/{run_id}`, `GET /api/clustering/runs/{run_id}/clusters`, `GET /api/hierarchy/{hierarchy_run_id}`, `GET /api/summaries`.
  - **Cluster detail** (`web/app/clusters/[runId]/[clusterId]`) → `GET /api/clustering/runs/{run_id}/clusters/{cluster_id}`, `GET /api/queries`, `GET /api/curation/clusters/{cluster_id}/history`.
  - **Search** (`web/app/search`) → `GET /api/search`, `GET /api/search/clusters`.
  - **Classification hub** (new pages) → `GET /api/templates`, `POST /api/templates`, `GET /api/classification/runs`, `GET /api/classification/runs/{run_id}`, `GET /api/classification/runs/{run_id}/results`, `GET /api/classification/runs/{run_id}/pivot`.
  - **Template creation/edit modal** → `POST /api/templates`, `PUT /api/templates/{template_id}`, `POST /api/classification/runs` (optional immediate execution).
  - **Curation panels** → `POST /api/curation/...` endpoints for rename/move/tags; `GET /api/curation/.../history` for audit trails.
  - **Admin / data management** (optional tooling) → `POST /api/datasets/load`, `GET /api/datasets/status`, `DELETE /api/datasets`.
- Provide shared client utilities (e.g., React Query hooks) that wrap the REST endpoints and handle polling for long-running jobs.

## 8. Operational Considerations
- **Authentication & Rate Limiting**: Start with unauthenticated API; design for easy insertion of API key middleware. Consider per-IP throttling for `/api/predict`.
- **Observability**: Add structured logging and metrics (`DatasetLoadStarted`, `ClusteringRunStarted`, `SummaryGenerated`, `ClassificationTemplateSynced`, `ClassificationRunStarted`, `PredictionServed`). Expose health check at `/api/health` returning DB connectivity + queue backlog status.
- **Migration**: Since database can be reset during development, provide `/api/dev/reset` (guarded by env flag) to drop and recreate tables.
- **Testing**: Add pytest integration tests hitting the FastAPI client, covering template lifecycle, run creation, prediction, and pivot endpoints.

## 9. Open Questions
- Do we need WebSocket/SSE streaming for long-running runs, or is polling sufficient initially?
- Should `/api/predict/batch` persist results for later retrieval, or remain stateless?
- How should authentication integrate with existing CLI workflows (shared API key vs. per-user tokens)?
- Do we want to expose verification/edit operations (currently lower priority) before or after the first API rollout?

---
This specification enumerates the FastAPI surface area needed to mirror the full CLI—including data ingestion, clustering, analysis, curation, and classification—so the web frontend and external clients can orchestrate the entire LMSYS workflow through a single service layer.

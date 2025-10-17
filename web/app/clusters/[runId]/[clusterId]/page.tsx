import Link from "next/link";
import { notFound } from "next/navigation";
import { apiFetch } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ClusterQueriesClient } from "./cluster-queries-client";
import { ClusterMetadataPanel } from "./cluster-metadata-panel";
import { EditHistoryPanel } from "./edit-history-panel";
import type { components } from "@/lib/api/types";

type ClusterSummary = components["schemas"]["ClusterSummaryResponse"];
type Query = components["schemas"]["QueryResponse"];
type ClusterMetadata = components["schemas"]["ClusterMetadata"];
type ClusterEdit = components["schemas"]["EditHistoryRecord"];

interface ClusterPageProps {
  params: Promise<{ runId: string; clusterId: string }>;
  searchParams: Promise<{ page?: string }>;
}

export default async function ClusterPage({ params, searchParams }: ClusterPageProps) {
  const { runId, clusterId: clusterIdStr } = await params;
  const { page: pageStr } = await searchParams;

  const clusterId = parseInt(clusterIdStr);
  const page = parseInt(pageStr || "1");

  // Single API call for cluster detail (summary + queries)
  const clusterDetail = await apiFetch<{
    cluster: ClusterSummary;
    queries: {
      items: Query[];
      total: number;
      page: number;
      pages: number;
      limit: number;
    };
  }>(`/api/clustering/runs/${runId}/clusters/${clusterId}?page=${page}&limit=50`);

  if (!clusterDetail || (clusterDetail.queries.items.length === 0 && page === 1)) {
    notFound();
  }

  const summary = clusterDetail.cluster;
  const queriesData = clusterDetail.queries;

  // Fetch cluster metadata and edit history
  const metadata = await apiFetch<ClusterMetadata>(
    `/api/curation/clusters/${clusterId}/metadata?run_id=${runId}`
  );
  const historyResponse = await apiFetch<{
    items: ClusterEdit[];
    total: number;
  }>(`/api/curation/clusters/${clusterId}/history?run_id=${runId}`);
  const editHistory = historyResponse.items;

  return (
    <div className="container mx-auto py-8 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <Link href={`/runs/${runId}`}>
            <Button variant="ghost" size="sm">
              ← Back to run
            </Button>
          </Link>
          <h1 className="text-3xl font-bold mt-2">{summary?.title || `Cluster ${clusterId}`}</h1>
          {!summary && (
            <p className="text-sm text-muted-foreground mt-1">
              Run{" "}
              <code className="text-xs bg-muted px-1 py-0.5 rounded">lmsys summarize {runId}</code>{" "}
              to generate cluster summaries
            </p>
          )}
        </div>
        <div className="flex gap-2">
          <Badge variant="outline">Cluster ID: {clusterId}</Badge>
          {summary?.num_queries ? (
            <Badge variant="secondary">{summary.num_queries} queries</Badge>
          ) : (
            <Badge variant="secondary">{queriesData.total} queries</Badge>
          )}
        </div>
      </div>

      {summary?.description && (
        <Card>
          <CardHeader>
            <CardTitle>Description</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm">{summary.description}</p>
          </CardContent>
        </Card>
      )}

      {summary?.representative_queries && summary.representative_queries.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Representative Queries</CardTitle>
            <CardDescription>Examples that best represent this cluster</CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {summary.representative_queries.slice(0, 5).map((query: string, idx: number) => (
                <li key={idx} className="text-sm border-l-2 border-muted pl-4">
                  {query}
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <ClusterMetadataPanel metadata={metadata} runId={runId} clusterId={clusterId} />
        <EditHistoryPanel edits={editHistory} runId={runId} clusterId={clusterId} />
      </div>

      <Card>
        <CardHeader>
          <CardTitle>All Queries</CardTitle>
          <CardDescription>Browse all queries in this cluster</CardDescription>
        </CardHeader>
        <CardContent>
          <ClusterQueriesClient runId={runId} clusterId={clusterId} initialData={queriesData} />
        </CardContent>
      </Card>
    </div>
  );
}

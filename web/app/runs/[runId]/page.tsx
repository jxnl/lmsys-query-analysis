import Link from "next/link";
import { notFound } from "next/navigation";
import { apiFetch } from "@/lib/api";
import { HierarchyTree } from "@/components/hierarchy-tree";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { components } from "@/lib/api/types";

type ClusteringRun = components["schemas"]["ClusteringRunSummary"];
type HierarchyListItem = components["schemas"]["HierarchyNode"];
type HierarchyNode = components["schemas"]["HierarchyNode"];

interface RunPageProps {
  params: Promise<{ runId: string }>;
}

export default async function RunPage({ params }: RunPageProps) {
  const { runId } = await params;
  const run = await apiFetch<ClusteringRun>(`/api/clustering/runs/${runId}`);

  if (!run) {
    notFound();
  }

  const hierarchiesResponse = await apiFetch<{
    items: HierarchyListItem[];
    total: number;
  }>(`/api/hierarchy?run_id=${runId}`);
  const latestHierarchy = hierarchiesResponse.items[0];

  let hierarchyTree = null;
  if (latestHierarchy) {
    const treeResponse = await apiFetch<{
      nodes: HierarchyNode[];
      hierarchy_run_id: string;
      run_id: string;
      total_queries: number;
    }>(`/api/hierarchy/${latestHierarchy.hierarchy_run_id}`);
    hierarchyTree = treeResponse.nodes;
  }

  const runParams = run.parameters as Record<string, unknown> | null;

  return (
    <div className="container mx-auto py-8 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <Link href="/">
            <Button variant="ghost" size="sm">
              ← Back to runs
            </Button>
          </Link>
          <h1 className="text-3xl font-bold mt-2">{run.run_id}</h1>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">Algorithm</CardTitle>
          </CardHeader>
          <CardContent>
            <Badge variant="outline">{run.algorithm}</Badge>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">Clusters</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{run.num_clusters || "N/A"}</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">Created</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm">
              {new Date(run.created_at).toLocaleDateString("en-US", {
                year: "numeric",
                month: "long",
                day: "numeric",
              })}
            </p>
          </CardContent>
        </Card>
      </div>

      {runParams && (
        <Card>
          <CardHeader>
            <CardTitle>Parameters</CardTitle>
          </CardHeader>
          <CardContent>
            <dl className="grid grid-cols-2 gap-4 text-sm">
              {Object.entries(runParams).map(([key, value]) => (
                <div key={key}>
                  <dt className="font-medium text-muted-foreground">{key}</dt>
                  <dd className="mt-1 font-mono text-xs">
                    {typeof value === "object" ? JSON.stringify(value) : String(value)}
                  </dd>
                </div>
              ))}
            </dl>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Cluster Hierarchy</CardTitle>
          {hierarchiesResponse.items.length > 1 && (
            <p className="text-sm text-muted-foreground">
              Showing latest hierarchy ({hierarchiesResponse.items.length} total)
            </p>
          )}
        </CardHeader>
        <CardContent>
          {hierarchyTree ? (
            <HierarchyTree
              nodes={hierarchyTree}
              runId={runId}
              hierarchyRunId={latestHierarchy.hierarchy_run_id}
            />
          ) : (
            <p className="text-muted-foreground text-center py-8">
              No hierarchy found. Run{" "}
              <code className="bg-muted px-2 py-1 rounded">lmsys merge-clusters {runId}</code> to
              create one.
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
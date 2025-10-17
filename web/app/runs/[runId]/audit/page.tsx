import Link from "next/link";
import { notFound } from "next/navigation";
import { apiFetch } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { components } from "@/lib/api/types";

type ClusteringRun = components["schemas"]["ClusteringRunSummary"];
type ClusterEdit = components["schemas"]["EditHistoryRecord"];

interface AuditLogPageProps {
  params: Promise<{ runId: string }>;
}

export default async function AuditLogPage({ params }: AuditLogPageProps) {
  const { runId } = await params;

  // Fetch run metadata
  const run = await apiFetch<ClusteringRun>(`/api/clustering/runs/${runId}`);
  if (!run) {
    notFound();
  }

  // Fetch all edits for this run
  const editsResponse = await apiFetch<{
    items: ClusterEdit[];
    total: number;
  }>(`/api/curation/runs/${runId}/audit`);
  const edits = editsResponse.items;

  const getEditTypeBadge = (editType: string) => {
    const variants: Record<
      string,
      {
        variant: "default" | "secondary" | "destructive" | "outline";
        label: string;
      }
    > = {
      rename: { variant: "secondary", label: "Rename" },
      move_query: { variant: "default", label: "Move Query" },
      merge: { variant: "secondary", label: "Merge" },
      split: { variant: "secondary", label: "Split" },
      delete: { variant: "destructive", label: "Delete" },
      tag: { variant: "outline", label: "Tag" },
    };

    const config = variants[editType] || {
      variant: "outline" as const,
      label: editType,
    };
    return <Badge variant={config.variant}>{config.label}</Badge>;
  };

  return (
    <div className="container mx-auto py-8 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <Link href={`/runs/${runId}`}>
            <Button variant="ghost" size="sm">
              ← Back to run
            </Button>
          </Link>
          <h1 className="text-3xl font-bold mt-2">Audit Log</h1>
          <p className="text-muted-foreground mt-1">Edit history for run {runId}</p>
        </div>
        <Badge variant="secondary">{edits.length} edits</Badge>
      </div>

      {edits.length === 0 ? (
        <Card>
          <CardHeader>
            <CardTitle>No Edits</CardTitle>
            <CardDescription>
              No curation operations have been performed on this run
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Use <code className="text-xs bg-muted px-1 py-0.5 rounded">lmsys edit</code> commands
              to curate clusters. All changes will appear here.
            </p>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardHeader>
            <CardTitle>All Edits ({edits.length})</CardTitle>
            <CardDescription>Complete audit trail of all curation operations</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {edits.map((edit, index) => (
                <div key={index} className="border rounded-lg p-4">
                  <div className="flex items-start justify-between gap-4 mb-2">
                    <div className="flex items-center gap-2">
                      {getEditTypeBadge(edit.edit_type)}
                      {edit.cluster_id && (
                        <Link href={`/clusters/${runId}/${edit.cluster_id}`}>
                          <Badge variant="outline" className="hover:bg-accent cursor-pointer">
                            Cluster {edit.cluster_id}
                          </Badge>
                        </Link>
                      )}
                      <span className="text-xs text-muted-foreground">by {edit.editor}</span>
                    </div>
                    <span className="text-xs text-muted-foreground">
                      {new Date(edit.timestamp).toLocaleString()}
                    </span>
                  </div>

                  {edit.reason && (
                    <p className="text-sm text-muted-foreground mb-2">{edit.reason}</p>
                  )}

                  {/* Show detailed changes */}
                  {edit.edit_type === "rename" && edit.old_value && edit.new_value && (
                    <div className="mt-2 p-2 bg-muted rounded text-xs">
                      <div className="font-medium mb-1">Title Change:</div>
                      <div className="line-through text-muted-foreground">
                        {(edit.old_value as Record<string, unknown>).title as string}
                      </div>
                      <div>→ {(edit.new_value as Record<string, unknown>).title as string}</div>
                    </div>
                  )}

                  {edit.edit_type === "move_query" && edit.old_value && edit.new_value && (
                    <div className="mt-2 p-2 bg-muted rounded text-xs">
                      <div className="font-medium mb-1">Query Movement:</div>
                      <div>
                        Query {(edit.old_value as Record<string, unknown>).query_id as string}: Cluster{" "}
                        {(edit.old_value as Record<string, unknown>).cluster_id as number} → {(edit.new_value as Record<string, unknown>).cluster_id as number}
                      </div>
                    </div>
                  )}

                  {edit.edit_type === "merge" && edit.old_value && edit.new_value && (
                    <div className="mt-2 p-2 bg-muted rounded text-xs">
                      <div className="font-medium mb-1">Cluster Merge:</div>
                      <div>
                        Merged {((edit.old_value as Record<string, unknown>).source_clusters as string[] || []).join(", ")} →
                        Cluster {(edit.new_value as Record<string, unknown>).target_cluster as number}
                      </div>
                      <div className="text-muted-foreground">
                        {(edit.new_value as Record<string, unknown>).queries_moved as number} queries moved
                      </div>
                    </div>
                  )}

                  {edit.edit_type === "tag" && edit.new_value && (
                    <div className="mt-2 p-2 bg-muted rounded text-xs">
                      <div className="font-medium mb-1">Metadata Update:</div>
                      {(edit.new_value as Record<string, unknown>).quality && (
                        <div>Quality: {(edit.new_value as Record<string, unknown>).quality as string}</div>
                      )}
                      {(edit.new_value as Record<string, unknown>).coherence_score && (
                        <div>Coherence: {(edit.new_value as Record<string, unknown>).coherence_score as number}/5</div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      <div className="text-sm text-muted-foreground">
        <p>
          View the full audit log with:{" "}
          <code className="bg-muted px-1 py-0.5 rounded text-xs">lmsys edit audit {runId}</code>
        </p>
      </div>
    </div>
  );
}

import Link from 'next/link';
import { notFound } from 'next/navigation';
import { getRun, getClusterEditHistory } from '@/app/actions';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface AuditLogPageProps {
  params: Promise<{ runId: string }>;
}

export default async function AuditLogPage({ params }: AuditLogPageProps) {
  const { runId } = await params;

  // Fetch run metadata
  const run = await getRun(runId);
  if (!run) {
    notFound();
  }

  // Fetch all edits for this run
  const edits = await getClusterEditHistory(runId);

  const getEditTypeBadge = (editType: string) => {
    const variants: Record<string, { variant: 'default' | 'secondary' | 'destructive' | 'outline'; label: string }> = {
      rename: { variant: 'secondary', label: 'Rename' },
      move_query: { variant: 'default', label: 'Move Query' },
      merge: { variant: 'secondary', label: 'Merge' },
      split: { variant: 'secondary', label: 'Split' },
      delete: { variant: 'destructive', label: 'Delete' },
      tag: { variant: 'outline', label: 'Tag' },
    };

    const config = variants[editType] || { variant: 'outline' as const, label: editType };
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
          <p className="text-muted-foreground mt-1">
            Edit history for run {runId}
          </p>
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
              Use <code className="text-xs bg-muted px-1 py-0.5 rounded">lmsys edit</code> commands to curate clusters. All changes will appear here.
            </p>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardHeader>
            <CardTitle>All Edits ({edits.length})</CardTitle>
            <CardDescription>
              Complete audit trail of all curation operations
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {edits.map((edit) => (
                <div key={edit.id} className="border rounded-lg p-4">
                  <div className="flex items-start justify-between gap-4 mb-2">
                    <div className="flex items-center gap-2">
                      {getEditTypeBadge(edit.editType)}
                      {edit.clusterId && (
                        <Link href={`/clusters/${runId}/${edit.clusterId}`}>
                          <Badge variant="outline" className="hover:bg-accent cursor-pointer">
                            Cluster {edit.clusterId}
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
                    <p className="text-sm text-muted-foreground mb-2">
                      {edit.reason}
                    </p>
                  )}

                  {/* Show detailed changes */}
                  {edit.editType === 'rename' && edit.oldValue && edit.newValue && (
                    <div className="mt-2 p-2 bg-muted rounded text-xs">
                      <div className="font-medium mb-1">Title Change:</div>
                      <div className="line-through text-muted-foreground">{(edit.oldValue as any).title}</div>
                      <div>→ {(edit.newValue as any).title}</div>
                    </div>
                  )}

                  {edit.editType === 'move_query' && edit.oldValue && edit.newValue && (
                    <div className="mt-2 p-2 bg-muted rounded text-xs">
                      <div className="font-medium mb-1">Query Movement:</div>
                      <div>
                        Query {(edit.oldValue as any).query_id}: Cluster {(edit.oldValue as any).cluster_id} → {(edit.newValue as any).cluster_id}
                      </div>
                    </div>
                  )}

                  {edit.editType === 'merge' && edit.oldValue && edit.newValue && (
                    <div className="mt-2 p-2 bg-muted rounded text-xs">
                      <div className="font-medium mb-1">Cluster Merge:</div>
                      <div>
                        Merged {((edit.oldValue as any).source_clusters || []).join(', ')} → Cluster {(edit.newValue as any).target_cluster}
                      </div>
                      <div className="text-muted-foreground">
                        {(edit.newValue as any).queries_moved} queries moved
                      </div>
                    </div>
                  )}

                  {edit.editType === 'tag' && edit.newValue && (
                    <div className="mt-2 p-2 bg-muted rounded text-xs">
                      <div className="font-medium mb-1">Metadata Update:</div>
                      {(edit.newValue as any).quality && (
                        <div>Quality: {(edit.newValue as any).quality}</div>
                      )}
                      {(edit.newValue as any).coherence_score && (
                        <div>Coherence: {(edit.newValue as any).coherence_score}/5</div>
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
          View the full audit log with:{' '}
          <code className="bg-muted px-1 py-0.5 rounded text-xs">
            lmsys edit audit {runId}
          </code>
        </p>
      </div>
    </div>
  );
}

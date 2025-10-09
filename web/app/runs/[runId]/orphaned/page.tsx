import Link from 'next/link';
import { notFound } from 'next/navigation';
import { getRun, getOrphanedQueries } from '@/app/actions';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface OrphanedQueriesPageProps {
  params: Promise<{ runId: string }>;
}

export default async function OrphanedQueriesPage({ params }: OrphanedQueriesPageProps) {
  const { runId } = await params;

  // Fetch run metadata
  const run = await getRun(runId);
  if (!run) {
    notFound();
  }

  // Fetch orphaned queries
  const orphanedResults = await getOrphanedQueries(runId);

  return (
    <div className="container mx-auto py-8 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <Link href={`/runs/${runId}`}>
            <Button variant="ghost" size="sm">
              ← Back to run
            </Button>
          </Link>
          <h1 className="text-3xl font-bold mt-2">Orphaned Queries</h1>
          <p className="text-muted-foreground mt-1">
            Queries removed from clusters for run {runId}
          </p>
        </div>
        <Badge variant="secondary">{orphanedResults.length} orphaned</Badge>
      </div>

      {orphanedResults.length === 0 ? (
        <Card>
          <CardHeader>
            <CardTitle>No Orphaned Queries</CardTitle>
            <CardDescription>
              All queries in this run are assigned to clusters
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Orphaned queries appear here when clusters are deleted with the <code className="text-xs bg-muted px-1 py-0.5 rounded">--orphan</code> flag.
            </p>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardHeader>
            <CardTitle>Orphaned Queries ({orphanedResults.length})</CardTitle>
            <CardDescription>
              These queries were removed from their original clusters
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {orphanedResults.map(({ orphan, query }) => (
                <div key={orphan.id} className="border rounded-lg p-4">
                  <div className="flex items-start justify-between gap-4 mb-2">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-xs text-muted-foreground">Query {query.id}</span>
                        {orphan.originalClusterId && (
                          <Badge variant="outline" className="text-xs">
                            From cluster {orphan.originalClusterId}
                          </Badge>
                        )}
                      </div>
                      <p className="text-sm">{query.queryText}</p>
                    </div>
                    <div className="text-right">
                      <div className="text-xs text-muted-foreground">
                        {new Date(orphan.orphanedAt).toLocaleDateString()}
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center gap-4 text-xs text-muted-foreground">
                    <span>Model: {query.model}</span>
                    {query.language && <span>Language: {query.language}</span>}
                  </div>

                  {orphan.reason && (
                    <div className="mt-2 text-xs text-muted-foreground border-t pt-2">
                      Reason: {orphan.reason}
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
          To reassign orphaned queries, use:{' '}
          <code className="bg-muted px-1 py-0.5 rounded text-xs">
            lmsys edit move-query {runId} --query-id &lt;ID&gt; --to-cluster &lt;CLUSTER_ID&gt;
          </code>
        </p>
      </div>
    </div>
  );
}

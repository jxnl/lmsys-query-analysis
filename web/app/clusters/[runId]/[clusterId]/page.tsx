import Link from 'next/link';
import { notFound } from 'next/navigation';
import { getClusterSummary, getClusterQueries } from '@/app/actions';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ClusterQueriesClient } from './cluster-queries-client';

interface ClusterPageProps {
  params: Promise<{ runId: string; clusterId: string }>;
  searchParams: Promise<{ page?: string }>;
}

export default async function ClusterPage({ params, searchParams }: ClusterPageProps) {
  const { runId, clusterId: clusterIdStr } = await params;
  const { page: pageStr } = await searchParams;

  const clusterId = parseInt(clusterIdStr);
  const page = parseInt(pageStr || '1');

  console.log('[ClusterPage] Loading cluster:', runId, clusterId, 'page:', page);

  // Fetch cluster summary (optional - may not exist if summarization hasn't been run)
  const summary = await getClusterSummary(runId, clusterId);
  console.log('[ClusterPage] Summary:', summary ? 'Found' : 'Not found');

  // Fetch initial page of queries
  const initialData = await getClusterQueries(runId, clusterId, page);
  console.log('[ClusterPage] Initial data:', {
    queries: initialData.queries.length,
    total: initialData.total,
    page: initialData.page,
    pages: initialData.pages
  });

  // If no queries found, return 404
  if (initialData.queries.length === 0 && page === 1) {
    console.log('[ClusterPage] No queries found, returning 404');
    notFound();
  }

  return (
    <div className="container mx-auto py-8 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <Link href={`/runs/${runId}`}>
            <Button variant="ghost" size="sm">
              ← Back to run
            </Button>
          </Link>
          <h1 className="text-3xl font-bold mt-2">
            {summary?.title || `Cluster ${clusterId}`}
          </h1>
          {!summary && (
            <p className="text-sm text-muted-foreground mt-1">
              Run <code className="text-xs bg-muted px-1 py-0.5 rounded">lmsys summarize {runId}</code> to generate cluster summaries
            </p>
          )}
        </div>
        <div className="flex gap-2">
          <Badge variant="outline">Cluster ID: {clusterId}</Badge>
          {summary?.numQueries ? (
            <Badge variant="secondary">{summary.numQueries} queries</Badge>
          ) : (
            <Badge variant="secondary">{initialData.total} queries</Badge>
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

      {summary?.representativeQueries && summary.representativeQueries.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Representative Queries</CardTitle>
            <CardDescription>Examples that best represent this cluster</CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {summary.representativeQueries.slice(0, 5).map((query: any, idx: number) => (
                <li key={idx} className="text-sm border-l-2 border-muted pl-4">
                  {typeof query === 'string' ? query : query.query_text || JSON.stringify(query)}
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>All Queries</CardTitle>
          <CardDescription>Browse all queries in this cluster</CardDescription>
        </CardHeader>
        <CardContent>
          <ClusterQueriesClient
            runId={runId}
            clusterId={clusterId}
            initialData={initialData}
          />
        </CardContent>
      </Card>
    </div>
  );
}

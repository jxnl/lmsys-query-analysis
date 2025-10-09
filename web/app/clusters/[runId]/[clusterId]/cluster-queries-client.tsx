'use client'

import { useState, useTransition } from 'react';
import { useRouter } from 'next/navigation';
import { DataViewer, type DataViewerData } from '@/components/data-viewer';
import type { PaginatedQueries } from '@/lib/types/schemas';
import { getClusterQueries } from '@/app/actions';

interface ClusterQueriesClientProps {
  runId: string;
  clusterId: number;
  initialData: PaginatedQueries;
}

export function ClusterQueriesClient({
  runId,
  clusterId,
  initialData,
}: ClusterQueriesClientProps) {
  // Convert PaginatedQueries to DataViewerData format
  const convertData = (data: PaginatedQueries): DataViewerData => ({
    queries: data.queries.map(q => ({
      ...q,
      clusters: [], // Queries in cluster view don't show other cluster associations
    })),
    total: data.total,
    page: data.page,
    pages: data.pages,
    limit: data.limit,
  });

  const [data, setData] = useState(convertData(initialData));
  const [isPending, startTransition] = useTransition();
  const router = useRouter();

  const handlePageChange = (newPage: number) => {
    startTransition(async () => {
      const newData = await getClusterQueries(runId, clusterId, newPage);
      setData(convertData(newData));
      // Update URL to reflect current page
      router.push(`/clusters/${runId}/${clusterId}?page=${newPage}`, { scroll: false });
    });
  };

  return (
    <div className={isPending ? 'opacity-50 pointer-events-none' : ''}>
      <DataViewer data={data} onPageChange={handlePageChange} showClusters={false} />
    </div>
  );
}

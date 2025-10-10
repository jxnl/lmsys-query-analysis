'use client'

import { useState, useTransition } from 'react';
import { useRouter } from 'next/navigation';
import { DataViewer, type DataViewerData } from '@/components/data-viewer';
import type { PaginatedQueries } from '@/lib/types';
import { getClusterQueries, searchQueriesInCluster } from '@/app/actions';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Search, X } from 'lucide-react';

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
  const [searchText, setSearchText] = useState('');
  const [isSearchMode, setIsSearchMode] = useState(false);
  const [currentSearchText, setCurrentSearchText] = useState('');
  const [searchError, setSearchError] = useState<string | null>(null);
  const router = useRouter();

  const handlePageChange = (newPage: number) => {
    startTransition(async () => {
      const newData = isSearchMode
        ? await searchQueriesInCluster(currentSearchText, runId, clusterId, newPage)
        : await getClusterQueries(runId, clusterId, newPage);
      setData(convertData(newData));
      // Update URL to reflect current page
      const params = new URLSearchParams({ page: newPage.toString() });
      if (isSearchMode) params.set('q', currentSearchText);
      router.push(`/clusters/${runId}/${clusterId}?${params}`, { scroll: false });
    });
  };

  const handleSearch = () => {
    if (!searchText.trim()) return;

    startTransition(async () => {
      setIsSearchMode(true);
      setCurrentSearchText(searchText);
      setSearchError(null);
      const searchResults = await searchQueriesInCluster(searchText, runId, clusterId, 1);
      
      setData(convertData(searchResults));
      // Update URL with search query
      router.push(`/clusters/${runId}/${clusterId}?q=${encodeURIComponent(searchText)}`, { scroll: false });
    });
  };

  const handleClearSearch = () => {
    setSearchText('');
    setIsSearchMode(false);
    setCurrentSearchText('');

    startTransition(async () => {
      const allQueries = await getClusterQueries(runId, clusterId, 1);
      setData(convertData(allQueries));
      router.push(`/clusters/${runId}/${clusterId}`, { scroll: false });
    });
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <div className="space-y-4">
      {/* Search Bar */}
      <div className="flex gap-2">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            type="text"
            placeholder="Search queries in this cluster..."
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            onKeyDown={handleKeyDown}
            className="pl-10 pr-10"
            disabled={isPending}
          />
          {searchText && (
            <button
              onClick={() => setSearchText('')}
              className="absolute right-3 top-1/2 transform -translate-y-1/2 text-muted-foreground hover:text-foreground"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>
        <Button onClick={handleSearch} disabled={!searchText.trim() || isPending}>
          Search
        </Button>
        {isSearchMode && (
          <Button variant="outline" onClick={handleClearSearch} disabled={isPending}>
            Show All
          </Button>
        )}
      </div>

      {/* Search Status */}
      {isSearchMode && (
        <div className="text-sm text-muted-foreground bg-muted/50 px-3 py-2 rounded-md">
          Showing search results for: <span className="font-medium text-foreground">"{currentSearchText}"</span>
          {' '} — {data.total} {data.total === 1 ? 'result' : 'results'} found in this cluster
        </div>
      )}

      {/* Query List */}
      <div className={isPending ? 'opacity-50 pointer-events-none' : ''}>
        <DataViewer data={data} onPageChange={handlePageChange} showClusters={false} />
      </div>
    </div>
  );
}

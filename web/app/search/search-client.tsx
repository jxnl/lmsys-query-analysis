'use client'

import { useState, useTransition } from 'react';
import { useRouter } from 'next/navigation';
import { DataViewer, type DataViewerData } from '@/components/data-viewer';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import type { ClusteringRun } from '@/lib/types';
import { searchQueries } from '../actions';
import { Search, Loader2 } from 'lucide-react';

interface SearchClientProps {
  runs: ClusteringRun[];
}

export function SearchClient({ runs }: SearchClientProps) {
  // Default to latest run (first in array since ordered by newest first)
  const defaultRunId = runs.length > 0 ? runs[0].runId : 'all';

  const [searchText, setSearchText] = useState('');
  const [selectedRunId, setSelectedRunId] = useState<string>(defaultRunId);
  const [results, setResults] = useState<DataViewerData | null>(null);
  const [isPending, startTransition] = useTransition();
  const [hasSearched, setHasSearched] = useState(false);
  const router = useRouter();

  const handleSearch = () => {
    if (!searchText.trim()) return;

    setHasSearched(true);
    startTransition(async () => {
      const runId = selectedRunId === 'all' ? undefined : selectedRunId;
      const data = await searchQueries(searchText, runId, 1, 50);
      setResults(data);

      // Update URL with search params
      const params = new URLSearchParams();
      params.set('q', searchText);
      if (runId) params.set('runId', runId);
      params.set('page', '1');
      router.push(`/search?${params.toString()}`, { scroll: false });
    });
  };

  const handlePageChange = (newPage: number) => {
    startTransition(async () => {
      const runId = selectedRunId === 'all' ? undefined : selectedRunId;
      const data = await searchQueries(searchText, runId, newPage, 50);
      setResults(data);

      // Update URL with new page
      const params = new URLSearchParams();
      params.set('q', searchText);
      if (runId) params.set('runId', runId);
      params.set('page', newPage.toString());
      router.push(`/search?${params.toString()}`, { scroll: false });
    });
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardContent className="pt-6">
          <div className="space-y-4">
            <div className="flex gap-4">
              <div className="flex-1">
                <Input
                  type="text"
                  placeholder="Search queries..."
                  value={searchText}
                  onChange={(e) => setSearchText(e.target.value)}
                  onKeyPress={handleKeyPress}
                  className="w-full"
                />
              </div>
              <div className="w-64">
                <Select value={selectedRunId} onValueChange={setSelectedRunId}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select run" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Runs</SelectItem>
                    {runs.map((run) => (
                      <SelectItem key={run.runId} value={run.runId}>
                        {run.runId} ({run.algorithm})
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <Button onClick={handleSearch} disabled={isPending || !searchText.trim()}>
                {isPending ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Searching...
                  </>
                ) : (
                  <>
                    <Search className="mr-2 h-4 w-4" />
                    Search
                  </>
                )}
              </Button>
            </div>
            
            <div className="text-sm text-muted-foreground">
              <p>Search uses full-text search to find queries matching your search terms.</p>
              {selectedRunId !== 'all' && (
                <p className="mt-1">Filtering by run: <span className="font-mono">{selectedRunId}</span></p>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      {isPending && !results && (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      )}

      {!isPending && hasSearched && results && results.queries.length === 0 && (
        <Card>
          <CardContent className="py-12 text-center space-y-3">
            <p className="text-muted-foreground">No results found for "{searchText}"</p>
            <div className="text-sm text-muted-foreground">
              <p>Try different search terms or check that queries have been loaded into the database.</p>
            </div>
          </CardContent>
        </Card>
      )}

      {results && results.queries.length > 0 && (
        <div className={isPending ? 'opacity-50 pointer-events-none' : ''}>
          <DataViewer
            data={results}
            onPageChange={handlePageChange}
            showClusters={true}
            filterRunId={selectedRunId === 'all' ? undefined : selectedRunId}
          />
        </div>
      )}
    </div>
  );
}


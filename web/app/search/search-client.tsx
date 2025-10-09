'use client'

import { useState, useTransition } from 'react';
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
import type { ClusteringRun } from '@/lib/types/schemas';
import { searchQueries } from '../actions';
import { Search, Loader2 } from 'lucide-react';

interface SearchClientProps {
  runs: ClusteringRun[];
}

export function SearchClient({ runs }: SearchClientProps) {
  const [searchText, setSearchText] = useState('');
  const [selectedRunId, setSelectedRunId] = useState<string>('all');
  const [results, setResults] = useState<DataViewerData | null>(null);
  const [isPending, startTransition] = useTransition();
  const [hasSearched, setHasSearched] = useState(false);

  const handleSearch = () => {
    if (!searchText.trim()) return;

    setHasSearched(true);
    startTransition(async () => {
      const runId = selectedRunId === 'all' ? undefined : selectedRunId;
      const data = await searchQueries(searchText, runId, 1, 50);
      setResults(data);
    });
  };

  const handlePageChange = (newPage: number) => {
    startTransition(async () => {
      const runId = selectedRunId === 'all' ? undefined : selectedRunId;
      const data = await searchQueries(searchText, runId, newPage, 50);
      setResults(data);
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
              <p>Search uses ChromaDB semantic search to find similar queries by meaning.</p>
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
          <CardContent className="py-12 text-center">
            <p className="text-muted-foreground">No results found for "{searchText}"</p>
          </CardContent>
        </Card>
      )}

      {results && results.queries.length > 0 && (
        <div className={isPending ? 'opacity-50 pointer-events-none' : ''}>
          <DataViewer data={results} onPageChange={handlePageChange} showClusters={true} />
        </div>
      )}
    </div>
  );
}


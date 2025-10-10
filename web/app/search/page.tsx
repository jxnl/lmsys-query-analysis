import { clusteringApi } from '@/lib/api/client';
import { SearchClient } from './search-client';

export default async function SearchPage() {
  const response = await clusteringApi.listRuns({ limit: 100 });
  const runs = response.items;

  return (
    <div className="container mx-auto py-8 space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Search Queries</h1>
        <p className="text-muted-foreground mt-2">
          Search across all queries or filter by clustering run
        </p>
      </div>

      <SearchClient runs={runs} />
    </div>
  );
}


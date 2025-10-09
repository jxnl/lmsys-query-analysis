import { getRuns } from '../actions';
import { SearchClient } from './search-client';

export default async function SearchPage() {
  const runs = await getRuns();

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


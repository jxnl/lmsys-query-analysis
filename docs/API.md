# FastAPI Backend Documentation

## Quick Start

### Running Both Services Together (Recommended)

The easiest way to run the full stack (FastAPI + Next.js) is using the provided scripts:

```bash
# Start both FastAPI and Next.js in the background
./scripts/start-dev.sh

# View logs from both services
./scripts/logs.sh

# Stop both services
./scripts/stop-dev.sh
```

**What happens:**
- FastAPI starts on `http://localhost:8000` (docs at `/docs`)
- Next.js starts on `http://localhost:3000`
- Logs are written to `logs/fastapi.log` and `logs/nextjs.log`
- Process IDs saved to `logs/*.pid` for cleanup

### Running Services Individually

**FastAPI only:**
```bash
# Using the convenience command
uv run lmsys-api

# Or using uvicorn directly
uv run uvicorn lmsys_query_analysis.api.app:app --reload
```

**Next.js only:**
```bash
cd web
npm run dev
```

---

## API Endpoints

### Base URL
- **Development:** `http://localhost:8000`
- **Docs:** `http://localhost:8000/docs` (Swagger UI)
- **OpenAPI Spec:** `http://localhost:8000/openapi.json`

### Categories

| Category | Prefix | Description |
|----------|--------|-------------|
| **Clustering** | `/api/clustering` | Manage clustering runs and view cluster lists |
| **Analysis** | `/api` | Query listings and cluster details |
| **Search** | `/api/search` | Semantic and full-text search |
| **Hierarchy** | `/api/hierarchy` | Cluster hierarchy trees |
| **Summaries** | `/api/summaries` | LLM-generated summaries |
| **Curation** | `/api/curation` | Metadata, history, orphaned queries |

---

## Key Features

### 1. Dual Search Modes

All search endpoints support two modes:

**Full-text (default):** SQL LIKE search, no embeddings needed
```bash
curl "http://localhost:8000/api/search/queries?text=Python&mode=fulltext"
```

**Semantic:** ChromaDB vector search, requires embeddings
```bash
curl "http://localhost:8000/api/search/queries?text=Python&mode=semantic&run_id=kmeans-200-..."
```

### 2. Aggregations & Percentages

Cluster endpoints support enhanced aggregations:
```bash
# Get clusters with counts and percentages
curl "http://localhost:8000/api/clustering/runs/{run_id}/clusters?include_counts=true&include_percentages=true"
```

Response includes:
- `query_count`: Number of queries in cluster
- `percentage`: Percentage of total queries in run
- `total_queries`: Total queries in run (for context)

### 3. Pagination

All list endpoints support pagination:
```bash
curl "http://localhost:8000/api/queries?page=2&limit=50"
```

Response format:
```json
{
  "items": [...],
  "total": 1000,
  "page": 2,
  "pages": 20,
  "limit": 50
}
```

---

## Configuration

The API reads configuration from environment variables (same as CLI):

```bash
# Database path (default: ~/.lmsys-query-analysis/queries.db)
export DB_PATH=/path/to/queries.db

# ChromaDB path (default: ~/.lmsys-query-analysis/chroma)
export CHROMA_PATH=/path/to/chroma

# API keys for embeddings (server-side only)
export OPENAI_API_KEY=sk-...
export COHERE_API_KEY=...
export ANTHROPIC_API_KEY=...
```

---

## Next.js Integration

### 1. Install Dependencies
```bash
cd web
npm install
```

### 2. Configure API URL
```bash
# web/.env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 3. Generate TypeScript Types
```bash
# Requires FastAPI to be running on localhost:8000
npm run generate-types
```

This creates `web/lib/api/types.ts` from the OpenAPI spec.

### 4. Create Type-Safe Client

```typescript
// web/lib/api/client.ts
import type { paths } from './types';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

type RunsResponse = paths['/api/clustering/runs']['get']['responses']['200']['content']['application/json'];

export async function fetchRuns(): Promise<RunsResponse> {
  const res = await fetch(`${API_URL}/api/clustering/runs`);
  if (!res.ok) throw new Error('Failed to fetch runs');
  return res.json();
}
```

### 5. Use in Components

```tsx
// app/runs/page.tsx
import { fetchRuns } from '@/lib/api/client';

export default async function RunsPage() {
  const data = await fetchRuns();

  return (
    <div>
      {data.items.map(run => (
        <div key={run.run_id}>{run.run_id}</div>
      ))}
    </div>
  );
}
```

---

## Testing

```bash
# Run API tests
uv run pytest tests/api/ -v

# Run with coverage
uv run pytest tests/api/ --cov=src/lmsys_query_analysis/api

# Test specific endpoint
uv run pytest tests/api/test_endpoints.py::test_list_runs_with_data -v
```

---

## Development Workflow

**Typical development session:**

1. **Start services:**
   ```bash
   ./scripts/start-dev.sh
   ```

2. **Make changes to FastAPI code**
   - Changes auto-reload (uvicorn `--reload` flag)

3. **Regenerate types if schema changes:**
   ```bash
   cd web
   npm run generate-types
   ```

4. **View logs:**
   ```bash
   ./scripts/logs.sh
   # or individually:
   tail -f logs/fastapi.log
   tail -f logs/nextjs.log
   ```

5. **Stop services:**
   ```bash
   ./scripts/stop-dev.sh
   ```

---

## Troubleshooting

### Port already in use
```bash
# Kill processes on ports 8000 or 3000
lsof -ti:8000 | xargs kill -9
lsof -ti:3000 | xargs kill -9
```

### FastAPI not starting
Check logs:
```bash
cat logs/fastapi.log
```

Common issues:
- Missing dependencies: `uv sync`
- Database not found: Check `DB_PATH`
- ChromaDB issues: Check `CHROMA_PATH`

### Type generation fails
Ensure FastAPI is running:
```bash
curl http://localhost:8000/api/health
```

---

## Architecture

```
┌─────────────────────┐
│   Next.js (3000)    │  ← User Interface
│  - TypeScript types │
│  - API client       │
└──────────┬──────────┘
           │ HTTP/REST
           ↓
┌─────────────────────┐
│  FastAPI (8000)     │  ← REST API
│  - Routers          │
│  - Shared services  │
└──────────┬──────────┘
           │
           ├→ SQLite (queries.db)      ← Persistent data
           └→ ChromaDB (chroma/)        ← Embeddings
```

**Key principles:**
- CLI and API share same services (`query_service`, `cluster_service`, etc.)
- Both read from same database and ChromaDB
- API keys handled server-side only (Next.js never sees them)
- TypeScript types auto-generated from OpenAPI spec

---

## Future Enhancements (Phase 2)

POST endpoints are currently stubbed (return 501). Future work:

- `POST /api/clustering/kmeans` - Create clustering run
- `POST /api/clustering/hdbscan` - Create HDBSCAN run
- `POST /api/summaries` - Generate summaries
- `POST /api/hierarchy` - Create hierarchy
- `POST /api/curation/*` - Edit operations (rename, merge, move, etc.)

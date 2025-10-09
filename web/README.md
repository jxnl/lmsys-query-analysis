# LMSYS Query Analysis Viewer

Interactive web interface for exploring LMSYS clustering analysis results.

## Overview

This Next.js application provides a read-only visualization layer for the LMSYS query analysis tool. It connects directly to the SQLite database and ChromaDB vector store created by the Python CLI.

## Prerequisites

Before running the viewer, you need data from the Python CLI:

```bash
# From the project root
uv run lmsys load --limit 10000 --use-chroma
uv run lmsys cluster kmeans --n-clusters 200
uv run lmsys summarize <RUN_ID>
uv run lmsys merge-clusters <RUN_ID>
```

## Quick Start

```bash
cd web
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

## Features

- **Sidebar Navigation**: Persistent sidebar with navigation to all pages
- **Jobs Dashboard**: View all clustering runs with metadata and stats
- **Semantic Search**: Search queries using ChromaDB semantic search (finds similar queries by meaning)
- **Hierarchy Viewer**: Navigate cluster hierarchies with collapsible tree
- **Query Browser**: Paginated view of queries within each cluster
- **Cluster Summaries**: LLM-generated titles, descriptions, and representative queries
- **Data Viewer**: Reusable component for displaying queries with cluster associations

## Architecture

### Tech Stack

- **Next.js 15**: App Router with Server Components
- **Drizzle ORM**: Type-safe SQL queries (read-only)
- **ChromaDB JS Client**: Semantic search (requires server)
- **Zod**: Runtime validation
- **ShadCN**: UI components (Radix UI + Tailwind)

### Data Flow

```
Python CLI → SQLite + ChromaDB (data artifacts)
                ↓
      Next.js Server Components
                ↓
          Browser UI
```

## Configuration

### Default Paths

The viewer uses the same default paths as the Python CLI:

- **SQLite**: `~/.lmsys-query-analysis/queries.db`
- **ChromaDB**: `~/.lmsys-query-analysis/chroma/`

### Custom Paths

Create `.env.local` to override:

```bash
DB_PATH=/path/to/your/queries.db
CHROMA_PATH=/path/to/your/chroma
```

## Project Structure

```
web/
├── app/
│   ├── actions.ts                       # Server Actions (data fetching)
│   ├── page.tsx                         # Home (jobs list)
│   ├── runs/[runId]/page.tsx           # Run detail + hierarchy
│   └── clusters/[runId]/[clusterId]/   # Cluster detail + queries
│       ├── page.tsx
│       └── cluster-queries-client.tsx
├── components/
│   ├── jobs-table.tsx                  # Jobs list table
│   ├── hierarchy-tree.tsx              # Collapsible tree
│   └── query-list.tsx                  # Paginated query list
├── lib/
│   ├── db/
│   │   ├── schema.ts                   # Drizzle schema (mirrors SQLModel)
│   │   └── client.ts                   # SQLite connection
│   ├── chroma/
│   │   └── client.ts                   # ChromaDB client
│   └── types/
│       └── schemas.ts                  # Zod validation schemas
└── next.config.ts
```

## Development

### Running the Dev Server

```bash
npm run dev
```

The app runs on `http://localhost:3000` with hot reload.

### Type Safety

The application maintains full type safety from database to UI:

- **Drizzle**: Compile-time SQL type checking
- **Zod**: Runtime validation in Server Actions
- **TypeScript**: End-to-end type inference

### Database Schema

The Drizzle schema (`lib/db/schema.ts`) mirrors the Python SQLModel schema:

- `queries` - User queries from LMSYS-1M
- `clustering_runs` - Clustering experiments
- `query_clusters` - Query-to-cluster mappings
- `cluster_summaries` - LLM-generated summaries
- `cluster_hierarchies` - Multi-level hierarchies

## Deployment

### Build for Production

```bash
npm run build
npm start
```

### Environment Variables

For production deployments, set:

```bash
DB_PATH=/production/path/queries.db
CHROMA_PATH=/production/path/chroma
```

## Troubleshooting

### "No clustering runs found"

The Python CLI hasn't created any data yet. Run:

```bash
uv run lmsys load --limit 1000
uv run lmsys cluster kmeans --n-clusters 50
```

### ChromaDB Connection Errors

The JS client requires a running ChromaDB server for semantic search functionality:

```bash
chroma run --path ~/.lmsys-query-analysis/chroma
```

Then restart the Next.js dev server.

Note: Semantic search will only work if:
1. ChromaDB server is running on `localhost:8000`
2. Queries were loaded with `--use-chroma` flag
3. The embedding provider/model matches your clustering run

### SQLite Permission Errors

The viewer opens the database in read-only mode. Ensure the file exists and is readable:

```bash
ls -la ~/.lmsys-query-analysis/queries.db
```

## API Reference

See `app/actions.ts` for all Server Actions:

- `getRuns()` - List all clustering runs
- `getHierarchyTree(hierarchyRunId)` - Get full hierarchy tree
- `getClusterSummary(runId, clusterId)` - Get cluster metadata
- `getClusterQueries(runId, clusterId, page)` - Get paginated queries
- `searchClusters(query, runId)` - Semantic search via ChromaDB

## Contributing

This viewer is part of the `lmsys-query-analysis` project. See the main README for contribution guidelines.

## License

Same as parent project.

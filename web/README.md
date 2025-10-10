# LMSYS Query Analysis Viewer

Interactive web interface for exploring LMSYS clustering analysis results.

## Overview

This Next.js application provides a read-only visualization layer for the LMSYS query analysis tool. It connects to the FastAPI backend (port 8000) which provides access to clustering data and analysis results.

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

### Zero External Dependencies

**No ChromaDB server required!** All search functionality uses SQL LIKE queries for fast, simple operation. Just start the Next.js server and go.

### Core Features

- **Sidebar Navigation**: Persistent sidebar with navigation to all pages
- **Jobs Dashboard**: View all clustering runs with metadata and stats
- **Search (SQL LIKE queries)**:
  - Global search across all queries
  - Cluster-specific search within individual clusters
  - Search cluster summaries by title/description
- **Enhanced Hierarchy Viewer**: Navigate cluster hierarchies with:
  - Collapsible tree with expand/collapse all controls
  - Visual progress bars showing cluster size
  - Color coding (blue for large ≥10%, primary for medium 3-10%, gray for small <3%)
  - Summary statistics (total clusters, leaf count, levels, query count)
- **Query Browser**: Paginated view of queries within each cluster (50 per page)
- **Cluster Summaries**: LLM-generated titles, descriptions, and representative queries
- **Data Viewer**: Reusable component for displaying queries with cluster associations

## Architecture

### Tech Stack

- **Next.js 15**: App Router with Server Components
- **Drizzle ORM**: Type-safe SQL queries (read-only)
- **SQLite**: All data storage and search via LIKE queries (no external services)
- **Zod**: Runtime validation
- **ShadCN**: UI components (Radix UI + Tailwind)

### Data Flow

```
Python CLI → SQLite (clustering data)
                ↓
      Next.js Server Components (read-only SQLite access)
                ↓
          Browser UI
```

The web viewer **only reads from SQLite** and uses SQL LIKE queries for search. ChromaDB is only used by the Python CLI for semantic search.

## Configuration

### Default Paths

The viewer uses the same default path as the Python CLI:

- **SQLite**: `~/.lmsys-query-analysis/queries.db`

### Custom Paths

Create `.env.local` to override:

```bash
DB_PATH=/path/to/your/queries.db
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

### Testing

Run E2E tests with Playwright:

```bash
# Run all tests
npm test

# Run tests in headed mode (see browser)
npm run test:headed

# Run tests with UI mode (interactive)
npm run test:ui
```

**Test Coverage:**

- **API Integration** (6 tests): Health checks, fetch runs/queries, search, error handling
- **Homepage** (3 tests): Page load, navigation, responsive design
- **Clustering Runs** (2 tests): List display, run details page access
- **Search** (2 tests): Frontend search functionality, empty search handling

All tests use Playwright with Chromium browser and run against the live development server.

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
```

## Troubleshooting

### "No clustering runs found"

The Python CLI hasn't created any data yet. Run:

```bash
uv run lmsys load --limit 1000
uv run lmsys cluster kmeans --n-clusters 50
```

### SQLite Permission Errors

The viewer opens the database in read-only mode. Ensure the file exists and is readable:

```bash
ls -la ~/.lmsys-query-analysis/queries.db
```

## API Reference

See `app/actions.ts` for all Server Actions:

- `getRuns()` - List all clustering runs
- `getHierarchyTree(hierarchyRunId)` - Get full hierarchy tree
- `getClusterSummary(runId, clusterId, alias?)` - Get cluster metadata
- `getClusterQueries(runId, clusterId, page)` - Get paginated queries for a cluster
- `searchClusters(query, runId, nResults?)` - Search cluster summaries using SQL LIKE queries
- `searchQueries(searchText, runId?, page)` - Global query search using SQL LIKE queries
- `searchQueriesInCluster(searchText, runId, clusterId, page)` - Search queries within a specific cluster using SQL LIKE queries
- `getClusterQueryCounts(runId)` - Get query counts for all clusters in a run
- `getSummaryAliases(runId)` - Get all summary aliases for a run

## Contributing

This viewer is part of the `lmsys-query-analysis` project. See the main README for contribution guidelines.

## License

Same as parent project.

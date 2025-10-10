import { apiFetch } from "@/lib/api";
import { JobsTable } from "@/components/jobs-table";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Database, Search, FolderTree } from "lucide-react";
import type { components } from "@/lib/api/types";

type ClusteringRun = components["schemas"]["ClusteringRunSummary"];

export default async function HomePage() {
  const response = await apiFetch<{
    items: ClusteringRun[];
    total: number;
    page: number;
    pages: number;
  }>("/api/clustering/runs?limit=100");
  const runs = response.items;

  return (
    <div className="container mx-auto py-8 space-y-6">
      <div>
        <h1 className="text-3xl font-bold">LMSYS Query Analysis</h1>
        <p className="text-muted-foreground mt-2">
          View and explore clustering analysis runs from the LMSYS-1M dataset
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Clustering Runs
            </CardTitle>
            <FolderTree className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{runs.length}</div>
            <p className="text-xs text-muted-foreground">Total analysis runs</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">
              Search Queries
            </CardTitle>
            <Search className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">Available</div>
            <p className="text-xs text-muted-foreground">
              Find queries across runs
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Data Source</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">SQLite</div>
            <p className="text-xs text-muted-foreground">Read-only database</p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Clustering Runs</CardTitle>
          <CardDescription>
            All clustering experiments run on the LMSYS-1M dataset
          </CardDescription>
        </CardHeader>
        <CardContent>
          <JobsTable runs={runs} />
        </CardContent>
      </Card>
    </div>
  );
}

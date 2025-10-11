import { apiFetch } from "@/lib/api";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ArrowLeft, Clock, Database, Settings, GitBranch } from "lucide-react";
import Link from "next/link";
import type { components } from "@/lib/api/types";

type HierarchyRunDetail = components["schemas"]["HierarchyRunDetail"];

interface HierarchyRunDetailPageProps {
  params: {
    runId: string;
    hierarchyRunId: string;
  };
}

export default async function HierarchyRunDetailPage({
  params,
}: HierarchyRunDetailPageProps) {
  const hierarchyRun = await apiFetch<HierarchyRunDetail>(
    `/api/hierarchy/${params.hierarchyRunId}/metadata`
  );

  return (
    <div className="container mx-auto py-8 space-y-6">
      <div className="flex items-center gap-4">
        <Button variant="outline" size="sm" asChild>
          <Link href={`/runs/${params.runId}`}>
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Run
          </Link>
        </Button>
        <div>
          <h1 className="text-3xl font-bold">{hierarchyRun.hierarchy_run_id}</h1>
          <p className="text-muted-foreground">
            Hierarchy run details and configuration
          </p>
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Settings className="h-5 w-5" />
              LLM Configuration
            </CardTitle>
            <CardDescription>
              Language model used for hierarchical merging
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-sm font-medium text-muted-foreground">
                  LLM Provider
                </label>
                <p className="text-sm font-mono">{hierarchyRun.llm_provider}</p>
              </div>
              <div>
                <label className="text-sm font-medium text-muted-foreground">
                  LLM Model
                </label>
                <p className="text-sm font-mono">{hierarchyRun.llm_model}</p>
              </div>
              <div>
                <label className="text-sm font-medium text-muted-foreground">
                  Concurrency
                </label>
                <p className="text-sm">{hierarchyRun.concurrency}</p>
              </div>
              <div>
                <label className="text-sm font-medium text-muted-foreground">
                  RPM Limit
                </label>
                <p className="text-sm">{hierarchyRun.rpm || "None"}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="h-5 w-5" />
              Embedding Configuration
            </CardTitle>
            <CardDescription>
              Embedding model used for similarity search
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-sm font-medium text-muted-foreground">
                  Embedding Provider
                </label>
                <p className="text-sm font-mono">{hierarchyRun.embedding_provider}</p>
              </div>
              <div>
                <label className="text-sm font-medium text-muted-foreground">
                  Embedding Model
                </label>
                <p className="text-sm font-mono">{hierarchyRun.embedding_model}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <GitBranch className="h-5 w-5" />
            Merge Parameters
          </CardTitle>
          <CardDescription>
            Algorithm settings for hierarchical organization
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-4 gap-4">
            <div>
              <label className="text-sm font-medium text-muted-foreground">
                Target Levels
              </label>
              <p className="text-sm">{hierarchyRun.target_levels}</p>
            </div>
            <div>
              <label className="text-sm font-medium text-muted-foreground">
                Merge Ratio
              </label>
              <p className="text-sm">{hierarchyRun.merge_ratio}</p>
            </div>
            <div>
              <label className="text-sm font-medium text-muted-foreground">
                Neighborhood Size
              </label>
              <p className="text-sm">{hierarchyRun.neighborhood_size}</p>
            </div>
            <div>
              <label className="text-sm font-medium text-muted-foreground">
                Summary Run ID
              </label>
              <p className="text-sm font-mono">
                {hierarchyRun.summary_run_id || "None"}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Clock className="h-5 w-5" />
            Execution Details
          </CardTitle>
          <CardDescription>
            Runtime information and performance metrics
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="text-sm font-medium text-muted-foreground">
                Total Nodes
              </label>
              <p className="text-sm">{hierarchyRun.total_nodes || "Unknown"}</p>
            </div>
            <div>
              <label className="text-sm font-medium text-muted-foreground">
                Execution Time
              </label>
              <p className="text-sm flex items-center gap-1">
                <Clock className="h-3 w-3" />
                {hierarchyRun.execution_time_seconds
                  ? `${hierarchyRun.execution_time_seconds.toFixed(2)}s`
                  : "Unknown"}
              </p>
            </div>
            <div>
              <label className="text-sm font-medium text-muted-foreground">
                Created At
              </label>
              <p className="text-sm">
                {new Date(hierarchyRun.created_at).toLocaleString()}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Actions</CardTitle>
          <CardDescription>
            Navigate to related resources and views
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2">
            <Button variant="outline" asChild>
              <Link href={`/runs/${params.runId}`}>
                View Base Clustering Run
              </Link>
            </Button>
            <Button variant="outline" asChild>
              <Link href={`/runs/${params.runId}/hierarchy/${params.hierarchyRunId}`}>
                View Hierarchy Tree
              </Link>
            </Button>
            {hierarchyRun.summary_run_id && (
              <Button variant="outline" asChild>
                <Link href={`/runs/${params.runId}/summaries/${hierarchyRun.summary_run_id}`}>
                  View Summary Run
                </Link>
              </Button>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

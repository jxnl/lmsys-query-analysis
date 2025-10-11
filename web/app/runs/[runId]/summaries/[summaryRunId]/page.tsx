import { apiFetch } from "@/lib/api";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ArrowLeft, Clock, Database, Settings, Zap } from "lucide-react";
import Link from "next/link";
import type { components } from "@/lib/api/types";

type SummaryRunDetail = components["schemas"]["SummaryRunDetail"];

interface SummaryRunDetailPageProps {
  params: {
    runId: string;
    summaryRunId: string;
  };
}

export default async function SummaryRunDetailPage({
  params,
}: SummaryRunDetailPageProps) {
  const { runId, summaryRunId } = await params;
  const summaryRun = await apiFetch<SummaryRunDetail>(
    `/api/summaries/${summaryRunId}/metadata`
  );

  return (
    <div className="container mx-auto py-8 space-y-6">
      <div className="flex items-center gap-4">
        <Button variant="outline" size="sm" asChild>
          <Link href={`/runs/${runId}`}>
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Run
          </Link>
        </Button>
        <div>
          <h1 className="text-3xl font-bold">{summaryRun.summary_run_id}</h1>
          <p className="text-muted-foreground">
            Summary run details and configuration
          </p>
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Settings className="h-5 w-5" />
              Configuration
            </CardTitle>
            <CardDescription>
              LLM and summarization parameters used for this run
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-sm font-medium text-muted-foreground">
                  LLM Provider
                </label>
                <p className="text-sm font-mono">{summaryRun.llm_provider}</p>
              </div>
              <div>
                <label className="text-sm font-medium text-muted-foreground">
                  LLM Model
                </label>
                <p className="text-sm font-mono">{summaryRun.llm_model}</p>
              </div>
              <div>
                <label className="text-sm font-medium text-muted-foreground">
                  Max Queries
                </label>
                <p className="text-sm">{summaryRun.max_queries}</p>
              </div>
              <div>
                <label className="text-sm font-medium text-muted-foreground">
                  Concurrency
                </label>
                <p className="text-sm">{summaryRun.concurrency}</p>
              </div>
              <div>
                <label className="text-sm font-medium text-muted-foreground">
                  RPM Limit
                </label>
                <p className="text-sm">{summaryRun.rpm || "None"}</p>
              </div>
              <div>
                <label className="text-sm font-medium text-muted-foreground">
                  Alias
                </label>
                <p className="text-sm">
                  {summaryRun.alias ? (
                    <Badge variant="secondary">{summaryRun.alias}</Badge>
                  ) : (
                    "—"
                  )}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="h-5 w-5" />
              Execution Details
            </CardTitle>
            <CardDescription>
              Runtime information and performance metrics
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-sm font-medium text-muted-foreground">
                  Total Clusters
                </label>
                <p className="text-sm">{summaryRun.total_clusters || "Unknown"}</p>
              </div>
              <div>
                <label className="text-sm font-medium text-muted-foreground">
                  Execution Time
                </label>
                <p className="text-sm flex items-center gap-1">
                  <Clock className="h-3 w-3" />
                  {summaryRun.execution_time_seconds
                    ? `${summaryRun.execution_time_seconds.toFixed(2)}s`
                    : "Unknown"}
                </p>
              </div>
              <div>
                <label className="text-sm font-medium text-muted-foreground">
                  Created At
                </label>
                <p className="text-sm">
                  {new Date(summaryRun.created_at).toLocaleString()}
                </p>
              </div>
              <div>
                <label className="text-sm font-medium text-muted-foreground">
                  Base Run ID
                </label>
                <p className="text-sm font-mono">{summaryRun.run_id}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5" />
            Contrastive Analysis Settings
          </CardTitle>
          <CardDescription>
            Parameters for highlighting unique aspects of each cluster
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="text-sm font-medium text-muted-foreground">
                Contrast Neighbors
              </label>
              <p className="text-sm">{summaryRun.contrast_neighbors}</p>
            </div>
            <div>
              <label className="text-sm font-medium text-muted-foreground">
                Contrast Examples
              </label>
              <p className="text-sm">{summaryRun.contrast_examples}</p>
            </div>
            <div>
              <label className="text-sm font-medium text-muted-foreground">
                Contrast Mode
              </label>
              <p className="text-sm">{summaryRun.contrast_mode}</p>
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
              <Link href={`/runs/${runId}`}>
                View Base Clustering Run
              </Link>
            </Button>
            <Button variant="outline" asChild>
              <Link href={`/runs/${runId}/clusters`}>
                View Cluster Summaries
              </Link>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

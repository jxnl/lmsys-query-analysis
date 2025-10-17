import Link from "next/link";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { components } from "@/lib/api/types";

type ClusteringRun = components["schemas"]["ClusteringRunSummary"];

interface JobsTableProps {
  runs: ClusteringRun[];
}

export function JobsTable({ runs }: JobsTableProps) {
  return (
    <div className="rounded-md border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Run ID</TableHead>
            <TableHead>Algorithm</TableHead>
            <TableHead className="text-right">Clusters</TableHead>
            <TableHead>Embedding Model</TableHead>
            <TableHead>Merge Params</TableHead>
            <TableHead>Created</TableHead>
            <TableHead className="text-right">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {runs.length === 0 ? (
            <TableRow>
              <TableCell colSpan={7} className="text-center text-muted-foreground">
                No clustering runs found. Run `lmsys cluster` to create one.
              </TableCell>
            </TableRow>
          ) : (
            runs.map((run) => {
              // Extract embedding info from parameters
              const params = run.parameters as Record<string, unknown> | null | undefined;
              const embeddingProvider = params?.embedding_provider as string | undefined;
              const embeddingModel = params?.embedding_model as string | undefined;
              const embeddingDimension = params?.embedding_dimension as number | undefined;

              // Filter out embedding-related and redundant parameters
              const excludeKeys = new Set([
                "embedding_provider",
                "embedding_model",
                "embedding_dimension",
                "n_clusters",
                "num_clusters",
              ]);

              const otherParams = params
                ? Object.entries(params).filter(([key]) => !excludeKeys.has(key))
                : [];

              const formatParamValue = (value: unknown): string => {
                if (value === null || value === undefined) return "null";
                if (typeof value === "boolean") return value.toString();
                if (typeof value === "number") return value.toString();
                if (typeof value === "string") return value;
                if (Array.isArray(value)) return `[${value.length} items]`;
                if (typeof value === "object") return JSON.stringify(value);
                return String(value);
              };

              return (
                <TableRow key={run.run_id}>
                  <TableCell className="font-mono text-sm">{run.run_id}</TableCell>
                  <TableCell>
                    <Badge variant="outline">{run.algorithm}</Badge>
                  </TableCell>
                  <TableCell className="text-right">{run.num_clusters || "N/A"}</TableCell>
                  <TableCell className="text-sm">
                    {embeddingProvider && embeddingModel ? (
                      <div className="space-y-0.5">
                        <div className="font-medium">
                          {embeddingProvider}/{embeddingModel}
                        </div>
                        {embeddingDimension && (
                          <div className="text-xs text-muted-foreground">
                            dim: {embeddingDimension}
                          </div>
                        )}
                      </div>
                    ) : (
                      <span className="text-muted-foreground">N/A</span>
                    )}
                  </TableCell>
                  <TableCell className="text-xs">
                    {otherParams.length > 0 ? (
                      <div className="space-y-0.5 max-w-xs">
                        {otherParams.map(([key, value]) => (
                          <div key={key} className="flex items-baseline gap-1.5">
                            <span className="font-mono text-[10px] text-muted-foreground/70 whitespace-nowrap">
                              {key}:
                            </span>
                            <span
                              className="truncate font-medium text-foreground/90"
                              title={`${key}: ${formatParamValue(value)}`}
                            >
                              {formatParamValue(value)}
                            </span>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <span className="text-muted-foreground">—</span>
                    )}
                  </TableCell>
                  <TableCell>
                    {new Date(run.created_at).toLocaleDateString("en-US", {
                      year: "numeric",
                      month: "short",
                      day: "numeric",
                    })}
                  </TableCell>
                  <TableCell className="text-right">
                    <Link href={`/runs/${run.run_id}`}>
                      <Button variant="ghost" size="sm">
                        View
                      </Button>
                    </Link>
                  </TableCell>
                </TableRow>
              );
            })
          )}
        </TableBody>
      </Table>
    </div>
  );
}

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

type SummaryRun = components["schemas"]["SummaryRunSummary"];

interface SummaryRunsTableProps {
  runs: SummaryRun[];
}

export function SummaryRunsTable({ runs }: SummaryRunsTableProps) {
  return (
    <div className="rounded-md border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Summary Run ID</TableHead>
            <TableHead>Base Run ID</TableHead>
            <TableHead>Model</TableHead>
            <TableHead>Alias</TableHead>
            <TableHead>Status</TableHead>
            <TableHead>Created</TableHead>
            <TableHead className="text-right">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {runs.length === 0 ? (
            <TableRow>
              <TableCell
                colSpan={7}
                className="text-center text-muted-foreground"
              >
                No summary runs found. Run `lmsys summarize` to create one.
              </TableCell>
            </TableRow>
          ) : (
            runs.map((run) => {
              return (
                <TableRow key={run.summary_run_id}>
                  <TableCell className="font-mono text-sm">
                    {run.summary_run_id}
                  </TableCell>
                  <TableCell className="font-mono text-sm">
                    {run.run_id}
                  </TableCell>
                  <TableCell className="text-sm">
                    {run.model}
                  </TableCell>
                  <TableCell className="text-sm">
                    {run.alias ? (
                      <Badge variant="secondary">{run.alias}</Badge>
                    ) : (
                      <span className="text-muted-foreground">—</span>
                    )}
                  </TableCell>
                  <TableCell>
                    <Badge variant={run.status === "completed" ? "default" : "secondary"}>
                      {run.status}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-sm">
                    {new Date(run.generated_at).toLocaleString()}
                  </TableCell>
                  <TableCell className="text-right">
                    <div className="flex justify-end gap-2">
                      <Button variant="outline" size="sm" asChild>
                        <Link href={`/runs/${run.run_id}/summaries/${run.summary_run_id}`}>
                          View Details
                        </Link>
                      </Button>
                      <Button variant="outline" size="sm" asChild>
                        <Link href={`/runs/${run.run_id}`}>
                          View Base Run
                        </Link>
                      </Button>
                    </div>
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

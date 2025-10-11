import Link from "next/link";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import type { components } from "@/lib/api/types";

type HierarchyRun = components["schemas"]["HierarchyRunInfo"];

interface HierarchyRunsTableProps {
  runs: HierarchyRun[];
}

export function HierarchyRunsTable({ runs }: HierarchyRunsTableProps) {
  return (
    <div className="rounded-md border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Hierarchy Run ID</TableHead>
            <TableHead>Base Run ID</TableHead>
            <TableHead>Created</TableHead>
            <TableHead className="text-right">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {runs.length === 0 ? (
            <TableRow>
              <TableCell
                colSpan={4}
                className="text-center text-muted-foreground"
              >
                No hierarchy runs found. Run `lmsys merge-clusters` to create one.
              </TableCell>
            </TableRow>
          ) : (
            runs.map((run) => {
              return (
                <TableRow key={run.hierarchy_run_id}>
                  <TableCell className="font-mono text-sm">
                    {run.hierarchy_run_id}
                  </TableCell>
                  <TableCell className="font-mono text-sm">
                    {run.run_id}
                  </TableCell>
                  <TableCell className="text-sm">
                    {new Date(run.created_at).toLocaleString()}
                  </TableCell>
                  <TableCell className="text-right">
                    <div className="flex justify-end gap-2">
                      <Button variant="outline" size="sm" asChild>
                        <Link href={`/runs/${run.run_id}/hierarchies/${run.hierarchy_run_id}`}>
                          View Details
                        </Link>
                      </Button>
                      <Button variant="outline" size="sm" asChild>
                        <Link href={`/runs/${run.run_id}/hierarchy/${run.hierarchy_run_id}`}>
                          View Tree
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

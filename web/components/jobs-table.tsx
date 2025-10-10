import Link from 'next/link';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import type { ClusteringRun } from '@/lib/types';

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
            <TableHead>Created</TableHead>
            <TableHead className="text-right">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {runs.length === 0 ? (
            <TableRow>
              <TableCell colSpan={5} className="text-center text-muted-foreground">
                No clustering runs found. Run `lmsys cluster` to create one.
              </TableCell>
            </TableRow>
          ) : (
            runs.map((run) => (
              <TableRow key={run.runId}>
                <TableCell className="font-mono text-sm">{run.runId}</TableCell>
                <TableCell>
                  <Badge variant="outline">{run.algorithm}</Badge>
                </TableCell>
                <TableCell className="text-right">{run.numClusters || 'N/A'}</TableCell>
                <TableCell>
                  {new Date(run.createdAt).toLocaleDateString('en-US', {
                    year: 'numeric',
                    month: 'short',
                    day: 'numeric',
                  })}
                </TableCell>
                <TableCell className="text-right">
                  <Link href={`/runs/${run.runId}`}>
                    <Button variant="ghost" size="sm">
                      View
                    </Button>
                  </Link>
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
  );
}

'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ClusterEdit } from '@/lib/types';

interface EditHistoryPanelProps {
  edits: ClusterEdit[];
  runId: string;
  clusterId: number;
}

export function EditHistoryPanel({ edits, runId, clusterId }: EditHistoryPanelProps) {
  const getEditTypeBadge = (editType: string) => {
    const variants: Record<string, { variant: 'default' | 'secondary' | 'destructive' | 'outline'; label: string }> = {
      rename: { variant: 'secondary', label: 'Rename' },
      move_query: { variant: 'default', label: 'Move Query' },
      merge: { variant: 'secondary', label: 'Merge' },
      split: { variant: 'secondary', label: 'Split' },
      delete: { variant: 'destructive', label: 'Delete' },
      tag: { variant: 'outline', label: 'Tag' },
    };

    const config = variants[editType] || { variant: 'outline' as const, label: editType };
    return <Badge variant={config.variant}>{config.label}</Badge>;
  };

  if (edits.length === 0) {
    return (
      <Card className="border-dashed">
        <CardHeader>
          <CardTitle>Edit History</CardTitle>
          <CardDescription>No edits yet</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            Changes to this cluster will appear here
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Edit History</CardTitle>
        <CardDescription>
          {edits.length} {edits.length === 1 ? 'edit' : 'edits'} to this cluster
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {edits.map((edit) => (
            <div key={edit.id} className="border-l-2 border-muted pl-4 py-2">
              <div className="flex items-start justify-between gap-2 mb-1">
                <div className="flex items-center gap-2">
                  {getEditTypeBadge(edit.editType)}
                  <span className="text-xs text-muted-foreground">
                    by {edit.editor}
                  </span>
                </div>
                <span className="text-xs text-muted-foreground">
                  {new Date(edit.timestamp).toLocaleString()}
                </span>
              </div>

              {edit.reason && (
                <p className="text-sm text-muted-foreground mt-1">
                  {edit.reason}
                </p>
              )}

              {/* Show change details for specific edit types */}
              {edit.editType === 'rename' && edit.oldValue && edit.newValue && (
                <div className="mt-2 text-xs">
                  <div className="text-muted-foreground">
                    <span className="line-through">{(edit.oldValue as any).title}</span>
                    {' → '}
                    <span className="font-medium">{(edit.newValue as any).title}</span>
                  </div>
                </div>
              )}

              {edit.editType === 'move_query' && edit.oldValue && edit.newValue && (
                <div className="mt-2 text-xs text-muted-foreground">
                  Moved query {(edit.oldValue as any).query_id} from cluster {(edit.oldValue as any).cluster_id} → {(edit.newValue as any).cluster_id}
                </div>
              )}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

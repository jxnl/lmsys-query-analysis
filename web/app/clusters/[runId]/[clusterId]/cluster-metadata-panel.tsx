'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ClusterMetadata } from '@/lib/db/schema';

interface ClusterMetadataPanelProps {
  metadata: ClusterMetadata | null;
  runId: string;
  clusterId: number;
}

export function ClusterMetadataPanel({ metadata, runId, clusterId }: ClusterMetadataPanelProps) {
  if (!metadata) {
    return (
      <Card className="border-dashed">
        <CardHeader>
          <CardTitle>Cluster Quality</CardTitle>
          <CardDescription>No quality metadata yet</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            Use <code className="text-xs bg-muted px-1 py-0.5 rounded">lmsys edit tag-cluster {runId} --cluster-id {clusterId}</code> to add quality metadata
          </p>
        </CardContent>
      </Card>
    );
  }

  // Render quality badge with appropriate color
  const getQualityColor = (quality: string | null) => {
    if (!quality) return 'secondary';
    switch (quality) {
      case 'high':
        return 'default'; // green
      case 'medium':
        return 'secondary'; // yellow
      case 'low':
        return 'destructive'; // red
      default:
        return 'secondary';
    }
  };

  // Render coherence stars
  const renderCoherenceScore = (score: number | null) => {
    if (!score) return null;
    return (
      <div className="flex items-center gap-1">
        {[...Array(5)].map((_, i) => (
          <span key={i} className={i < score ? 'text-yellow-500' : 'text-gray-300'}>
            ★
          </span>
        ))}
        <span className="ml-2 text-sm text-muted-foreground">{score}/5</span>
      </div>
    );
  };

  const flags = Array.isArray(metadata.flags) ? metadata.flags : [];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Cluster Quality</CardTitle>
        <CardDescription>
          Quality metadata and annotations
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {metadata.coherenceScore && (
          <div>
            <div className="text-sm font-medium mb-1">Coherence Score</div>
            {renderCoherenceScore(metadata.coherenceScore)}
          </div>
        )}

        {metadata.quality && (
          <div>
            <div className="text-sm font-medium mb-1">Quality Rating</div>
            <Badge variant={getQualityColor(metadata.quality)}>
              {metadata.quality.toUpperCase()}
            </Badge>
          </div>
        )}

        {flags.length > 0 && (
          <div>
            <div className="text-sm font-medium mb-2">Flags</div>
            <div className="flex flex-wrap gap-2">
              {flags.map((flag, idx) => (
                <Badge key={idx} variant="outline" className="text-red-600 border-red-300">
                  {flag}
                </Badge>
              ))}
            </div>
          </div>
        )}

        {metadata.notes && (
          <div>
            <div className="text-sm font-medium mb-1">Notes</div>
            <p className="text-sm text-muted-foreground whitespace-pre-wrap">
              {metadata.notes}
            </p>
          </div>
        )}

        {metadata.lastEdited && (
          <div className="text-xs text-muted-foreground pt-2 border-t">
            Last edited: {new Date(metadata.lastEdited).toLocaleString()}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

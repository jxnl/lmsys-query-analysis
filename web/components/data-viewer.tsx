"use client";

import { ScrollArea } from "@/components/ui/scroll-area";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import Link from "next/link";
import { useState } from "react";
import { ChevronDown, ChevronUp, ExternalLink } from "lucide-react";

export interface QueryWithClusters {
  id: number;
  conversation_id: string;
  model: string;
  query_text: string;
  language: string | null;
  timestamp: string | null;
  clusters?: Array<{
    cluster_id: number;
    run_id: string;
    title: string | null;
    confidence_score?: number | null;
  }>;
}

export interface DataViewerData {
  queries: QueryWithClusters[];
  total: number;
  page: number;
  pages: number;
  limit: number;
}

interface DataViewerProps {
  data: DataViewerData;
  onPageChange: (page: number) => void;
  showClusters?: boolean;
  filterRunId?: string; // Filter clusters to only show those from this run
}

export function DataViewer({
  data,
  onPageChange,
  showClusters = true,
  filterRunId,
}: DataViewerProps) {
  const { queries, page, pages, total } = data;
  const [expandedQueries, setExpandedQueries] = useState<Set<number>>(new Set());

  const toggleExpand = (queryId: number) => {
    setExpandedQueries((prev) => {
      const next = new Set(prev);
      if (next.has(queryId)) {
        next.delete(queryId);
      } else {
        next.add(queryId);
      }
      return next;
    });
  };

  const truncateText = (text: string, maxLength: number = 200) => {
    if (text.length <= maxLength) return text;
    return text.slice(0, maxLength) + "...";
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <p className="text-sm text-muted-foreground">
          Showing {queries.length} of {total.toLocaleString()} queries
        </p>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => onPageChange(page - 1)}
            disabled={page === 1}
          >
            Previous
          </Button>
          <span className="text-sm">
            Page {page.toLocaleString()} of {pages.toLocaleString()}
          </span>
          <Button
            variant="outline"
            size="sm"
            onClick={() => onPageChange(page + 1)}
            disabled={page === pages}
          >
            Next
          </Button>
        </div>
      </div>

      <ScrollArea className="h-[calc(100vh-240px)] rounded-md border">
        <div className="space-y-3 p-4">
          {queries.map((query) => {
            const isExpanded = expandedQueries.has(query.id);
            const displayText = isExpanded ? query.query_text : truncateText(query.query_text);
            const needsTruncation = query.query_text.length > 200;

            return (
              <Card key={query.id}>
                <CardContent className="pt-4">
                  <div className="space-y-3">
                    {/* Query Text */}
                    <div className="space-y-2">
                      <p className="text-sm whitespace-pre-wrap">{displayText}</p>
                      {needsTruncation && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => toggleExpand(query.id)}
                          className="h-6 px-2 text-xs"
                        >
                          {isExpanded ? (
                            <>
                              <ChevronUp className="h-3 w-3 mr-1" />
                              Show less
                            </>
                          ) : (
                            <>
                              <ChevronDown className="h-3 w-3 mr-1" />
                              Show more
                            </>
                          )}
                        </Button>
                      )}
                    </div>

                    {/* Metadata Row */}
                    <div className="flex items-start justify-between gap-2 flex-wrap">
                      <div className="flex gap-2 items-center flex-wrap">
                        {query.model && (
                          <Badge variant="secondary" className="text-xs">
                            {query.model}
                          </Badge>
                        )}
                        {query.language && (
                          <Badge variant="outline" className="text-xs">
                            {query.language}
                          </Badge>
                        )}
                      </div>
                    </div>

                    {/* Cluster Associations */}
                    {showClusters &&
                      query.clusters &&
                      query.clusters.length > 0 &&
                      (() => {
                        // Filter clusters by run if filterRunId is provided
                        const displayClusters = filterRunId
                          ? query.clusters.filter((c) => c.run_id === filterRunId)
                          : query.clusters;

                        if (displayClusters.length === 0) return null;

                        return (
                          <div className="space-y-2 pt-2 border-t">
                            <p className="text-xs font-medium text-muted-foreground">
                              Topics ({displayClusters.length}):
                            </p>
                            <div className="flex flex-wrap gap-2">
                              {displayClusters.map((cluster, idx) => (
                                <Link
                                  key={`${cluster.run_id}-${cluster.cluster_id}-${idx}`}
                                  href={`/clusters/${cluster.run_id}/${cluster.cluster_id}`}
                                >
                                  <Badge
                                    variant="default"
                                    className="text-xs hover:bg-primary/80 cursor-pointer"
                                  >
                                    {cluster.title || `Cluster ${cluster.cluster_id}`}
                                    <ExternalLink className="ml-1 h-3 w-3" />
                                  </Badge>
                                </Link>
                              ))}
                            </div>
                          </div>
                        );
                      })()}

                    {/* Query Metadata */}
                    <div className="flex gap-4 text-xs text-muted-foreground pt-2 border-t">
                      <span>ID: {query.id}</span>
                      <span>Conversation: {query.conversation_id.substring(0, 8)}...</span>
                      {query.timestamp && (
                        <span>
                          {new Date(query.timestamp).toLocaleDateString("en-US", {
                            year: "numeric",
                            month: "short",
                            day: "numeric",
                          })}
                        </span>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </ScrollArea>

      <div className="flex items-center justify-center">
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => onPageChange(page - 1)}
            disabled={page === 1}
          >
            Previous
          </Button>
          <span className="text-sm px-4">
            Page {page.toLocaleString()} of {pages.toLocaleString()}
          </span>
          <Button
            variant="outline"
            size="sm"
            onClick={() => onPageChange(page + 1)}
            disabled={page === pages}
          >
            Next
          </Button>
        </div>
      </div>
    </div>
  );
}

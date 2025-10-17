"use client";

import { useState, useEffect, createContext, useContext } from "react";
import Link from "next/link";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import {
  ChevronRight,
  ChevronDown,
  FileText,
  Loader2,
  Copy,
  Check,
  Maximize2,
  Minimize2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import type { components } from "@/lib/api/types";
import { apiFetch } from "@/lib/api";

type ClusterHierarchy = components["schemas"]["HierarchyNode"];
type Query = components["schemas"]["QueryResponse"];

// Context for managing expand/collapse state
const ExpandContext = createContext<{
  expandAll: boolean;
  toggleExpandAll: () => void;
} | null>(null);

interface HierarchyTreeProps {
  nodes: ClusterHierarchy[];
  runId: string;
  queryCounts?: Record<number, number>;
  hierarchyRunId?: string;
}

export function HierarchyTree({
  nodes,
  runId,
  queryCounts = {},
  hierarchyRunId,
}: HierarchyTreeProps) {
  const [expandAll, setExpandAll] = useState(false);

  // Build tree structure - find root nodes (no parent)
  const rootNodes = nodes.filter((n) => n.parent_cluster_id === null);

  // Calculate total queries from hierarchy nodes (they now include query_count)
  const totalQueries = nodes
    .filter((n) => n.level === 0) // Only count leaf nodes to avoid double-counting
    .reduce((sum, n) => sum + (n.query_count || 0), 0);

  // Calculate hierarchy stats
  const maxLevel = Math.max(...nodes.map((n) => n.level), 0);
  const leafCount = nodes.filter((n) => !n.children_ids || n.children_ids.length === 0).length;

  // Helper to calculate total query count for any node (including descendants)
  const getTotalQueryCount = (nodeId: number): number => {
    const currentNode = nodes.find((n) => n.cluster_id === nodeId);
    if (!currentNode) return 0;

    // Use the query_count from the node directly (API now calculates this)
    return currentNode.query_count || 0;
  };

  // Sort root nodes by query count (descending)
  const sortedRootNodes = [...rootNodes].sort((a, b) => {
    const countA = getTotalQueryCount(a.cluster_id);
    const countB = getTotalQueryCount(b.cluster_id);
    return countB - countA;
  });

  if (sortedRootNodes.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        No hierarchy found for this run. Run `lmsys merge-clusters` to create one.
      </div>
    );
  }

  return (
    <ExpandContext.Provider value={{ expandAll, toggleExpandAll: () => setExpandAll(!expandAll) }}>
      <div className="space-y-4">
        {/* Header with stats and controls */}
        <div className="flex items-center justify-between pb-2 border-b">
          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            <span>{nodes.length} clusters</span>
            <span>•</span>
            <span>{leafCount} leaf clusters</span>
            <span>•</span>
            <span>{maxLevel + 1} levels</span>
            <span>•</span>
            <span className="font-medium text-foreground">
              {totalQueries.toLocaleString()} total queries
            </span>
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setExpandAll(true)}
              className="h-7 text-xs"
            >
              <Maximize2 className="h-3 w-3 mr-1" />
              Expand All
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setExpandAll(false)}
              className="h-7 text-xs"
            >
              <Minimize2 className="h-3 w-3 mr-1" />
              Collapse All
            </Button>
          </div>
        </div>

        {/* Tree */}
        <div className="space-y-3">
          {sortedRootNodes.map((node) => (
            <TreeNode
              key={node.cluster_id}
              node={node}
              nodes={nodes}
              runId={runId}
              queryCounts={queryCounts}
              totalQueries={totalQueries}
              getTotalQueryCount={getTotalQueryCount}
              hierarchyRunId={hierarchyRunId}
            />
          ))}
        </div>
      </div>
    </ExpandContext.Provider>
  );
}

interface TreeNodeProps {
  node: ClusterHierarchy;
  nodes: ClusterHierarchy[];
  runId: string;
  queryCounts?: Record<number, number>;
  totalQueries: number;
  getTotalQueryCount: (nodeId: number) => number;
  hierarchyRunId?: string;
}

function TreeNode({
  node,
  nodes,
  runId,
  queryCounts = {},
  totalQueries,
  getTotalQueryCount,
  hierarchyRunId,
}: TreeNodeProps) {
  const expandContext = useContext(ExpandContext);
  const [isOpen, setIsOpen] = useState(false);
  const [queries, setQueries] = useState<Query[]>([]);
  const [isLoadingQueries, setIsLoadingQueries] = useState(false);
  const [showAllQueries, setShowAllQueries] = useState(false);
  const [isCopied, setIsCopied] = useState(false);

  // Respond to expand/collapse all
  useEffect(() => {
    if (expandContext) {
      setIsOpen(expandContext.expandAll);
    }
  }, [expandContext]);

  // Find children nodes
  const children = node.children_ids
    ? nodes.filter((n) => node.children_ids?.includes(n.cluster_id))
    : [];

  // Sort children by total query count (descending)
  const sortedChildren = [...children].sort((a, b) => {
    const countA = getTotalQueryCount(a.cluster_id);
    const countB = getTotalQueryCount(b.cluster_id);
    return countB - countA;
  });

  const isLeaf = sortedChildren.length === 0;
  // Use the query_count from the node directly (API now includes it)
  const queryCount = node.query_count || 0;

  // Calculate total query count for this node (including all descendants)
  const totalQueryCount = getTotalQueryCount(node.cluster_id);

  // Calculate percentage relative to parent's total
  // For root nodes, use global totalQueries; for children, calculate parent's total
  let parentTotal = totalQueries;
  if (node.parent_cluster_id !== null && node.parent_cluster_id !== undefined) {
    parentTotal = getTotalQueryCount(node.parent_cluster_id);
  }
  const percentage = parentTotal > 0 ? (totalQueryCount / parentTotal) * 100 : 0;

  // Determine size category for color coding
  const getSizeCategory = (): "large" | "medium" | "small" => {
    if (percentage >= 10) return "large";
    if (percentage >= 3) return "medium";
    return "small";
  };
  const sizeCategory = getSizeCategory();

  // Build hierarchy path (from root to this node)
  const buildHierarchyPath = (): string[] => {
    const path: string[] = [];
    let currentId: number | null = node.cluster_id;

    while (currentId !== null) {
      const currentNode = nodes.find((n) => n.cluster_id === currentId);
      if (!currentNode) break;

      path.unshift(
        `${currentNode.title || `Cluster ${currentNode.cluster_id}`} (ID: ${currentNode.cluster_id}, Level: ${currentNode.level})`
      );
      currentId = currentNode.parent_cluster_id ?? null;
    }

    return path;
  };

  // Copy cluster metadata to clipboard
  const copyMetadata = async () => {
    const hierarchyPath = buildHierarchyPath();

    // Fetch sample queries if not already loaded
    let sampleQueries = queries;
    if (isLeaf && sampleQueries.length === 0) {
      try {
        const data = await apiFetch<{
          items: Query[];
          total: number;
          page: number;
          pages: number;
          limit: number;
        }>(`/api/queries?run_id=${runId}&cluster_id=${node.cluster_id}&page=1&limit=5`);
        sampleQueries = data.items;
      } catch (err) {
        console.error("Failed to fetch sample queries:", err);
      }
    }

    const queriesSection =
      sampleQueries.length > 0
        ? `## Sample Queries (${sampleQueries.length} of ${queryCount})
${sampleQueries
  .map(
    (q, i) => `
### Query ${i + 1}
- **Text**: ${q.query_text}
- **Model**: ${q.model || "N/A"}
- **Language**: ${q.language || "N/A"}
- **ID**: ${q.id}
`
  )
  .join("\n")}
`
        : isLeaf
          ? `## Sample Queries
No queries available
`
          : "";

    const metadata = `# Cluster Metadata

## Run Information
- Run ID: ${runId}
- Hierarchy Run ID: ${hierarchyRunId || node.hierarchy_run_id || "N/A"}

## Cluster Information
- Cluster ID: ${node.cluster_id}
- Title: ${node.title || "N/A"}
- Level: ${node.level}
- Type: ${isLeaf ? "Leaf" : "Parent"}

## Hierarchy Path
${hierarchyPath.map((p, i) => `${i + 1}. ${p}`).join("\n")}

## Statistics
- Direct Query Count: ${queryCount}
- Total Query Count (including descendants): ${totalQueryCount}
- Percentage of Parent: ${percentage.toFixed(1)}%
- Parent Total Queries: ${parentTotal}
- Global Total Queries: ${totalQueries}

## Structure
- Parent Cluster ID: ${node.parent_cluster_id ?? "None (Root)"}
- Children Count: ${sortedChildren.length}
- Children IDs: ${node.children_ids?.join(", ") || "None"}

## Description
${node.description || "N/A"}

${queriesSection}
## Database Record
\`\`\`json
${JSON.stringify(
  {
    hierarchy_run_id: node.hierarchy_run_id,
    run_id: node.run_id,
    cluster_id: node.cluster_id,
    parent_cluster_id: node.parent_cluster_id,
    level: node.level,
    children_ids: node.children_ids,
    title: node.title,
    description: node.description,
  },
  null,
  2
)}
\`\`\`
`;

    try {
      await navigator.clipboard.writeText(metadata);
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy metadata:", err);
    }
  };

  // Load queries when leaf node is opened
  useEffect(() => {
    if (isLeaf && isOpen && queries.length === 0) {
      setIsLoadingQueries(true);
      apiFetch<{
        items: Query[];
        total: number;
        page: number;
        pages: number;
        limit: number;
      }>(`/api/queries?run_id=${runId}&cluster_id=${node.cluster_id}&page=1&limit=10`)
        .then((data) => {
          setQueries(data.items);
        })
        .catch((err) => {
          console.error("Failed to load queries:", err);
        })
        .finally(() => {
          setIsLoadingQueries(false);
        });
    }
  }, [isLeaf, isOpen, runId, node.cluster_id, queries.length, expandContext]);

  const displayedQueries = showAllQueries ? queries : queries.slice(0, 5);

  return (
    <div className="border-l-2 border-border pl-4">
      {isLeaf ? (
        // Leaf node - collapsible to show queries
        <Collapsible open={isOpen} onOpenChange={setIsOpen}>
          <div
            className={`rounded-lg border transition-all ${
              sizeCategory === "large"
                ? "border-blue-200 bg-blue-50/30 dark:border-blue-900 dark:bg-blue-950/20"
                : sizeCategory === "medium"
                  ? "border-border bg-background"
                  : "border-border/50 bg-muted/20"
            }`}
          >
            <CollapsibleTrigger className="flex items-center gap-2 py-2.5 px-3 hover:bg-accent/50 rounded-lg w-full text-left">
              {isOpen ? (
                <ChevronDown className="h-4 w-4 flex-shrink-0" />
              ) : (
                <ChevronRight className="h-4 w-4 flex-shrink-0" />
              )}
              <FileText
                className={`h-4 w-4 flex-shrink-0 ${
                  sizeCategory === "large"
                    ? "text-blue-600 dark:text-blue-400"
                    : "text-muted-foreground"
                }`}
              />
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <span
                    className={`font-medium truncate ${
                      sizeCategory === "large" ? "text-base" : "text-sm"
                    }`}
                  >
                    {node.title || `Cluster ${node.cluster_id}`}
                  </span>
                  <span
                    className={`text-xs font-medium ${
                      sizeCategory === "large"
                        ? "text-blue-600 dark:text-blue-400"
                        : sizeCategory === "medium"
                          ? "text-foreground"
                          : "text-muted-foreground"
                    }`}
                  >
                    {queryCount.toLocaleString()} queries ({percentage.toFixed(1)}%)
                  </span>
                </div>
                {/* Visual progress bar */}
                <div className="w-full h-1.5 bg-muted rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all ${
                      sizeCategory === "large"
                        ? "bg-blue-500"
                        : sizeCategory === "medium"
                          ? "bg-primary"
                          : "bg-muted-foreground/40"
                    }`}
                    style={{ width: `${Math.min(percentage, 100)}%` }}
                  />
                </div>
              </div>
              <div className="flex items-center gap-1 flex-shrink-0">
                <div
                  onClick={(e) => {
                    e.stopPropagation();
                    copyMetadata();
                  }}
                  className="h-7 px-2 text-xs inline-flex items-center justify-center gap-1 rounded-md hover:bg-accent hover:text-accent-foreground cursor-pointer"
                >
                  {isCopied ? (
                    <>
                      <Check className="h-3 w-3" />
                      <span className="hidden sm:inline">Copied</span>
                    </>
                  ) : (
                    <>
                      <Copy className="h-3 w-3" />
                      <span className="hidden sm:inline">Copy</span>
                    </>
                  )}
                </div>
                <Link
                  href={`/clusters/${runId}/${node.cluster_id}`}
                  className="text-xs text-muted-foreground hover:text-foreground px-2 py-1 rounded-md hover:bg-accent"
                  onClick={(e) => e.stopPropagation()}
                >
                  View all →
                </Link>
              </div>
            </CollapsibleTrigger>

            <CollapsibleContent className="px-3 pb-3 pt-2 space-y-2">
              {isLoadingQueries ? (
                <div className="flex items-center gap-2 text-sm text-muted-foreground py-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Loading queries...
                </div>
              ) : queries.length > 0 ? (
                <>
                  <div className="space-y-2">
                    {displayedQueries.map((query) => (
                      <div
                        key={query.id}
                        className="text-sm p-2.5 bg-muted/50 rounded-md border border-border/50"
                      >
                        <p className="whitespace-pre-wrap text-foreground/90">
                          {query.query_text.length > 200
                            ? query.query_text.slice(0, 200) + "..."
                            : query.query_text}
                        </p>
                        <div className="flex gap-2 mt-1.5 text-xs text-muted-foreground">
                          {query.model && <span className="font-medium">{query.model}</span>}
                          {query.language && <span>• {query.language}</span>}
                        </div>
                      </div>
                    ))}
                  </div>
                  {queries.length > 5 && !showAllQueries && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setShowAllQueries(true)}
                      className="text-xs w-full"
                    >
                      Show {queries.length - 5} more queries
                    </Button>
                  )}
                  {showAllQueries && queries.length > 5 && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setShowAllQueries(false)}
                      className="text-xs w-full"
                    >
                      Show less
                    </Button>
                  )}
                </>
              ) : (
                <p className="text-sm text-muted-foreground py-2">No queries found</p>
              )}
            </CollapsibleContent>
          </div>
        </Collapsible>
      ) : (
        // Parent node - collapsible
        <Collapsible open={isOpen} onOpenChange={setIsOpen}>
          <div
            className={`rounded-lg border transition-all ${
              sizeCategory === "large"
                ? "border-emerald-200 bg-emerald-50/30 dark:border-emerald-900 dark:bg-emerald-950/20"
                : sizeCategory === "medium"
                  ? "border-border bg-background"
                  : "border-border/50 bg-muted/20"
            }`}
          >
            <CollapsibleTrigger className="flex items-center gap-2 py-2.5 px-3 hover:bg-accent/50 rounded-lg w-full text-left">
              {isOpen ? (
                <ChevronDown className="h-4 w-4 flex-shrink-0" />
              ) : (
                <ChevronRight className="h-4 w-4 flex-shrink-0" />
              )}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <span
                    className={`font-semibold truncate ${
                      sizeCategory === "large" ? "text-base" : "text-sm"
                    }`}
                  >
                    {node.title || `Cluster ${node.cluster_id}`}
                  </span>
                  <span
                    className={`text-xs font-medium ${
                      sizeCategory === "large"
                        ? "text-emerald-600 dark:text-emerald-400"
                        : sizeCategory === "medium"
                          ? "text-foreground"
                          : "text-muted-foreground"
                    }`}
                  >
                    {totalQueryCount.toLocaleString()} queries ({percentage.toFixed(1)}%)
                  </span>
                  <span className="text-xs text-muted-foreground">
                    • {sortedChildren.length} {sortedChildren.length === 1 ? "child" : "children"}
                  </span>
                </div>
                {/* Visual progress bar */}
                <div className="w-full h-1.5 bg-muted rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all ${
                      sizeCategory === "large"
                        ? "bg-emerald-500"
                        : sizeCategory === "medium"
                          ? "bg-primary"
                          : "bg-muted-foreground/40"
                    }`}
                    style={{ width: `${Math.min(percentage, 100)}%` }}
                  />
                </div>
              </div>
              <div
                onClick={(e) => {
                  e.stopPropagation();
                  copyMetadata();
                }}
                className="h-7 px-2 text-xs inline-flex items-center justify-center gap-1 rounded-md hover:bg-accent hover:text-accent-foreground cursor-pointer flex-shrink-0"
              >
                {isCopied ? (
                  <>
                    <Check className="h-3 w-3" />
                    <span className="hidden sm:inline">Copied</span>
                  </>
                ) : (
                  <>
                    <Copy className="h-3 w-3" />
                    <span className="hidden sm:inline">Copy</span>
                  </>
                )}
              </div>
            </CollapsibleTrigger>

            <CollapsibleContent className="px-3 pb-3 pt-2 space-y-2">
              {node.description && (
                <p className="text-sm text-muted-foreground mb-3 pb-3 border-b">
                  {node.description}
                </p>
              )}
              <div className="space-y-2 pl-2">
                {sortedChildren.map((child) => (
                  <TreeNode
                    key={child.cluster_id}
                    node={child}
                    nodes={nodes}
                    runId={runId}
                    queryCounts={queryCounts}
                    totalQueries={totalQueries}
                    getTotalQueryCount={getTotalQueryCount}
                    hierarchyRunId={hierarchyRunId}
                  />
                ))}
              </div>
            </CollapsibleContent>
          </div>
        </Collapsible>
      )}
    </div>
  );
}

"use client";

import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { components } from "@/lib/api/types";
import { Search } from "lucide-react";

type ClusteringRun = components["schemas"]["ClusteringRunSummary"];

interface SearchClientProps {
  runs: ClusteringRun[];
}

export function SearchClient({ runs }: SearchClientProps) {

  const defaultRunId = runs.length > 0 ? runs[0].run_id : "all";

  const [searchText, setSearchText] = useState("");
  const [selectedRunId, setSelectedRunId] = useState<string>(defaultRunId);

  const handleSearch = () => {
    if (!searchText.trim()) return;

  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSearch();
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardContent className="pt-6">
          <div className="space-y-4">
            <div className="flex gap-4">
              <div className="flex-1">
                <Input
                  type="text"
                  placeholder="Search queries..."
                  value={searchText}
                  onChange={(e) => setSearchText(e.target.value)}
                  onKeyDown={handleKeyDown}
                  className="w-full"
                />
              </div>
              <div className="w-64">
                <Select value={selectedRunId} onValueChange={setSelectedRunId}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select run" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Runs</SelectItem>
                    {runs.map((run) => (
                      <SelectItem key={run.run_id} value={run.run_id}>
                        {run.run_id} ({run.algorithm})
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <Button onClick={handleSearch} disabled={!searchText.trim()}>
                <Search className="mr-2 h-4 w-4" />
                Search
              </Button>
            </div>

            <div className="text-sm text-muted-foreground">
              <p>Search functionality coming soon!</p>
              <p className="mt-2">
                This will allow you to search queries semantically using vector similarity. Results will
                show matching queries with their cluster assignments and relevance scores.
              </p>
              {selectedRunId !== "all" && (
                <p className="mt-1">
                  Filtering by run: <span className="font-mono">{selectedRunId}</span>
                </p>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
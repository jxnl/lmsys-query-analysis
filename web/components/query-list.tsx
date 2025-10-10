'use client'

import { ScrollArea } from '@/components/ui/scroll-area';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import type { PaginatedQueries } from '@/lib/types';

interface QueryListProps {
  data: PaginatedQueries;
  onPageChange: (page: number) => void;
}

export function QueryList({ data, onPageChange }: QueryListProps) {
  const { queries, page, pages, total } = data;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <p className="text-sm text-muted-foreground">
          Showing {queries.length} of {total} queries
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
            Page {page} of {pages}
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

      <ScrollArea className="h-[600px] rounded-md border">
        <div className="space-y-3 p-4">
          {queries.map((query) => (
            <Card key={query.id}>
              <CardContent className="pt-4">
                <div className="space-y-2">
                  <div className="flex items-start justify-between gap-2">
                    <p className="text-sm flex-1">{query.queryText}</p>
                    <div className="flex gap-2 shrink-0">
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
                  <div className="flex gap-4 text-xs text-muted-foreground">
                    <span>ID: {query.id}</span>
                    <span>Conversation: {query.conversationId.substring(0, 8)}...</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
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
            Page {page} of {pages}
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

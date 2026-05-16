'use client';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import {
  getDocumentTable,
  getFigureMediaUrl,
  listDocumentFigures,
  listDocumentTables,
  reclassifyDocumentTables,
  type DocumentFigureListItem,
  type DocumentTableListItem,
} from '@/lib/api/edgequake';
import { useMutation, useQueries, useQuery, useQueryClient } from '@tanstack/react-query';
import { Images, Loader2, RefreshCw, Table2 } from 'lucide-react';
import { useMemo, useState } from 'react';
import { toast } from 'sonner';

interface FiguresTabContentProps {
  documentId: string;
}

type FilterKey =
  | 'all'
  | 'images'
  | 'performance'
  | 'quality'
  | 'complexity'
  | 'other';

const FILTER_OPTIONS: { key: FilterKey; label: string }[] = [
  { key: 'all', label: 'All' },
  { key: 'images', label: 'Images' },
  { key: 'performance', label: 'Performance' },
  { key: 'quality', label: 'Quality' },
  { key: 'complexity', label: 'Complexity' },
  { key: 'other', label: 'Other' },
];

/**
 * Gallery tab listing every image figure and every extracted table for the
 * document, ordered by (page, order_index). Tables show a `table_type`
 * badge once the classification stage runs; until then they're treated as
 * "Other" by the filter chips so they don't disappear.
 */
export function FiguresTabContent({ documentId }: FiguresTabContentProps) {
  const [filter, setFilter] = useState<FilterKey>('all');
  const queryClient = useQueryClient();

  const figuresQuery = useQuery({
    queryKey: ['document-figures', documentId],
    queryFn: () => listDocumentFigures(documentId),
    enabled: !!documentId,
    staleTime: 30 * 1000,
  });

  const tablesQuery = useQuery({
    queryKey: ['document-tables', documentId],
    queryFn: () => listDocumentTables(documentId),
    enabled: !!documentId,
    staleTime: 30 * 1000,
    // Tables get a classification stage that lands after PDF processing —
    // refetch a few times so newly-arrived `table_type` values show up
    // without forcing a manual reload.
    refetchInterval: (query) => {
      const tables = query.state.data?.tables ?? [];
      const anyUnclassified = tables.some((t) => !t.table_type);
      return anyUnclassified ? 5_000 : false;
    },
  });

  const reclassifyMutation = useMutation({
    mutationFn: () => reclassifyDocumentTables(documentId),
    onSuccess: (data) => {
      toast.success(
        data.tables_reset > 0
          ? `Re-classifying ${data.tables_reset} table${
              data.tables_reset === 1 ? '' : 's'
            }…`
          : 'Re-classification queued.',
      );
      // Reset the list immediately so the badges show "unclassified" while
      // the worker re-runs the LLM. The polling refetch will pick up the new
      // labels as they land.
      queryClient.invalidateQueries({
        queryKey: ['document-tables', documentId],
      });
    },
    onError: (err) => {
      toast.error('Re-classification failed', {
        description: err instanceof Error ? err.message : String(err),
      });
    },
  });

  const figures = figuresQuery.data?.figures ?? [];
  const tables = tablesQuery.data?.tables ?? [];

  // Fetch the HTML for each visible table on demand so the card preview can
  // render the actual table (not just a caption). `useQueries` so the list
  // of fetches scales with `tables.length` without violating hook rules.
  const tableDetails = useQueries({
    queries: tables.map((t) => ({
      queryKey: ['document-table', documentId, t.table_id],
      queryFn: () => getDocumentTable(documentId, t.table_id),
      enabled: !!documentId,
      staleTime: 60 * 1000,
    })),
  });

  const items = useMemo(() => mergeItems(figures, tables), [figures, tables]);
  const filtered = useMemo(
    () => items.filter((it) => matchesFilter(it, filter)),
    [items, filter],
  );

  const isLoading = figuresQuery.isLoading || tablesQuery.isLoading;
  const isError = figuresQuery.isError || tablesQuery.isError;

  return (
    <div className="p-4 space-y-4 overflow-auto h-full">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-sm font-semibold text-muted-foreground">
          {isLoading
            ? 'Loading figures and tables…'
            : `${items.length} item${items.length === 1 ? '' : 's'}`}
        </h2>
        <div className="flex items-center gap-2 flex-wrap">
          <div className="flex flex-wrap gap-1.5">
            {FILTER_OPTIONS.map((opt) => (
              <Button
                key={opt.key}
                variant={filter === opt.key ? 'default' : 'outline'}
                size="sm"
                className="h-7 text-xs"
                onClick={() => setFilter(opt.key)}
              >
                {opt.label}
              </Button>
            ))}
          </div>
          <Button
            variant="outline"
            size="sm"
            className="h-7 text-xs gap-1"
            disabled={
              reclassifyMutation.isPending || tables.length === 0
            }
            onClick={() => reclassifyMutation.mutate()}
            title="Reset table types and re-run the LLM classifier"
          >
            {reclassifyMutation.isPending ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : (
              <RefreshCw className="h-3 w-3" />
            )}
            Re-classify
          </Button>
        </div>
      </div>

      {isLoading && <GallerySkeleton />}

      {isError && !isLoading && (
        <div className="text-destructive text-sm">
          Failed to load figures or tables. Try refreshing the page.
        </div>
      )}

      {!isLoading && !isError && filtered.length === 0 && (
        <div className="flex flex-col items-center justify-center py-16 text-muted-foreground gap-2">
          <Images className="h-10 w-10 opacity-50" />
          <p className="text-sm">
            {items.length === 0
              ? 'No figures or tables were extracted from this document.'
              : 'No items match this filter.'}
          </p>
        </div>
      )}

      {!isLoading && !isError && filtered.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
          {filtered.map((item) =>
            item.kind === 'figure' ? (
              <FigureCard
                key={`fig:${item.figure.figure_id}`}
                documentId={documentId}
                figure={item.figure}
              />
            ) : (
              <TableCard
                key={`tbl:${item.table.table_id}`}
                table={item.table}
                html={
                  tableDetails[item.tableIndex]?.data?.html ?? null
                }
                isLoading={tableDetails[item.tableIndex]?.isLoading ?? false}
              />
            ),
          )}
        </div>
      )}
    </div>
  );
}

type GalleryItem =
  | { kind: 'figure'; figure: DocumentFigureListItem; sortKey: number }
  | {
      kind: 'table';
      table: DocumentTableListItem;
      tableIndex: number;
      sortKey: number;
    };

function mergeItems(
  figures: DocumentFigureListItem[],
  tables: DocumentTableListItem[],
): GalleryItem[] {
  // (page * 10000) + order_index lets a single sort handle both lists.
  // Order_index reaches into the hundreds at most, so 10k headroom is fine.
  const out: GalleryItem[] = [];
  for (const f of figures) {
    out.push({
      kind: 'figure',
      figure: f,
      sortKey: f.page * 10000 + f.order_index,
    });
  }
  tables.forEach((t, idx) => {
    out.push({
      kind: 'table',
      table: t,
      tableIndex: idx,
      sortKey: t.page * 10000 + t.order_index,
    });
  });
  out.sort((a, b) => a.sortKey - b.sortKey);
  return out;
}

function matchesFilter(item: GalleryItem, filter: FilterKey): boolean {
  if (filter === 'all') return true;
  if (filter === 'images') return item.kind === 'figure';
  if (item.kind !== 'table') return false;
  const type = item.table.table_type ?? 'other';
  return type === filter;
}

function FigureCard({
  documentId,
  figure,
}: {
  documentId: string;
  figure: DocumentFigureListItem;
}) {
  return (
    <div className="rounded-xl border bg-card p-3 flex flex-col gap-2">
      <div className="flex items-center justify-between gap-2">
        <Badge variant="secondary" className="text-[10px] uppercase">
          <Images className="h-3 w-3 mr-1" /> Figure
        </Badge>
        <span className="text-[11px] text-muted-foreground">
          p.{figure.page || '?'}
        </span>
      </div>
      <div className="bg-muted/40 rounded-lg overflow-hidden flex items-center justify-center min-h-32">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={getFigureMediaUrl(documentId, figure.figure_id)}
          alt={figure.caption || figure.figure_id}
          className="max-h-64 w-auto object-contain"
        />
      </div>
      <p className="text-xs text-foreground/90 line-clamp-3">
        {figure.caption || (
          <span className="text-muted-foreground italic">
            (no caption detected)
          </span>
        )}
      </p>
      <p className="text-[10px] font-mono text-muted-foreground">
        {figure.figure_id}
      </p>
    </div>
  );
}

function TableCard({
  table,
  html,
  isLoading,
}: {
  table: DocumentTableListItem;
  html: string | null;
  isLoading: boolean;
}) {
  return (
    <div className="rounded-xl border bg-card p-3 flex flex-col gap-2">
      <div className="flex items-center justify-between gap-2">
        <Badge variant="secondary" className="text-[10px] uppercase">
          <Table2 className="h-3 w-3 mr-1" /> Table
        </Badge>
        <span className="text-[11px] text-muted-foreground">
          p.{table.page || '?'}
        </span>
      </div>
      <div className="bg-muted/30 rounded-lg p-2 overflow-auto max-h-64">
        {isLoading || html === null ? (
          <div className="flex items-center justify-center py-6 text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
          </div>
        ) : (
          // Backend-rendered HTML from VLM-OCR's RecognitionTask::Table.
          // It's the same HTML stored in chunks.table_html — trusted output
          // of our own extractor, not user input.
          <div
            className="prose prose-sm dark:prose-invert max-w-none [&_table]:border [&_th]:px-2 [&_td]:px-2 [&_th]:py-1 [&_td]:py-1"
            dangerouslySetInnerHTML={{ __html: html }}
          />
        )}
      </div>
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs text-foreground/90 line-clamp-2">
          {table.caption || (
            <span className="text-muted-foreground italic">
              (no caption detected)
            </span>
          )}
        </p>
        <TableTypeBadge type={table.table_type} />
      </div>
      <p className="text-[10px] font-mono text-muted-foreground">
        {table.table_id}
      </p>
    </div>
  );
}

function TableTypeBadge({ type }: { type: string | null }) {
  if (!type) {
    return (
      <Badge variant="outline" className="text-[10px] whitespace-nowrap">
        unclassified
      </Badge>
    );
  }
  const color =
    type === 'performance'
      ? 'bg-blue-500/15 text-blue-700 dark:text-blue-300 border-blue-500/30'
      : type === 'quality'
      ? 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300 border-emerald-500/30'
      : type === 'complexity'
      ? 'bg-purple-500/15 text-purple-700 dark:text-purple-300 border-purple-500/30'
      : 'bg-muted text-muted-foreground border-border';
  return (
    <Badge variant="outline" className={`text-[10px] whitespace-nowrap ${color}`}>
      {type}
    </Badge>
  );
}

function GallerySkeleton() {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
      {Array.from({ length: 6 }).map((_, i) => (
        <div key={i} className="rounded-xl border p-3 space-y-2">
          <Skeleton className="h-4 w-16" />
          <Skeleton className="h-32 w-full" />
          <Skeleton className="h-3 w-4/5" />
          <Skeleton className="h-3 w-2/3" />
        </div>
      ))}
    </div>
  );
}

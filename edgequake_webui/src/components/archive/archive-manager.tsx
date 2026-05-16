/**
 * @module ArchiveManager
 * @description Lists archived documents for the active workspace.
 *
 * Archived documents keep the original PDF, converted Markdown, extracted
 * algorithms, and references (`document_repos`). Their chunks, embeddings,
 * knowledge-graph contributions, and indexed code are gone. The list is
 * read-only with view actions; the only mutations are unarchive and
 * permanent delete.
 */
'use client';

import { ArchiveActionsMenu } from '@/components/archive/archive-actions-menu';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from '@/components/ui/table';
import { useDocumentMutations } from '@/hooks/use-document-mutations';
import { getDocuments } from '@/lib/api/edgequake';
import { useTenantStore } from '@/stores/use-tenant-store';
import { useQuery } from '@tanstack/react-query';
import { Archive } from 'lucide-react';
import { useTranslation } from 'react-i18next';

function formatDate(value?: string | null): string {
  if (!value) return '—';
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? value : d.toLocaleString();
}

export function ArchiveManager() {
  const { t } = useTranslation();
  const { selectedTenantId, selectedWorkspaceId } = useTenantStore();
  const { deleteMutation, unarchiveMutation } = useDocumentMutations();

  const { data, isLoading } = useQuery({
    queryKey: ['archived-documents', selectedTenantId, selectedWorkspaceId],
    queryFn: () =>
      getDocuments({
        page: 1,
        page_size: 200,
        archived: 'true',
      }),
    enabled: !!selectedTenantId && !!selectedWorkspaceId,
    staleTime: 30 * 1000,
  });

  const documents = data?.items ?? [];

  return (
    <div className="flex flex-col h-full">
      <header className="shrink-0 border-b px-4 py-3">
        <div className="flex items-center gap-2">
          <Archive className="h-5 w-5 text-muted-foreground" />
          <h1 className="text-lg font-semibold">
            {t('archive.title', 'Archive')}
          </h1>
          <Badge variant="outline" className="ml-2">
            {documents.length}
          </Badge>
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          {t(
            'archive.subtitle',
            'Archived documents keep their PDF, Markdown, algorithms, and references. They are excluded from queries and workspace rebuilds.',
          )}
        </p>
      </header>

      <div className="flex-1 overflow-auto">
        {isLoading ? (
          <div className="p-4 space-y-2">
            <Skeleton className="h-10 w-full" />
            <Skeleton className="h-10 w-full" />
            <Skeleton className="h-10 w-full" />
          </div>
        ) : documents.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center px-8">
            <Archive className="h-12 w-12 text-muted-foreground/40 mb-3" />
            <h2 className="text-base font-semibold mb-1">
              {t('archive.empty.title', 'No archived documents')}
            </h2>
            <p className="text-sm text-muted-foreground max-w-md">
              {t(
                'archive.empty.body',
                'Archive a document from the three-dot menu on the documents page to keep its PDF, Markdown, algorithms, and references while dropping the rest.',
              )}
            </p>
          </div>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>{t('archive.col.title', 'Title')}</TableHead>
                <TableHead>{t('archive.col.source', 'Source')}</TableHead>
                <TableHead>
                  {t('archive.col.archivedAt', 'Archived')}
                </TableHead>
                <TableHead>
                  {t('archive.col.created', 'Created')}
                </TableHead>
                <TableHead className="w-12" />
              </TableRow>
            </TableHeader>
            <TableBody>
              {documents.map((doc) => (
                <TableRow key={doc.id}>
                  <TableCell className="font-medium">
                    <div className="flex flex-col gap-0.5">
                      <span className="truncate">
                        {doc.title ||
                          doc.file_name ||
                          `Document ${doc.id.slice(0, 8)}`}
                      </span>
                      <span className="text-xs text-muted-foreground truncate">
                        {doc.id}
                      </span>
                    </div>
                  </TableCell>
                  <TableCell>
                    <Badge variant="outline" className="text-xs">
                      {doc.source_type || 'text'}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground">
                    {formatDate(doc.archived_at)}
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground">
                    {formatDate(doc.created_at)}
                  </TableCell>
                  <TableCell>
                    <ArchiveActionsMenu
                      doc={doc}
                      onUnarchive={(id) => unarchiveMutation.mutate(id)}
                      onDelete={(id) => deleteMutation.mutate(id)}
                    />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </div>
    </div>
  );
}

export default ArchiveManager;

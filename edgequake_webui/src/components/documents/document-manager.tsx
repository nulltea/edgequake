/**
 * @module DocumentManager
 * @description Document ingestion and management interface.
 * Supports file upload, progress tracking, status monitoring, and batch operations.
 * 
 * @implements UC0001 - User uploads documents for ingestion
 * @implements UC0007 - User monitors document processing progress
 * @implements UC0008 - User reprocesses failed documents
 * @implements UC0009 - User deletes documents from knowledge graph
 * @implements FEAT0001 - Document ingestion with entity extraction
 * @implements FEAT0003 - Batch document processing
 * @implements FEAT0004 - Processing status tracking per document
 * @implements FEAT0602 - Real-time progress indicators
 * 
 * @enforces BR0302 - Failed documents can be reprocessed
 * @enforces BR0303 - Document deletion cascades to related entities
 * @enforces BR0305 - Cost tracking per document ingestion
 * 
 * @see {@link docs/use_cases.md} UC0001, UC0007-UC0009
 * @see {@link docs/features.md} FEAT0001, FEAT0003
 */
'use client';

import { useTenantStore } from '@/stores/use-tenant-store';
import type { Document } from '@/types';

import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useRouter } from 'next/navigation';
import { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import {
  getAlgorithmCounts,
  getCodeArtifactCounts,
  uploadPdfFromUrl,
} from '@/lib/api/edgequake';
import { toast as sonnerToast } from 'sonner';

import { useBulkSelection } from '@/hooks/use-bulk-selection';
import { useDocumentDropzone } from '@/hooks/use-document-dropzone';
import { useDocumentFiltering } from '@/hooks/use-document-filtering';
import { useDocumentHandlers } from '@/hooks/use-document-handlers';
import { useDocumentKeyboard } from '@/hooks/use-document-keyboard';
import { useDocumentMutations } from '@/hooks/use-document-mutations';
import { useDocumentPreferences } from '@/hooks/use-document-preferences';
import { useDocumentQueries } from '@/hooks/use-document-queries';
import { useDocumentTitle } from '@/hooks/use-document-title';
import { useDocumentWebSocket } from '@/hooks/use-document-websocket';
import { useFileUpload } from '@/hooks/use-file-upload';
import { useStuckDetection } from '@/hooks/use-stuck-detection';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { DocumentErrorAlert } from './document-error-alert';
import { DocumentHeader } from './document-header';
import { DocumentPreviewRightPanel } from './document-preview-right-panel';
import { DocumentTableSection } from './document-table-section';
import { DocumentToolbarSection } from './document-toolbar-section';
import { DuplicateUploadDialog } from './duplicate-upload-dialog';
import { SetLabelDialog } from './set-label-dialog';
import { isProcessingStatus } from './status-badge';

export function DocumentManager() {
  const { t } = useTranslation();
  const router = useRouter();

  // Get tenant context for query key
  const { selectedTenantId, selectedWorkspaceId } = useTenantStore();
  const queryClient = useQueryClient();

  // Selected document for preview panel
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);
  const [previewPanelOpen, setPreviewPanelOpen] = useState(false);

  // SPEC-002: Document viewer dialog state for PDF/Markdown side-by-side view
  const [viewerDialogOpen, setViewerDialogOpen] = useState(false);
  const [viewerPdfId, setViewerPdfId] = useState<string | null>(null);

  // Search state
  const [searchQuery, setSearchQuery] = useState('');
  const [pdfParserBackend, setPdfParserBackend] = useState<'default' | 'vision' | 'edgeparse' | 'vlmocr'>('default');
  // Chunks-only upload mode: skip heavy LLM extraction at upload time.
  const [skipExtraction, setSkipExtraction] = useState(false);

  // Pagination state
  const [currentPage, setCurrentPage] = useState(1);

  // OODA-17: Filter, sort, and pagination preferences with localStorage persistence
  const {
    pageSize, setPageSize,
    statusFilter, setStatusFilter,
    sortField, setSortField,
    sortDirection, setSortDirection,
  } = useDocumentPreferences();

  // Pipeline status dialog state
  const [pipelineDialogOpen, setPipelineDialogOpen] = useState(false);

  // Per-document algorithm counts: drives the row-level "Algorithms" button
  // visibility (only show `</>` when extracted algos exist for that doc).
  // One fetch per workspace, memoised into a Set for O(1) lookup.
  const { data: algoCountsData } = useQuery({
    queryKey: ['algorithm-counts', selectedTenantId, selectedWorkspaceId],
    queryFn: getAlgorithmCounts,
    enabled: !!selectedTenantId && !!selectedWorkspaceId,
    staleTime: 30_000,
  });
  const docsWithAlgorithms: Set<string> = useMemo(() => {
    const set = new Set<string>();
    for (const entry of algoCountsData?.counts ?? []) {
      if (entry.count > 0) set.add(entry.document_id);
    }
    return set;
  }, [algoCountsData]);

  // Per-document code-artifact counts — gates the "Reference code" button.
  const { data: codeCountsData } = useQuery({
    queryKey: ['code-artifact-counts', selectedTenantId, selectedWorkspaceId],
    queryFn: getCodeArtifactCounts,
    enabled: !!selectedTenantId && !!selectedWorkspaceId,
    staleTime: 30_000,
  });
  const docsWithCodeArtifacts: Set<string> = useMemo(() => {
    const set = new Set<string>();
    for (const entry of codeCountsData?.counts ?? []) {
      if (entry.count > 0) set.add(entry.document_id);
    }
    return set;
  }, [codeCountsData]);

  // OODA-13: Upload state extracted to useFileUpload hook
  const {
    uploadingFiles,
    isUploading,
    handleFilesUpload,
    removeUploadingFile,
    handleUploadComplete,
    handleUploadFailed,
    pendingDuplicates,
    resolvePendingDuplicates,
  } = useFileUpload({
    tenantId: selectedTenantId,
    workspaceId: selectedWorkspaceId,
    onUploadStart: () => setStatusFilter('all'),
    pdfParserBackend:
      pdfParserBackend === 'default' ? undefined : pdfParserBackend,
    skipExtraction,
  });

  // OODA-14: Document mutations extracted to useDocumentMutations hook
  const {
    deleteMutation,
    reprocessMutation,
    cancelMutation,
    archiveMutation,
  } = useDocumentMutations({
    onReprocessSuccess: () => setPipelineDialogOpen(true),
  });

  // Confirmation dialog for archive. WHY: archive is non-trivial (drops chunks,
  // embeddings, KG contributions, indexed code) so we want an explicit confirm.
  const [archiveTargetId, setArchiveTargetId] = useState<string | null>(null);

  // Dialog state for assigning / editing a document's free-form label.
  const [labelTargetDoc, setLabelTargetDoc] = useState<Document | null>(null);

  // Navigate to document's algorithms tab
  const handleViewAlgorithms = (doc: Document) => {
    router.push(`/documents/${doc.id}?tab=algorithms`);
  };

  // URL-initiated PDF upload. Server-side fetch + ingestion — the row
  // appears in the list when the normal polling picks it up, so there's
  // no per-file progress in the uploads strip. The toast here is the
  // only feedback; the post-OCR rename kicks in a few seconds later.
  const handleUrlUpload = async (url: string) => {
    setStatusFilter('all');
    try {
      const resp = await uploadPdfFromUrl(url, {
        skip_extraction: skipExtraction,
      });
      if (resp.status === 'duplicate') {
        sonnerToast.info('PDF already in workspace', {
          description: 'Existing document preserved.',
        });
      } else {
        sonnerToast.success('PDF queued', {
          description: 'Processing will finish in the background.',
        });
      }
      // Invalidate the documents list so the new row (or status change
      // on a reprocess) shows up immediately; all other list-related
      // queries already key off "documents" as the first segment.
      queryClient.invalidateQueries({
        predicate: (query) => query.queryKey[0] === 'documents',
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unknown error';
      sonnerToast.error('URL upload failed', { description: msg });
      throw err; // let the Dropzone's form see the failure
    }
  };

  // Navigate to document's References tab (repo detection + approval flow).
  // WHY not `?tab=code-matches`: the row-level button's gate unions
  // `document_repos` + `code_artifacts` so it surfaces docs that only
  // have a detected-but-unapproved repo. Those docs have nothing to show
  // on the Code Matches tab yet; the user needs to land on References to
  // approve the repo (and trigger analysis from there).
  const handleViewCodeArtifacts = (doc: Document) => {
    router.push(`/documents/${doc.id}?tab=repos`);
  };

  // OODA-29: Document queries extracted to useDocumentQueries hook.
  // `queryClient` is already declared above via useQueryClient(); the
  // hook returns the same instance, so we skip it here to avoid a
  // "defined multiple times" SWC/Turbopack build error.
  const { data, isLoading, isError, error, refetch, pipelineStatus } = useDocumentQueries({
    tenantId: selectedTenantId,
    workspaceId: selectedWorkspaceId,
    currentPage,
    pageSize,
    statusFilter,
  });

  // OODA-05: WebSocket subscription for real-time document status updates
  // WHY: Extracted to useDocumentWebSocket hook for SRP compliance
  useDocumentWebSocket(data?.items, queryClient);

  // Algorithm extraction handler — checks for existing algorithms and confirms re-extraction
  const handleExtractAlgorithms = async (documentId: string) => {
    try {
      const { extractAlgorithms, getAlgorithms, deleteAlgorithms } = await import('@/lib/api/edgequake');
      const { toast } = await import('sonner');

      // Check if document already has algorithms
      const existing = await getAlgorithms(documentId).catch(() => null);
      if (existing && existing.algorithms.length > 0) {
        const confirmed = window.confirm(
          `This document has ${existing.algorithms.length} extracted algorithm(s). ` +
          `Re-extracting will delete them and start fresh. Continue?`
        );
        if (!confirmed) return;
        await deleteAlgorithms(documentId);
      }

      await extractAlgorithms(documentId);
      toast.success('Algorithm extraction started');
      // Refetch documents list so the algo extraction status shows immediately
      refetch();
    } catch {
      const { toast } = await import('sonner');
      toast.error('Failed to start algorithm extraction');
    }
  };

  // Trigger the heavy LLM extraction stages for a single chunks-only document.
  const handleTriggerExtraction = async (documentId: string) => {
    try {
      const { triggerExtraction } = await import('@/lib/api/edgequake');
      const { toast } = await import('sonner');
      const res = await triggerExtraction(documentId);
      toast.success(`Extraction started (${res.queued.length} stage(s) queued)`);
      refetch();
    } catch {
      const { toast } = await import('sonner');
      toast.error('Failed to start extraction');
    }
  };

  // Reprocess a single document chunks-only: re-embed without the heavy LLM
  // stages. Keeps any existing entities/relationships untouched.
  const handleReprocessChunksOnly = async (documentId: string) => {
    try {
      const { reprocessDocumentChunksOnly } = await import('@/lib/api/edgequake');
      const { toast } = await import('sonner');
      await reprocessDocumentChunksOnly(documentId, true);
      toast.success('Reprocessing chunks (skip extraction)');
      refetch();
    } catch {
      const { toast } = await import('sonner');
      toast.error('Failed to start reprocess');
    }
  };

  // (Re-)run reference-implementation detection (repo search) for a document.
  const handleDetectRepos = async (documentId: string) => {
    try {
      const { detectRepos } = await import('@/lib/api/edgequake');
      const { toast } = await import('sonner');
      const res = await detectRepos(documentId);
      toast.success('Searching for reference implementations', {
        description: `Track ID: ${res.track_id}`,
      });
      refetch();
    } catch {
      const { toast } = await import('sonner');
      toast.error('Failed to start reference-implementation search');
    }
  };

  // Trigger extraction for every chunks-only document in the workspace.
  const handleExtractPending = async () => {
    if (!selectedWorkspaceId) return;
    try {
      const { extractPendingDocuments } = await import('@/lib/api/edgequake');
      const { toast } = await import('sonner');
      const res = await extractPendingDocuments(selectedWorkspaceId);
      if (res.documents_queued > 0) {
        toast.success(
          `Extraction started for ${res.documents_queued} document(s)`,
        );
      } else {
        toast.info('No documents are awaiting extraction');
      }
      refetch();
    } catch {
      const { toast } = await import('sonner');
      toast.error('Failed to start extraction for pending documents');
    }
  };

  // Count of documents awaiting extraction (chunks-only ingestion).
  const pendingExtractionCount = (data?.items || []).filter(
    (d) => d.extraction_skipped,
  ).length;

  // OODA-04: Detect stuck documents using extracted hook
  useStuckDetection(data?.items, {
    timeout: 30000,
    checkInterval: 30000,
  });

  // OODA-21: Document dropzone with file validation
  const { getRootProps, getInputProps, isDragActive, openFileDialog } = useDocumentDropzone({
    onFilesAccepted: handleFilesUpload,
    t,
  });

  // OODA-19: Filter and sort documents using extracted hook
  // OODA-20: Also compute status counts in hook
  const { documents, totalCount, totalPages, statusCounts } = useDocumentFiltering({
    documents: data?.items || [],
    searchQuery,
    statusFilter,
    sortField,
    sortDirection,
    pageSize,
    serverStatusCounts: data?.status_counts,
  });

  // OODA-16: Bulk selection extracted to useBulkSelection hook
  const {
    selectedIds,
    selectedCount,
    isAllSelected,
    handleSelectAll,
    handleSelectOne,
    handleClearSelection,
    handleBulkDelete,
    handleBulkReprocess,
  } = useBulkSelection({ documents });

  // OODA-28: Document handlers extracted to useDocumentHandlers hook
  const {
    handleDocumentClick,
    handleDocumentDoubleClick,
    handleViewDetails,
    handlePreviewClose,
    handleViewInGraph,
    handleViewPdf,
  } = useDocumentHandlers({
    setSelectedDocument,
    setPreviewPanelOpen,
    setViewerDialogOpen,
    setViewerPdfId,
  });

  /**
   * OODA-19: Keyboard shortcuts for power users
   * WHY: Keyboard shortcuts improve efficiency and accessibility
   * 
   * Shortcuts:
   * - Escape: Clear selection or close preview panel
   * - Ctrl/Cmd + A: Select all documents
   * - R: Refresh document list (when not in input)
   */
  // OODA-18: Document keyboard shortcuts (Escape, Ctrl+A, R)
  useDocumentKeyboard({
    previewPanelOpen,
    selectedCount,
    onPreviewClose: handlePreviewClose,
    onSelectAll: handleSelectAll,
    onClearSelection: handleClearSelection,
    onRefresh: refetch,
    t,
  });

  // OODA-22: Dynamic page title with document count
  // WHY: Use document-level processing count (not task count) so the title
  // reflects what users see in the table. Tasks can be "processing" while
  // their documents are already "failed" or "completed" (e.g., after restart).
  const processingDocCount = documents?.filter(
    (d: Document) => d.status && isProcessingStatus(d.status)
  ).length ?? 0;
  useDocumentTitle({
    totalCount,
    processingCount: processingDocCount,
  });

  if (isError) {
    return <DocumentErrorAlert error={error} onRetry={refetch} />;
  }

  return (
    <div className="flex h-full overflow-hidden">
      {/* Main Content - Flex column for proper scroll zones */}
      <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
        {/* Fixed Header Zone */}
        <div className="shrink-0 px-4 pt-4 space-y-3 bg-background">
          <DocumentHeader
            totalCount={totalCount}
            failedCount={statusCounts.failed + statusCounts.cancelled}
            pipelineIsBusy={!!pipelineStatus?.is_busy}
            pipelineDialogOpen={pipelineDialogOpen}
            onPipelineDialogChange={setPipelineDialogOpen}
            onRefresh={refetch}
            tenantId={selectedTenantId ?? undefined}
            workspaceId={selectedWorkspaceId ?? undefined}
          />

          {/* OODA-30: Toolbar section extracted to DocumentToolbarSection */}
          <DocumentToolbarSection
            searchQuery={searchQuery}
            onSearchChange={setSearchQuery}
            statusFilter={statusFilter}
            onStatusFilterChange={setStatusFilter}
            sortField={sortField}
            onSortFieldChange={setSortField}
            sortDirection={sortDirection}
            onSortDirectionChange={setSortDirection}
            statusCounts={statusCounts}
            pipelineStatus={pipelineStatus}
            documents={documents}
            onOpenPipelineDetails={() => setPipelineDialogOpen(true)}
            getRootProps={getRootProps}
            getInputProps={getInputProps}
            isDragActive={isDragActive}
            openFileDialog={openFileDialog}
            pdfParserBackend={pdfParserBackend}
            onPdfParserBackendChange={setPdfParserBackend}
            skipExtraction={skipExtraction}
            onSkipExtractionChange={setSkipExtraction}
            onExtractPending={handleExtractPending}
            pendingExtractionCount={pendingExtractionCount}
            onUrlSubmit={handleUrlUpload}
            selectedCount={selectedCount}
            onBulkReprocess={handleBulkReprocess}
            onBulkDelete={handleBulkDelete}
            onClearSelection={handleClearSelection}
            uploadingFiles={uploadingFiles}
            isUploading={isUploading}
            onRemoveUpload={removeUploadingFile}
            onUploadComplete={handleUploadComplete}
            onUploadFailed={handleUploadFailed}
          />

        </div>

      {/* OODA-26: Table section extracted to DocumentTableSection */}
      <DocumentTableSection
        documents={documents}
        totalCount={totalCount}
        isLoading={isLoading}
        selectedIds={selectedIds}
        selectedDocument={selectedDocument}
        searchQuery={searchQuery}
        statusFilter={statusFilter}
        isAllSelected={isAllSelected}
        onSelectAll={handleSelectAll}
        onSelectOne={handleSelectOne}
        onRowClick={handleDocumentClick}
        onRowDoubleClick={handleDocumentDoubleClick}
        onViewDetails={handleViewDetails}
        onViewInGraph={handleViewInGraph}
        onViewPdf={handleViewPdf}
        onRetry={(id) => reprocessMutation.mutate(id)}
        onCancel={(trackId) => cancelMutation.mutate(trackId)}
        onDelete={(id) => deleteMutation.mutate(id)}
        onArchive={(id) => setArchiveTargetId(id)}
        onExtractAlgorithms={handleExtractAlgorithms}
        onTriggerExtraction={handleTriggerExtraction}
        onReprocessChunksOnly={handleReprocessChunksOnly}
        onDetectRepos={handleDetectRepos}
        onSetLabel={(doc) => setLabelTargetDoc(doc)}
        onViewAlgorithms={handleViewAlgorithms}
        docsWithAlgorithms={docsWithAlgorithms}
        onViewCodeArtifacts={handleViewCodeArtifacts}
        docsWithCodeArtifacts={docsWithCodeArtifacts}
        isRetrying={reprocessMutation.isPending}
        isCancelling={cancelMutation.isPending}
        onUploadClick={openFileDialog}
        currentPage={currentPage}
        totalPages={totalPages}
        pageSize={pageSize}
        onPageChange={setCurrentPage}
        onPageSizeChange={setPageSize}
        onClearFilter={() => {
          setStatusFilter('all');
          setSearchQuery('');
        }}
      />
      </div>

      {/* OODA-27: Right panel extracted to DocumentPreviewRightPanel */}
      <DocumentPreviewRightPanel
        isOpen={previewPanelOpen}
        onToggle={() => setPreviewPanelOpen(!previewPanelOpen)}
        onClose={handlePreviewClose}
        selectedDocument={selectedDocument}
        onDelete={(id) => deleteMutation.mutate(id)}
        onReprocess={(id) => reprocessMutation.mutate(id)}
        onViewInGraph={handleViewInGraph}
        onViewFull={(doc) => router.push(`/documents/${doc.id}`)}
        isDeleting={deleteMutation.isPending}
        isReprocessing={reprocessMutation.isPending}
        viewerDialogOpen={viewerDialogOpen}
        onViewerDialogChange={setViewerDialogOpen}
        viewerPdfId={viewerPdfId}
      />

      {/* Duplicate upload dialog — shown when backend returns duplicate_of */}
      <DuplicateUploadDialog
        open={pendingDuplicates.length > 0}
        duplicates={pendingDuplicates}
        onResolve={resolvePendingDuplicates}
      />

      {/* Archive confirmation — archive removes derived data but keeps the
          PDF, Markdown, algorithms, and references. */}
      <AlertDialog
        open={archiveTargetId !== null}
        onOpenChange={(open) => !open && setArchiveTargetId(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {t('documents.archive.title', 'Archive this document?')}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {t(
                'documents.archive.confirm',
                'Archiving removes the chunks, embeddings, knowledge graph contributions, and indexed code derived from this document. The original PDF, converted Markdown, extracted algorithms, and references will be kept. Archived documents are excluded from queries and workspace rebuilds. You can find them on the Archive page.',
              )}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>
              {t('common.cancel', 'Cancel')}
            </AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                if (archiveTargetId) {
                  archiveMutation.mutate(archiveTargetId);
                  setArchiveTargetId(null);
                }
              }}
            >
              {t('documents.actions.archive', 'Archive')}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Label assign / edit dialog. Free text, max 80 chars, empty clears. */}
      <SetLabelDialog
        doc={labelTargetDoc}
        open={labelTargetDoc !== null}
        onClose={() => setLabelTargetDoc(null)}
      />
    </div>
  );
}

export default DocumentManager;

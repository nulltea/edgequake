'use client';

import { Button } from '@/components/ui/button';
import type { StatusCounts } from '@/hooks/use-document-filtering';
import type { Document, PipelineStatus } from '@/types';
import { Zap } from 'lucide-react';
import { BatchActionsBar } from './batch-actions-bar';
import { DocumentDropzone, type DocumentDropzoneProps } from './document-dropzone';
import type { DocStatus, SortField } from './document-filters';
import { DocumentFilters } from './document-filters';
import { DocumentSearchBar } from './document-search-bar';
import { ProcessingStatusSummary } from './processing-status-summary';
import type { UploadingFile } from './types';
import { UploadProgressList } from './upload-progress-list';

/**
 * OODA-30: Document toolbar section component
 * 
 * WHY: Single Responsibility Principle - isolate toolbar UI from main component.
 * Contains search, filters, status summary, dropzone, batch actions, and upload progress.
 */

export interface DocumentToolbarSectionProps {
  // Search
  searchQuery: string;
  onSearchChange: (value: string) => void;
  
  // Filters
  statusFilter: DocStatus;
  onStatusFilterChange: (value: DocStatus) => void;
  sortField: SortField;
  onSortFieldChange: (value: SortField) => void;
  sortDirection: 'asc' | 'desc';
  onSortDirectionChange: (value: 'asc' | 'desc') => void;
  statusCounts: StatusCounts;
  
  // Pipeline status
  pipelineStatus: PipelineStatus | undefined;
  documents: Document[];
  onOpenPipelineDetails: () => void;
  
  // Dropzone
  getRootProps: DocumentDropzoneProps['getRootProps'];
  getInputProps: DocumentDropzoneProps['getInputProps'];
  isDragActive: boolean;
  openFileDialog: () => void;
  pdfParserBackend: 'default' | 'vision' | 'edgeparse' | 'vlmocr';
  onPdfParserBackendChange: (value: 'default' | 'vision' | 'edgeparse' | 'vlmocr') => void;
  /** Whether uploads skip heavy LLM extraction (chunks-only indexing). */
  skipExtraction: boolean;
  onSkipExtractionChange: (value: boolean) => void;
  /** Trigger extraction for all chunks-only documents in the workspace. */
  onExtractPending: () => void;
  /** Number of documents awaiting extraction (drives the Extract pending button). */
  pendingExtractionCount: number;
  /** URL-upload handler exposed on the Dropzone (shows the URL input row). */
  onUrlSubmit?: (url: string) => Promise<void>;

  // Bulk actions
  selectedCount: number;
  onBulkReprocess: () => void;
  onBulkDelete: () => void;
  onClearSelection: () => void;
  
  // Upload progress
  uploadingFiles: UploadingFile[];
  isUploading: boolean;
  onRemoveUpload: (index: number) => void;
  onUploadComplete: (index: number) => void;
  onUploadFailed: (index: number, error: string) => void;
}

export function DocumentToolbarSection({
  searchQuery,
  onSearchChange,
  statusFilter,
  onStatusFilterChange,
  sortField,
  onSortFieldChange,
  sortDirection,
  onSortDirectionChange,
  statusCounts,
  pipelineStatus,
  documents,
  onOpenPipelineDetails,
  getRootProps,
  getInputProps,
  isDragActive,
  openFileDialog,
  pdfParserBackend,
  onPdfParserBackendChange,
  skipExtraction,
  onSkipExtractionChange,
  onExtractPending,
  pendingExtractionCount,
  onUrlSubmit,
  selectedCount,
  onBulkReprocess,
  onBulkDelete,
  onClearSelection,
  uploadingFiles,
  isUploading,
  onRemoveUpload,
  onUploadComplete,
  onUploadFailed,
}: DocumentToolbarSectionProps) {
  return (
    <>
      {/* Search and Filters */}
      <div className="flex flex-col sm:flex-row sm:items-center gap-3 pb-3 border-b">
        <DocumentSearchBar
          value={searchQuery}
          onChange={onSearchChange}
        />
        <DocumentFilters
          status={statusFilter}
          onStatusChange={onStatusFilterChange}
          sortField={sortField}
          onSortFieldChange={onSortFieldChange}
          sortDirection={sortDirection}
          onSortDirectionChange={onSortDirectionChange}
          statusCounts={statusCounts}
        />
      </div>

      {/* Processing Status Summary */}
      {pipelineStatus && (
        <ProcessingStatusSummary
          pipelineStatus={pipelineStatus}
          documents={documents}
          onOpenDetails={onOpenPipelineDetails}
        />
      )}

      {/* Compact Upload Zone */}
      <DocumentDropzone
        getRootProps={getRootProps}
        getInputProps={getInputProps}
        isDragActive={isDragActive}
        openFileDialog={openFileDialog}
        pdfParserBackend={pdfParserBackend}
        onPdfParserBackendChange={onPdfParserBackendChange}
        skipExtraction={skipExtraction}
        onSkipExtractionChange={onSkipExtractionChange}
        onUrlSubmit={onUrlSubmit}
      />

      {/* Workspace-wide trigger for documents ingested in chunks-only mode. */}
      {pendingExtractionCount > 0 && (
        <div className="flex items-center justify-between gap-3 px-3 py-2 rounded-lg border border-amber-500/30 bg-amber-500/5">
          <span className="text-sm text-muted-foreground">
            {pendingExtractionCount} document
            {pendingExtractionCount === 1 ? '' : 's'} indexed for chunk search
            only — extraction not yet run.
          </span>
          <Button size="sm" variant="outline" onClick={onExtractPending}>
            <Zap className="h-4 w-4 mr-2" />
            Extract pending
          </Button>
        </div>
      )}

      {/* Bulk Actions Bar */}
      <BatchActionsBar
        selectedCount={selectedCount}
        onReprocess={onBulkReprocess}
        onDelete={onBulkDelete}
        onClear={onClearSelection}
      />

      {/* Upload Progress */}
      <UploadProgressList
        uploadingFiles={uploadingFiles}
        isUploading={isUploading}
        onRemove={onRemoveUpload}
        onComplete={onUploadComplete}
        onFailed={onUploadFailed}
      />
    </>
  );
}

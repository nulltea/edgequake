'use client';

import { Button } from '@/components/ui/button';
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { getAlgorithms } from '@/lib/api/edgequake';
import type { Document } from '@/types';
import { useQuery } from '@tanstack/react-query';
import { Archive, Brain, Copy, Eye, MoreVertical, RefreshCcw, RefreshCw, StopCircle, Trash2, Zap } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { toast } from 'sonner';
import { ResetDocumentStatusButton } from './reset-document-status-button';

/**
 * Props for the DocumentActionsMenu component.
 */
interface DocumentActionsMenuProps {
  /** The document this menu acts on */
  doc: Document;
  /** Callback to view PDF document */
  onViewPdf: (doc: Document) => void;
  /** Callback to cancel document processing */
  onCancel: (trackId: string) => void;
  /** Callback to reprocess document */
  onReprocess: (id: string) => void;
  /** Callback to delete document */
  onDelete: (id: string) => void;
  /** Callback to archive document (keeps PDF/MD/algorithms/refs, drops derived data) */
  onArchive: (id: string) => void;
  /** Callback to extract algorithms from document */
  onExtractAlgorithms?: (id: string) => void;
  /** Callback to run all heavy LLM extraction stages for a chunks-only document */
  onTriggerExtraction?: (id: string) => void;
  /** Whether a cancel operation is in progress */
  isCancelling?: boolean;
}

/** Processing status values that allow cancellation */
const CANCELLABLE_STATUSES = ['pending', 'processing'];
/** Processing stages that allow cancellation */
const CANCELLABLE_STAGES = [
  'converting', 'uploading', 'preprocessing', 'chunking',
  'extracting', 'gleaning', 'merging', 'summarizing', 'embedding', 'storing'
];

/**
 * Dropdown menu with document actions.
 *
 * WHY: Extracted from DocumentManager for SRP compliance (OODA-09).
 * This component handles the actions dropdown for each document row.
 *
 * @implements FEAT0001 - Document ingestion with entity extraction
 */
export function DocumentActionsMenu({
  doc,
  onViewPdf,
  onCancel,
  onReprocess,
  onDelete,
  onArchive,
  onExtractAlgorithms,
  onTriggerExtraction,
  isCancelling = false,
}: DocumentActionsMenuProps) {
  const { t } = useTranslation();

  const isCompleted = doc.status === 'completed' || doc.status === 'indexed';

  // Check if document already has algorithms (cached, only for completed docs)
  const { data: algoData } = useQuery({
    queryKey: ['algorithms', doc.id],
    queryFn: () => getAlgorithms(doc.id),
    enabled: isCompleted && !!onExtractAlgorithms,
    staleTime: 60 * 1000,
  });
  const hasAlgorithms = (algoData?.algorithms?.length ?? 0) > 0;

  const handleCopyId = () => {
    navigator.clipboard.writeText(doc.id);
    toast.success(t('documents.actions.idCopied', 'Document ID copied'));
  };

  const canCancel =
    ((CANCELLABLE_STATUSES.includes(doc.status || '')) ||
    (CANCELLABLE_STAGES.includes(doc.current_stage || ''))) &&
    doc.track_id;

  const showViewPdf = doc.source_type === 'pdf' || doc.pdf_id;
  const showExtractAlgorithms = onExtractAlgorithms && isCompleted;
  // Chunks-only documents (uploaded with "Skip extraction") expose a
  // "Run extraction" action that kicks off all heavy LLM stages.
  const showTriggerExtraction = onTriggerExtraction && doc.extraction_skipped;
  // WHY: Cancelled documents should also show the reset/reprocess option
  const showReset = doc.status === 'failed' || doc.status === 'partial_failure' || doc.status === 'cancelled';

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="icon" className="h-8 w-8" aria-label="More actions">
          <MoreVertical className="h-4 w-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        {/* OODA-31: Copy document ID */}
        <DropdownMenuItem onClick={handleCopyId}>
          <Copy className="h-4 w-4 mr-2" />
          {t('documents.actions.copyId', 'Copy ID')}
        </DropdownMenuItem>

        {/* SPEC-002: View PDF/Markdown for PDF documents */}
        {showViewPdf && (
          <DropdownMenuItem onClick={() => onViewPdf(doc)}>
            <Eye className="h-4 w-4 mr-2" />
            {t('documents.actions.viewPdf', 'View PDF')}
          </DropdownMenuItem>
        )}

        {/* Run extraction for a chunks-only document */}
        {showTriggerExtraction && (
          <DropdownMenuItem onClick={() => onTriggerExtraction!(doc.id)}>
            <Zap className="h-4 w-4 mr-2" />
            {t('documents.actions.runExtraction', 'Run extraction')}
          </DropdownMenuItem>
        )}

        {/* Extract / Re-extract Algorithms */}
        {showExtractAlgorithms && (
          <DropdownMenuItem onClick={() => onExtractAlgorithms!(doc.id)}>
            {hasAlgorithms ? (
              <>
                <RefreshCcw className="h-4 w-4 mr-2" />
                Re-extract Algorithms
              </>
            ) : (
              <>
                <Brain className="h-4 w-4 mr-2" />
                Extract Algorithms
              </>
            )}
          </DropdownMenuItem>
        )}

        {/* Reset status option for failed documents */}
        {showReset && (
          <DropdownMenuItem asChild>
            <div className="p-0">
              <ResetDocumentStatusButton document={doc} iconOnly={false} size="sm" />
            </div>
          </DropdownMenuItem>
        )}

        {/* Cancel option for processing documents */}
        {canCancel && (
          <DropdownMenuItem
            onClick={() => onCancel(doc.track_id!)}
            className="text-orange-600"
            disabled={isCancelling}
          >
            <StopCircle className="h-4 w-4 mr-2" />
            {t('documents.actions.cancel', 'Cancel Extraction')}
          </DropdownMenuItem>
        )}

        {/* Reprocess */}
        <DropdownMenuItem onClick={() => onReprocess(doc.id)}>
          <RefreshCw className="h-4 w-4 mr-2" />
          {t('documents.actions.reprocess')}
        </DropdownMenuItem>

        {/* Archive — soft-delete that keeps PDF/MD/algorithms/references */}
        <DropdownMenuItem onClick={() => onArchive(doc.id)}>
          <Archive className="h-4 w-4 mr-2" />
          {t('documents.actions.archive', 'Archive')}
        </DropdownMenuItem>

        {/* Delete */}
        <DropdownMenuItem
          onClick={() => onDelete(doc.id)}
          className="text-destructive"
        >
          <Trash2 className="h-4 w-4 mr-2" />
          {t('documents.actions.delete')}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

'use client';

import { Button } from '@/components/ui/button';
import { deleteAlgorithms, extractAlgorithms } from '@/lib/api/edgequake';
import { useAlgorithmStore } from '@/stores/use-algorithm-store';
import { Brain, Loader2, RefreshCcw } from 'lucide-react';
import { useCallback } from 'react';

interface AlgorithmExtractionButtonProps {
  documentId: string;
  /** Number of existing algorithms (if > 0, shows re-extract with confirmation) */
  existingCount?: number;
  onExtracted?: () => void;
}

export function AlgorithmExtractionButton({
  documentId,
  existingCount = 0,
  onExtracted,
}: AlgorithmExtractionButtonProps) {
  const { startExtraction, finishExtraction, isExtracting } = useAlgorithmStore();
  const extracting = isExtracting(documentId);
  const isReextract = existingCount > 0;

  const handleExtract = useCallback(async () => {
    if (extracting) return;

    if (isReextract) {
      const confirmed = window.confirm(
        `This document has ${existingCount} extracted algorithm(s). ` +
        `Re-extracting will delete them and start fresh. Continue?`
      );
      if (!confirmed) return;
    }

    startExtraction(documentId);
    try {
      if (isReextract) {
        await deleteAlgorithms(documentId);
      }
      await extractAlgorithms(documentId);
      onExtracted?.();
    } catch (err) {
      console.error('[AlgorithmExtractionButton] Extraction failed:', err);
    } finally {
      finishExtraction(documentId);
    }
  }, [documentId, extracting, isReextract, existingCount, startExtraction, finishExtraction, onExtracted]);

  return (
    <Button
      variant="outline"
      size="sm"
      disabled={extracting}
      onClick={handleExtract}
    >
      {extracting ? (
        <Loader2 className="h-4 w-4 animate-spin" />
      ) : isReextract ? (
        <RefreshCcw className="h-4 w-4" />
      ) : (
        <Brain className="h-4 w-4" />
      )}
      {extracting ? 'Extracting...' : isReextract ? 'Re-extract' : 'Extract'}
    </Button>
  );
}

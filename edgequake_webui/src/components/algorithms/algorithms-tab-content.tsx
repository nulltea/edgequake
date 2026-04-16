'use client';

import { getAlgorithms } from '@/lib/api/edgequake';
import { useAlgorithmStore } from '@/stores/use-algorithm-store';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { CodeXml, Loader2 } from 'lucide-react';
import { useCallback } from 'react';
import { AlgorithmExtractionButton } from './algorithm-extraction-button';
import { AlgorithmList } from './algorithm-list';

interface AlgorithmsTabContentProps {
  documentId: string;
}

export function AlgorithmsTabContent({ documentId }: AlgorithmsTabContentProps) {
  const queryClient = useQueryClient();
  const isExtracting = useAlgorithmStore((s) => s.isExtracting(documentId));

  const { data, isLoading } = useQuery({
    queryKey: ['algorithms', documentId],
    queryFn: () => getAlgorithms(documentId),
    enabled: !!documentId,
    staleTime: 30 * 1000,
  });

  const handleExtracted = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ['algorithms', documentId] });
  }, [queryClient, documentId]);

  const hasAlgorithms = (data?.algorithms?.length ?? 0) > 0;

  // Extracting state
  if (isExtracting) {
    return (
      <div className="flex flex-col items-center justify-center py-16 px-4">
        <div className="rounded-full bg-primary/10 p-4 mb-4">
          <Loader2 className="h-8 w-8 text-primary animate-spin" />
        </div>
        <h3 className="text-lg font-semibold mb-1">Extraction in Progress</h3>
        <p className="text-sm text-muted-foreground text-center max-w-sm">
          Algorithms are being extracted from this document. This may take a
          moment depending on document length and complexity.
        </p>
      </div>
    );
  }

  // Loading initial data
  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-16">
        <Loader2 className="h-6 w-6 text-muted-foreground animate-spin" />
      </div>
    );
  }

  // Empty state -- no algorithms yet
  if (!hasAlgorithms) {
    return (
      <div className="flex flex-col items-center justify-center py-16 px-4">
        <div className="rounded-full bg-muted p-4 mb-4">
          <CodeXml className="h-8 w-8 text-muted-foreground" />
        </div>
        <h3 className="text-lg font-semibold mb-1">No Algorithms Found</h3>
        <p className="text-sm text-muted-foreground text-center max-w-sm mb-4">
          Extract algorithms, procedures, and computational methods described in
          this document using AI-powered analysis.
        </p>
        <AlgorithmExtractionButton
          documentId={documentId}
          onExtracted={handleExtracted}
        />
      </div>
    );
  }

  // Algorithms exist
  return (
    <div className="p-4 space-y-4 overflow-auto h-full">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-muted-foreground">
          {data!.total} algorithm{data!.total !== 1 ? 's' : ''} extracted
        </h2>
        <AlgorithmExtractionButton
          documentId={documentId}
          existingCount={data!.total}
          onExtracted={handleExtracted}
        />
      </div>
      <AlgorithmList documentId={documentId} />
    </div>
  );
}

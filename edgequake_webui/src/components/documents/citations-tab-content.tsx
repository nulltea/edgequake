'use client';

import { getDocumentReferences } from '@/lib/api/edgequake';
import { useQuery } from '@tanstack/react-query';
import { Link2, Loader2, Quote } from 'lucide-react';

interface CitationsTabContentProps {
  documentId: string;
}

/**
 * Read-only list of references (citations) parsed from a document's reference
 * section during ingestion. Ordered by reference number; DOI/URL render as
 * links. References are deterministic parser output — there is no review step.
 */
export function CitationsTabContent({ documentId }: CitationsTabContentProps) {
  const { data, isLoading } = useQuery({
    queryKey: ['references', documentId],
    queryFn: () => getDocumentReferences(documentId),
    enabled: !!documentId,
    staleTime: 30 * 1000,
  });

  const references = data?.references ?? [];

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-16">
        <Loader2 className="h-6 w-6 text-muted-foreground animate-spin" />
      </div>
    );
  }

  if (references.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 px-4">
        <div className="rounded-full bg-muted p-4 mb-4">
          <Quote className="h-8 w-8 text-muted-foreground" />
        </div>
        <h3 className="text-lg font-semibold mb-1">No References Found</h3>
        <p className="text-sm text-muted-foreground text-center max-w-sm">
          No numbered reference section was detected in this document. References
          are parsed automatically when a document is ingested.
        </p>
      </div>
    );
  }

  return (
    <div className="p-3">
      <ol className="divide-y rounded-lg border">
        {references.map((ref) => (
          <li key={ref.reference_number} className="flex gap-3 p-3 text-sm">
            <span className="shrink-0 font-mono text-xs text-muted-foreground mt-0.5">
              [{ref.reference_number}]
            </span>
            <div className="min-w-0 flex-1 space-y-1">
              <p className="break-words">{ref.raw_text}</p>
              {(ref.doi || ref.url) && (
                <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs">
                  {ref.doi && (
                    <a
                      href={`https://doi.org/${ref.doi}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 text-primary hover:underline"
                    >
                      <Quote className="h-3 w-3" />
                      {ref.doi}
                    </a>
                  )}
                  {ref.url && (
                    <a
                      href={ref.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 text-primary hover:underline truncate max-w-full"
                    >
                      <Link2 className="h-3 w-3 shrink-0" />
                      <span className="truncate">{ref.url}</span>
                    </a>
                  )}
                </div>
              )}
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}

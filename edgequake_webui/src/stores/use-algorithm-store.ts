/**
 * Algorithm Extraction Store
 *
 * Lightweight Zustand store for tracking which documents currently have
 * algorithm extraction in progress. Actual data fetching is handled by
 * TanStack Query in the components.
 */

import { create } from "zustand";
import { devtools } from "zustand/middleware";

interface AlgorithmState {
  extractingDocuments: Set<string>;
}

interface AlgorithmActions {
  startExtraction: (documentId: string) => void;
  finishExtraction: (documentId: string) => void;
  isExtracting: (documentId: string) => boolean;
}

type AlgorithmStore = AlgorithmState & AlgorithmActions;

export const useAlgorithmStore = create<AlgorithmStore>()(
  devtools(
    (set, get) => ({
      extractingDocuments: new Set(),

      startExtraction: (documentId: string) => {
        set((state) => {
          const next = new Set(state.extractingDocuments);
          next.add(documentId);
          return { extractingDocuments: next };
        });
      },

      finishExtraction: (documentId: string) => {
        set((state) => {
          const next = new Set(state.extractingDocuments);
          next.delete(documentId);
          return { extractingDocuments: next };
        });
      },

      isExtracting: (documentId: string) => {
        return get().extractingDocuments.has(documentId);
      },
    }),
    { name: "algorithm-store" },
  ),
);

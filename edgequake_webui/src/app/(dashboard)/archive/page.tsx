/**
 * @module ArchivePage
 * @description Lists workspace documents that have been soft-archived.
 *
 * Archived documents keep their PDF, Markdown, extracted algorithms, and
 * references; chunks, embeddings, KG contributions, and indexed code have
 * been removed. They are excluded from queries and workspace rebuilds.
 */
import { ArchiveManager } from '@/components/archive/archive-manager';

export default function ArchivePage() {
  return <ArchiveManager />;
}

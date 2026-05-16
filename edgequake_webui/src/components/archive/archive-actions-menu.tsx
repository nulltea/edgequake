'use client';

import { Button } from '@/components/ui/button';
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuSeparator,
    DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { getPdfDownloadUrl } from '@/lib/api/edgequake';
import type { Document } from '@/types';
import {
    ArchiveRestore,
    Brain,
    Eye,
    FileText,
    GitBranch,
    MoreVertical,
    Trash2,
} from 'lucide-react';
import { useRouter } from 'next/navigation';
import { useTranslation } from 'react-i18next';

interface ArchiveActionsMenuProps {
  doc: Document;
  onUnarchive: (id: string) => void;
  onDelete: (id: string) => void;
}

/**
 * Per-row action menu on the Archive page. View-only for the kept assets
 * (PDF, Markdown, algorithms, references) plus Unarchive and permanent
 * Delete. No reprocess / upload / extract — archived docs are inert until
 * unarchived.
 */
export function ArchiveActionsMenu({
  doc,
  onUnarchive,
  onDelete,
}: ArchiveActionsMenuProps) {
  const { t } = useTranslation();
  const router = useRouter();

  // pdf_id falls back to the document id for legacy PDF docs where
  // pdf_id wasn't populated separately.
  const pdfIdForViewer =
    doc.pdf_id || (doc.source_type === 'pdf' ? doc.id : null);

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8"
          aria-label="More actions"
        >
          <MoreVertical className="h-4 w-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        {pdfIdForViewer && (
          <DropdownMenuItem asChild>
            <a
              href={getPdfDownloadUrl(pdfIdForViewer)}
              target="_blank"
              rel="noopener noreferrer"
            >
              <Eye className="h-4 w-4 mr-2" />
              {t('archive.actions.viewPdf', 'View PDF')}
            </a>
          </DropdownMenuItem>
        )}

        <DropdownMenuItem
          onClick={() => router.push(`/documents/${doc.id}?tab=content`)}
        >
          <FileText className="h-4 w-4 mr-2" />
          {t('archive.actions.viewMarkdown', 'View Markdown')}
        </DropdownMenuItem>

        <DropdownMenuItem
          onClick={() => router.push(`/documents/${doc.id}?tab=algorithms`)}
        >
          <Brain className="h-4 w-4 mr-2" />
          {t('archive.actions.viewAlgorithms', 'View Algorithms')}
        </DropdownMenuItem>

        <DropdownMenuItem
          onClick={() => router.push(`/documents/${doc.id}?tab=repos`)}
        >
          <GitBranch className="h-4 w-4 mr-2" />
          {t('archive.actions.viewReferences', 'View References')}
        </DropdownMenuItem>

        <DropdownMenuSeparator />

        <DropdownMenuItem onClick={() => onUnarchive(doc.id)}>
          <ArchiveRestore className="h-4 w-4 mr-2" />
          {t('archive.actions.unarchive', 'Unarchive')}
        </DropdownMenuItem>

        <DropdownMenuItem
          onClick={() => onDelete(doc.id)}
          className="text-destructive"
        >
          <Trash2 className="h-4 w-4 mr-2" />
          {t('archive.actions.delete', 'Delete permanently')}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export default ArchiveActionsMenu;

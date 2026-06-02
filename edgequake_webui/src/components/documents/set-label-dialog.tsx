'use client';

import { Button } from '@/components/ui/button';
import {
    Dialog,
    DialogContent,
    DialogFooter,
    DialogHeader,
    DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { setDocumentLabel } from '@/lib/api/edgequake';
import type { Document } from '@/types';
import { useQueryClient } from '@tanstack/react-query';
import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { toast } from 'sonner';

const LABEL_MAX_LEN = 80;

interface SetLabelDialogProps {
  doc: Document | null;
  open: boolean;
  onClose: () => void;
}

/**
 * Dialog for assigning / editing / clearing a document's free-form label.
 *
 * Saves via `PUT /documents/{id}/label`, then invalidates the documents
 * query so the title cell re-renders with the new "(Label)" suffix.
 */
export function SetLabelDialog({ doc, open, onClose }: SetLabelDialogProps) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [value, setValue] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Reset the input when a new doc is opened.
  useEffect(() => {
    if (open) {
      setValue(doc?.label ?? '');
    }
  }, [open, doc]);

  if (!doc) return null;

  const trimmed = value.trim();
  const isClearing = trimmed.length === 0;
  const isUnchanged = trimmed === (doc.label ?? '').trim();
  const isTooLong = trimmed.length > LABEL_MAX_LEN;

  const handleSubmit = async () => {
    if (isTooLong) {
      toast.error(
        t(
          'documents.label.tooLong',
          `Label must be at most ${LABEL_MAX_LEN} characters`,
        ),
      );
      return;
    }
    setIsSubmitting(true);
    try {
      await setDocumentLabel(doc.id, isClearing ? null : trimmed);
      toast.success(
        isClearing
          ? t('documents.label.cleared', 'Label cleared')
          : t('documents.label.saved', 'Label saved'),
      );
      // Refresh the documents list so the "(Label)" suffix updates.
      queryClient.invalidateQueries({
        predicate: (q) => q.queryKey[0] === 'documents',
      });
      onClose();
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unknown error';
      toast.error(t('documents.label.failed', 'Failed to update label'), {
        description: msg,
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>
            {doc.label
              ? t('documents.label.editTitle', 'Edit document label')
              : t('documents.label.setTitle', 'Set document label')}
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-3 py-2">
          <p className="text-sm text-muted-foreground">
            {t(
              'documents.label.help',
              'Short tag shown next to the title (e.g. "Nexus", "AloePri"). Leave blank to clear.',
            )}
          </p>
          <div className="space-y-1.5">
            <Label htmlFor="document-label">
              {t('documents.label.field', 'Label')}
            </Label>
            <Input
              id="document-label"
              value={value}
              onChange={(e) => setValue(e.target.value)}
              maxLength={LABEL_MAX_LEN + 1}
              placeholder={t('documents.label.placeholder', 'e.g. Nexus')}
              autoFocus
              onKeyDown={(e) => {
                if (
                  e.key === 'Enter' &&
                  !e.shiftKey &&
                  !isSubmitting &&
                  !isUnchanged &&
                  !isTooLong
                ) {
                  e.preventDefault();
                  handleSubmit();
                }
              }}
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>
                {isTooLong
                  ? t(
                      'documents.label.tooLong',
                      `Label must be at most ${LABEL_MAX_LEN} characters`,
                    )
                  : ''}
              </span>
              <span>
                {trimmed.length}/{LABEL_MAX_LEN}
              </span>
            </div>
          </div>
        </div>

        <DialogFooter>
          <Button variant="ghost" onClick={onClose} disabled={isSubmitting}>
            {t('common.cancel', 'Cancel')}
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={isSubmitting || isUnchanged || isTooLong}
          >
            {isClearing
              ? t('documents.label.clear', 'Clear label')
              : t('common.save', 'Save')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

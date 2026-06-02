'use client';

import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { cn } from '@/lib/utils';
import { Link as LinkIcon, Loader2, Upload } from 'lucide-react';
import type React from 'react';
import { useState } from 'react';
import type { DropzoneInputProps, DropzoneRootProps } from 'react-dropzone';
import { useTranslation } from 'react-i18next';

/**
 * Props for the DocumentDropzone component.
 */
export interface DocumentDropzoneProps {
  /** Props to spread on the dropzone container */
  getRootProps: <T extends DropzoneRootProps>(props?: T) => T;
  /** Props to spread on the hidden file input */
  getInputProps: <T extends DropzoneInputProps>(props?: T) => T;
  /** Whether a drag operation is currently active over the zone */
  isDragActive: boolean;
  /** Function to programmatically open file dialog (explicit click handler) */
  openFileDialog: () => void;
  /** Per-upload PDF parser backend override. */
  pdfParserBackend: 'default' | 'vision' | 'edgeparse' | 'vlmocr';
  /** Change handler for the PDF parser override selector. */
  onPdfParserBackendChange: (
    value: 'default' | 'vision' | 'edgeparse' | 'vlmocr',
  ) => void;
  /** When true, uploads skip heavy LLM extraction (chunks-only indexing). */
  skipExtraction: boolean;
  /** Change handler for the skip-extraction checkbox. */
  onSkipExtractionChange: (value: boolean) => void;
  /**
   * Handler for URL-based PDF uploads. When provided, the dropzone
   * surface shows an additional URL input row. Returning a promise lets
   * the dropzone show a spinner while the server-side fetch is in flight.
   * Optional — older integrations that don't support URL upload can
   * omit it.
   */
  onUrlSubmit?: (url: string) => Promise<void>;
}

/**
 * Compact file upload dropzone with drag-and-drop support.
 *
 * WHY: Extracted from DocumentManager for SRP compliance (OODA-08).
 * This component handles only the visual presentation of the dropzone.
 *
 * WHY explicit onClick: react-dropzone's internal click handler (noClick: false)
 * can silently fail with the File System Access API in certain browsers/contexts.
 * We disable noClick and use an explicit onClick → openFileDialog() for reliable
 * cross-browser file dialog opening. See:
 * - https://github.com/react-dropzone/react-dropzone/issues/1127
 * - https://github.com/react-dropzone/react-dropzone/issues/1349
 *
 * @implements FEAT0001 - Document ingestion with entity extraction
 */
export function DocumentDropzone({
  getRootProps,
  getInputProps,
  isDragActive,
  openFileDialog,
  pdfParserBackend,
  onPdfParserBackendChange,
  skipExtraction,
  onSkipExtractionChange,
  onUrlSubmit,
}: DocumentDropzoneProps) {
  const { t } = useTranslation();

  const [urlValue, setUrlValue] = useState('');
  const [urlBusy, setUrlBusy] = useState(false);
  const [urlError, setUrlError] = useState<string | null>(null);

  const handleUrlSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!onUrlSubmit) return;
    const trimmed = urlValue.trim();
    if (!trimmed) {
      setUrlError('Enter a URL');
      return;
    }
    try {
      new URL(trimmed); // structural validation only; backend enforces http(s)
    } catch {
      setUrlError('Not a valid URL');
      return;
    }
    setUrlError(null);
    setUrlBusy(true);
    try {
      await onUrlSubmit(trimmed);
      setUrlValue('');
    } catch (err) {
      setUrlError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUrlBusy(false);
    }
  };

  return (
    <div className="space-y-2">
      <div
        {...getRootProps({
          onClick: (e: React.MouseEvent) => {
            e.stopPropagation();
            openFileDialog();
          },
          role: 'button' as const,
          'aria-label': t(
            'documents.upload.uploadDrop',
            'Upload files by clicking or dragging',
          ),
          tabIndex: 0,
        })}
        className={cn(
          'border-2 border-dashed rounded-lg cursor-pointer transition-all duration-200',
          'flex items-center gap-4 px-4 py-3',
          isDragActive
            ? 'border-primary bg-primary/5 ring-2 ring-primary/20 animate-pulse'
            : 'border-muted-foreground/20 hover:border-primary/50 hover:bg-muted/30',
        )}
      >
        <input {...getInputProps()} />
        <div
          className={cn(
            'p-2 rounded-lg transition-all',
            isDragActive ? 'bg-primary/10' : 'bg-muted/50',
          )}
        >
          <Upload
            className={cn(
              'h-5 w-5 transition-all duration-200',
              isDragActive ? 'text-primary scale-110' : 'text-muted-foreground',
            )}
          />
        </div>
        <div className="flex-1 min-w-0">
          {isDragActive ? (
            <p className="text-sm font-medium text-primary">
              {t('documents.upload.uploadDropActive', 'Drop files here')}
            </p>
          ) : (
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">
                {t('documents.upload.uploadDrop', 'Drag & drop or click to upload')}{' '}
                • TXT, MD, JSON, PDF (max 100MB)
              </p>
              <p className="text-xs text-muted-foreground">
                {t(
                  'documents.upload.pdfParserHint',
                  'Choose a PDF parser override for this upload, or keep the workspace default.',
                )}
              </p>
            </div>
          )}
        </div>
        <div
          className="flex items-center gap-2"
          onClick={(event) => event.stopPropagation()}
          onKeyDown={(event) => event.stopPropagation()}
        >
          <span className="text-xs text-muted-foreground whitespace-nowrap">
            {t('documents.upload.pdfParser', 'Parser for this upload')}
          </span>
          <Select
            value={pdfParserBackend}
            onValueChange={(value: 'default' | 'vision' | 'edgeparse' | 'vlmocr') =>
              onPdfParserBackendChange(value)
            }
          >
            <SelectTrigger className="w-[190px] h-9 bg-background">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="default">
                {t('documents.upload.pdfParserDefault', 'Workspace Default')}
              </SelectItem>
              <SelectItem value="vision">
                {t('documents.upload.pdfParserVision', 'Vision')}
              </SelectItem>
              <SelectItem value="edgeparse">
                {t('documents.upload.pdfParserEdgeParse', 'EdgeParse')}
              </SelectItem>
              <SelectItem value="vlmocr">
                {t('documents.upload.pdfParserVlmOcr', 'VLM-OCR')}
              </SelectItem>
            </SelectContent>
          </Select>
          <div className="flex items-center gap-2 pl-2 border-l">
            <Checkbox
              id="skip-extraction"
              checked={skipExtraction}
              onCheckedChange={(checked) =>
                onSkipExtractionChange(checked === true)
              }
            />
            <Label
              htmlFor="skip-extraction"
              className="text-xs text-muted-foreground whitespace-nowrap cursor-pointer"
              title={t(
                'documents.upload.skipExtractionHint',
                'Index chunks for semantic search only. Skips entity, algorithm, reference and table extraction — you can run them later from the document menu.',
              )}
            >
              {t('documents.upload.skipExtraction', 'Skip extraction')}
            </Label>
          </div>
        </div>
      </div>

      {/* URL-upload row. Rendered inline below the Dropzone so the two
          entry points share one visual surface. Its click handlers
          stopPropagation so interacting with the input doesn't trigger
          the parent Dropzone's file picker. */}
      {onUrlSubmit && (
        <form
          onSubmit={handleUrlSubmit}
          onClick={(event) => event.stopPropagation()}
          onKeyDown={(event) => event.stopPropagation()}
          className="flex items-center gap-2 px-4 py-2 rounded-lg border bg-muted/30"
        >
          <LinkIcon className="h-4 w-4 text-muted-foreground shrink-0" />
          <Input
            type="url"
            inputMode="url"
            placeholder="https://arxiv.org/pdf/2506.09452"
            value={urlValue}
            onChange={(e) => {
              setUrlValue(e.target.value);
              if (urlError) setUrlError(null);
            }}
            disabled={urlBusy}
            className="flex-1 h-8 text-sm bg-background"
            aria-label="PDF URL"
            aria-invalid={urlError ? true : undefined}
          />
          <Button
            type="submit"
            size="sm"
            disabled={urlBusy || urlValue.trim().length === 0}
          >
            {urlBusy ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <LinkIcon className="h-4 w-4" />
            )}
            Fetch
          </Button>
          {urlError && (
            <span className="text-xs text-destructive" role="alert">
              {urlError}
            </span>
          )}
        </form>
      )}
    </div>
  );
}

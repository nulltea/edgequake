'use client';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import type { RepoCandidate } from '@/types/document-repos';
import {
  CheckCircle,
  ExternalLink,
  FileText,
  Globe,
  GitBranch,
  Trash2,
  XCircle,
} from 'lucide-react';

interface RepoCardProps {
  repo: RepoCandidate;
  onApprove?: (id: string) => void;
  onReject?: (id: string) => void;
  /**
   * Remove the repo regardless of current status. Backend uses the
   * same hard-delete path as `reject` (rejected rows are deleted,
   * not flagged), so this also tears down any analyzer artifacts
   * via the `ON DELETE CASCADE` on `code_artifacts.document_repo_id`.
   */
  onDelete?: (id: string) => void;
}

function confidenceColor(c: RepoCandidate['confidence']): string {
  switch (c) {
    case 'high':
      return 'bg-green-500/15 text-green-700 dark:text-green-400 border-green-500/30';
    case 'medium':
      return 'bg-yellow-500/15 text-yellow-700 dark:text-yellow-400 border-yellow-500/30';
    case 'low':
      return 'bg-red-500/15 text-red-700 dark:text-red-400 border-red-500/30';
  }
}

function statusVariant(
  s: RepoCandidate['status'],
): 'default' | 'secondary' | 'destructive' | 'outline' {
  switch (s) {
    case 'approved':
      return 'default';
    case 'rejected':
      return 'destructive';
    default:
      return 'secondary';
  }
}

function methodLabel(m: RepoCandidate['detection_method']): string {
  switch (m) {
    case 'pdf_link':
      return 'PDF link';
    case 'github_api':
      return 'GitHub API';
    case 'web_search':
      return 'Web search';
    case 'manual':
      return 'Manual';
  }
}

function MethodIcon({ method }: { method: RepoCandidate['detection_method'] }) {
  switch (method) {
    case 'pdf_link':
      return <FileText className="h-3.5 w-3.5" />;
    case 'github_api':
      return <GitBranch className="h-3.5 w-3.5" />;
    case 'web_search':
    case 'manual':
      return <Globe className="h-3.5 w-3.5" />;
  }
}

export function RepoCard({ repo, onApprove, onReject, onDelete }: RepoCardProps) {
  return (
    <div className="rounded-lg border bg-card p-4 shadow-sm">
      {/* Header: host/owner/repo + status */}
      <div className="flex items-start justify-between gap-3 mb-2">
        <div className="flex items-center gap-2 min-w-0">
          <GitBranch className="h-5 w-5 text-primary shrink-0" />
          <h3 className="text-base font-semibold truncate">
            <a
              href={repo.url}
              target="_blank"
              rel="noopener noreferrer"
              className="hover:underline"
            >
              {repo.owner}/{repo.repo}
            </a>
          </h3>
          <ExternalLink className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          <Badge className={confidenceColor(repo.confidence)}>
            {repo.confidence}
          </Badge>
          <Badge variant={statusVariant(repo.status)}>{repo.status}</Badge>
          {onDelete && (
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7 text-muted-foreground hover:text-destructive"
              onClick={() => onDelete(repo.id)}
              title="Delete this reference repository"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </Button>
          )}
        </div>
      </div>

      {/* Provenance row */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground mb-3">
        <span className="inline-flex items-center gap-1">
          <MethodIcon method={repo.detection_method} />
          {methodLabel(repo.detection_method)}
        </span>
        {repo.detection_method === 'pdf_link' && repo.pdf_page_index !== null && (
          <span>page {repo.pdf_page_index + 1}</span>
        )}
        {(repo.detection_method === 'web_search' ||
          repo.detection_method === 'github_api') &&
          repo.search_rank !== null && (
            <span>search rank #{repo.search_rank + 1}</span>
          )}
        <span className="uppercase tracking-wide">{repo.host}</span>
      </div>

      {/* Source attribution for web-search results */}
      {repo.source_url && repo.source_url !== repo.url && (
        <div className="text-xs text-muted-foreground truncate mb-3">
          via{' '}
          <a
            href={repo.source_url}
            target="_blank"
            rel="noopener noreferrer"
            className="hover:underline"
          >
            {repo.source_url}
          </a>
        </div>
      )}

      {/* Actions */}
      {(onApprove || onReject) && repo.status === 'pending' && (
        <div className="flex items-center gap-2 pt-3 border-t">
          {onApprove && (
            <Button
              variant="outline"
              size="sm"
              className="text-green-700 dark:text-green-400 hover:bg-green-500/10"
              onClick={() => onApprove(repo.id)}
            >
              <CheckCircle className="h-4 w-4" />
              Approve
            </Button>
          )}
          {onReject && (
            <Button
              variant="outline"
              size="sm"
              className="text-red-700 dark:text-red-400 hover:bg-red-500/10"
              onClick={() => onReject(repo.id)}
            >
              <XCircle className="h-4 w-4" />
              Reject
            </Button>
          )}
        </div>
      )}
    </div>
  );
}

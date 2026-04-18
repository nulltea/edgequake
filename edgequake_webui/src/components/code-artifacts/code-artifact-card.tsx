'use client';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import type { CodeArtifact } from '@/types/code-artifacts';
import { CheckCircle, ExternalLink, FileCode2, XCircle } from 'lucide-react';

interface CodeArtifactCardProps {
  artifact: CodeArtifact;
  repoUrl?: string;
  onApprove?: (id: string) => void;
  onReject?: (id: string) => void;
}

function confidenceColor(c: CodeArtifact['match_confidence']): string {
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
  s: CodeArtifact['status'],
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

/** Build a deep-link URL to the matched lines on GitHub at the pinned commit. */
function buildGithubLink(
  repoUrl: string | undefined,
  commit: string,
  file: string,
  start: number,
  end: number,
): string | null {
  if (!repoUrl) return null;
  // Accept both https://github.com/… and github.com/… forms.
  const base = repoUrl.replace(/\.git$/, '').replace(/\/+$/, '');
  return `${base}/blob/${commit}/${file}#L${start}-L${end}`;
}

export function CodeArtifactCard({
  artifact,
  repoUrl,
  onApprove,
  onReject,
}: CodeArtifactCardProps) {
  const link = buildGithubLink(
    repoUrl,
    artifact.repo_commit,
    artifact.file_path,
    artifact.start_line,
    artifact.end_line,
  );

  return (
    <div className="rounded-lg border bg-card p-4 shadow-sm">
      {/* Header */}
      <div className="flex items-start justify-between gap-3 mb-2">
        <div className="flex items-center gap-2 min-w-0">
          <FileCode2 className="h-5 w-5 text-primary shrink-0" />
          <h3 className="text-sm font-mono font-semibold truncate">
            {link ? (
              <a
                href={link}
                target="_blank"
                rel="noopener noreferrer"
                className="hover:underline"
              >
                {artifact.file_path}:{artifact.start_line}-{artifact.end_line}
              </a>
            ) : (
              <span>
                {artifact.file_path}:{artifact.start_line}-{artifact.end_line}
              </span>
            )}
          </h3>
          {link && (
            <ExternalLink className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
          )}
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          <Badge className={confidenceColor(artifact.match_confidence)}>
            {artifact.match_confidence}
          </Badge>
          <Badge variant={statusVariant(artifact.status)}>
            {artifact.status}
          </Badge>
        </div>
      </div>

      {/* Language + commit */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground mb-3">
        <span className="uppercase tracking-wide">{artifact.language}</span>
        <span className="font-mono truncate">
          @ {artifact.repo_commit.slice(0, 8)}
        </span>
        {artifact.repo_license && (
          <span className="uppercase">{artifact.repo_license}</span>
        )}
      </div>

      {/* Rationale */}
      {artifact.match_rationale && (
        <p className="text-sm text-muted-foreground mb-3 leading-relaxed">
          {artifact.match_rationale}
        </p>
      )}

      {/* Snippet */}
      <pre className="rounded-md bg-muted/50 border p-3 overflow-x-auto text-xs font-mono leading-relaxed max-h-80">
        {artifact.snippet}
      </pre>

      {/* Actions */}
      {(onApprove || onReject) && artifact.status === 'pending' && (
        <div className="flex items-center gap-2 mt-4 pt-3 border-t">
          {onApprove && (
            <Button
              variant="outline"
              size="sm"
              className="text-green-700 dark:text-green-400 hover:bg-green-500/10"
              onClick={() => onApprove(artifact.id)}
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
              onClick={() => onReject(artifact.id)}
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

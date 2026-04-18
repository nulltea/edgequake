'use client';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import type { Algorithm } from '@/types/algorithms';
import { CheckCircle, ChevronDown, CodeXml, Trash2, XCircle } from 'lucide-react';
import { InlineMath } from './inline-math';

interface AlgorithmCardProps {
  algorithm: Algorithm;
  onApprove?: (id: string) => void;
  onReject?: (id: string) => void;
  onDelete?: (id: string) => void;
}

function confidenceColor(confidence: string): string {
  switch (confidence.toLowerCase()) {
    case 'high':
      return 'bg-green-500/15 text-green-700 dark:text-green-400 border-green-500/30';
    case 'medium':
      return 'bg-yellow-500/15 text-yellow-700 dark:text-yellow-400 border-yellow-500/30';
    case 'low':
      return 'bg-red-500/15 text-red-700 dark:text-red-400 border-red-500/30';
    default:
      return 'bg-muted text-muted-foreground';
  }
}

function statusVariant(status: string): 'default' | 'secondary' | 'destructive' | 'outline' {
  switch (status) {
    case 'approved':
      return 'default';
    case 'rejected':
      return 'destructive';
    default:
      return 'secondary';
  }
}


export function AlgorithmCard({ algorithm, onApprove, onReject, onDelete }: AlgorithmCardProps) {
  return (
    <div className="rounded-lg border bg-card p-4 shadow-sm">
      {/* Header */}
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="flex items-center gap-2 min-w-0">
          <CodeXml className="h-5 w-5 text-primary shrink-0" />
          {/* break-words only — no truncate/line-clamp. KaTeX renders the
              math in algorithm names (e.g. "Functionality ℱ_{B2A}") as
              inline-block spans; `truncate` hides them past the container
              edge because text-overflow: ellipsis doesn't apply to non-text
              children. Most titles fit on one line; rare long ones wrap
              naturally. */}
          <h3 className="text-base font-semibold break-words leading-tight">
            <InlineMath text={algorithm.name} />
          </h3>
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          <Badge className={confidenceColor(algorithm.confidence)}>
            {algorithm.confidence}
          </Badge>
          <Badge variant={statusVariant(algorithm.status)}>
            {algorithm.status}
          </Badge>
          {onDelete && (
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7 text-muted-foreground hover:text-destructive"
              onClick={() => onDelete(algorithm.id)}
            >
              <Trash2 className="h-3.5 w-3.5" />
            </Button>
          )}
        </div>
      </div>

      {/* Description */}
      {algorithm.description && (
        <p className="text-sm text-muted-foreground mb-3 leading-relaxed">
          <InlineMath text={algorithm.description} />
        </p>
      )}

      {/* Tags + Complexity */}
      <div className="flex flex-wrap items-center gap-1.5 mb-3">
        {algorithm.complexity && (
          <Badge variant="outline" className="text-xs">
            O({algorithm.complexity})
          </Badge>
        )}
        {algorithm.tags.map((tag) => (
          <Badge key={tag} variant="secondary" className="text-xs">
            {tag}
          </Badge>
        ))}
      </div>

      {/* Collapsible Sections */}
      <div className="space-y-1">
        {/* Steps */}
        {algorithm.steps.length > 0 && (
          <details className="group">
            <summary className="flex cursor-pointer items-center gap-1.5 rounded-md px-2 py-1.5 text-sm font-medium hover:bg-muted/50 select-none">
              <ChevronDown className="h-4 w-4 shrink-0 transition-transform group-open:rotate-180" />
              Steps ({algorithm.steps.length})
            </summary>
            <div className="mt-1 ml-2 pl-4 border-l space-y-2">
              {algorithm.steps.map((step) => (
                <div key={step.number} className="text-sm">
                  <div className="flex items-start gap-2">
                    <span className="text-muted-foreground font-mono text-xs mt-0.5 shrink-0">
                      {step.number}.
                    </span>
                    <div className="min-w-0">
                      <span className="font-semibold"><InlineMath text={step.action} /></span>
                      {step.details && (
                        <span className="text-muted-foreground"> &mdash; <InlineMath text={step.details} /></span>
                      )}
                      {step.math && (
                        <div className="mt-1 text-primary/80">
                          <InlineMath text={step.math} />
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </details>
        )}

        {/* Inputs */}
        {algorithm.inputs.length > 0 && (
          <details className="group">
            <summary className="flex cursor-pointer items-center gap-1.5 rounded-md px-2 py-1.5 text-sm font-medium hover:bg-muted/50 select-none">
              <ChevronDown className="h-4 w-4 shrink-0 transition-transform group-open:rotate-180" />
              Inputs ({algorithm.inputs.length})
            </summary>
            <div className="mt-1 ml-2 overflow-x-auto">
              <table className="w-full text-sm border-collapse">
                <thead>
                  <tr className="border-b text-left">
                    <th className="py-1.5 pr-4 font-medium text-muted-foreground">Name</th>
                    <th className="py-1.5 pr-4 font-medium text-muted-foreground">Type</th>
                    <th className="py-1.5 font-medium text-muted-foreground">Description</th>
                  </tr>
                </thead>
                <tbody>
                  {algorithm.inputs.map((io) => (
                    <tr key={io.name} className="border-b last:border-0">
                      <td className="py-1.5 pr-4 font-mono text-xs"><InlineMath text={io.name} /></td>
                      <td className="py-1.5 pr-4 font-mono text-xs text-muted-foreground"><InlineMath text={io.type} /></td>
                      <td className="py-1.5 text-muted-foreground"><InlineMath text={io.description} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        )}

        {/* Outputs */}
        {algorithm.outputs.length > 0 && (
          <details className="group">
            <summary className="flex cursor-pointer items-center gap-1.5 rounded-md px-2 py-1.5 text-sm font-medium hover:bg-muted/50 select-none">
              <ChevronDown className="h-4 w-4 shrink-0 transition-transform group-open:rotate-180" />
              Outputs ({algorithm.outputs.length})
            </summary>
            <div className="mt-1 ml-2 overflow-x-auto">
              <table className="w-full text-sm border-collapse">
                <thead>
                  <tr className="border-b text-left">
                    <th className="py-1.5 pr-4 font-medium text-muted-foreground">Name</th>
                    <th className="py-1.5 pr-4 font-medium text-muted-foreground">Type</th>
                    <th className="py-1.5 font-medium text-muted-foreground">Description</th>
                  </tr>
                </thead>
                <tbody>
                  {algorithm.outputs.map((io) => (
                    <tr key={io.name} className="border-b last:border-0">
                      <td className="py-1.5 pr-4 font-mono text-xs"><InlineMath text={io.name} /></td>
                      <td className="py-1.5 pr-4 font-mono text-xs text-muted-foreground"><InlineMath text={io.type} /></td>
                      <td className="py-1.5 text-muted-foreground"><InlineMath text={io.description} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        )}

        {/* Preconditions */}
        {algorithm.preconditions.length > 0 && (
          <details className="group">
            <summary className="flex cursor-pointer items-center gap-1.5 rounded-md px-2 py-1.5 text-sm font-medium hover:bg-muted/50 select-none">
              <ChevronDown className="h-4 w-4 shrink-0 transition-transform group-open:rotate-180" />
              Preconditions ({algorithm.preconditions.length})
            </summary>
            <ul className="mt-1 ml-6 space-y-1 list-disc">
              {algorithm.preconditions.map((cond, i) => (
                <li key={i} className="text-sm text-muted-foreground">
                  <InlineMath text={cond} />
                </li>
              ))}
            </ul>
          </details>
        )}

        {/* Mathematical Notation */}
        {algorithm.mathematical_notation && (
          <details className="group">
            <summary className="flex cursor-pointer items-center gap-1.5 rounded-md px-2 py-1.5 text-sm font-medium hover:bg-muted/50 select-none">
              <ChevronDown className="h-4 w-4 shrink-0 transition-transform group-open:rotate-180" />
              Mathematical Notation
            </summary>
            <div className="mt-1 ml-2 p-3 rounded-md bg-muted/30 overflow-x-auto">
              <InlineMath text={algorithm.mathematical_notation} />
            </div>
          </details>
        )}

        {/* Pseudocode */}
        {algorithm.pseudocode && (
          <details className="group">
            <summary className="flex cursor-pointer items-center gap-1.5 rounded-md px-2 py-1.5 text-sm font-medium hover:bg-muted/50 select-none">
              <ChevronDown className="h-4 w-4 shrink-0 transition-transform group-open:rotate-180" />
              Pseudocode
            </summary>
            <div className="mt-1 ml-2">
              <pre className="rounded-md bg-muted/50 border p-3 overflow-x-auto text-sm font-mono leading-relaxed whitespace-pre-wrap">
                {algorithm.pseudocode}
              </pre>
            </div>
          </details>
        )}
      </div>

      {/* Actions */}
      {(onApprove || onReject) && algorithm.status === 'pending' && (
        <div className="flex items-center gap-2 mt-4 pt-3 border-t">
          {onApprove && (
            <Button
              variant="outline"
              size="sm"
              className="text-green-700 dark:text-green-400 hover:bg-green-500/10"
              onClick={() => onApprove(algorithm.id)}
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
              onClick={() => onReject(algorithm.id)}
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

'use client';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Skeleton } from '@/components/ui/skeleton';
import { LLMModelSelector, type LLMSelection } from '@/components/workspace/llm-model-selector';
import { getWorkspace, updateWorkspace } from '@/lib/api/edgequake';
import {
  getWorkspaceAlgorithmAnalysisSelection,
  getWorkspaceAlgorithmExtractionSelection,
} from '@/lib/workspace/drafts';
import { useTenantStore } from '@/stores/use-tenant-store';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Brain, Cloud, CodeXml, Cpu, Pencil, Save, Sparkles, X } from 'lucide-react';
import { useState } from 'react';
import { toast } from 'sonner';

function getProviderIcon(providerId: string | undefined) {
  switch (providerId?.toLowerCase()) {
    case 'openai':
      return <Cloud className="h-4 w-4 text-green-600" />;
    case 'ollama':
      return <Cpu className="h-4 w-4 text-blue-600" />;
    case 'lmstudio':
      return <Brain className="h-4 w-4 text-purple-600" />;
    default:
      return <Sparkles className="h-4 w-4 text-muted-foreground" />;
  }
}

function ModelDisplay({
  label,
  provider,
  model,
}: {
  label: string;
  provider?: string;
  model?: string;
}) {
  return (
    <div className="flex items-center gap-3 p-3 bg-muted/50 rounded-lg">
      {getProviderIcon(provider)}
      <div className="flex-1 min-w-0">
        <div className="text-xs text-muted-foreground mb-0.5">{label}</div>
        <div className="font-medium truncate">
          {model || 'Workspace Default'}
        </div>
        <div className="text-sm text-muted-foreground capitalize">
          {provider || 'Auto-detected'}
        </div>
      </div>
      {provider && model && (
        <Badge variant="outline" className="ml-auto shrink-0">
          {provider}/{model}
        </Badge>
      )}
    </div>
  );
}

export function AlgorithmLLMSettingsCard() {
  const queryClient = useQueryClient();
  const { selectedTenantId, selectedWorkspaceId } = useTenantStore();

  const [isEditing, setIsEditing] = useState(false);
  const [analysisLLM, setAnalysisLLM] = useState<LLMSelection | undefined>(undefined);
  const [extractionLLM, setExtractionLLM] = useState<LLMSelection | undefined>(undefined);
  const [reviewMode, setReviewMode] = useState<string>('manual');

  const { data: workspace, isLoading } = useQuery({
    queryKey: ['workspace', selectedTenantId, selectedWorkspaceId],
    queryFn: () => getWorkspace(selectedTenantId!, selectedWorkspaceId!),
    enabled: !!selectedTenantId && !!selectedWorkspaceId,
    staleTime: 60000,
    retry: 1,
  });

  const updateMutation = useMutation({
    mutationFn: () =>
      updateWorkspace(selectedTenantId!, selectedWorkspaceId!, {
        algorithm_analysis_llm_provider: analysisLLM?.provider ?? '',
        algorithm_analysis_llm_model: analysisLLM?.model ?? '',
        algorithm_extraction_llm_provider: extractionLLM?.provider ?? '',
        algorithm_extraction_llm_model: extractionLLM?.model ?? '',
        algorithm_review_mode: reviewMode,
      }),
    onSuccess: () => {
      toast.success('Algorithm configuration updated');
      queryClient.invalidateQueries({
        queryKey: ['workspace', selectedTenantId, selectedWorkspaceId],
      });
      setIsEditing(false);
    },
    onError: (error) => {
      toast.error('Failed to update algorithm configuration', {
        description: error instanceof Error ? error.message : 'Unknown error',
      });
    },
  });

  const handleEdit = () => {
    setAnalysisLLM(getWorkspaceAlgorithmAnalysisSelection(workspace));
    setExtractionLLM(getWorkspaceAlgorithmExtractionSelection(workspace));
    setReviewMode(workspace?.algorithm_review_mode || 'manual');
    setIsEditing(true);
  };

  const handleCancel = () => {
    setIsEditing(false);
  };

  if (!selectedTenantId || !selectedWorkspaceId) {
    return null;
  }

  const currentReviewMode = workspace?.algorithm_review_mode || 'manual';

  return (
    <Card>
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CodeXml className="h-5 w-5 text-violet-600" />
            <CardTitle>Algorithm Extraction</CardTitle>
          </div>
          {!isEditing && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleEdit}
              aria-label="Edit"
            >
              <Pencil className="h-4 w-4" />
            </Button>
          )}
        </div>
        <CardDescription>
          Configure LLM models and review behavior for the algorithm extraction pipeline.
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-4">
        {isLoading ? (
          <Skeleton className="h-28 w-full" />
        ) : isEditing ? (
          <>
            <div className="space-y-3">
              <div>
                <label className="text-sm font-medium mb-1.5 block">
                  Analysis LLM (Stages 1 &amp; 3)
                </label>
                <LLMModelSelector value={analysisLLM} onChange={setAnalysisLLM} />
              </div>
              <div>
                <label className="text-sm font-medium mb-1.5 block">
                  Extraction LLM (Stage 2)
                </label>
                <LLMModelSelector value={extractionLLM} onChange={setExtractionLLM} />
              </div>
              <div>
                <label className="text-sm font-medium mb-1.5 block">
                  Algorithm Review Mode
                </label>
                <Select value={reviewMode} onValueChange={setReviewMode}>
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="manual">Manual — Review before embedding</SelectItem>
                    <SelectItem value="auto">Auto-approve — Embed immediately after extraction</SelectItem>
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground mt-1">
                  {reviewMode === 'auto'
                    ? 'Algorithms will be automatically approved and embedded in the vector database after extraction.'
                    : 'Algorithms require manual approval before being embedded in the vector database.'}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2 pt-2">
              <Button
                size="sm"
                onClick={() => updateMutation.mutate()}
                disabled={updateMutation.isPending}
              >
                <Save className="h-4 w-4 mr-2" />
                Save
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={handleCancel}
                disabled={updateMutation.isPending}
              >
                <X className="h-4 w-4 mr-2" />
                Cancel
              </Button>
            </div>
          </>
        ) : workspace ? (
          <div className="space-y-2">
            <ModelDisplay
              label="Analysis LLM (Stages 1 & 3)"
              provider={workspace.algorithm_analysis_llm_provider}
              model={workspace.algorithm_analysis_llm_model}
            />
            <ModelDisplay
              label="Extraction LLM (Stage 2)"
              provider={workspace.algorithm_extraction_llm_provider}
              model={workspace.algorithm_extraction_llm_model}
            />
            <div className="flex items-center gap-3 p-3 bg-muted/50 rounded-lg">
              <div className="flex-1">
                <div className="text-xs text-muted-foreground mb-0.5">Review Mode</div>
                <div className="font-medium">
                  {currentReviewMode === 'auto' ? 'Auto-approve' : 'Manual Review'}
                </div>
                <div className="text-sm text-muted-foreground">
                  {currentReviewMode === 'auto'
                    ? 'Algorithms are auto-approved and embedded after extraction'
                    : 'Algorithms require manual approval before embedding'}
                </div>
              </div>
              <Badge variant={currentReviewMode === 'auto' ? 'default' : 'secondary'}>
                {currentReviewMode}
              </Badge>
            </div>
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">
            Configure LLM models and review behavior for algorithm extraction.
          </p>
        )}
      </CardContent>
    </Card>
  );
}

/**
 * @module RerankerModelSelector
 * @description Dropdown selector for the workspace's reranker model.
 *
 * Structurally mirrors `EmbeddingModelSelector` and `LLMModelSelector`:
 * - Models fetched via `useRerankerModels()` (`GET /api/v1/models/rerankers`).
 * - Grouped by provider, each provider rendered as a `<SelectGroup>`.
 * - The selected `name` is sent as the `model` field of the rerank HTTP
 *   request — llama-swap (or any OpenAI-compatible rerank front) routes
 *   by that string.
 * - "Server default" entry at the top maps to `undefined` (workspace
 *   inherits the API's `RERANKER_MODEL` env value at query time).
 */
'use client';

import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { useRerankerModels } from '@/hooks/use-providers';
import { cn } from '@/lib/utils';
import { Brain, Cloud, Cpu, FlaskConical, HelpCircle, Loader2 } from 'lucide-react';

export interface RerankerSelection {
  /** Model id (e.g. `"jina-reranker-v3"`). Sent as the `model` field. */
  model: string;
  /** Provider id from models.toml (e.g. `"lmstudio"`). Stored for context. */
  provider: string;
}

interface RerankerModelSelectorProps {
  value?: RerankerSelection;
  onChange?: (selection: RerankerSelection | undefined) => void;
  disabled?: boolean;
  className?: string;
}

function getProviderIcon(providerId: string) {
  switch (providerId.toLowerCase()) {
    case 'openai':
      return <Cloud className="h-4 w-4 text-green-600" />;
    case 'ollama':
      return <Cpu className="h-4 w-4 text-blue-600" />;
    case 'lmstudio':
      return <Brain className="h-4 w-4 text-purple-600" />;
    case 'mock':
      return <FlaskConical className="h-4 w-4 text-gray-500" />;
    default:
      return <Brain className="h-4 w-4 text-muted-foreground" />;
  }
}

export function RerankerModelSelector({
  value,
  onChange,
  disabled,
  className,
}: RerankerModelSelectorProps) {
  const { data: rerankerData, isLoading, error } = useRerankerModels();

  if (isLoading) {
    return (
      <div className={cn('flex items-center gap-2 px-3 py-2 bg-muted rounded-lg', className)}>
        <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
        <span className="text-sm text-muted-foreground">Loading reranker models...</span>
      </div>
    );
  }

  if (error || !rerankerData) {
    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <div className={cn('flex items-center gap-2 px-3 py-2 bg-muted rounded-lg cursor-help', className)}>
              <HelpCircle className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm text-muted-foreground">Using server default</span>
            </div>
          </TooltipTrigger>
          <TooltipContent>
            <p>Could not load reranker models. Will use server default.</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  }

  // Group models by provider — identical reduce pattern to EmbeddingModelSelector.
  const modelsByProvider = rerankerData.models.reduce((acc, model) => {
    if (!acc[model.provider]) {
      acc[model.provider] = {
        displayName: model.provider_display_name,
        models: [],
      };
    }
    acc[model.provider].models.push(model);
    return acc;
  }, {} as Record<string, { displayName: string; models: typeof rerankerData.models }>);

  const currentValue = value ? `${value.provider}:${value.model}` : undefined;

  const handleChange = (selectedId: string) => {
    if (selectedId === 'default') {
      onChange?.(undefined);
      return;
    }
    const colonIdx = selectedId.indexOf(':');
    if (colonIdx === -1) return;
    const provider = selectedId.slice(0, colonIdx);
    const modelName = selectedId.slice(colonIdx + 1);
    const modelInfo = rerankerData.models.find(
      (m) => m.provider === provider && m.name === modelName,
    );
    if (modelInfo) {
      onChange?.({ model: modelName, provider });
    }
  };

  return (
    <Select
      value={currentValue || 'default'}
      onValueChange={handleChange}
      disabled={disabled || rerankerData.models.length === 0}
    >
      <SelectTrigger className={cn('w-full', className)}>
        <SelectValue placeholder="Server default">
          {currentValue ? (
            <div className="flex items-center gap-2">
              {getProviderIcon(value?.provider || '')}
              <span className="text-sm truncate">{value?.model}</span>
            </div>
          ) : (
            <span className="text-sm text-muted-foreground">Server default</span>
          )}
        </SelectValue>
      </SelectTrigger>
      <SelectContent className="max-h-[400px]">
        <SelectItem value="default">
          <div className="flex items-center gap-2">
            <HelpCircle className="h-4 w-4 text-muted-foreground" />
            <div className="flex flex-col">
              <span className="text-sm">Server Default</span>
              <span className="text-xs text-muted-foreground">
                {rerankerData.default_model ?? '(RERANKER_MODEL unset)'}
              </span>
            </div>
          </div>
        </SelectItem>

        {Object.entries(modelsByProvider).map(([providerId, { displayName, models }]) => (
          <SelectGroup key={providerId}>
            <SelectLabel className="text-xs font-semibold uppercase tracking-wide text-muted-foreground px-2 flex items-center gap-1">
              {getProviderIcon(providerId)}
              {displayName}
            </SelectLabel>
            {models.map((model) => {
              const selectId = `${providerId}:${model.name}`;
              return (
                <SelectItem
                  key={selectId}
                  value={selectId}
                  disabled={model.deprecated}
                >
                  <div className="flex items-center gap-2 w-full">
                    <div className="flex flex-col flex-1 min-w-0">
                      <div className="flex items-center gap-1.5">
                        <span className="text-sm font-medium truncate">{model.display_name}</span>
                      </div>
                      <span className="text-xs text-muted-foreground truncate">
                        {model.name}
                        {model.capabilities.context_length > 0 &&
                          ` · ${(model.capabilities.context_length / 1024).toFixed(0)}K ctx`}
                      </span>
                    </div>
                  </div>
                </SelectItem>
              );
            })}
          </SelectGroup>
        ))}
      </SelectContent>
    </Select>
  );
}

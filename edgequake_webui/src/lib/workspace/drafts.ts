import type { PdfParserBackendChoice } from "@/components/settings/pdf-parser-backend-field";
import type { EmbeddingSelection } from "@/components/workspace/embedding-model-selector";
import type { LLMSelection } from "@/components/workspace/llm-model-selector";
import type { Workspace } from "@/types";

export function getWorkspaceLlmSelection(
  workspace?: Workspace | null,
): LLMSelection | undefined {
  if (!workspace?.llm_provider || !workspace.llm_model) {
    return undefined;
  }

  return {
    model: workspace.llm_model,
    provider: workspace.llm_provider,
    fullId: `${workspace.llm_provider}/${workspace.llm_model}`,
  };
}

export function getWorkspaceEmbeddingSelection(
  workspace?: Workspace | null,
): EmbeddingSelection | undefined {
  if (!workspace?.embedding_provider || !workspace.embedding_model) {
    return undefined;
  }

  return {
    model: workspace.embedding_model,
    provider: workspace.embedding_provider,
    dimension: workspace.embedding_dimension ?? 768,
  };
}

export function getWorkspaceVisionSelection(
  workspace?: Workspace | null,
): LLMSelection | undefined {
  if (!workspace?.vision_llm_provider || !workspace.vision_llm_model) {
    return undefined;
  }

  return {
    model: workspace.vision_llm_model,
    provider: workspace.vision_llm_provider,
    fullId: `${workspace.vision_llm_provider}/${workspace.vision_llm_model}`,
  };
}

export function getWorkspacePdfParserBackend(
  workspace?: Workspace | null,
): PdfParserBackendChoice {
  return workspace?.pdf_parser_backend ?? "none";
}

export function getWorkspaceAlgorithmAnalysisSelection(
  workspace?: Workspace | null,
): LLMSelection | undefined {
  if (!workspace?.algorithm_analysis_llm_provider || !workspace.algorithm_analysis_llm_model) {
    return undefined;
  }
  return {
    model: workspace.algorithm_analysis_llm_model,
    provider: workspace.algorithm_analysis_llm_provider,
    fullId: `${workspace.algorithm_analysis_llm_provider}/${workspace.algorithm_analysis_llm_model}`,
  };
}

export function getWorkspaceAlgorithmExtractionSelection(
  workspace?: Workspace | null,
): LLMSelection | undefined {
  if (!workspace?.algorithm_extraction_llm_provider || !workspace.algorithm_extraction_llm_model) {
    return undefined;
  }
  return {
    model: workspace.algorithm_extraction_llm_model,
    provider: workspace.algorithm_extraction_llm_provider,
    fullId: `${workspace.algorithm_extraction_llm_provider}/${workspace.algorithm_extraction_llm_model}`,
  };
}

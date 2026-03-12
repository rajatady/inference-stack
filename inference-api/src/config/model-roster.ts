/**
 * Model Roster — maps all known models to their type, VRAM estimate, and default GPU.
 *
 * Used by:
 * - ModelManager: GPU affinity for load placement
 * - Router: model type detection
 * - Endpoints: validation of model IDs
 */

export type ModelType =
  | 'text_gen'
  | 'vision_language'
  | 'tts'
  | 'image_gen'
  | 'video_gen';

export interface ModelRosterEntry {
  type: ModelType;
  vramEstimateBytes: number;
  defaultGpu: string; // Worker ID preference (e.g., 'worker-0', 'worker-1')
  tensorParallel?: boolean;       // Requires tensor parallelism across multiple GPUs
  tensorParallelSize?: number;    // Number of GPUs needed for TP
}

export const MODEL_ROSTER: Record<string, ModelRosterEntry> = {
  'HuggingFaceTB/SmolLM2-135M-Instruct': {
    type: 'text_gen',
    vramEstimateBytes: 0.3e9,
    defaultGpu: 'worker-0',
  },
  'HuggingFaceTB/SmolLM2-360M-Instruct': {
    type: 'text_gen',
    vramEstimateBytes: 0.7e9,
    defaultGpu: 'worker-0',
  },
  'HuggingFaceTB/SmolLM2-1.7B-Instruct': {
    type: 'text_gen',
    vramEstimateBytes: 3.5e9,
    defaultGpu: 'worker-0',
  },
  'Qwen/Qwen2.5-VL-3B-Instruct': {
    type: 'vision_language',
    vramEstimateBytes: 7e9,
    defaultGpu: 'worker-0',
  },
  'hexgrad/Kokoro-82M': {
    type: 'tts',
    vramEstimateBytes: 0.5e9,
    defaultGpu: 'worker-0',
  },
  'stabilityai/sd-turbo': {
    type: 'image_gen',
    vramEstimateBytes: 5.5e9,
    defaultGpu: 'worker-1',
  },
  'THUDM/CogVideoX-2b': {
    type: 'video_gen',
    vramEstimateBytes: 7e9,
    defaultGpu: 'worker-1',
  },
  'Qwen/Qwen3-14B': {
    type: 'text_gen',
    vramEstimateBytes: 29e9,
    defaultGpu: 'tp-worker-0',
    tensorParallel: true,
    tensorParallelSize: 2,
  },
};

export function getModelType(modelId: string): ModelType | undefined {
  return MODEL_ROSTER[modelId]?.type;
}

export function getDefaultGpu(modelId: string): string | undefined {
  return MODEL_ROSTER[modelId]?.defaultGpu;
}

export function isTensorParallel(modelId: string): boolean {
  return MODEL_ROSTER[modelId]?.tensorParallel === true;
}

export function getTensorParallelSize(modelId: string): number {
  return MODEL_ROSTER[modelId]?.tensorParallelSize ?? 1;
}

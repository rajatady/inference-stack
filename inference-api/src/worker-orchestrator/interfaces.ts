import { GpuWorkerService } from '../gpu-worker/gpu-worker.service';

export type WorkerMode = 'individual' | 'tensor-parallel';

export interface WorkerConfig {
  id: string;
  url: string;
  mode?: WorkerMode;
}

export interface WorkerHandle {
  config: WorkerConfig;
  service: GpuWorkerService;
  close: () => void;
}

export interface GpuSnapshot {
  vramTotalBytes: number;
  vramUsedBytes: number;
  vramAvailableBytes: number;
  utilization: number;
  temperatureC: number;
  healthy: boolean;
}

export interface LoadedModelSnapshot {
  modelId: string;
  ready: boolean;
  vramUsedBytes: number;
}

export interface WorkerSnapshot {
  workerId: string;
  healthy: boolean;
  gpu: GpuSnapshot;
  models: LoadedModelSnapshot[];
  activeInferences: number;
  queuedInferences: number;
  lastUpdated: number;
}

export interface ModelCapabilities {
  maxContextLength: number;
  vocabSize: number;
  supportsLogprobs: boolean;
  supportsJsonMode: boolean;
  supportsGrammar: boolean;
}

export interface RoutingDecision {
  worker: GpuWorkerService;
  workerId: string;
  action: 'direct' | 'load-then-infer';
}

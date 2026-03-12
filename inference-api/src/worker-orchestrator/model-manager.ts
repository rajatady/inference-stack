import { Injectable, Logger } from '@nestjs/common';
import { firstValueFrom } from 'rxjs';
import { GpuWorkerService } from '../gpu-worker/gpu-worker.service';
import { WorkerRegistry } from './worker-registry';
import { WorkerSnapshot, ModelCapabilities } from './interfaces';
import { getDefaultGpu, isTensorParallel } from '../config/model-roster';

@Injectable()
export class ModelManager {
  private readonly logger = new Logger(ModelManager.name);
  private readonly inFlightLoads = new Map<string, Promise<{ workerId: string; worker: GpuWorkerService }>>();
  private readonly capabilities = new Map<string, ModelCapabilities>();

  constructor(private readonly registry: WorkerRegistry) {}

  /**
   * Returns cached capabilities for a model (populated after loadModel succeeds).
   */
  getModelCapabilities(modelId: string): ModelCapabilities | undefined {
    return this.capabilities.get(modelId);
  }

  /**
   * Returns snapshots of healthy workers that have the given model loaded and ready.
   */
  getWorkersWithModel(modelId: string): WorkerSnapshot[] {
    return this.registry.getAllSnapshots().filter(
      (s) =>
        s.healthy &&
        s.models.some((m) => m.modelId === modelId && m.ready),
    );
  }

  /**
   * Ensures the model is loaded on at least one worker. Returns the worker to use.
   *
   * 1. If model already loaded → return least-busy worker
   * 2. If not → pick worker with most free VRAM, load model
   * 3. If VRAM tight → evict idle model first, then load
   * 4. If no capacity → throw
   *
   * Concurrent calls for the same model coalesce into a single load operation.
   */
  async ensureModelLoaded(
    modelId: string,
    quantization?: string,
  ): Promise<{ workerId: string; worker: GpuWorkerService }> {
    // Auto mode switching: if model needs TP and we're in individual mode (or vice versa)
    const needsTP = isTensorParallel(modelId);
    const currentMode = this.registry.getCurrentMode();

    if (needsTP && currentMode !== 'tensor-parallel') {
      this.logger.log(`Model ${modelId} requires tensor parallelism — switching mode`);
      await this.registry.switchMode('tensor-parallel');
    } else if (!needsTP && currentMode === 'tensor-parallel') {
      this.logger.log(`Model ${modelId} is single-GPU — switching to individual mode`);
      await this.registry.switchMode('individual');
    }

    // Check if already loaded
    const ready = this.getWorkersWithModel(modelId);
    if (ready.length > 0) {
      const best = this.leastBusy(ready);
      const worker = this.registry.getWorker(best.workerId)!;
      return { workerId: best.workerId, worker };
    }

    // Coalesce concurrent loads for the same model
    const existing = this.inFlightLoads.get(modelId);
    if (existing) {
      return existing;
    }

    const loadPromise = this.loadOnBestWorker(modelId, quantization);
    this.inFlightLoads.set(modelId, loadPromise);

    try {
      return await loadPromise;
    } finally {
      this.inFlightLoads.delete(modelId);
    }
  }

  private async loadOnBestWorker(
    modelId: string,
    quantization?: string,
  ): Promise<{ workerId: string; worker: GpuWorkerService }> {
    const allSnapshots = this.registry
      .getAllSnapshots()
      .filter((s) => s.healthy);

    if (allSnapshots.length === 0) {
      throw new Error(
        `No worker has capacity to load model ${modelId}: no healthy workers available`,
      );
    }

    // Sort by GPU affinity first, then most available VRAM
    const preferredWorker = getDefaultGpu(modelId);
    const candidates = [...allSnapshots].sort((a, b) => {
      // Preferred worker goes first
      if (preferredWorker) {
        const aPreferred = a.workerId === preferredWorker ? 1 : 0;
        const bPreferred = b.workerId === preferredWorker ? 1 : 0;
        if (aPreferred !== bPreferred) return bPreferred - aPreferred;
      }
      // Then sort by most available VRAM
      return b.gpu.vramAvailableBytes - a.gpu.vramAvailableBytes;
    });

    this.logger.log(
      `Loading ${modelId}: preferred=${preferredWorker ?? 'none'}, candidates=[${candidates.map((c) => `${c.workerId}(${(c.gpu.vramAvailableBytes / 1e9).toFixed(1)}GB)`).join(', ')}]`,
    );

    // Try loading on the worker with most free VRAM
    for (const candidate of candidates) {
      const worker = this.registry.getWorker(candidate.workerId);
      if (!worker) continue;

      try {
        const response = await firstValueFrom(
          worker.loadModel({
            model_id: modelId,
            quantization: quantization ?? 'fp16',
          }),
        );

        if (response.success) {
          this.cacheCapabilities(modelId, response.capabilities);
          this.logger.log(
            `Loaded ${modelId} on ${candidate.workerId} (${response.vram_used_bytes} bytes)`,
          );
          return { workerId: candidate.workerId, worker };
        }
      } catch (err) {
        this.logger.warn(
          `Failed to load ${modelId} on ${candidate.workerId}: ${err.message}`,
        );
      }

      // Load failed — try evicting an idle model on this worker
      const evicted = await this.tryEvictIdle(candidate);
      if (evicted) {
        try {
          const response = await firstValueFrom(
            worker.loadModel({
              model_id: modelId,
              quantization: quantization ?? 'fp16',
            }),
          );

          if (response.success) {
            this.cacheCapabilities(modelId, response.capabilities);
            this.logger.log(
              `Loaded ${modelId} on ${candidate.workerId} after eviction`,
            );
            return { workerId: candidate.workerId, worker };
          }
        } catch {
          // Continue to next candidate
        }
      }
    }

    throw new Error(
      `No worker has capacity to load model ${modelId}: all workers full or load failed`,
    );
  }

  /**
   * Try to evict an idle model (0 active inferences) on the given worker.
   * Returns true if something was evicted.
   */
  private async tryEvictIdle(snapshot: WorkerSnapshot): Promise<boolean> {
    if (snapshot.activeInferences > 0) return false;

    const idleModels = snapshot.models.filter((m) => m.ready);
    if (idleModels.length === 0) return false;

    // Evict the model using the least VRAM (smallest loss)
    const toEvict = idleModels.sort(
      (a, b) => a.vramUsedBytes - b.vramUsedBytes,
    )[0];

    const worker = this.registry.getWorker(snapshot.workerId);
    if (!worker) return false;

    try {
      const response = await firstValueFrom(
        worker.unloadModel({ model_id: toEvict.modelId }),
      );
      if (response.success) {
        this.logger.log(
          `Evicted ${toEvict.modelId} from ${snapshot.workerId} (freed ${response.vram_freed_bytes} bytes)`,
        );
        return true;
      }
    } catch (err) {
      this.logger.warn(
        `Failed to evict ${toEvict.modelId} from ${snapshot.workerId}: ${err.message}`,
      );
    }

    return false;
  }

  private cacheCapabilities(modelId: string, caps: any): void {
    if (!caps) return;
    this.capabilities.set(modelId, {
      maxContextLength: caps.max_context_length ?? 0,
      vocabSize: caps.vocab_size ?? 0,
      supportsLogprobs: caps.supports_logprobs ?? false,
      supportsJsonMode: caps.supports_json_mode ?? false,
      supportsGrammar: caps.supports_grammar ?? false,
    });
  }

  private leastBusy(snapshots: WorkerSnapshot[]): WorkerSnapshot {
    return [...snapshots].sort(
      (a, b) =>
        a.activeInferences + a.queuedInferences -
        (b.activeInferences + b.queuedInferences),
    )[0];
  }
}

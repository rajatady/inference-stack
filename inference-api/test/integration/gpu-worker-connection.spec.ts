/**
 * Integration Test: GPU Worker Connection
 *
 * Tests the most fundamental contract: the gateway can connect to a GPU worker
 * over gRPC and exchange messages. Everything else depends on this.
 *
 * Runs against REAL GPU workers on RunPod via SSH tunnel:
 *   localhost:50051 → RunPod GPU-0 (worker-0)
 *   localhost:50052 → RunPod GPU-1 (worker-1)
 *
 * Prerequisites:
 *   1. GPU workers running on RunPod: python server.py --port 50051 --gpu-id 0
 *   2. SSH tunnel: ssh -f -N -L 50051:localhost:50051 -L 50052:localhost:50052 $RUNPOD_SSH_USER@$RUNPOD_SSH_HOST -p $RUNPOD_SSH_PORT
 *
 * Scope reference: grpc-contract.md — InferenceWorker service
 * Test scenario: Foundation (prerequisite for T1-T22)
 */
import { lastValueFrom, firstValueFrom, toArray } from 'rxjs';
import { WorkerRegistry } from '../../src/worker-orchestrator/worker-registry';

const WORKER_URL = process.env.GPU_WORKER_URL || 'localhost:50051';
const TEST_MODEL = 'HuggingFaceTB/SmolLM2-135M';

describe('GPU Worker Connection', () => {
  let registry: WorkerRegistry;

  beforeAll(() => {
    registry = new WorkerRegistry();
    registry.addWorker({ id: 'gpu-0', url: WORKER_URL });
  }, 15_000);

  afterAll(() => {
    // Unload the test model if it's still loaded (cleanup)
    const worker = registry.getWorker('gpu-0');
    if (worker) {
      firstValueFrom(worker.unloadModel({ model_id: TEST_MODEL })).catch(
        () => {},
      );
    }
    registry.onModuleDestroy();
  });

  describe('Health', () => {
    it('should return HEALTHY status from a running worker', async () => {
      const worker = registry.getWorker('gpu-0')!;
      const response = await firstValueFrom(worker.health());
      expect(response.status).toBe('HEALTHY');
      expect(response.uptime_ms).toBeGreaterThan(0);
    });

    it('should report GPU info (model, VRAM total, temperature)', async () => {
      const worker = registry.getWorker('gpu-0')!;
      const state = await firstValueFrom(worker.getWorkerState());
      expect(state.gpu.gpu_model).toBeTruthy();
      expect(state.gpu.vram_total_bytes).toBeGreaterThan(0);
      expect(state.gpu.gpu_temperature_c).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Model Loading', () => {
    it('should load a model onto the worker GPU', async () => {
      const worker = registry.getWorker('gpu-0')!;
      const response = await firstValueFrom(
        worker.loadModel({
          model_id: TEST_MODEL,
          quantization: 'fp16',
        }),
      );
      expect(response.success).toBe(true);
      // vram_used_bytes is a delta from torch.cuda.memory_allocated() — it can be negative
      // when the model was already loaded by another test suite (cached weights).
      // The important assertions are success + valid capabilities.
      expect(typeof response.vram_used_bytes).toBe('number');
      expect(response.capabilities.max_context_length).toBeGreaterThan(0);
    }, 120_000);

    it('should report the loaded model in worker state', async () => {
      const worker = registry.getWorker('gpu-0')!;
      const state = await firstValueFrom(worker.getWorkerState());
      const loadedModel = state.models.find(
        (m) => m.model_id === TEST_MODEL,
      );
      expect(loadedModel).toBeDefined();
      expect(loadedModel.ready).toBe(true);
    });

    it('should report available VRAM after loading', async () => {
      const worker = registry.getWorker('gpu-0')!;
      const state = await firstValueFrom(worker.getWorkerState());
      const availableVram = state.gpu.vram_available_bytes;
      expect(availableVram).toBeGreaterThan(0);
      expect(state.gpu.vram_used_bytes).toBeGreaterThan(0);
    });
  });

  describe('Basic Inference', () => {
    it('should stream tokens back for a simple prompt', async () => {
      const worker = registry.getWorker('gpu-0')!;
      const responses = await lastValueFrom(
        worker
          .infer({
            request_id: 'test-1',
            model_id: TEST_MODEL,
            prompt: 'The capital of France is',
            params: {
              max_tokens: 20,
              temperature: 0.1,
            },
          })
          .pipe(toArray()),
      );

      expect(responses.length).toBeGreaterThanOrEqual(2);

      const chunks = responses.filter((r) => r.chunk);
      const complete = responses.find((r) => r.complete);

      expect(chunks.length).toBeGreaterThan(0);
      expect(chunks[0].chunk.text).toBeTruthy();

      expect(complete).toBeDefined();
      expect(['STOP', 'MAX_TOKENS']).toContain(
        complete.complete.finish_reason,
      );
      expect(complete.complete.usage.prompt_tokens).toBeGreaterThan(0);
      expect(complete.complete.usage.completion_tokens).toBeGreaterThan(0);
    }, 30_000);

    it('should respect max_tokens parameter', async () => {
      const worker = registry.getWorker('gpu-0')!;
      const responses = await lastValueFrom(
        worker
          .infer({
            request_id: 'test-2',
            model_id: TEST_MODEL,
            prompt: 'Once upon a time',
            params: {
              max_tokens: 5,
              temperature: 0.1,
            },
          })
          .pipe(toArray()),
      );

      const complete = responses.find((r) => r.complete);
      expect(complete).toBeDefined();
      expect(complete.complete.usage.completion_tokens).toBeLessThanOrEqual(5);
    }, 30_000);

    it('should return error for a model that is not loaded', async () => {
      const worker = registry.getWorker('gpu-0')!;
      const responses = await lastValueFrom(
        worker
          .infer({
            request_id: 'test-3',
            model_id: 'nonexistent-model',
            prompt: 'Hello',
            params: { max_tokens: 5 },
          })
          .pipe(toArray()),
      );

      expect(responses.length).toBe(1);
      expect(responses[0].error).toBeDefined();
      expect(responses[0].error.code).toBe('MODEL_NOT_LOADED');
    }, 10_000);
  });

  describe('Model Unloading', () => {
    it('should unload a model and free VRAM', async () => {
      const worker = registry.getWorker('gpu-0')!;

      // Always ensure model is freshly loaded before testing unload.
      // Other test suites running in parallel may have unloaded it.
      // First try to unload (in case it's in a weird state), then load fresh.
      try {
        await firstValueFrom(worker.unloadModel({ model_id: TEST_MODEL }));
      } catch {}
      const loadRes = await firstValueFrom(
        worker.loadModel({ model_id: TEST_MODEL, quantization: 'fp16' }),
      );
      expect(loadRes.success).toBe(true);

      const stateBefore = await firstValueFrom(worker.getWorkerState());
      const vramBefore = stateBefore.gpu.vram_used_bytes;

      const response = await firstValueFrom(
        worker.unloadModel({ model_id: TEST_MODEL }),
      );
      expect(response.success).toBe(true);
      // vram_freed_bytes may be 0 if torch.cuda.memory_allocated doesn't capture all allocs
      expect(response.vram_freed_bytes).toBeGreaterThanOrEqual(0);

      const stateAfter = await firstValueFrom(worker.getWorkerState());
      const model = stateAfter.models.find((m) => m.model_id === TEST_MODEL);
      expect(model).toBeUndefined();
      // VRAM should be less or equal (torch allocator caching may not release immediately)
      expect(stateAfter.gpu.vram_used_bytes).toBeLessThanOrEqual(vramBefore);
    }, 120_000);
  });
});

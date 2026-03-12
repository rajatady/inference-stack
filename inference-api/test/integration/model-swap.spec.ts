/**
 * Integration Test: Model Swap
 *
 * Tests loading model A, inferring, unloading, loading model B, inferring on B.
 * Verifies the full model lifecycle on a single worker — critical for multi-model scenarios.
 *
 * Runs against REAL GPU workers on RunPod via SSH tunnel:
 *   localhost:50051 → RunPod GPU-0 (worker-0)
 *
 * Prerequisites:
 *   1. GPU workers running on RunPod
 *   2. SSH tunnel: ssh -f -N -L 50051:localhost:50051 $RUNPOD_SSH_USER@$RUNPOD_SSH_HOST -p $RUNPOD_SSH_PORT
 *
 * Scope reference: test-scenarios.md — T17 (model swap cost accounting)
 */
import { firstValueFrom, lastValueFrom, toArray } from 'rxjs';
import { WorkerRegistry } from '../../src/worker-orchestrator/worker-registry';
import { ModelManager } from '../../src/worker-orchestrator/model-manager';
import { Router } from '../../src/worker-orchestrator/router';

const WORKER_URL = process.env.GPU_WORKER_0_URL || 'localhost:50051';
const MODEL_A = 'HuggingFaceTB/SmolLM2-135M';
const MODEL_B = 'HuggingFaceTB/SmolLM2-360M';

describe('Model Swap', () => {
  let registry: WorkerRegistry;
  let modelManager: ModelManager;
  let router: Router;

  beforeAll(async () => {
    registry = new WorkerRegistry();
    modelManager = new ModelManager(registry);
    router = new Router(modelManager, registry);

    registry.addWorker({ id: 'swap-worker', url: WORKER_URL });

    // Clean state: unload both models
    const worker = registry.getWorker('swap-worker')!;
    try {
      await firstValueFrom(worker.unloadModel({ model_id: MODEL_A }));
    } catch {}
    try {
      await firstValueFrom(worker.unloadModel({ model_id: MODEL_B }));
    } catch {}

    await registry.pollAllWorkers();
  }, 30_000);

  afterAll(async () => {
    // Cleanup: unload both models
    const worker = registry.getWorker('swap-worker');
    if (worker) {
      try {
        await firstValueFrom(worker.unloadModel({ model_id: MODEL_A }));
      } catch {}
      try {
        await firstValueFrom(worker.unloadModel({ model_id: MODEL_B }));
      } catch {}
    }
    registry.onModuleDestroy();
  });

  describe('Load Model A, infer, verify', () => {
    it('should load SmolLM2-135M onto the worker', async () => {
      const worker = registry.getWorker('swap-worker')!;
      const response = await firstValueFrom(
        worker.loadModel({ model_id: MODEL_A, quantization: 'fp16' }),
      );
      expect(response.success).toBe(true);
      expect(response.capabilities.max_context_length).toBeGreaterThan(0);
    }, 120_000);

    it('should infer successfully with SmolLM2-135M', async () => {
      const worker = registry.getWorker('swap-worker')!;
      const responses = await lastValueFrom(
        worker
          .infer({
            request_id: 'swap-test-a',
            model_id: MODEL_A,
            prompt: 'Hello world',
            params: { max_tokens: 5, temperature: 0.1 },
          })
          .pipe(toArray()),
      );

      const chunks = responses.filter((r) => r.chunk);
      const complete = responses.find((r) => r.complete);
      expect(chunks.length).toBeGreaterThan(0);
      expect(complete).toBeDefined();
      expect(complete.complete.usage.completion_tokens).toBeGreaterThan(0);
    }, 30_000);

    it('should show Model A in worker state', async () => {
      await registry.pollAllWorkers();
      const snap = registry.getSnapshot('swap-worker')!;
      const loadedA = snap.models.find((m) => m.modelId === MODEL_A);
      expect(loadedA).toBeDefined();
      expect(loadedA!.ready).toBe(true);
    });
  });

  describe('Unload Model A, load Model B', () => {
    it('should unload SmolLM2-135M', async () => {
      const worker = registry.getWorker('swap-worker')!;
      const response = await firstValueFrom(
        worker.unloadModel({ model_id: MODEL_A }),
      );
      expect(response.success).toBe(true);
    }, 30_000);

    it('should no longer show Model A in worker state', async () => {
      await registry.pollAllWorkers();
      const snap = registry.getSnapshot('swap-worker')!;
      const loadedA = snap.models.find((m) => m.modelId === MODEL_A);
      expect(loadedA).toBeUndefined();
    });

    it('should load SmolLM2-360M onto the worker', async () => {
      const worker = registry.getWorker('swap-worker')!;
      const response = await firstValueFrom(
        worker.loadModel({ model_id: MODEL_B, quantization: 'fp16' }),
      );
      expect(response.success).toBe(true);
      expect(response.capabilities.max_context_length).toBeGreaterThan(0);
    }, 120_000);

    it('should infer successfully with SmolLM2-360M', async () => {
      const worker = registry.getWorker('swap-worker')!;
      const responses = await lastValueFrom(
        worker
          .infer({
            request_id: 'swap-test-b',
            model_id: MODEL_B,
            prompt: 'The capital of France is',
            params: { max_tokens: 5, temperature: 0.1 },
          })
          .pipe(toArray()),
      );

      const chunks = responses.filter((r) => r.chunk);
      const complete = responses.find((r) => r.complete);
      expect(chunks.length).toBeGreaterThan(0);
      expect(complete).toBeDefined();
      expect(complete.complete.usage.completion_tokens).toBeGreaterThan(0);
    }, 30_000);

    it('should show Model B (not A) in worker state after swap', async () => {
      await registry.pollAllWorkers();
      const snap = registry.getSnapshot('swap-worker')!;
      const loadedA = snap.models.find((m) => m.modelId === MODEL_A);
      const loadedB = snap.models.find((m) => m.modelId === MODEL_B);
      expect(loadedA).toBeUndefined();
      expect(loadedB).toBeDefined();
      expect(loadedB!.ready).toBe(true);
    });
  });

  describe('Router-driven model swap', () => {
    it('should route to the worker with Model B already loaded', async () => {
      await registry.pollAllWorkers();
      const decision = await router.route(MODEL_B);
      expect(decision.action).toBe('direct');
      expect(decision.workerId).toBe('swap-worker');
    });

    it('should auto-load Model A via router (triggers load on same worker)', async () => {
      // Model A is not loaded anywhere — router should trigger load
      const decision = await router.route(MODEL_A);
      expect(decision.action).toBe('load-then-infer');
      expect(decision.worker).toBeDefined();
    }, 120_000);

    it('should now have both models loaded on the worker', async () => {
      await registry.pollAllWorkers();
      const snap = registry.getSnapshot('swap-worker')!;
      const loadedA = snap.models.find((m) => m.modelId === MODEL_A);
      const loadedB = snap.models.find((m) => m.modelId === MODEL_B);
      expect(loadedA).toBeDefined();
      expect(loadedB).toBeDefined();
    });
  });
});

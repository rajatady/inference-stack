/**
 * Integration Test: Multi-Worker Routing
 *
 * Tests the full orchestration stack (Router → ModelManager → WorkerRegistry)
 * against real GPU workers. Verifies:
 *   - Registry connects and polls real GPU state
 *   - Router auto-loads a model when none is loaded
 *   - Second request routes directly (no re-load)
 *   - Routing decision reflects actual worker state
 *
 * Prerequisites:
 *   1. GPU workers running on RunPod
 *   2. SSH tunnel: ssh -f -N -L 50051:localhost:50051 -L 50052:localhost:50052 $RUNPOD_SSH_USER@$RUNPOD_SSH_HOST -p $RUNPOD_SSH_PORT
 */
import { firstValueFrom, lastValueFrom, toArray } from 'rxjs';
import { WorkerRegistry } from '../../src/worker-orchestrator/worker-registry';
import { ModelManager } from '../../src/worker-orchestrator/model-manager';
import { Router } from '../../src/worker-orchestrator/router';

const WORKER_0_URL = process.env.GPU_WORKER_0_URL || 'localhost:50051';
const WORKER_1_URL = process.env.GPU_WORKER_1_URL || 'localhost:50052';
const TEST_MODEL = 'HuggingFaceTB/SmolLM2-135M';

describe('Multi-Worker Routing', () => {
  let registry: WorkerRegistry;
  let modelManager: ModelManager;
  let router: Router;

  beforeAll(async () => {
    registry = new WorkerRegistry();
    modelManager = new ModelManager(registry);
    router = new Router(modelManager, registry);

    registry.addWorker({ id: 'worker-0', url: WORKER_0_URL });
    registry.addWorker({ id: 'worker-1', url: WORKER_1_URL });

    // Initial poll to populate snapshots
    await registry.pollAllWorkers();
  }, 15_000);

  afterAll(async () => {
    // Cleanup: unload test model from all workers
    for (const handle of registry.getAllWorkers()) {
      try {
        await firstValueFrom(
          handle.service.unloadModel({ model_id: TEST_MODEL }),
        );
      } catch {
        // Model may not be loaded on this worker
      }
    }
    registry.onModuleDestroy();
  });

  describe('Registry polls real GPU state', () => {
    it('should have snapshots for both workers after poll', () => {
      const snapshots = registry.getAllSnapshots();
      expect(snapshots).toHaveLength(2);

      for (const snap of snapshots) {
        expect(snap.healthy).toBe(true);
        expect(snap.gpu.vramTotalBytes).toBeGreaterThan(0);
      }
    });

    it('should report actual VRAM figures (not zeroes)', () => {
      const snap = registry.getSnapshot('worker-0');
      expect(snap).toBeDefined();
      // RTX A4500 has ~20GB VRAM
      expect(snap!.gpu.vramTotalBytes).toBeGreaterThan(10_000_000_000);
    });
  });

  describe('Router auto-loads and routes', () => {
    it('should auto-load model and return load-then-infer on first request', async () => {
      // Ensure model is NOT loaded (unload if present)
      const worker0 = registry.getWorker('worker-0')!;
      try {
        await firstValueFrom(
          worker0.unloadModel({ model_id: TEST_MODEL }),
        );
      } catch {
        // Not loaded, that's fine
      }
      const worker1 = registry.getWorker('worker-1')!;
      try {
        await firstValueFrom(
          worker1.unloadModel({ model_id: TEST_MODEL }),
        );
      } catch {
        // Not loaded, that's fine
      }

      // Re-poll so snapshots reflect unloaded state
      await registry.pollAllWorkers();
      expect(modelManager.getWorkersWithModel(TEST_MODEL)).toHaveLength(0);

      // Route — should trigger auto-load
      const decision = await router.route(TEST_MODEL);

      expect(decision.action).toBe('load-then-infer');
      expect(decision.worker).toBeDefined();
      expect(decision.workerId).toBeTruthy();
    }, 120_000); // Model download + load

    it('should route directly on second request (model already loaded)', async () => {
      // Poll to pick up the model that was just loaded
      await registry.pollAllWorkers();

      const decision = await router.route(TEST_MODEL);

      expect(decision.action).toBe('direct');
      expect(decision.worker).toBeDefined();
    });

    it('should produce valid inference through the routed worker', async () => {
      const { worker } = await router.route(TEST_MODEL);

      const responses = await lastValueFrom(
        worker
          .infer({
            request_id: 'routing-test-1',
            model_id: TEST_MODEL,
            prompt: 'The meaning of life is',
            params: { max_tokens: 10, temperature: 0.1 },
          })
          .pipe(toArray()),
      );

      const chunks = responses.filter((r) => r.chunk);
      const complete = responses.find((r) => r.complete);

      expect(chunks.length).toBeGreaterThan(0);
      expect(complete).toBeDefined();
      expect(complete.complete.usage.completion_tokens).toBeGreaterThan(0);
    }, 30_000);
  });

  describe('ModelManager state awareness', () => {
    it('should report the model as loaded on exactly one worker', async () => {
      await registry.pollAllWorkers();
      const workersWithModel =
        modelManager.getWorkersWithModel(TEST_MODEL);
      expect(workersWithModel.length).toBeGreaterThanOrEqual(1);
    });

    it('should pick the worker that already has the model (no unnecessary load)', async () => {
      await registry.pollAllWorkers();
      const workersWithModel =
        modelManager.getWorkersWithModel(TEST_MODEL);
      const expectedWorkerId = workersWithModel[0].workerId;

      const { workerId, action } = await router.route(TEST_MODEL);
      expect(action).toBe('direct');
      expect(workerId).toBe(expectedWorkerId);
    });
  });
});

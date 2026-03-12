import { ModelManager } from './model-manager';
import { WorkerRegistry } from './worker-registry';
import { WorkerSnapshot } from './interfaces';
import { of, throwError } from 'rxjs';

function makeSnapshot(overrides: Partial<WorkerSnapshot> = {}): WorkerSnapshot {
  return {
    workerId: 'w-0',
    healthy: true,
    gpu: {
      vramTotalBytes: 20_000_000_000,
      vramUsedBytes: 2_000_000_000,
      vramAvailableBytes: 18_000_000_000,
      utilization: 0.1,
      temperatureC: 45,
      healthy: true,
    },
    models: [],
    activeInferences: 0,
    queuedInferences: 0,
    lastUpdated: Date.now(),
    ...overrides,
  };
}

function mockRegistry(snapshots: WorkerSnapshot[]) {
  const workers = new Map<string, any>();
  for (const snap of snapshots) {
    workers.set(snap.workerId, {
      service: {
        loadModel: jest.fn(() =>
          of({ success: true, vram_used_bytes: 500_000_000 }),
        ),
        unloadModel: jest.fn(() =>
          of({ success: true, vram_freed_bytes: 500_000_000 }),
        ),
        getWorkerState: jest.fn(),
      },
    });
  }

  return {
    getAllSnapshots: jest.fn(() => snapshots),
    getSnapshot: jest.fn((id: string) => snapshots.find((s) => s.workerId === id)),
    getWorker: jest.fn((id: string) => workers.get(id)?.service),
    getAllWorkers: jest.fn(() => Array.from(workers.values())),
    pollAllWorkers: jest.fn(),
    getCurrentMode: jest.fn(() => 'individual'),
    switchMode: jest.fn(),
    _workers: workers,
  } as unknown as WorkerRegistry & { _workers: Map<string, any> };
}

describe('ModelManager', () => {
  describe('getWorkersWithModel', () => {
    it('returns only workers with the model loaded and ready', () => {
      const registry = mockRegistry([
        makeSnapshot({
          workerId: 'w-0',
          models: [{ modelId: 'model-A', ready: true, vramUsedBytes: 500_000_000 }],
        }),
        makeSnapshot({
          workerId: 'w-1',
          models: [], // no model
        }),
        makeSnapshot({
          workerId: 'w-2',
          models: [{ modelId: 'model-A', ready: false, vramUsedBytes: 500_000_000 }], // loading
        }),
      ]);
      const manager = new ModelManager(registry);

      const result = manager.getWorkersWithModel('model-A');
      expect(result).toHaveLength(1);
      expect(result[0].workerId).toBe('w-0');
    });

    it('excludes unhealthy workers', () => {
      const registry = mockRegistry([
        makeSnapshot({
          workerId: 'w-0',
          healthy: false,
          models: [{ modelId: 'model-A', ready: true, vramUsedBytes: 500_000_000 }],
        }),
      ]);
      const manager = new ModelManager(registry);

      expect(manager.getWorkersWithModel('model-A')).toHaveLength(0);
    });
  });

  describe('ensureModelLoaded', () => {
    it('returns existing worker without calling loadModel when model is loaded', async () => {
      const registry = mockRegistry([
        makeSnapshot({
          workerId: 'w-0',
          models: [{ modelId: 'model-A', ready: true, vramUsedBytes: 500_000_000 }],
          activeInferences: 2,
        }),
      ]);
      const manager = new ModelManager(registry);
      const workerService = registry.getWorker('w-0')!;

      const result = await manager.ensureModelLoaded('model-A');

      expect(result.workerId).toBe('w-0');
      expect(result.worker).toBe(workerService);
      expect(workerService.loadModel).not.toHaveBeenCalled();
    });

    it('prefers worker with fewer active inferences when model on multiple workers', async () => {
      const registry = mockRegistry([
        makeSnapshot({
          workerId: 'w-0',
          models: [{ modelId: 'model-A', ready: true, vramUsedBytes: 500_000_000 }],
          activeInferences: 10,
        }),
        makeSnapshot({
          workerId: 'w-1',
          models: [{ modelId: 'model-A', ready: true, vramUsedBytes: 500_000_000 }],
          activeInferences: 2,
        }),
      ]);
      const manager = new ModelManager(registry);

      const result = await manager.ensureModelLoaded('model-A');
      expect(result.workerId).toBe('w-1');
    });

    it('picks worker with most free VRAM when model needs loading', async () => {
      const registry = mockRegistry([
        makeSnapshot({
          workerId: 'w-0',
          gpu: {
            vramTotalBytes: 20e9, vramUsedBytes: 15e9, vramAvailableBytes: 5e9,
            utilization: 0.5, temperatureC: 50, healthy: true,
          },
        }),
        makeSnapshot({
          workerId: 'w-1',
          gpu: {
            vramTotalBytes: 20e9, vramUsedBytes: 5e9, vramAvailableBytes: 15e9,
            utilization: 0.2, temperatureC: 40, healthy: true,
          },
        }),
      ]);
      const manager = new ModelManager(registry);

      const result = await manager.ensureModelLoaded('model-A');

      expect(result.workerId).toBe('w-1');
      const w1Service = registry.getWorker('w-1')!;
      expect(w1Service.loadModel).toHaveBeenCalledWith(
        expect.objectContaining({ model_id: 'model-A' }),
      );
    });

    it('evicts idle model when VRAM is tight, then loads requested model', async () => {
      const registry = mockRegistry([
        makeSnapshot({
          workerId: 'w-0',
          gpu: {
            vramTotalBytes: 20e9, vramUsedBytes: 19e9, vramAvailableBytes: 1e9,
            utilization: 0.9, temperatureC: 70, healthy: true,
          },
          models: [
            { modelId: 'old-model', ready: true, vramUsedBytes: 5e9 },
          ],
          activeInferences: 0,
        }),
      ]);
      const manager = new ModelManager(registry);
      const w0Service = registry.getWorker('w-0')!;

      // First loadModel fails (OOM), second succeeds (after eviction freed VRAM)
      let loadAttempt = 0;
      (w0Service.loadModel as jest.Mock).mockImplementation(() => {
        loadAttempt++;
        if (loadAttempt === 1) {
          return throwError(() => new Error('CUDA out of memory'));
        }
        return of({ success: true, vram_used_bytes: 500_000_000 });
      });

      const result = await manager.ensureModelLoaded('model-B');

      expect(w0Service.unloadModel).toHaveBeenCalledWith(
        expect.objectContaining({ model_id: 'old-model' }),
      );
      expect(w0Service.loadModel).toHaveBeenCalledTimes(2);
      expect(result.workerId).toBe('w-0');
    });

    it('refuses to evict model with active inferences', async () => {
      const registry = mockRegistry([
        makeSnapshot({
          workerId: 'w-0',
          gpu: {
            vramTotalBytes: 20e9, vramUsedBytes: 19e9, vramAvailableBytes: 1e9,
            utilization: 0.9, temperatureC: 70, healthy: true,
          },
          models: [
            { modelId: 'busy-model', ready: true, vramUsedBytes: 15e9 },
          ],
          activeInferences: 5, // busy — cannot evict
        }),
      ]);
      const manager = new ModelManager(registry);
      const w0Service = registry.getWorker('w-0')!;

      // Load always fails (VRAM tight, and we can't evict)
      (w0Service.loadModel as jest.Mock).mockReturnValue(
        throwError(() => new Error('CUDA out of memory')),
      );

      await expect(manager.ensureModelLoaded('model-B')).rejects.toThrow(
        /no worker.*capacity/i,
      );
      // Should NOT have tried to evict the busy model
      expect(w0Service.unloadModel).not.toHaveBeenCalled();
    });

    it('coalesces concurrent loads for the same model', async () => {
      const registry = mockRegistry([
        makeSnapshot({ workerId: 'w-0' }),
      ]);
      const manager = new ModelManager(registry);

      const w0Service = registry.getWorker('w-0')!;
      let loadCallCount = 0;
      (w0Service.loadModel as jest.Mock).mockImplementation(() => {
        loadCallCount++;
        return of({ success: true, vram_used_bytes: 500_000_000 });
      });

      // Fire two concurrent loads for the same model
      const [r1, r2] = await Promise.all([
        manager.ensureModelLoaded('model-A'),
        manager.ensureModelLoaded('model-A'),
      ]);

      expect(loadCallCount).toBe(1);
      expect(r1.workerId).toBe(r2.workerId);
    });

    it('throws descriptive error when no healthy workers exist', async () => {
      const registry = mockRegistry([
        makeSnapshot({ workerId: 'w-0', healthy: false }),
      ]);
      const manager = new ModelManager(registry);

      await expect(manager.ensureModelLoaded('model-A')).rejects.toThrow(
        /no worker.*capacity/i,
      );
    });
  });
});

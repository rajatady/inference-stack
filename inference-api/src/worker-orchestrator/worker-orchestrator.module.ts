import { Module, OnModuleInit } from '@nestjs/common';
import { WorkerRegistry } from './worker-registry';
import { ModelManager } from './model-manager';
import { Router } from './router';

/**
 * Worker Orchestrator Module
 *
 * Manages GPU workers, model placement, and request routing.
 * Replaces the old GpuWorkerModule (single static client) with
 * dynamic multi-worker support.
 */
@Module({
  providers: [WorkerRegistry, ModelManager, Router],
  exports: [Router, WorkerRegistry, ModelManager],
})
export class WorkerOrchestratorModule implements OnModuleInit {
  constructor(private readonly registry: WorkerRegistry) {}

  async onModuleInit() {
    // Register initial workers based on mode
    const initialMode = process.env.WORKER_MODE || 'individual';

    if (initialMode === 'tensor-parallel') {
      this.registry.setCurrentMode('tensor-parallel');
      const workers = [
        { id: 'tp-worker-0', url: process.env.GPU_TP_WORKER_URL || 'localhost:50051', mode: 'tensor-parallel' as const },
      ];
      for (const w of workers) {
        this.registry.addWorker(w);
      }
    } else {
      const workers = [
        { id: 'worker-0', url: process.env.GPU_WORKER_0_URL || 'localhost:50051', mode: 'individual' as const },
        { id: 'worker-1', url: process.env.GPU_WORKER_1_URL || 'localhost:50052', mode: 'individual' as const },
      ];
      for (const w of workers) {
        this.registry.addWorker(w);
      }
    }

    // Initial poll + start periodic polling
    await this.registry.pollAllWorkers();
    this.registry.startPolling(5000);
  }
}

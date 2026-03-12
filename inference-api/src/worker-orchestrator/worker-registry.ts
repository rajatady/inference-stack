import { Injectable, Logger, OnModuleDestroy } from '@nestjs/common';
import { ClientProxyFactory, Transport } from '@nestjs/microservices';
import { join } from 'path';
import { firstValueFrom } from 'rxjs';
import { GpuWorkerService } from '../gpu-worker/gpu-worker.service';
import {
  WorkerConfig,
  WorkerHandle,
  WorkerMode,
  WorkerSnapshot,
  LoadedModelSnapshot,
} from './interfaces';

const PROTO_PATH = join(__dirname, '../../proto/inference_worker.proto');
const LOADER_OPTIONS = {
  keepCase: true,
  longs: Number,
  enums: String,
  defaults: true,
  oneofs: true,
};

@Injectable()
export class WorkerRegistry implements OnModuleDestroy {
  private readonly logger = new Logger(WorkerRegistry.name);
  private readonly workers = new Map<string, WorkerHandle>();
  private readonly snapshots = new Map<string, WorkerSnapshot>();
  private pollTimer: ReturnType<typeof setInterval> | null = null;
  private currentMode: WorkerMode = 'individual';
  private switching = false;

  /** SSH config for GPU worker management */
  private readonly sshHost = process.env.RUNPOD_SSH_HOST || 'localhost';
  private readonly sshPort = process.env.RUNPOD_SSH_PORT || '22';
  private readonly sshUser = process.env.RUNPOD_SSH_USER || 'root';
  private readonly gpuWorkerPath = process.env.GPU_WORKER_PATH || '/workspace/gpu-worker';
  private readonly hfCachePath = process.env.HF_HOME || '/workspace/huggingface_cache';

  getCurrentMode(): WorkerMode {
    return this.currentMode;
  }

  setCurrentMode(mode: WorkerMode): void {
    this.currentMode = mode;
  }

  isSwitching(): boolean {
    return this.switching;
  }

  /**
   * Switch worker mode: kills current workers via SSH, starts new ones in target mode.
   * - 'individual': 2 separate workers (worker-0 on GPU-0, worker-1 on GPU-1)
   * - 'tensor-parallel': 1 TP worker spanning both GPUs via torchrun
   */
  async switchMode(targetMode: WorkerMode): Promise<void> {
    if (this.currentMode === targetMode) {
      this.logger.log(`Already in ${targetMode} mode`);
      return;
    }
    if (this.switching) {
      throw new Error('Mode switch already in progress');
    }

    this.switching = true;
    this.logger.log(`Switching worker mode: ${this.currentMode} → ${targetMode}`);

    try {
      // 1. Stop polling during switch
      this.stopPolling();

      // 2. Remove all current workers from registry
      const workerIds = Array.from(this.workers.keys());
      for (const id of workerIds) {
        this.removeWorker(id);
      }

      // 3. Kill existing worker processes on RunPod
      // kill python workers and torchrun separately to avoid pkill killing SSH session
      await this.sshExec('kill $(pgrep -f "python.*server.py") 2>/dev/null; kill $(pgrep -f torchrun) 2>/dev/null; true');
      // Wait for processes to die
      await new Promise((r) => setTimeout(r, 3000));

      // 4. Start workers in new mode
      const w0Url = process.env.GPU_WORKER_0_URL || 'localhost:50051';
      const w1Url = process.env.GPU_WORKER_1_URL || 'localhost:50052';
      const tpUrl = process.env.GPU_TP_WORKER_URL || 'localhost:50051';

      if (targetMode === 'tensor-parallel') {
        await this.sshExec(
          `cd ${this.gpuWorkerPath} && HF_HOME=${this.hfCachePath} nohup torchrun --nproc_per_node=2 server.py --port 50051 --worker-id tp-worker-0 --tp > /tmp/tp-worker.log 2>&1 &`,
        );
        // Wait for startup (torchrun init + NCCL setup takes longer)
        await new Promise((r) => setTimeout(r, 15000));
        this.addWorker({ id: 'tp-worker-0', url: tpUrl, mode: 'tensor-parallel' });
      } else {
        await this.sshExec(
          `cd ${this.gpuWorkerPath} && ` +
          `HF_HOME=${this.hfCachePath} CUDA_VISIBLE_DEVICES=0 nohup python3 server.py --port 50051 --gpu-id 0 --worker-id worker-0 > /tmp/worker-0.log 2>&1 & ` +
          `HF_HOME=${this.hfCachePath} CUDA_VISIBLE_DEVICES=1 nohup python3 server.py --port 50052 --gpu-id 0 --worker-id worker-1 > /tmp/worker-1.log 2>&1 &`,
        );
        await new Promise((r) => setTimeout(r, 3000));
        this.addWorker({ id: 'worker-0', url: w0Url, mode: 'individual' });
        this.addWorker({ id: 'worker-1', url: w1Url, mode: 'individual' });
      }

      // 5. Poll and verify health
      await this.pollAllWorkers();
      this.startPolling(5000);

      this.currentMode = targetMode;
      this.logger.log(`Mode switch complete: now in ${targetMode} mode`);
    } finally {
      this.switching = false;
    }
  }

  private async sshExec(command: string): Promise<string> {
    const { execSync } = await import('child_process');
    const { writeFileSync, unlinkSync } = await import('fs');
    const { join } = await import('path');
    const tmpScript = join('/tmp', `ssh-cmd-${Date.now()}.sh`);
    writeFileSync(tmpScript, command, { mode: 0o755 });
    const sshCmd = `cat ${tmpScript} | ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 ${this.sshUser}@${this.sshHost} -p ${this.sshPort} bash`;
    this.logger.log(`SSH: ${command}`);
    try {
      return execSync(sshCmd, { encoding: 'utf-8', timeout: 60000 });
    } catch (err) {
      this.logger.warn(`SSH command failed: ${err.message}`);
      return '';
    } finally {
      try { unlinkSync(tmpScript); } catch {}
    }
  }

  addWorker(config: WorkerConfig): GpuWorkerService {
    // If worker with this ID exists, close old one first
    if (this.workers.has(config.id)) {
      this.removeWorker(config.id);
    }

    const { service, close } = this.createWorkerService(config);
    const handle: WorkerHandle = { config, service, close };
    this.workers.set(config.id, handle);
    this.logger.log(`Added worker ${config.id} at ${config.url}`);
    return service;
  }

  removeWorker(id: string): void {
    const handle = this.workers.get(id);
    if (!handle) return;

    handle.close();
    this.workers.delete(id);
    this.snapshots.delete(id);
    this.logger.log(`Removed worker ${id}`);
  }

  getWorker(id: string): GpuWorkerService | undefined {
    return this.workers.get(id)?.service;
  }

  getAllWorkers(): WorkerHandle[] {
    return Array.from(this.workers.values());
  }

  getSnapshot(id: string): WorkerSnapshot | undefined {
    return this.snapshots.get(id);
  }

  getAllSnapshots(): WorkerSnapshot[] {
    return Array.from(this.snapshots.values());
  }

  async pollAllWorkers(): Promise<void> {
    const entries = Array.from(this.workers.entries());
    await Promise.all(
      entries.map(([id, handle]) => this.pollWorker(id, handle)),
    );
  }

  startPolling(intervalMs = 5000): void {
    this.stopPolling();
    this.pollTimer = setInterval(() => this.pollAllWorkers(), intervalMs);
  }

  stopPolling(): void {
    if (this.pollTimer) {
      clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
  }

  onModuleDestroy(): void {
    this.stopPolling();
    for (const [id] of this.workers) {
      this.removeWorker(id);
    }
  }

  /**
   * Creates a GpuWorkerService with its own gRPC client.
   * Extracted as a method so tests can override it.
   */
  protected createWorkerService(
    config: WorkerConfig,
  ): { service: GpuWorkerService; close: () => void } {
    const grpcClient = ClientProxyFactory.create({
      transport: Transport.GRPC,
      options: {
        package: 'inference.worker.v1',
        protoPath: PROTO_PATH,
        url: config.url,
        loader: LOADER_OPTIONS,
      },
    });

    const service = new GpuWorkerService(grpcClient as any);
    service.init();

    return {
      service,
      close: () => grpcClient.close(),
    };
  }

  private async pollWorker(id: string, handle: WorkerHandle): Promise<void> {
    try {
      const state = await firstValueFrom(handle.service.getWorkerState());
      this.snapshots.set(id, this.toSnapshot(id, state));
    } catch (err) {
      this.logger.warn(`Poll failed for worker ${id}: ${err.message}`);
      // Preserve last known GPU info if we have it, but mark unhealthy
      const existing = this.snapshots.get(id);
      this.snapshots.set(id, {
        workerId: id,
        healthy: false,
        gpu: existing?.gpu ?? {
          vramTotalBytes: 0,
          vramUsedBytes: 0,
          vramAvailableBytes: 0,
          utilization: 0,
          temperatureC: 0,
          healthy: false,
        },
        models: existing?.models ?? [],
        activeInferences: existing?.activeInferences ?? 0,
        queuedInferences: existing?.queuedInferences ?? 0,
        lastUpdated: Date.now(),
      });
    }
  }

  private toSnapshot(workerId: string, state: any): WorkerSnapshot {
    const gpu = state.gpu ?? {};
    const models: LoadedModelSnapshot[] = (state.models ?? []).map(
      (m: any) => ({
        modelId: m.model_id,
        ready: m.ready,
        vramUsedBytes: m.vram_used_bytes ?? 0,
      }),
    );

    return {
      workerId,
      healthy: true,
      gpu: {
        vramTotalBytes: gpu.vram_total_bytes ?? 0,
        vramUsedBytes: gpu.vram_used_bytes ?? 0,
        vramAvailableBytes: gpu.vram_available_bytes ?? 0,
        utilization: gpu.gpu_utilization ?? 0,
        temperatureC: gpu.gpu_temperature_c ?? 0,
        healthy: gpu.healthy ?? true,
      },
      models,
      activeInferences: state.active_inferences ?? 0,
      queuedInferences: state.queued_inferences ?? 0,
      lastUpdated: Date.now(),
    };
  }
}

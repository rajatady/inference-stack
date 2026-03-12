import { ClientGrpc } from '@nestjs/microservices';
import { Observable } from 'rxjs';

/**
 * gRPC service interface matching the InferenceWorker proto definition.
 * Using `any` for message types since we're using proto-loader (dynamic).
 */
interface InferenceWorkerGrpc {
  health(data: Record<string, never>): Observable<any>;
  loadModel(data: any): Observable<any>;
  unloadModel(data: any): Observable<any>;
  getWorkerState(data: Record<string, never>): Observable<any>;
  watchWorkerState(data: any): Observable<any>;
  infer(data: any): Observable<any>;
  batchInfer(data: any): Observable<any>;
  getCacheEntries(data: any): Observable<any>;
  evictCache(data: any): Observable<any>;
}

/**
 * Thin wrapper around a single GPU worker's gRPC client.
 *
 * This is a plain class — NOT a NestJS provider. One instance is created
 * per worker by WorkerRegistry, each with its own gRPC connection.
 */
export class GpuWorkerService {
  private worker: InferenceWorkerGrpc;

  constructor(private readonly client: ClientGrpc) {}

  init(): void {
    this.worker =
      this.client.getService<InferenceWorkerGrpc>('InferenceWorker');
  }

  health(): Observable<any> {
    return this.worker.health({});
  }

  loadModel(request: {
    model_id: string;
    model_path?: string;
    quantization?: string;
    estimated_vram_bytes?: number;
    max_batch_size?: number;
  }): Observable<any> {
    return this.worker.loadModel(request);
  }

  unloadModel(request: {
    model_id: string;
    force?: boolean;
  }): Observable<any> {
    return this.worker.unloadModel(request);
  }

  getWorkerState(): Observable<any> {
    return this.worker.getWorkerState({});
  }

  watchWorkerState(minIntervalMs?: number): Observable<any> {
    return this.worker.watchWorkerState({
      min_interval_ms: minIntervalMs ?? 100,
    });
  }

  infer(request: {
    request_id: string;
    model_id: string;
    token_ids?: number[];
    prompt?: string;
    params?: {
      max_tokens?: number;
      temperature?: number;
      top_p?: number;
    };
    cache_hint?: {
      session_id?: string;
      prefix_hash?: string;
      prefix_length?: number;
    };
  }): Observable<any> {
    return this.worker.infer(request);
  }

  batchInfer(request: {
    requests: Array<{
      request_id: string;
      model_id: string;
      token_ids?: number[];
      prompt?: string;
      params?: {
        max_tokens?: number;
        temperature?: number;
        top_p?: number;
      };
    }>;
  }): Observable<any> {
    return this.worker.batchInfer(request);
  }

  getCacheEntries(request: {
    model_id?: string;
    session_id?: string;
  }): Observable<any> {
    return this.worker.getCacheEntries(request);
  }

  evictCache(request: { cache_id: string }): Observable<any> {
    return this.worker.evictCache(request);
  }
}

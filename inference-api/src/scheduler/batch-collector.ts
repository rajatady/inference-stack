import { Injectable, Logger } from '@nestjs/common';

export interface BatchConfig {
  /** Time window in ms to accumulate requests before dispatching a batch */
  windowMs: number;
  /** Max requests per batch — dispatch immediately when reached */
  maxBatchSize: number;
  /** Max ratio between longest and shortest prompt in a batch */
  maxSeqLengthRatio: number;
  /** Whether batching is enabled */
  enabled: boolean;
  /** Called when a batch of >1 requests is ready. If not set, falls back to individual dispatch. */
  batchDispatch?: (requests: BatchableRequest[]) => void;
}

export const DEFAULT_BATCH_CONFIG: BatchConfig = {
  windowMs: 50,
  maxBatchSize: 256,
  maxSeqLengthRatio: 4.0,
  enabled: true,
};

export interface BatchableRequest {
  id: string;
  modelId: string;
  estimatedTokens: number;
  dispatch: () => void;
  /** Opaque reference to the original request — used by batchDispatch */
  context?: any;
}

interface Bucket {
  requests: BatchableRequest[];
  timer: ReturnType<typeof setTimeout> | null;
}

@Injectable()
export class BatchCollector {
  private readonly logger = new Logger(BatchCollector.name);
  private config: BatchConfig;

  /** Per-model buckets for accumulating compatible requests */
  private readonly buckets = new Map<string, Bucket[]>();

  constructor() {
    this.config = { ...DEFAULT_BATCH_CONFIG };
  }

  setConfig(config: Partial<BatchConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Submit a request for batching.
   * If batching is disabled, dispatches immediately.
   * Otherwise, adds to a compatible bucket and dispatches when:
   *   - maxBatchSize is reached (immediate), or
   *   - windowMs timer fires
   */
  submit(request: BatchableRequest): void {
    if (!this.config.enabled) {
      request.dispatch();
      return;
    }

    const bucket = this.findOrCreateBucket(request);
    bucket.requests.push(request);

    // Dispatch immediately if batch is full
    if (bucket.requests.length >= this.config.maxBatchSize) {
      this.dispatchBucket(request.modelId, bucket);
      return;
    }

    // Start timer if this is the first request in the bucket
    if (bucket.requests.length === 1) {
      bucket.timer = setTimeout(() => {
        this.dispatchBucket(request.modelId, bucket);
      }, this.config.windowMs);
    }
  }

  /**
   * Find a compatible bucket or create a new one.
   * Compatibility: same model, and token count ratio within maxSeqLengthRatio.
   */
  private findOrCreateBucket(request: BatchableRequest): Bucket {
    let modelBuckets = this.buckets.get(request.modelId);
    if (!modelBuckets) {
      modelBuckets = [];
      this.buckets.set(request.modelId, modelBuckets);
    }

    // Find a compatible bucket
    for (const bucket of modelBuckets) {
      if (bucket.requests.length >= this.config.maxBatchSize) continue;
      if (this.isCompatible(bucket, request)) {
        return bucket;
      }
    }

    // No compatible bucket — create a new one
    const newBucket: Bucket = { requests: [], timer: null };
    modelBuckets.push(newBucket);
    return newBucket;
  }

  private isCompatible(bucket: Bucket, request: BatchableRequest): boolean {
    if (bucket.requests.length === 0) return true;

    const tokens = bucket.requests.map((r) => r.estimatedTokens);
    const min = Math.min(...tokens, request.estimatedTokens);
    const max = Math.max(...tokens, request.estimatedTokens);

    if (min === 0) return true;
    return max / min <= this.config.maxSeqLengthRatio;
  }

  private dispatchBucket(modelId: string, bucket: Bucket): void {
    if (bucket.timer) {
      clearTimeout(bucket.timer);
      bucket.timer = null;
    }

    const requests = bucket.requests.splice(0);

    // Remove empty bucket from model's list
    const modelBuckets = this.buckets.get(modelId);
    if (modelBuckets) {
      const idx = modelBuckets.indexOf(bucket);
      if (idx !== -1) modelBuckets.splice(idx, 1);
      if (modelBuckets.length === 0) this.buckets.delete(modelId);
    }

    this.logger.debug(
      `Dispatching batch of ${requests.length} for model ${modelId}`,
    );

    if (requests.length > 1 && this.config.batchDispatch) {
      this.config.batchDispatch(requests);
    } else {
      for (const req of requests) {
        req.dispatch();
      }
    }
  }

  /** Cancel all pending timers (for cleanup) */
  destroy(): void {
    for (const modelBuckets of this.buckets.values()) {
      for (const bucket of modelBuckets) {
        if (bucket.timer) {
          clearTimeout(bucket.timer);
          bucket.timer = null;
        }
      }
    }
    this.buckets.clear();
  }
}

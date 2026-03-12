import { Injectable, Logger, HttpException, HttpStatus, OnModuleDestroy } from '@nestjs/common';
import { v4 as uuidv4 } from 'uuid';
import { Router } from '../worker-orchestrator/router';
import { WorkerRegistry } from '../worker-orchestrator/worker-registry';
import { TokenizerService } from '../tokenizer/tokenizer.service';
import { BatchCollector } from './batch-collector';
import {
  Priority,
  QueuedRequest,
  DEFAULT_SCHEDULER_CONFIG,
} from './interfaces';
import { CreateCompletionDto } from '../completions/dto/create-completion.dto';

@Injectable()
export class SchedulerService implements OnModuleDestroy {
  private readonly logger = new Logger(SchedulerService.name);

  /** Per-user FIFO queues */
  private readonly userQueues = new Map<string, QueuedRequest[]>();
  /** Round-robin index: tracks which user to dequeue from next */
  private userKeys: string[] = [];
  private rrIndex = 0;

  /** Counters */
  private totalQueued = 0;
  private totalQueuedTokens = 0;
  private activeCount = 0;

  /** Active requests (for cancellation) */
  private readonly activeRequests = new Map<string, QueuedRequest>();

  /** Config (mutable for testing) */
  private maxQueueDepth = DEFAULT_SCHEDULER_CONFIG.maxQueueDepth;
  private maxQueuedTokens = DEFAULT_SCHEDULER_CONFIG.maxQueuedTokens;
  private maxConcurrent = Infinity; // no limit by default; workers self-regulate
  private requestTimeoutMs = DEFAULT_SCHEDULER_CONFIG.requestTimeoutMs;

  /** Aging timer */
  private agingTimer: ReturnType<typeof setInterval> | null = null;
  private agingBoostPerSecond = DEFAULT_SCHEDULER_CONFIG.agingBoostPerSecond;

  constructor(
    private readonly router: Router,
    private readonly registry: WorkerRegistry,
    private readonly tokenizer: TokenizerService,
    private readonly batchCollector: BatchCollector,
  ) {
    // Wire batch dispatch callback so BatchCollector can trigger batch inference
    this.batchCollector.setConfig({
      batchDispatch: (batchRequests) => {
        const queuedRequests = batchRequests
          .map((br) => br.context as QueuedRequest)
          .filter(Boolean);
        if (queuedRequests.length > 1) {
          this.dispatchBatch(queuedRequests);
        } else if (queuedRequests.length === 1) {
          this.dispatchRequest(queuedRequests[0]);
        }
      },
    });
  }

  onModuleDestroy(): void {
    if (this.agingTimer) {
      clearInterval(this.agingTimer);
      this.agingTimer = null;
    }
    this.batchCollector.destroy();
  }

  // ── Config setters (for testing) ──────────────────────────

  setMaxQueueDepth(n: number): void {
    this.maxQueueDepth = n;
  }

  setMaxQueuedTokens(n: number): void {
    this.maxQueuedTokens = n;
  }

  setMaxConcurrent(n: number): void {
    this.maxConcurrent = n;
  }

  setRequestTimeout(ms: number): void {
    this.requestTimeoutMs = ms;
  }

  // ── Public API ────────────────────────────────────────────

  /**
   * Enqueue a non-streaming request. Returns a promise that resolves
   * with the completion result when inference finishes.
   */
  async enqueue(opts: {
    dto: CreateCompletionDto;
    userId: string;
    priority: Priority;
  }): Promise<any> {
    const textForEstimation = opts.dto.messages?.length
      ? opts.dto.messages.map((m) => m.content).join(' ')
      : opts.dto.prompt || '';
    const estimatedTokens = this.tokenizer.estimateTokenCount(textForEstimation);
    this.admissionCheck(estimatedTokens);

    return new Promise<any>((resolve, reject) => {
      const request: QueuedRequest = {
        id: uuidv4(),
        dto: opts.dto,
        userId: opts.userId,
        priority: opts.priority,
        estimatedTokens,
        enqueuedAt: Date.now(),
        effectivePriority: opts.priority,
        state: 'queued',
        resolve,
        reject,
      };

      this.addToQueue(request);
      this.tryDispatch();
    });
  }

  /**
   * Cancel a queued or active request.
   */
  cancel(requestId: string): void {
    // Check queued requests
    for (const [userId, queue] of this.userQueues) {
      const idx = queue.findIndex((r) => r.id === requestId);
      if (idx !== -1) {
        const request = queue.splice(idx, 1)[0];
        this.totalQueued--;
        this.totalQueuedTokens -= request.estimatedTokens;
        if (queue.length === 0) {
          this.userQueues.delete(userId);
          this.refreshUserKeys();
        }
        request.state = 'cancelled';
        request.reject({ error: { message: 'Request cancelled' } });
        return;
      }
    }

    // Check active requests
    const active = this.activeRequests.get(requestId);
    if (active) {
      active.subscription?.unsubscribe();
      active.state = 'cancelled';
      this.finishRequest(active);
      active.reject({ error: { message: 'Request cancelled' } });
    }
  }

  /**
   * Get IDs of all queued requests (for testing/debugging).
   */
  getQueuedRequestIds(): string[] {
    const ids: string[] = [];
    for (const queue of this.userQueues.values()) {
      for (const r of queue) {
        ids.push(r.id);
      }
    }
    return ids;
  }

  /**
   * Get scheduler stats.
   */
  getStats(): { queueDepth: number; totalQueuedTokens: number; activeCount: number } {
    return {
      queueDepth: this.totalQueued,
      totalQueuedTokens: this.totalQueuedTokens,
      activeCount: this.activeCount,
    };
  }

  // ── Internal ──────────────────────────────────────────────

  private admissionCheck(estimatedTokens: number): void {
    if (this.totalQueued >= this.maxQueueDepth) {
      throw new HttpException(
        {
          error: {
            message: 'Too many requests — queue is full',
            type: 'rate_limit',
          },
          retryAfter: this.estimateRetryAfter(),
        },
        HttpStatus.TOO_MANY_REQUESTS,
      );
    }

    if (this.totalQueuedTokens + estimatedTokens > this.maxQueuedTokens) {
      throw new HttpException(
        {
          error: {
            message: 'Token budget exceeded — too many tokens queued',
            type: 'rate_limit',
          },
          retryAfter: this.estimateRetryAfter(),
        },
        HttpStatus.TOO_MANY_REQUESTS,
      );
    }
  }

  private addToQueue(request: QueuedRequest): void {
    let queue = this.userQueues.get(request.userId);
    if (!queue) {
      queue = [];
      this.userQueues.set(request.userId, queue);
      this.refreshUserKeys();
    }
    queue.push(request);
    this.totalQueued++;
    this.totalQueuedTokens += request.estimatedTokens;
  }

  private refreshUserKeys(): void {
    this.userKeys = Array.from(this.userQueues.keys());
    if (this.rrIndex >= this.userKeys.length) {
      this.rrIndex = 0;
    }
  }

  /**
   * Try to dispatch the next request from the queue.
   * Uses priority-aware round-robin: pick the highest-priority request
   * across users, round-robin within the same priority tier.
   */
  private tryDispatch(): void {
    while (this.activeCount < this.maxConcurrent && this.totalQueued > 0) {
      const request = this.dequeueNext();
      if (!request) break;
      this.batchCollector.submit({
        id: request.id,
        modelId: request.dto.model,
        estimatedTokens: request.estimatedTokens,
        dispatch: () => this.dispatchRequest(request),
        context: request,
      });
    }
  }

  /**
   * Dequeue: pick highest priority, round-robin within tier.
   */
  private dequeueNext(): QueuedRequest | null {
    if (this.userKeys.length === 0) return null;

    // Find the best (lowest) effective priority across all queue heads
    let bestPriority = Infinity;
    for (const key of this.userKeys) {
      const queue = this.userQueues.get(key)!;
      if (queue.length > 0 && queue[0].effectivePriority < bestPriority) {
        bestPriority = queue[0].effectivePriority;
      }
    }

    // Round-robin among users whose head matches the best priority (within 0.5 tolerance for aging)
    const startIdx = this.rrIndex;
    for (let i = 0; i < this.userKeys.length; i++) {
      const idx = (startIdx + i) % this.userKeys.length;
      const key = this.userKeys[idx];
      const queue = this.userQueues.get(key)!;
      if (queue.length > 0 && queue[0].effectivePriority <= bestPriority + 0.5) {
        const request = queue.shift()!;
        this.totalQueued--;
        this.totalQueuedTokens -= request.estimatedTokens;
        if (queue.length === 0) {
          this.userQueues.delete(key);
          this.refreshUserKeys();
        }
        this.rrIndex = (idx + 1) % Math.max(this.userKeys.length, 1);
        return request;
      }
    }

    return null;
  }

  private async dispatchRequest(request: QueuedRequest): Promise<void> {
    request.state = 'routing';
    request.dispatchStartTime = Date.now();
    this.activeCount++;
    this.activeRequests.set(request.id, request);

    try {
      const { worker, workerId } = await this.router.route(request.dto.model);
      request.routingTimeMs = Date.now() - request.dispatchStartTime;
      request.state = 'active';

      let fullText = '';
      let usage: any = null;
      let finishReason = '';

      // Pass image data if present (for vision models)
      const imageData = request.dto.images?.[0]
        ? Buffer.from(request.dto.images[0], 'base64')
        : undefined;

      const subscription = worker
        .infer({
          request_id: request.id,
          model_id: request.dto.model,
          prompt: request.dto.prompt || '',
          params: {
            max_tokens: request.dto.max_tokens ?? 50,
            temperature: request.dto.temperature ?? 1.0,
            top_p: request.dto.top_p ?? 1.0,
          },
          ...(imageData && { image_data: imageData, image_mime_type: 'image/png' }),
          ...(request.dto.messages?.length && {
            messages: request.dto.messages.map((m) => ({ role: m.role, content: m.content })),
          }),
          ...(request.dto.session_id && {
            cache_hint: { session_id: request.dto.session_id },
          }),
        })
        .subscribe({
          next: (response) => {
            if (response.chunk) {
              fullText += response.chunk.text || '';
            }
            if (response.complete) {
              finishReason = response.complete.finish_reason;
              usage = response.complete.usage;
            }
            if (response.error) {
              this.finishRequest(request);
              request.reject({
                error: {
                  message: response.error.message,
                  code: response.error.code,
                },
              });
            }
          },
          complete: () => {
            request.state = 'completed';
            this.finishRequest(request);
            request.resolve({
              id: request.id,
              object: 'text_completion',
              created: Math.floor(Date.now() / 1000),
              model: request.dto.model,
              choices: [
                { text: fullText, index: 0, finish_reason: finishReason },
              ],
              usage: {
                prompt_tokens: usage?.prompt_tokens ?? 0,
                completion_tokens: usage?.completion_tokens ?? 0,
                total_tokens:
                  (usage?.prompt_tokens ?? 0) + (usage?.completion_tokens ?? 0),
                prefill_time_ms: usage?.prefill_time_ms ?? 0,
                decode_time_ms: usage?.decode_time_ms ?? 0,
                total_time_ms: usage?.total_time_ms ?? 0,
                cached_tokens: usage?.cached_tokens ?? 0,
                cache_load_ms: usage?.cache_load_ms ?? 0,
                cache_save_ms: usage?.cache_save_ms ?? 0,
              },
              _timing: {
                queueWaitMs: (request.dispatchStartTime ?? request.enqueuedAt) - request.enqueuedAt,
                routingTimeMs: request.routingTimeMs ?? 0,
              },
              workerId,
            });
          },
          error: (err) => {
            request.state = 'error';
            this.finishRequest(request);
            request.reject({
              error: {
                message: err.message || 'Inference failed',
                type: 'server_error',
              },
            });
          },
        });

      request.subscription = subscription;

      // Per-request timeout: reject if inference takes too long
      if (this.requestTimeoutMs > 0) {
        request.timeoutTimer = setTimeout(() => {
          request.subscription?.unsubscribe();
          request.state = 'timeout';
          this.finishRequest(request);
          request.reject({
            error: { message: 'Request timed out', type: 'timeout' },
          });
        }, this.requestTimeoutMs);
      }
    } catch (err) {
      request.state = 'error';
      this.finishRequest(request);
      request.reject(err);
    }
  }

  /**
   * Dispatch a batch of requests to the same worker using batchInfer.
   * All requests must target the same model (BatchCollector ensures this).
   */
  private async dispatchBatch(requests: QueuedRequest[]): Promise<void> {
    const now = Date.now();

    // Mark all requests as routing/active
    for (const req of requests) {
      req.state = 'routing';
      req.dispatchStartTime = now;
      this.activeCount++;
      this.activeRequests.set(req.id, req);
    }

    try {
      const { worker, workerId } = await this.router.route(requests[0].dto.model);
      const routingTimeMs = Date.now() - now;

      for (const req of requests) {
        req.routingTimeMs = routingTimeMs;
        req.state = 'active';
      }

      // Build map to track per-request results
      const requestMap = new Map<string, { request: QueuedRequest; text: string; usage: any }>();
      for (const req of requests) {
        requestMap.set(req.id, { request: req, text: '', usage: null });
      }

      const subscription = worker
        .batchInfer({
          requests: requests.map((req) => ({
            request_id: req.id,
            model_id: req.dto.model,
            prompt: req.dto.prompt || '',
            params: {
              max_tokens: req.dto.max_tokens ?? 50,
              temperature: req.dto.temperature ?? 1.0,
              top_p: req.dto.top_p ?? 1.0,
            },
            ...(req.dto.messages?.length && {
              messages: req.dto.messages.map((m) => ({ role: m.role, content: m.content })),
            }),
            ...(req.dto.session_id && {
              cache_hint: { session_id: req.dto.session_id },
            }),
          })),
        })
        .subscribe({
          next: (response) => {
            const entry = requestMap.get(response.request_id);
            if (!entry) return;

            if (response.chunk) {
              entry.text += response.chunk.text || '';
            }
            if (response.complete) {
              entry.usage = response.complete.usage;
              entry.request.state = 'completed';
              this.finishRequest(entry.request);
              entry.request.resolve({
                id: entry.request.id,
                object: 'text_completion',
                created: Math.floor(Date.now() / 1000),
                model: entry.request.dto.model,
                choices: [
                  { text: entry.text, index: 0, finish_reason: response.complete.finish_reason },
                ],
                usage: {
                  prompt_tokens: entry.usage?.prompt_tokens ?? 0,
                  completion_tokens: entry.usage?.completion_tokens ?? 0,
                  total_tokens:
                    (entry.usage?.prompt_tokens ?? 0) + (entry.usage?.completion_tokens ?? 0),
                  prefill_time_ms: entry.usage?.prefill_time_ms ?? 0,
                  decode_time_ms: entry.usage?.decode_time_ms ?? 0,
                  total_time_ms: entry.usage?.total_time_ms ?? 0,
                  cached_tokens: entry.usage?.cached_tokens ?? 0,
                },
                _timing: {
                  queueWaitMs: (entry.request.dispatchStartTime ?? entry.request.enqueuedAt) - entry.request.enqueuedAt,
                  routingTimeMs: entry.request.routingTimeMs ?? 0,
                },
                workerId,
              });
            }
            if (response.error) {
              this.finishRequest(entry.request);
              entry.request.reject({
                error: {
                  message: response.error.message,
                  code: response.error.code,
                },
              });
            }
          },
          complete: () => {
            // If any requests weren't completed by individual messages, resolve them now
            for (const entry of requestMap.values()) {
              if (entry.request.state === 'active') {
                entry.request.state = 'completed';
                this.finishRequest(entry.request);
                entry.request.resolve({
                  id: entry.request.id,
                  object: 'text_completion',
                  created: Math.floor(Date.now() / 1000),
                  model: entry.request.dto.model,
                  choices: [
                    { text: entry.text, index: 0, finish_reason: 'STOP' },
                  ],
                  usage: {
                    prompt_tokens: entry.usage?.prompt_tokens ?? 0,
                    completion_tokens: entry.usage?.completion_tokens ?? 0,
                    total_tokens: 0,
                    prefill_time_ms: entry.usage?.prefill_time_ms ?? 0,
                    decode_time_ms: entry.usage?.decode_time_ms ?? 0,
                    total_time_ms: entry.usage?.total_time_ms ?? 0,
                    cached_tokens: 0,
                  },
                  workerId,
                });
              }
            }
          },
          error: (err) => {
            // Reject all pending requests in the batch
            for (const entry of requestMap.values()) {
              if (entry.request.state !== 'completed' && entry.request.state !== 'error') {
                entry.request.state = 'error';
                this.finishRequest(entry.request);
                entry.request.reject({
                  error: {
                    message: err.message || 'Batch inference failed',
                    type: 'server_error',
                  },
                });
              }
            }
          },
        });

      // Store subscription + per-request timeout on each request
      for (const req of requests) {
        req.subscription = subscription;
        if (this.requestTimeoutMs > 0) {
          req.timeoutTimer = setTimeout(() => {
            req.state = 'timeout';
            this.finishRequest(req);
            req.reject({
              error: { message: 'Request timed out', type: 'timeout' },
            });
          }, this.requestTimeoutMs);
        }
      }

      this.logger.log(
        `Dispatched batch of ${requests.length} to ${workerId} for model ${requests[0].dto.model}`,
      );
    } catch (err) {
      // Reject all requests on routing failure
      for (const req of requests) {
        req.state = 'error';
        this.finishRequest(req);
        req.reject(err);
      }
    }
  }

  private finishRequest(request: QueuedRequest): void {
    if (!this.activeRequests.has(request.id)) return; // already finished (idempotent)
    if (request.timeoutTimer) clearTimeout(request.timeoutTimer);
    this.activeRequests.delete(request.id);
    this.activeCount--;
    // Try to dispatch next queued request now that a slot freed up
    this.tryDispatch();
  }

  private estimateRetryAfter(): number {
    // Rough estimate: assume 1s per active request
    return Math.max(1, Math.ceil(this.activeCount));
  }
}

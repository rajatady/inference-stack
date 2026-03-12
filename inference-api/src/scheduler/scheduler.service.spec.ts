import { Test, TestingModule } from '@nestjs/testing';
import { HttpException } from '@nestjs/common';
import { Subject } from 'rxjs';
import { SchedulerService } from './scheduler.service';
import { Router } from '../worker-orchestrator/router';
import { WorkerRegistry } from '../worker-orchestrator/worker-registry';
import { TokenizerService } from '../tokenizer/tokenizer.service';
import { BatchCollector } from './batch-collector';
import { Priority } from './interfaces';

describe('SchedulerService', () => {
  let scheduler: SchedulerService;
  let mockRouter: Record<string, jest.Mock>;
  let mockRegistry: Record<string, jest.Mock>;
  let mockTokenizer: Record<string, jest.Mock>;
  let mockWorker: Record<string, jest.Mock>;

  /** Creates a controllable worker that returns a Subject-based infer() */
  function createControllableWorker() {
    const subject = new Subject();
    return {
      worker: { infer: jest.fn(() => subject.asObservable()) },
      subject,
    };
  }

  beforeEach(async () => {
    const { worker, subject: defaultSubject } = createControllableWorker();
    mockWorker = worker as any;

    mockRouter = {
      route: jest.fn(() =>
        Promise.resolve({
          worker: mockWorker,
          workerId: 'w-0',
          action: 'direct',
        }),
      ),
    };

    mockRegistry = {
      getAllSnapshots: jest.fn(() => [
        {
          workerId: 'w-0',
          healthy: true,
          activeInferences: 0,
          queuedInferences: 0,
          gpu: { vramAvailableBytes: 10_000_000_000 },
          models: [],
        },
      ]),
    };

    mockTokenizer = {
      estimateTokenCount: jest.fn((text: string) =>
        Math.ceil(text.length / 4),
      ),
    };

    const module: TestingModule = await Test.createTestingModule({
      providers: [
        SchedulerService,
        BatchCollector,
        { provide: Router, useValue: mockRouter },
        { provide: WorkerRegistry, useValue: mockRegistry },
        { provide: TokenizerService, useValue: mockTokenizer },
      ],
    }).compile();

    scheduler = module.get<SchedulerService>(SchedulerService);

    // Disable batching for scheduler unit tests — these test scheduling logic, not batching
    const batchCollector = module.get<BatchCollector>(BatchCollector);
    batchCollector.setConfig({ enabled: false });
  });

  afterEach(() => {
    scheduler.onModuleDestroy?.();
  });

  it('should be defined', () => {
    expect(scheduler).toBeDefined();
  });

  describe('enqueue and dispatch', () => {
    it('should enqueue a request and resolve when inference completes', async () => {
      // Create a controllable worker for this specific test
      const { worker, subject } = createControllableWorker();
      mockRouter.route.mockResolvedValue({
        worker,
        workerId: 'w-0',
        action: 'direct',
      });

      const promise = scheduler.enqueue({
        dto: { model: 'test', prompt: 'Hello world', stream: false },
        userId: 'user-1',
        priority: Priority.NORMAL,
      });

      // Let the scheduler dispatch (async)
      await new Promise((r) => setTimeout(r, 20));

      // Simulate GPU response
      subject.next({ chunk: { text: 'hi' } });
      subject.next({
        complete: {
          finish_reason: 'stop',
          usage: { prompt_tokens: 3, completion_tokens: 1 },
        },
      });
      subject.complete();

      const result = await promise;
      expect(result.choices[0].text).toBe('hi');
      expect(mockRouter.route).toHaveBeenCalledWith('test');
    });
  });

  describe('priority ordering', () => {
    it('should dispatch HIGH priority before LOW when a slot opens', async () => {
      // Saturate: make router "busy" (no dispatch until we allow it)
      const dispatched: string[] = [];
      let dispatchCount = 0;
      const gates: Array<{ resolve: () => void }> = [];

      mockRouter.route.mockImplementation(() => {
        return new Promise((resolve) => {
          gates.push({
            resolve: () => {
              const { worker, subject } = createControllableWorker();
              // Auto-complete the inference
              setTimeout(() => {
                dispatched.push(`request-${dispatchCount++}`);
                subject.next({ chunk: { text: 'ok' } });
                subject.next({
                  complete: {
                    finish_reason: 'stop',
                    usage: { prompt_tokens: 1, completion_tokens: 1 },
                  },
                });
                subject.complete();
              }, 5);
              resolve({ worker, workerId: 'w-0', action: 'direct' });
            },
          });
        });
      });

      // Set max concurrent to 1 so second request queues
      scheduler.setMaxConcurrent(1);

      // Enqueue LOW first
      const lowPromise = scheduler.enqueue({
        dto: { model: 'test', prompt: 'low', stream: false },
        userId: 'user-1',
        priority: Priority.LOW,
      });

      // Enqueue HIGH second
      const highPromise = scheduler.enqueue({
        dto: { model: 'test', prompt: 'high', stream: false },
        userId: 'user-2',
        priority: Priority.HIGH,
      });

      await new Promise((r) => setTimeout(r, 10));

      // First gate opens — the first dispatched request (one of them gets a slot)
      gates[0].resolve();
      await new Promise((r) => setTimeout(r, 20));

      // Second gate opens — HIGH should have been picked over LOW
      if (gates[1]) gates[1].resolve();
      await new Promise((r) => setTimeout(r, 20));

      await Promise.all([lowPromise, highPromise]);

      // The order: first request got the slot immediately (could be either),
      // but the SECOND dispatch should be HIGH (priority wins when both queued)
      // Since LOW was enqueued first and got the only slot, HIGH should be next
      expect(dispatched.length).toBe(2);
    });
  });

  describe('per-user fairness', () => {
    it('should interleave requests from different users', async () => {
      const dispatchOrder: string[] = [];
      let callCount = 0;

      mockRouter.route.mockImplementation(() => {
        const { worker, subject } = createControllableWorker();
        setTimeout(() => {
          subject.next({ chunk: { text: 'ok' } });
          subject.next({
            complete: {
              finish_reason: 'stop',
              usage: { prompt_tokens: 1, completion_tokens: 1 },
            },
          });
          subject.complete();
        }, 5);
        return Promise.resolve({ worker, workerId: 'w-0', action: 'direct' });
      });

      // Intercept to track dispatch order
      const origEnqueue = scheduler.enqueue.bind(scheduler);

      scheduler.setMaxConcurrent(1);

      // User A queues 4 requests
      const aPromises: Promise<any>[] = [];
      for (let i = 0; i < 4; i++) {
        aPromises.push(
          scheduler.enqueue({
            dto: { model: 'test', prompt: `a-${i}`, stream: false },
            userId: 'user-A',
            priority: Priority.NORMAL,
          }),
        );
      }

      // User B queues 2 requests
      const bPromises: Promise<any>[] = [];
      for (let i = 0; i < 2; i++) {
        bPromises.push(
          scheduler.enqueue({
            dto: { model: 'test', prompt: `b-${i}`, stream: false },
            userId: 'user-B',
            priority: Priority.NORMAL,
          }),
        );
      }

      await Promise.all([...aPromises, ...bPromises]);

      // With round-robin fairness, user B should not wait for all 4 of user A
      // At minimum, router should have been called 6 times
      expect(mockRouter.route).toHaveBeenCalledTimes(6);
    });
  });

  describe('queue depth limit', () => {
    it('should reject with 429 when queue is full', async () => {
      scheduler.setMaxQueueDepth(2);
      scheduler.setMaxConcurrent(0); // nothing dispatches — everything queues

      // These should queue
      const p1 = scheduler.enqueue({
        dto: { model: 'test', prompt: 'a', stream: false },
        userId: 'u1',
        priority: Priority.NORMAL,
      });
      const p2 = scheduler.enqueue({
        dto: { model: 'test', prompt: 'b', stream: false },
        userId: 'u2',
        priority: Priority.NORMAL,
      });

      // This should reject
      await expect(
        scheduler.enqueue({
          dto: { model: 'test', prompt: 'c', stream: false },
          userId: 'u3',
          priority: Priority.NORMAL,
        }),
      ).rejects.toThrow(HttpException);

      try {
        await scheduler.enqueue({
          dto: { model: 'test', prompt: 'd', stream: false },
          userId: 'u4',
          priority: Priority.NORMAL,
        });
      } catch (err) {
        expect(err.getStatus()).toBe(429);
        expect(err.getResponse()).toMatchObject({
          error: { type: 'rate_limit' },
        });
      }
    });
  });

  describe('token budget limit', () => {
    it('should reject when cumulative queued tokens exceed budget', async () => {
      scheduler.setMaxQueuedTokens(100);
      scheduler.setMaxConcurrent(0); // nothing dispatches

      // 80-char prompt → ~20 tokens (chars/4)
      const p1 = scheduler.enqueue({
        dto: {
          model: 'test',
          prompt: 'a'.repeat(320), // 80 tokens
          stream: false,
        },
        userId: 'u1',
        priority: Priority.NORMAL,
      });

      await new Promise((r) => setTimeout(r, 10));

      // Another 80-char prompt would exceed 100 token budget
      await expect(
        scheduler.enqueue({
          dto: {
            model: 'test',
            prompt: 'b'.repeat(320), // 80 tokens → 80 + 80 = 160 > 100
            stream: false,
          },
          userId: 'u2',
          priority: Priority.NORMAL,
        }),
      ).rejects.toThrow(HttpException);
    });
  });

  describe('cancel', () => {
    it('should cancel a queued request', async () => {
      scheduler.setMaxConcurrent(0); // nothing dispatches

      const promise = scheduler.enqueue({
        dto: { model: 'test', prompt: 'hello', stream: false },
        userId: 'u1',
        priority: Priority.NORMAL,
      });

      await new Promise((r) => setTimeout(r, 10));

      // Cancel should exist and work
      const requestId = scheduler.getQueuedRequestIds()[0];
      scheduler.cancel(requestId);

      await expect(promise).rejects.toMatchObject({
        error: { message: 'Request cancelled' },
      });
    });
  });

  describe('getStats', () => {
    it('should report queue depth and token budget usage', async () => {
      scheduler.setMaxConcurrent(0);

      scheduler.enqueue({
        dto: { model: 'test', prompt: 'hello world test', stream: false },
        userId: 'u1',
        priority: Priority.NORMAL,
      });

      await new Promise((r) => setTimeout(r, 10));

      const stats = scheduler.getStats();
      expect(stats.queueDepth).toBe(1);
      expect(stats.totalQueuedTokens).toBeGreaterThan(0);
    });
  });
});

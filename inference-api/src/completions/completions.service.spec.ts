import { Test, TestingModule } from '@nestjs/testing';
import { getRepositoryToken } from '@nestjs/typeorm';
import { CompletionsService } from './completions.service';
import { Completion } from './entities/completion.entity';
import { SchedulerService } from '../scheduler/scheduler.service';
import { TokenizerService } from '../tokenizer/tokenizer.service';
import { MetricsService } from '../metrics/metrics.service';

describe('CompletionsService', () => {
  let service: CompletionsService;
  let mockRepo: Record<string, jest.Mock>;
  let mockScheduler: Record<string, jest.Mock>;
  let mockTokenizer: Record<string, jest.Mock>;

  beforeEach(async () => {
    mockRepo = {
      create: jest.fn((data) => ({ ...data })),
      save: jest.fn((entity) => Promise.resolve(entity)),
      find: jest.fn(() => Promise.resolve([])),
      findOneBy: jest.fn(() => Promise.resolve(null)),
      delete: jest.fn(() => Promise.resolve({ affected: 1 })),
    };

    mockScheduler = {
      enqueue: jest.fn(),
      cancel: jest.fn(),
      getQueuedRequestIds: jest.fn(() => []),
    };

    mockTokenizer = {
      estimateTokenCount: jest.fn((text: string) =>
        Math.ceil(text.length / 4),
      ),
    };

    const module: TestingModule = await Test.createTestingModule({
      providers: [
        CompletionsService,
        { provide: getRepositoryToken(Completion), useValue: mockRepo },
        { provide: SchedulerService, useValue: mockScheduler },
        { provide: TokenizerService, useValue: mockTokenizer },
        { provide: MetricsService, useValue: { recordInference: jest.fn() } },
      ],
    }).compile();

    service = module.get<CompletionsService>(CompletionsService);
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  describe('create (non-streaming)', () => {
    it('should enqueue via scheduler and return result', async () => {
      const fakeResult = {
        id: 'req-1',
        object: 'text_completion',
        choices: [{ text: 'world', index: 0, finish_reason: 'stop' }],
        usage: { prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 },
      };
      mockScheduler.enqueue.mockResolvedValue(fakeResult);

      const { promise } = service.create({
        model: 'test-model',
        prompt: 'Hello',
        max_tokens: 10,
      });

      const result = await promise;

      expect(mockScheduler.enqueue).toHaveBeenCalledWith(
        expect.objectContaining({
          dto: expect.objectContaining({ model: 'test-model', prompt: 'Hello' }),
          userId: 'anonymous',
        }),
      );
      expect(result.choices[0].text).toBe('world');
    });

    it('should save completed status to DB', async () => {
      mockScheduler.enqueue.mockResolvedValue({
        choices: [{ text: 'ok', finish_reason: 'stop' }],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        workerId: 'w-0',
      });

      const { promise } = service.create({
        model: 'test-model',
        prompt: 'Hi',
      });

      await promise;

      // save called at least twice: initial + completed
      expect(mockRepo.save).toHaveBeenCalled();
      const lastSave =
        mockRepo.save.mock.calls[mockRepo.save.mock.calls.length - 1][0];
      expect(lastSave.status).toBe('completed');
      expect(lastSave.completion_text).toBe('ok');
    });

    it('should save error status on scheduler rejection', async () => {
      mockScheduler.enqueue.mockRejectedValue({
        error: { message: 'Inference failed' },
      });

      const { promise } = service.create({
        model: 'test-model',
        prompt: 'Hello',
      });

      await expect(promise).rejects.toMatchObject({
        error: { message: 'Inference failed' },
      });

      const lastSave =
        mockRepo.save.mock.calls[mockRepo.save.mock.calls.length - 1][0];
      expect(lastSave.status).toBe('error');
    });

    it('should return a cancel function', () => {
      mockScheduler.enqueue.mockReturnValue(new Promise(() => {})); // never resolves

      const { cancel } = service.create({
        model: 'test-model',
        prompt: 'Hello',
      });

      expect(typeof cancel).toBe('function');
      cancel();
      expect(mockScheduler.cancel).toHaveBeenCalled();
    });

    it('should use dto.user as userId when provided', async () => {
      mockScheduler.enqueue.mockResolvedValue({
        choices: [{ text: 'ok', finish_reason: 'stop' }],
        usage: {},
      });

      const { promise } = service.create({
        model: 'test-model',
        prompt: 'Hello',
        user: 'user-42',
      });

      await promise;

      expect(mockScheduler.enqueue).toHaveBeenCalledWith(
        expect.objectContaining({ userId: 'user-42' }),
      );
    });
  });

  describe('createStream', () => {
    it('should return stream$ and cancel from scheduler result', async () => {
      // Use a deferred promise so we control when enqueue resolves
      let resolveEnqueue: (v: any) => void;
      mockScheduler.enqueue.mockReturnValue(
        new Promise((r) => {
          resolveEnqueue = r;
        }),
      );

      const { stream$, cancel } = await service.createStream({
        model: 'test-model',
        prompt: 'Hello',
        max_tokens: 5,
      });

      expect(stream$).toBeDefined();
      expect(typeof cancel).toBe('function');

      // Subscribe BEFORE resolving the enqueue
      const events: any[] = [];
      const completed = new Promise<void>((resolve) => {
        stream$.subscribe({
          next: (event) => events.push(event),
          complete: () => resolve(),
        });
      });

      // Now resolve the scheduler — triggers .then() which emits into subject
      resolveEnqueue!({
        choices: [{ text: 'hi', finish_reason: 'stop' }],
        usage: { prompt_tokens: 1, completion_tokens: 1 },
      });

      await completed;

      expect(events.length).toBeGreaterThanOrEqual(2);
      const lastEvent = events[events.length - 1];
      expect(lastEvent.data).toBe('[DONE]');

      const firstData = JSON.parse(events[0].data);
      expect(firstData.object).toBe('text_completion');
      expect(firstData.choices[0].text).toBe('hi');
    });

    it('should emit error event on scheduler rejection', async () => {
      let rejectEnqueue: (v: any) => void;
      mockScheduler.enqueue.mockReturnValue(
        new Promise((_, r) => {
          rejectEnqueue = r;
        }),
      );

      const { stream$ } = await service.createStream({
        model: 'test-model',
        prompt: 'Hello',
      });

      const events: any[] = [];
      const completed = new Promise<void>((resolve) => {
        stream$.subscribe({
          next: (event) => events.push(event),
          complete: () => resolve(),
        });
      });

      rejectEnqueue!({ error: { message: 'GPU OOM' } });

      await completed;

      const errorEvent = events.find((e) => {
        const parsed = JSON.parse(e.data);
        return parsed.error;
      });
      expect(errorEvent).toBeDefined();
    });
  });

  describe('findAll', () => {
    it('should return completions ordered by created_at DESC', async () => {
      mockRepo.find.mockResolvedValue([
        { id: '1', prompt: 'a' },
        { id: '2', prompt: 'b' },
      ]);

      const result = await service.findAll();
      expect(result).toHaveLength(2);
      expect(mockRepo.find).toHaveBeenCalledWith({
        order: { created_at: 'DESC' },
        take: 100,
      });
    });
  });

  describe('findOne', () => {
    it('should return a completion by ID', async () => {
      mockRepo.findOneBy.mockResolvedValue({ id: 'abc', prompt: 'test' });

      const result = await service.findOne('abc');
      expect(result).toMatchObject({ id: 'abc', prompt: 'test' });
    });

    it('should return null for non-existent ID', async () => {
      const result = await service.findOne('missing');
      expect(result).toBeNull();
    });
  });

  describe('remove', () => {
    it('should delete by ID', async () => {
      await service.remove('abc');
      expect(mockRepo.delete).toHaveBeenCalledWith('abc');
    });
  });
});

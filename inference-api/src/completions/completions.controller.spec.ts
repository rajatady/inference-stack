import { Test, TestingModule } from '@nestjs/testing';
import { HttpException, HttpStatus } from '@nestjs/common';
import { CompletionsController } from './completions.controller';
import { CompletionsService } from './completions.service';
import { SchedulerService } from '../scheduler/scheduler.service';
import { Subject } from 'rxjs';

describe('CompletionsController', () => {
  let controller: CompletionsController;
  let mockService: Record<string, jest.Mock>;
  let mockScheduler: Record<string, jest.Mock>;

  beforeEach(async () => {
    mockService = {
      create: jest.fn(),
      createStream: jest.fn(),
      findAll: jest.fn(() => Promise.resolve([])),
      findOne: jest.fn(() => Promise.resolve(null)),
      remove: jest.fn(() => Promise.resolve()),
    };

    mockScheduler = {
      getStats: jest.fn(() => ({
        queueDepth: 3,
        totalQueuedTokens: 500,
        activeCount: 2,
      })),
    };

    const module: TestingModule = await Test.createTestingModule({
      controllers: [CompletionsController],
      providers: [
        { provide: CompletionsService, useValue: mockService },
        { provide: SchedulerService, useValue: mockScheduler },
      ],
    }).compile();

    controller = module.get<CompletionsController>(CompletionsController);
  });

  const mockRes = () => {
    const res: any = {
      setHeader: jest.fn(),
      write: jest.fn(),
      end: jest.fn(),
      json: jest.fn(),
      status: jest.fn(() => res),
      on: jest.fn(),
    };
    return res;
  };

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });

  describe('create — validation', () => {
    it('should throw 400 when model is missing', async () => {
      const res = mockRes();
      await expect(
        controller.create({ prompt: 'Hello' } as any, res),
      ).rejects.toThrow(HttpException);
    });

    it('should throw 400 when prompt is missing', async () => {
      const res = mockRes();
      await expect(
        controller.create({ model: 'test' } as any, res),
      ).rejects.toThrow(HttpException);
    });
  });

  describe('create — non-streaming', () => {
    it('should destructure { promise, cancel } and return result', async () => {
      const res = mockRes();
      const cancel = jest.fn();
      mockService.create.mockReturnValue({
        promise: Promise.resolve({ id: '1', choices: [] }),
        cancel,
      });

      await controller.create(
        { model: 'test', prompt: 'Hello', stream: false },
        res,
      );

      expect(mockService.create).toHaveBeenCalled();
      expect(res.json).toHaveBeenCalledWith(expect.objectContaining({ id: '1', choices: [] }));
      expect(res.on).toHaveBeenCalledWith('close', expect.any(Function));
    });

    it('should wire res.on("close") to cancel', async () => {
      const res = mockRes();
      const cancel = jest.fn();
      // Promise that never resolves — simulates in-flight request
      mockService.create.mockReturnValue({
        promise: new Promise(() => {}),
        cancel,
      });

      // Don't await — it would hang
      controller.create(
        { model: 'test', prompt: 'Hello', stream: false },
        res,
      );

      await new Promise((r) => setTimeout(r, 10));

      // Find the 'close' callback and invoke it
      const closeCall = res.on.mock.calls.find((c: any) => c[0] === 'close');
      expect(closeCall).toBeDefined();
      closeCall[1](); // trigger disconnect
      expect(cancel).toHaveBeenCalled();
    });

    it('should handle 429 from scheduler with Retry-After header', async () => {
      const res = mockRes();
      const err = new HttpException(
        { error: { message: 'Queue full', type: 'rate_limit' }, retryAfter: 5 },
        HttpStatus.TOO_MANY_REQUESTS,
      );
      mockService.create.mockReturnValue({
        promise: Promise.reject(err),
        cancel: jest.fn(),
      });

      await controller.create(
        { model: 'test', prompt: 'Hello', stream: false },
        res,
      );

      expect(res.setHeader).toHaveBeenCalledWith('Retry-After', '5');
      expect(res.status).toHaveBeenCalledWith(429);
    });
  });

  describe('create — streaming', () => {
    it('should set SSE headers and wire cancel on disconnect', async () => {
      const res = mockRes();
      const subject = new Subject();
      const cancel = jest.fn();
      mockService.createStream.mockResolvedValue({
        stream$: subject.asObservable(),
        cancel,
      });

      // Don't await — streaming is async
      controller.create(
        { model: 'test', prompt: 'Hello', stream: true },
        res,
      );

      await new Promise((r) => setTimeout(r, 10));

      expect(res.setHeader).toHaveBeenCalledWith(
        'Content-Type',
        'text/event-stream',
      );
      expect(res.on).toHaveBeenCalledWith('close', expect.any(Function));

      // Trigger disconnect
      const closeCall = res.on.mock.calls.find((c: any) => c[0] === 'close');
      closeCall[1]();
      expect(cancel).toHaveBeenCalled();

      subject.complete();
    });

    it('should handle 429 for streaming requests', async () => {
      const res = mockRes();
      const err = new HttpException(
        { error: { message: 'Queue full', type: 'rate_limit' }, retryAfter: 3 },
        HttpStatus.TOO_MANY_REQUESTS,
      );
      mockService.createStream.mockRejectedValue(err);

      await controller.create(
        { model: 'test', prompt: 'Hello', stream: true },
        res,
      );

      expect(res.setHeader).toHaveBeenCalledWith('Retry-After', '3');
      expect(res.status).toHaveBeenCalledWith(429);
    });
  });

  describe('getStats', () => {
    it('should return scheduler stats', () => {
      const result = controller.getStats();
      expect(result).toEqual({
        queueDepth: 3,
        totalQueuedTokens: 500,
        activeCount: 2,
      });
      expect(mockScheduler.getStats).toHaveBeenCalled();
    });
  });

  describe('findAll', () => {
    it('should return service.findAll()', async () => {
      const data = [{ id: '1' }, { id: '2' }];
      mockService.findAll.mockResolvedValue(data);

      const result = await controller.findAll();
      expect(result).toEqual(data);
    });
  });

  describe('findOne', () => {
    it('should return service.findOne(id)', async () => {
      mockService.findOne.mockResolvedValue({ id: 'abc' });
      const result = await controller.findOne('abc');
      expect(result).toEqual({ id: 'abc' });
    });
  });

  describe('remove', () => {
    it('should call service.remove(id)', async () => {
      await controller.remove('abc');
      expect(mockService.remove).toHaveBeenCalledWith('abc');
    });
  });
});

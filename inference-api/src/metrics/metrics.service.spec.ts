import { Test, TestingModule } from '@nestjs/testing';
import { MetricsService } from './metrics.service';
import { ClickHouseService } from './clickhouse.service';
import { InferenceMetricEvent } from './interfaces';

describe('MetricsService', () => {
  let service: MetricsService;
  let mockCh: Record<string, jest.Mock>;

  beforeEach(async () => {
    mockCh = {
      insert: jest.fn().mockResolvedValue(undefined),
      query: jest.fn().mockResolvedValue([]),
    };

    const module: TestingModule = await Test.createTestingModule({
      providers: [
        MetricsService,
        { provide: ClickHouseService, useValue: mockCh },
      ],
    }).compile();

    service = module.get<MetricsService>(MetricsService);
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  describe('recordInference', () => {
    const baseEvent: InferenceMetricEvent = {
      requestId: 'req-1',
      model: 'SmolLM2-135M',
      modelType: 'text_gen',
      workerId: 'w-0',
      prefillTimeMs: 50,
      decodeTimeMs: 200,
      totalTimeMs: 250,
      queueWaitMs: 10,
      routingTimeMs: 5,
      e2eTimeMs: 300,
      promptTokens: 100,
      completionTokens: 40,
      cachedTokens: 0,
      userId: 'user-1',
      priority: 'normal',
      isStream: false,
      finishReason: 'stop',
      isError: false,
    };

    it('should compute decode_tps correctly', () => {
      service.recordInference(baseEvent);

      expect(mockCh.insert).toHaveBeenCalledWith(
        'inference_metrics',
        expect.arrayContaining([
          expect.objectContaining({
            // 40 tokens / (200ms / 1000) = 200 TPS
            decode_tps: 200,
          }),
        ]),
      );
    });

    it('should compute prefill_tps correctly', () => {
      service.recordInference(baseEvent);

      expect(mockCh.insert).toHaveBeenCalledWith(
        'inference_metrics',
        expect.arrayContaining([
          expect.objectContaining({
            // 100 tokens / (50ms / 1000) = 2000 TPS
            prefill_tps: 2000,
          }),
        ]),
      );
    });

    it('should set TPS to 0 when timing is 0', () => {
      service.recordInference({
        ...baseEvent,
        prefillTimeMs: 0,
        decodeTimeMs: 0,
      });

      expect(mockCh.insert).toHaveBeenCalledWith(
        'inference_metrics',
        expect.arrayContaining([
          expect.objectContaining({
            decode_tps: 0,
            prefill_tps: 0,
          }),
        ]),
      );
    });

    it('should include all fields in the insert', () => {
      service.recordInference(baseEvent);

      const row = mockCh.insert.mock.calls[0][1][0];
      expect(row.request_id).toBe('req-1');
      expect(row.model).toBe('SmolLM2-135M');
      expect(row.model_type).toBe('text_gen');
      expect(row.worker_id).toBe('w-0');
      expect(row.prompt_tokens).toBe(100);
      expect(row.completion_tokens).toBe(40);
      expect(row.queue_wait_ms).toBe(10);
      expect(row.routing_time_ms).toBe(5);
      expect(row.e2e_time_ms).toBe(300);
      expect(row.user_id).toBe('user-1');
      expect(row.priority).toBe('normal');
      expect(row.is_stream).toBe(false);
      expect(row.is_error).toBe(false);
    });

    it('should default optional media fields', () => {
      service.recordInference(baseEvent);

      const row = mockCh.insert.mock.calls[0][1][0];
      expect(row.media_type).toBe('');
      expect(row.media_size_bytes).toBe(0);
      expect(row.error_message).toBe('');
    });

    it('should include media fields when provided', () => {
      service.recordInference({
        ...baseEvent,
        mediaType: 'image/png',
        mediaSizeBytes: 102400,
      });

      const row = mockCh.insert.mock.calls[0][1][0];
      expect(row.media_type).toBe('image/png');
      expect(row.media_size_bytes).toBe(102400);
    });

    it('should not throw when insert fails', () => {
      mockCh.insert.mockRejectedValue(new Error('ClickHouse down'));

      // Should not throw — fire-and-forget
      expect(() => service.recordInference(baseEvent)).not.toThrow();
    });
  });

  describe('getTps', () => {
    it('should return defaults when no data', async () => {
      mockCh.query.mockResolvedValue([]);

      const result = await service.getTps('5m');

      expect(result.current.decode_tps).toBe(0);
      expect(result.current.prefill_tps).toBe(0);
      expect(result.current.requests_per_minute).toBe(0);
      expect(result.by_model).toEqual([]);
      expect(result.window).toBe('5m');
    });

    it('should parse TPS results from ClickHouse', async () => {
      mockCh.query
        .mockResolvedValueOnce([
          { avg_decode_tps: '150.5', avg_prefill_tps: '3200', rpm: '42' },
        ])
        .mockResolvedValueOnce([
          {
            model: 'SmolLM2-135M',
            avg_decode_tps: '185.2',
            avg_prefill_tps: '4100',
            avg_total_ms: '250',
            cnt: '30',
          },
        ]);

      const result = await service.getTps('5m');

      expect(result.current.decode_tps).toBe(150.5);
      expect(result.current.prefill_tps).toBe(3200);
      expect(result.current.requests_per_minute).toBe(42);
      expect(result.by_model).toHaveLength(1);
      expect(result.by_model[0].model).toBe('SmolLM2-135M');
      expect(result.by_model[0].decode_tps).toBe(185.2);
      expect(result.by_model[0].request_count).toBe(30);
    });

    it('should use correct interval for window parsing', async () => {
      mockCh.query.mockResolvedValue([]);

      await service.getTps('1h');

      // First query should use 3600 seconds
      const sql = mockCh.query.mock.calls[0][0];
      expect(sql).toContain('3600');
    });
  });

  describe('getLatencyPercentiles', () => {
    it('should return zeros when no data', async () => {
      mockCh.query.mockResolvedValue([]);

      const result = await service.getLatencyPercentiles('5m');

      expect(result.e2e.p50).toBe(0);
      expect(result.e2e.p95).toBe(0);
      expect(result.e2e.p99).toBe(0);
      expect(result.window).toBe('5m');
    });

    it('should parse percentile results', async () => {
      mockCh.query.mockResolvedValue([
        {
          e2e_p50: '120.5',
          e2e_p95: '350',
          e2e_p99: '820',
          prefill_p50: '45',
          prefill_p95: '120',
          prefill_p99: '350',
          decode_p50: '80',
          decode_p95: '250',
          decode_p99: '600',
          queue_p50: '5',
          queue_p95: '50',
          queue_p99: '200',
          ttft_p50: '80',
          ttft_p95: '200',
          ttft_p99: '500',
        },
      ]);

      const result = await service.getLatencyPercentiles('5m');

      expect(result.e2e.p50).toBe(120.5);
      expect(result.e2e.p95).toBe(350);
      expect(result.prefill.p50).toBe(45);
      expect(result.decode.p99).toBe(600);
      expect(result.queue_wait.p95).toBe(50);
      expect(result.time_to_first_token.p50).toBe(80);
    });
  });

  describe('getBreakdown', () => {
    it('should return empty arrays when no data', async () => {
      mockCh.query.mockResolvedValue([]);

      const result = await service.getBreakdown('5m');

      expect(result.by_model).toEqual([]);
      expect(result.by_worker).toEqual([]);
      expect(result.window).toBe('5m');
    });

    it('should parse breakdown results', async () => {
      mockCh.query
        .mockResolvedValueOnce([
          {
            model: 'SmolLM2-135M',
            cnt: '25',
            avg_decode_tps: '185',
            avg_prefill_tps: '4100',
            avg_e2e: '300',
            p95_e2e: '500',
            sum_prompt: '2500',
            sum_completion: '1000',
            err_cnt: '2',
          },
        ])
        .mockResolvedValueOnce([
          {
            worker_id: 'w-0',
            cnt: '40',
            avg_e2e: '280',
            sum_prompt: '4000',
            sum_completion: '1600',
          },
        ]);

      const result = await service.getBreakdown('5m');

      expect(result.by_model).toHaveLength(1);
      expect(result.by_model[0].model).toBe('SmolLM2-135M');
      expect(result.by_model[0].error_count).toBe(2);
      expect(result.by_worker).toHaveLength(1);
      expect(result.by_worker[0].worker_id).toBe('w-0');
      expect(result.by_worker[0].request_count).toBe(40);
    });
  });

  describe('getHistory', () => {
    it('should default to decode_tps metric', async () => {
      mockCh.query.mockResolvedValue([]);

      await service.getHistory({});

      const sql = mockCh.query.mock.calls[0][0];
      expect(sql).toContain('avg(decode_tps)');
    });

    it('should support latency metric', async () => {
      mockCh.query.mockResolvedValue([]);

      await service.getHistory({ metric: 'latency' });

      const sql = mockCh.query.mock.calls[0][0];
      expect(sql).toContain('avg(e2e_time_ms)');
    });

    it('should support throughput metric', async () => {
      mockCh.query.mockResolvedValue([]);

      await service.getHistory({ metric: 'throughput' });

      const sql = mockCh.query.mock.calls[0][0];
      expect(sql).toContain('count()');
    });

    it('should filter by model when provided', async () => {
      mockCh.query.mockResolvedValue([]);

      await service.getHistory({ model: 'SmolLM2-135M' });

      const sql = mockCh.query.mock.calls[0][0];
      expect(sql).toContain("model = 'SmolLM2-135M'");
    });

    it('should return parsed time-series data', async () => {
      mockCh.query.mockResolvedValue([
        { ts: '2026-03-12 00:00:00', val: '150.5' },
        { ts: '2026-03-12 00:01:00', val: '160.2' },
      ]);

      const result = await service.getHistory({});

      expect(result).toHaveLength(2);
      expect(result[0].timestamp).toBe('2026-03-12 00:00:00');
      expect(result[0].value).toBe(150.5);
    });
  });
});

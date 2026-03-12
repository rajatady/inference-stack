import { Injectable, Logger } from '@nestjs/common';
import { ClickHouseService } from './clickhouse.service';
import {
  InferenceMetricEvent,
  TpsResult,
  LatencyResult,
  BreakdownResult,
} from './interfaces';

@Injectable()
export class MetricsService {
  private readonly logger = new Logger(MetricsService.name);

  constructor(private readonly ch: ClickHouseService) {}

  /**
   * Record an inference metric event. Fire-and-forget — never blocks the response.
   */
  recordInference(event: InferenceMetricEvent): void {
    const decodeTps =
      event.decodeTimeMs > 0
        ? event.completionTokens / (event.decodeTimeMs / 1000)
        : 0;
    const prefillTps =
      event.prefillTimeMs > 0
        ? event.promptTokens / (event.prefillTimeMs / 1000)
        : 0;

    this.ch
      .insert('inference_metrics', [
        {
          timestamp: Math.round(Date.now() / 1000),
          request_id: event.requestId,
          model: event.model,
          model_type: event.modelType,
          worker_id: event.workerId,
          prefill_time_ms: event.prefillTimeMs,
          decode_time_ms: event.decodeTimeMs,
          total_time_ms: event.totalTimeMs,
          queue_wait_ms: event.queueWaitMs,
          routing_time_ms: event.routingTimeMs,
          e2e_time_ms: event.e2eTimeMs,
          time_to_first_token_ms: event.timeToFirstTokenMs ?? 0,
          prompt_tokens: event.promptTokens,
          completion_tokens: event.completionTokens,
          cached_tokens: event.cachedTokens,
          decode_tps: decodeTps,
          prefill_tps: prefillTps,
          user_id: event.userId,
          priority: event.priority,
          is_stream: event.isStream,
          finish_reason: event.finishReason,
          media_type: event.mediaType ?? '',
          media_size_bytes: event.mediaSizeBytes ?? 0,
          is_error: event.isError,
          error_message: event.errorMessage ?? '',
        },
      ])
      .catch((err) =>
        this.logger.error(`Failed to record metric: ${err.message}`),
      );
  }

  async getTps(window = '5m'): Promise<TpsResult> {
    const interval = this.parseWindow(window);

    const currentRows = await this.ch.query<{
      avg_decode_tps: string;
      avg_prefill_tps: string;
      rpm: string;
    }>(`
      SELECT
        avg(decode_tps) AS avg_decode_tps,
        avg(prefill_tps) AS avg_prefill_tps,
        count() * 60 / ${interval} AS rpm
      FROM inference_metrics
      WHERE timestamp > now() - INTERVAL ${interval} SECOND
        AND is_error = false
    `);

    const byModelRows = await this.ch.query<{
      model: string;
      avg_decode_tps: string;
      avg_prefill_tps: string;
      avg_total_ms: string;
      cnt: string;
    }>(`
      SELECT
        model,
        avg(decode_tps) AS avg_decode_tps,
        avg(prefill_tps) AS avg_prefill_tps,
        avg(total_time_ms) AS avg_total_ms,
        count() AS cnt
      FROM inference_metrics
      WHERE timestamp > now() - INTERVAL ${interval} SECOND
        AND is_error = false
      GROUP BY model
      ORDER BY cnt DESC
    `);

    const current = currentRows[0] || {
      avg_decode_tps: '0',
      avg_prefill_tps: '0',
      rpm: '0',
    };

    return {
      current: {
        decode_tps: parseFloat(current.avg_decode_tps),
        prefill_tps: parseFloat(current.avg_prefill_tps),
        requests_per_minute: parseFloat(current.rpm),
      },
      by_model: byModelRows.map((r) => ({
        model: r.model,
        decode_tps: parseFloat(r.avg_decode_tps),
        prefill_tps: parseFloat(r.avg_prefill_tps),
        avg_generation_ms: parseFloat(r.avg_total_ms),
        request_count: parseInt(r.cnt, 10),
      })),
      window,
    };
  }

  async getLatencyPercentiles(window = '5m'): Promise<LatencyResult> {
    const interval = this.parseWindow(window);

    const rows = await this.ch.query<{
      e2e_p50: string;
      e2e_p95: string;
      e2e_p99: string;
      prefill_p50: string;
      prefill_p95: string;
      prefill_p99: string;
      decode_p50: string;
      decode_p95: string;
      decode_p99: string;
      queue_p50: string;
      queue_p95: string;
      queue_p99: string;
      ttft_p50: string;
      ttft_p95: string;
      ttft_p99: string;
    }>(`
      SELECT
        quantile(0.50)(e2e_time_ms) AS e2e_p50,
        quantile(0.95)(e2e_time_ms) AS e2e_p95,
        quantile(0.99)(e2e_time_ms) AS e2e_p99,
        quantile(0.50)(prefill_time_ms) AS prefill_p50,
        quantile(0.95)(prefill_time_ms) AS prefill_p95,
        quantile(0.99)(prefill_time_ms) AS prefill_p99,
        quantile(0.50)(decode_time_ms) AS decode_p50,
        quantile(0.95)(decode_time_ms) AS decode_p95,
        quantile(0.99)(decode_time_ms) AS decode_p99,
        quantile(0.50)(queue_wait_ms) AS queue_p50,
        quantile(0.95)(queue_wait_ms) AS queue_p95,
        quantile(0.99)(queue_wait_ms) AS queue_p99,
        quantile(0.50)(time_to_first_token_ms) AS ttft_p50,
        quantile(0.95)(time_to_first_token_ms) AS ttft_p95,
        quantile(0.99)(time_to_first_token_ms) AS ttft_p99
      FROM inference_metrics
      WHERE timestamp > now() - INTERVAL ${interval} SECOND
        AND is_error = false
    `);

    const r = rows[0] || {};
    const p = (v: string | undefined) => parseFloat(v || '0');

    return {
      e2e: { p50: p(r.e2e_p50), p95: p(r.e2e_p95), p99: p(r.e2e_p99) },
      prefill: {
        p50: p(r.prefill_p50),
        p95: p(r.prefill_p95),
        p99: p(r.prefill_p99),
      },
      decode: {
        p50: p(r.decode_p50),
        p95: p(r.decode_p95),
        p99: p(r.decode_p99),
      },
      queue_wait: {
        p50: p(r.queue_p50),
        p95: p(r.queue_p95),
        p99: p(r.queue_p99),
      },
      time_to_first_token: {
        p50: p(r.ttft_p50),
        p95: p(r.ttft_p95),
        p99: p(r.ttft_p99),
      },
      window,
    };
  }

  async getBreakdown(window = '5m'): Promise<BreakdownResult> {
    const interval = this.parseWindow(window);

    const byModel = await this.ch.query<{
      model: string;
      cnt: string;
      avg_decode_tps: string;
      avg_prefill_tps: string;
      avg_e2e: string;
      p95_e2e: string;
      sum_prompt: string;
      sum_completion: string;
      err_cnt: string;
    }>(`
      SELECT
        model,
        count() AS cnt,
        avg(decode_tps) AS avg_decode_tps,
        avg(prefill_tps) AS avg_prefill_tps,
        avg(e2e_time_ms) AS avg_e2e,
        quantile(0.95)(e2e_time_ms) AS p95_e2e,
        sum(prompt_tokens) AS sum_prompt,
        sum(completion_tokens) AS sum_completion,
        countIf(is_error = true) AS err_cnt
      FROM inference_metrics
      WHERE timestamp > now() - INTERVAL ${interval} SECOND
      GROUP BY model
      ORDER BY cnt DESC
    `);

    const byWorker = await this.ch.query<{
      worker_id: string;
      cnt: string;
      avg_e2e: string;
      sum_prompt: string;
      sum_completion: string;
    }>(`
      SELECT
        worker_id,
        count() AS cnt,
        avg(e2e_time_ms) AS avg_e2e,
        sum(prompt_tokens) AS sum_prompt,
        sum(completion_tokens) AS sum_completion
      FROM inference_metrics
      WHERE timestamp > now() - INTERVAL ${interval} SECOND
      GROUP BY worker_id
      ORDER BY cnt DESC
    `);

    return {
      by_model: byModel.map((r) => ({
        model: r.model,
        request_count: parseInt(r.cnt, 10),
        avg_decode_tps: parseFloat(r.avg_decode_tps),
        avg_prefill_tps: parseFloat(r.avg_prefill_tps),
        avg_e2e_ms: parseFloat(r.avg_e2e),
        p95_e2e_ms: parseFloat(r.p95_e2e),
        total_prompt_tokens: parseInt(r.sum_prompt, 10),
        total_completion_tokens: parseInt(r.sum_completion, 10),
        error_count: parseInt(r.err_cnt, 10),
      })),
      by_worker: byWorker.map((r) => ({
        worker_id: r.worker_id,
        request_count: parseInt(r.cnt, 10),
        avg_e2e_ms: parseFloat(r.avg_e2e),
        total_prompt_tokens: parseInt(r.sum_prompt, 10),
        total_completion_tokens: parseInt(r.sum_completion, 10),
      })),
      window,
    };
  }

  async getHistory(opts: {
    model?: string;
    metric?: string;
    interval?: string;
    from?: string;
    to?: string;
  }): Promise<Array<{ timestamp: string; value: number }>> {
    const bucketSeconds = this.parseWindow(opts.interval || '1m');
    const fromClause = opts.from
      ? `AND timestamp >= '${opts.from}'`
      : `AND timestamp > now() - INTERVAL 1 HOUR`;
    const toClause = opts.to ? `AND timestamp <= '${opts.to}'` : '';
    const modelClause = opts.model ? `AND model = '${opts.model}'` : '';

    let metricExpr: string;
    switch (opts.metric) {
      case 'prefill_tps':
        metricExpr = 'avg(prefill_tps)';
        break;
      case 'latency':
        metricExpr = 'avg(e2e_time_ms)';
        break;
      case 'throughput':
        metricExpr = 'count()';
        break;
      default:
        metricExpr = 'avg(decode_tps)';
    }

    const rows = await this.ch.query<{ ts: string; val: string }>(`
      SELECT
        toStartOfInterval(timestamp, INTERVAL ${bucketSeconds} SECOND) AS ts,
        ${metricExpr} AS val
      FROM inference_metrics
      WHERE is_error = false
        ${fromClause}
        ${toClause}
        ${modelClause}
      GROUP BY ts
      ORDER BY ts
    `);

    return rows.map((r) => ({
      timestamp: r.ts,
      value: parseFloat(r.val),
    }));
  }

  private parseWindow(window: string): number {
    const match = window.match(/^(\d+)(s|m|h|d)$/);
    if (!match) return 300; // default 5 minutes
    const n = parseInt(match[1], 10);
    switch (match[2]) {
      case 's':
        return n;
      case 'm':
        return n * 60;
      case 'h':
        return n * 3600;
      case 'd':
        return n * 86400;
      default:
        return 300;
    }
  }
}

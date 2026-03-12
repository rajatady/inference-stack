import {
  Injectable,
  Logger,
  OnModuleInit,
  OnModuleDestroy,
} from '@nestjs/common';
import { createClient, ClickHouseClient } from '@clickhouse/client';

@Injectable()
export class ClickHouseService implements OnModuleInit, OnModuleDestroy {
  private readonly logger = new Logger(ClickHouseService.name);
  private client: ClickHouseClient | null = null;

  async onModuleInit(): Promise<void> {
    const url = process.env.CLICKHOUSE_URL || 'http://localhost:8123';
    this.client = createClient({
      url,
      database: 'inference',
      clickhouse_settings: {
        async_insert: 1,
        wait_for_async_insert: 0,
      },
    });

    try {
      const result = await this.client.query({ query: 'SELECT 1' });
      await result.text();
      this.logger.log(`Connected to ClickHouse at ${url}`);
      await this.ensureSchema();
    } catch (err) {
      this.logger.warn(
        `ClickHouse not available at ${url} — metrics will be dropped. Error: ${err.message}`,
      );
      this.client = null;
    }
  }

  async onModuleDestroy(): Promise<void> {
    if (this.client) {
      await this.client.close();
    }
  }

  isConnected(): boolean {
    return this.client !== null;
  }

  async insert(table: string, values: Record<string, any>[]): Promise<void> {
    if (!this.client || values.length === 0) return;
    try {
      await this.client.insert({ table, values, format: 'JSONEachRow' });
    } catch (err) {
      this.logger.error(`ClickHouse insert failed: ${err.message}`);
    }
  }

  async query<T = Record<string, any>>(sql: string): Promise<T[]> {
    if (!this.client) return [];
    try {
      const result = await this.client.query({ query: sql, format: 'JSONEachRow' });
      return await result.json();
    } catch (err) {
      this.logger.error(`ClickHouse query failed: ${err.message}`);
      return [];
    }
  }

  private async ensureSchema(): Promise<void> {
    const ddl = `
      CREATE TABLE IF NOT EXISTS inference_metrics (
        timestamp        DateTime64(3),
        request_id       String,
        model            String,
        model_type       String,
        worker_id        String,
        prefill_time_ms  Float64,
        decode_time_ms   Float64,
        total_time_ms    Float64,
        queue_wait_ms    Float64,
        routing_time_ms  Float64,
        e2e_time_ms      Float64,
        time_to_first_token_ms Float64,
        prompt_tokens    UInt32,
        completion_tokens UInt32,
        cached_tokens    UInt32,
        decode_tps       Float64,
        prefill_tps      Float64,
        user_id          String,
        priority         String,
        is_stream        Bool,
        finish_reason    String,
        media_type       String,
        media_size_bytes UInt64,
        is_error         Bool,
        error_message    String
      ) ENGINE = MergeTree()
      ORDER BY (model, timestamp)
      PARTITION BY toYYYYMM(timestamp)
    `;
    try {
      const result = await this.client!.query({ query: ddl });
      await result.text();
      this.logger.log('inference_metrics table ensured');
    } catch (err) {
      this.logger.error(`Schema creation failed: ${err.message}`);
    }
  }
}

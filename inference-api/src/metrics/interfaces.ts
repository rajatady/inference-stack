export interface InferenceMetricEvent {
  requestId: string;
  model: string;
  modelType: string;
  workerId: string;

  // GPU timing (from worker UsageStats)
  prefillTimeMs: number;
  decodeTimeMs: number;
  totalTimeMs: number;

  // Gateway timing (measured in NestJS)
  queueWaitMs: number;
  routingTimeMs: number;
  e2eTimeMs: number;
  timeToFirstTokenMs?: number;

  // Tokens
  promptTokens: number;
  completionTokens: number;
  cachedTokens: number;

  // Request context
  userId: string;
  priority: string;
  isStream: boolean;
  finishReason: string;

  // Media (for non-text modalities)
  mediaType?: string;
  mediaSizeBytes?: number;

  // Error
  isError: boolean;
  errorMessage?: string;
}

export interface TpsResult {
  current: {
    decode_tps: number;
    prefill_tps: number;
    requests_per_minute: number;
  };
  by_model: Array<{
    model: string;
    decode_tps: number;
    prefill_tps: number;
    avg_generation_ms: number;
    request_count: number;
  }>;
  window: string;
}

export interface LatencyResult {
  e2e: { p50: number; p95: number; p99: number };
  prefill: { p50: number; p95: number; p99: number };
  decode: { p50: number; p95: number; p99: number };
  queue_wait: { p50: number; p95: number; p99: number };
  time_to_first_token: { p50: number; p95: number; p99: number };
  window: string;
}

export interface BreakdownResult {
  by_model: Array<{
    model: string;
    request_count: number;
    avg_decode_tps: number;
    avg_prefill_tps: number;
    avg_e2e_ms: number;
    p95_e2e_ms: number;
    total_prompt_tokens: number;
    total_completion_tokens: number;
    error_count: number;
  }>;
  by_worker: Array<{
    worker_id: string;
    request_count: number;
    avg_e2e_ms: number;
    total_prompt_tokens: number;
    total_completion_tokens: number;
  }>;
  window: string;
}

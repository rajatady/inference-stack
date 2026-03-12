# inference-api

NestJS gateway that sits between HTTP clients and GPU workers. Handles scheduling, batching, routing, model management, KV cache coordination, and metrics — all the CPU-side logic that makes inference work at scale.

## Architecture

```
HTTP Request
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  CompletionsController / ImagesController / AudioController / VideoController
│  (OpenAI-compatible endpoints, SSE streaming, cancel on disconnect)
│                          │
│                          ▼
│  ┌─────────────────────────────────────────────────┐
│  │  SchedulerService                                │
│  │  Per-user FIFO queues, priority tiers (H/N/L),  │
│  │  aging, backpressure (429 + Retry-After),        │
│  │  per-request timeout                             │
│  │                    │                             │
│  │                    ▼                             │
│  │  BatchCollector                                  │
│  │  50ms time window, max 256, seq-length compat   │
│  └────────────────────┬────────────────────────────┘
│                       │
│                       ▼
│  ┌─────────────────────────────────────────────────┐
│  │  Router → ModelManager → WorkerRegistry          │
│  │  Model affinity → least loaded → trigger load   │
│  │  VRAM-aware placement, auto mode switch (TP)    │
│  └────────────────────┬────────────────────────────┘
│                       │ gRPC
└───────────────────────┼─────────────────────────────┘
                        ▼
                   GPU Workers
```

## Modules

### `src/worker-orchestrator/`

The brain — manages GPU worker connections and decides where requests go.

| File | What it does |
|------|-------------|
| **worker-registry.ts** | Manages N gRPC worker connections. Polls worker state every 5s. Handles runtime mode switching (individual ↔ tensor-parallel) by SSHing to the GPU host to kill/start worker processes. |
| **model-manager.ts** | Decides which worker loads which model. VRAM-aware placement with GPU affinity (prefers the worker listed in model-roster). Coalesces concurrent load requests (if 10 requests arrive for an unloaded model, only one `LoadModel` RPC fires). Evicts idle models on OOM. Auto-switches to TP mode when a tensor-parallel model is requested. |
| **router.ts** | Routes a request to a worker. If the model is already loaded somewhere, picks the least-busy worker with it. If not, triggers ModelManager to load it, then routes. |
| **interfaces.ts** | `WorkerConfig`, `WorkerHandle`, `WorkerSnapshot`, `RoutingDecision`, `ModelCapabilities`, `GpuSnapshot` |

### `src/scheduler/`

Fair queuing, batching, and backpressure.

| File | What it does |
|------|-------------|
| **scheduler.service.ts** | Priority queue with 3 tiers (HIGH/NORMAL/LOW). Per-user FIFO queues with round-robin across users within each tier. Aging boosts priority over time. Admission control: rejects with 429 + Retry-After when queue depth (100) or token budget (50K) exceeded. Dispatches single requests via `worker.infer()` or batches via `worker.batchInfer()`, demuxing results back to individual promises. Per-request timeout (default 120s). |
| **batch-collector.ts** | Accumulates compatible requests in per-model buckets. Dispatches when the 50ms window expires or batch reaches 256. Compatibility = same model + sequence length ratio ≤ 4.0. |
| **interfaces.ts** | `QueuedRequest`, `Priority`, `RequestState`, `SchedulerConfig` |

### `src/completions/`

OpenAI-compatible `/v1/completions` endpoint.

| File | What it does |
|------|-------------|
| **completions.controller.ts** | `POST /v1/completions` — streaming (SSE) or JSON. `GET /v1/completions/stats` — queue depth, active count. `GET /v1/completions` — list recent. Wires `res.on('close')` → cancel for both modes. Auto-generates `session_id` for KV cache routing. |
| **completions.service.ts** | Creates DB entity, enqueues via scheduler, saves result on complete. Handles `thinking_content` from Qwen3-14B (separate field in response). Fires metrics to ClickHouse (fire-and-forget). |
| **dto/create-completion.dto.ts** | `model`, `prompt`, `messages` (chat format), `max_tokens`, `temperature`, `top_p`, `stream`, `user`, `priority`, `images` (base64 for vision), `session_id` |
| **entities/completion.entity.ts** | TypeORM/SQLite entity: id, model, prompt, completion_text, status, usage stats, timing, worker_id |

### `src/images/`, `src/audio/`, `src/video/`

Multi-modal endpoints — same pattern for all three:

| Module | Endpoint | Input | Output |
|--------|----------|-------|--------|
| **images** | `POST /v1/images/generations` | `{ model, prompt, n, size }` | `{ created, data: [{ b64_json }] }` |
| **audio** | `POST /v1/audio/speech` | `{ model, input, voice }` | Raw WAV binary (`audio/wav`) |
| **video** | `POST /v1/video/generations` | `{ model, prompt, num_frames }` | Raw MP4 binary (`video/mp4`) |

Each routes via `Router.route()`, calls `worker.infer()`, collects `MediaOutput` bytes, records metrics.

### `src/gpu-worker/`

| File | What it does |
|------|-------------|
| **gpu-worker.service.ts** | Plain class (not NestJS-managed) wrapping a single gRPC connection. One instance per worker, created by WorkerRegistry. Exposes 9 RPC methods as RxJS Observables: `infer`, `batchInfer`, `loadModel`, `unloadModel`, `getWorkerState`, `watchWorkerState`, `getCacheEntries`, `evictCache`, `health`. |

### `src/metrics/`

ClickHouse-backed inference analytics.

| File | What it does |
|------|-------------|
| **clickhouse.service.ts** | Connection wrapper. Creates `inference_metrics` table (25 columns, MergeTree engine, monthly partitions). Degrades gracefully if ClickHouse unavailable. |
| **metrics.service.ts** | `recordInference()` — fire-and-forget insert. Query methods: `getTps()`, `getLatencyPercentiles()` (p50/p95/p99), `getBreakdown()` (per model/worker), `getHistory()` (time-series). |
| **metrics.controller.ts** | `GET /v1/metrics/{tps,latency,breakdown,history}` |

### `src/tokenizer/`

| File | What it does |
|------|-------------|
| **tokenizer.service.ts** | `estimateTokenCount(text)` — chars/4 heuristic. `validateContextWindow(promptTokens, maxTokens, maxContextLength)` — rejects if total exceeds model limit. |

### `src/config/`

| File | What it does |
|------|-------------|
| **model-roster.ts** | Central registry of all 8 models. Maps model ID → `{ type, vramEstimateBytes, defaultGpu, tensorParallel?, tensorParallelSize? }`. Helpers: `getModelType()`, `getDefaultGpu()`, `isTensorParallel()`. |

## Request flow

```
POST /v1/completions { model: "Qwen/Qwen3-14B", messages: [...] }
  │
  ├─ Controller: validate, generate session_id
  ├─ Service: create DB entity (status: pending)
  ├─ Scheduler.enqueue(): estimate tokens, admission check, add to queue
  │   ├─ tryDispatch(): dequeue highest priority (round-robin within tier)
  │   └─ BatchCollector.submit(): buffer 50ms or dispatch immediately
  │       └─ dispatchRequest():
  │           ├─ Router.route("Qwen/Qwen3-14B")
  │           │   └─ ModelManager.ensureModelLoaded()
  │           │       ├─ isTensorParallel? → switchMode("tensor-parallel")
  │           │       └─ worker.loadModel() → wait for ready
  │           ├─ worker.infer({ messages, cache_hint: { session_id } })
  │           └─ Subscribe to gRPC stream → accumulate text + thinking_content
  │
  ├─ Service: save result to DB, record metrics to ClickHouse
  └─ Return { id, model, choices, thinking_content, usage, session_id }
```

## Configuration

All via environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 3000 | API server port |
| `WORKER_MODE` | individual | `individual` or `tensor-parallel` |
| `GPU_WORKER_0_URL` | localhost:50051 | Worker 0 gRPC endpoint |
| `GPU_WORKER_1_URL` | localhost:50052 | Worker 1 gRPC endpoint |
| `GPU_TP_WORKER_URL` | localhost:50051 | TP worker gRPC endpoint |
| `RUNPOD_SSH_HOST` | localhost | GPU host for mode switching |
| `RUNPOD_SSH_PORT` | 22 | SSH port |
| `CLICKHOUSE_URL` | http://localhost:8123 | ClickHouse (optional) |

## Tests

```bash
# Unit tests (107 passing, 16 suites)
npx jest --testPathPattern='src/'

# E2E tests (requires GPU workers + SSH tunnel)
npm run test:e2e

# Load tests (18 scenarios)
npm run test:load
```

| Suite | Count | What it covers |
|-------|-------|---------------|
| Unit | 107 | All services/controllers in isolation with mocked workers |
| Integration | 17+1 | Real GPU gRPC calls: connection, routing, fairness, cancel, batching |
| E2E | 18 | Full HTTP → scheduler → GPU → response, streaming, priority |
| Load | 18 | Concurrency scaling, model thrashing, queue overflow, mixed modality, timeout |

## Not yet implemented

These are scoped in the architecture docs (`.claude/skills/scope/`) but not built:

| Feature | What's missing | Why it matters |
|---------|---------------|---------------|
| **Prefix sharing** | Shared system prompt KV cache across requests on the same GPU | S3 load test showed 96% wasted prefill — 20 requests with identical 134-token system prompt each recompute from scratch |
| **Weighted KV cache eviction** | Eviction policy using `recompute_cost × reuse_probability` instead of pure LRU | LRU evicts a 10K-token cache (expensive to recompute) over a 100-token one (cheap). Weighted eviction prevents this. |
| **Cascade dampening** | Rate-limiting evictions to prevent chain reactions | Evicting one cache can cause a miss, which triggers recompute, which evicts another cache, etc. |
| **Continuous batching** | New requests joining in-flight batches mid-decode | Current batching waits for all requests in a batch to finish. Continuous batching would let new arrivals join immediately, keeping the GPU fully utilized. |
| **Speculative decoding** | Draft model generates candidate tokens, main model verifies in batch | Trades GPU compute for latency — small model drafts fast, large model verifies in parallel. |
| **OOM retry** | Retry failed inference on a different GPU | Currently OOM during inference returns an error. Should retry on another worker with more VRAM headroom. |
| **Worker crash recovery** | Redistribute queued requests when a worker dies | Worker crash currently marks it unhealthy but queued requests are lost. |
| **Session migration** | Move KV cache to another GPU after crash (without thundering herd) | When a GPU with active sessions crashes, all those sessions need to restart somewhere. |
| **Content filtering** | Pre-inference safety checks, post-inference output filtering | No safety pipeline — requests go straight to GPU. |
| **Request deduplication** | Identical simultaneous prompts share one inference | 100 users sending the same prompt = 100 separate GPU inferences today. |
| **Structured output** | JSON mode, constrained decoding, grammar-guided generation | No way to guarantee output format (valid JSON, function calls, etc.). |
| **Token-level rate limiting** | Per-user tokens/sec limits (not just requests/sec) | Current backpressure is queue-depth based. A single user could monopolize GPU with long-context requests. |
| **API versioning** | Endpoint versioning, model version pinning, deprecation policy | No versioning strategy — breaking changes would affect all clients. |
| **Idempotency keys** | At-most-once delivery for non-streaming requests | Client retries can cause duplicate inference work. |
| **Quantization-aware routing** | Same model at FP16 on one GPU, INT4 on another; router picks based on quality/speed tradeoff | Currently all models loaded at FP16. No mixed-precision routing. |

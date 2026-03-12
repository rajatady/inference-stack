---
name: scope
description: "Source of truth for the LLM inference API architecture, design decisions, resource layers, test scenarios, and open questions. Reference this skill whenever making architectural or implementation decisions for the inference API."
user-invocable: true
argument-hint: "[section]"
---

# LLM Inference API — Architecture Scope

> Source of truth. All architectural decisions, constraints, and test scenarios live here.
> Append new discoveries. Never remove without discussion.

For detailed breakdowns of each section, see the supporting files:
- [infrastructure.md](infrastructure.md) — physical topology, network architecture, control/data plane separation, deployment model
- [grpc-contract.md](grpc-contract.md) — protobuf schema for gateway ↔ GPU worker communication
- [resource-layers.md](resource-layers.md) — GPU, VRAM, model, KV cache layers
- [kv-cache.md](kv-cache.md) — KV cache routing, eviction, prefix sharing
- [scheduler.md](scheduler.md) — scheduling, batching, queue, fairness
- [failure-modes.md](failure-modes.md) — OOM, crashes, cascading failures, migration
- [test-scenarios.md](test-scenarios.md) — critical integration and e2e test invariants
- [open-questions.md](open-questions.md) — unresolved design decisions

---

## 1. Problem Statement

Build a production-grade LLM inference API in TypeScript/NestJS that handles the same class of problems as OpenAI/Anthropic: GPU resource scheduling, KV cache-aware routing, streaming, dynamic batching, backpressure, and graceful failure — all under real GPU constraints.

---

## 2. Infrastructure & Network Architecture

See [infrastructure.md](infrastructure.md) for full detail.

**Key principle**: The API server NEVER runs on GPU machines. Three separate planes:

- **Control Plane** (CPU-only): model registry, health monitoring, autoscaler, deployment orchestration. Does NOT touch inference data.
- **Data Plane — Gateway** (CPU-only): API gateway, LLM-aware router, scheduler, KV cache manager. Handles HTTP, auth, tokenization, routing.
- **Data Plane — GPU Workers** (GPU nodes): only do inference. One worker process per GPU. Communicate via gRPC.

**Load balancing is 4-5 layers deep**: Global DNS → Regional API Gateway → Model-level routing → LLM-aware replica routing → (optional) disaggregated prefill/decode routing.

**A single inference request NEVER spans data centers.** Cross-DC latency is too high for tensor parallelism. Each DC runs complete model replicas; global router picks the DC.

---

## 3. System Overview

```
CONTROL PLANE (CPU-only, separate cluster)
┌──────────────────────────────────────────────────────────┐
│  Model Registry │ Health Monitor │ Autoscaler │ Deploy   │
│  (does NOT touch inference data path)                    │
└──────────────────────────┬───────────────────────────────┘
                           │ config, metrics
DATA PLANE                 │
┌──────────────────────────▼───────────────────────────────┐
│                      API Gateway                          │
│  (auth, API versioning, token-level rate limiting,        │
│   pre-flight tokenization, request deduplication)         │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│                  Safety / Filter Pipeline                  │
│          (pre-inference content filtering)                 │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│                    Request Router                          │
│  (session affinity, prefix matching, model selection,     │
│   KV cache lookup, quantization-aware routing)            │
│  *** Standard L7 LBs fail here — needs LLM awareness *** │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│                      Scheduler                            │
│  (priority queue, fairness, backpressure, batch           │
│   formation, GPU assignment, speculative decoding)        │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│                  KV Cache Manager                         │
│  (cache registry, VRAM accounting, eviction policy,       │
│   prefix tree, cache migration, cold-start tracking)      │
└────────────────────────┬─────────────────────────────────┘
                         │ gRPC (the network boundary)
                         │
GPU CLUSTER(S)           │
┌────────────────────────▼─────────────────────────────────┐
│                      GPU Pool                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │  GPU 0   │  │  GPU 1   │  │  GPU N   │               │
│  │ weights  │  │ weights  │  │ weights  │               │
│  │ kv[$a,b] │  │ kv[$d,e] │  │ kv[$g]   │               │
│  └──────────┘  └──────────┘  └──────────┘               │
│  1 worker process per GPU                                │
│  Exposes: infer, cancel, health, cache_state,            │
│           load_model, unload_model                       │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│                 Post-Inference Pipeline                    │
│  (safety filtering, structured output validation,         │
│   token counting, usage/billing, streaming delivery)      │
└──────────────────────────────────────────────────────────┘
```

---

## 4. Resource Layers

See [resource-layers.md](resource-layers.md) for full detail.

**Layer 1 — GPU VRAM Budget**: Fixed per GPU. Split between model weights and KV cache. Ratio is dynamic.

**Layer 2 — Model Placement**: Which GPUs hold which models. Includes quantization level (FP16, INT8, INT4). Loading/unloading is expensive (seconds to minutes).

**Layer 3 — KV Cache Placement**: Per-request/session computed context on GPU VRAM. The critical routing factor. Expensive to recompute, impossible to ignore.

**Layer 4 — Request Routing**: Given all of the above, which GPU, when, and batched with what.

---

## 5. KV Cache Routing

See [kv-cache.md](kv-cache.md) for full detail.

Three scenarios:
- **Session continuity** — multi-turn chat, sticky to GPU holding prior turns' KV cache
- **Prefix sharing** — common system prompts computed once, shared across requests on same GPU
- **Long context pinning** — large document KV caches pinned, follow-up queries routed there

Routing decision tree:
1. Warm cache exists? → Route to that GPU if available or wait is short
2. Shared prefix exists? → Route to GPU with prefix
3. Neither? → Least-loaded GPU with model loaded

---

## 6. KV Cache Eviction

Weighted eviction: `recompute_cost x reuse_probability`

Factors: recency, recompute cost (token count), reuse likelihood (active session vs one-shot), size.

Must prevent **eviction cascades** — evict cache, request arrives, recompute evicts another cache, chain reaction.

---

## 7. Scheduling & Batching

See [scheduler.md](scheduler.md) for full detail.

- **Priority tiers**: paid > free, realtime > batch
- **Fairness**: no user starves, even under load
- **Backpressure**: queue depth limit → 429 + retry-after
- **Dynamic batching**: accumulate within time window, continuous batching (new requests join in-flight batches)
- **Speculative decoding**: smaller draft model generates candidate tokens, larger model verifies in batch — trades GPU compute for latency

---

## 8. Failure Modes

See [failure-modes.md](failure-modes.md) for full detail.

- GPU OOM mid-inference → retry on different GPU, eventual error
- Worker process crash → redistribute queued requests
- Model load failure → mark GPU as degraded, route elsewhere
- Timeout on long sequences → configurable per-model limits
- Cascading eviction → dampening / eviction rate limiting
- GPU crash with active KV caches → session migration without thundering herd
- Model swap invalidating KV caches → account for downstream cost of invalidated sessions

---

## 9. Streaming & Client Interaction

- SSE for token streaming
- Client disconnect mid-stream → cancel GPU inference, free resources
- Backpressure if client consumes slowly → buffer limit, then cancel
- Request cancellation propagation through full stack (API → scheduler → GPU worker)

---

## 10. Pre/Post Inference Pipeline

### Pre-inference
- **Tokenization**: validate input, count tokens, reject if over model context window
- **Content filtering**: safety checks before GPU work is spent
- **Request deduplication**: identical prompts arriving simultaneously share one inference
- **Context window management**: truncation strategy vs rejection when input exceeds limits

### Post-inference
- **Safety filtering**: output content checks
- **Structured output validation**: JSON mode, tool use / constrained decoding
- **Token counting**: actual usage for billing (cached vs uncached tokens have different costs)
- **Usage tracking**: per API key, per model

---

## 11. Multi-GPU / Multi-Node

- **Tensor parallelism**: single model split across multiple GPUs (large models that don't fit on one GPU)
- **Pipeline parallelism**: different layers on different GPUs
- **Quantization awareness**: same model at FP16 on GPU-0, INT4 on GPU-3 — different quality/speed/VRAM tradeoffs, router must be aware
- **Cold start / warm-up**: first inference after model load is slower — scheduler should account for this

---

## 12. API Surface Concerns

- **API versioning**: model versions, endpoint versions, deprecation policy
- **Token-level rate limiting**: not just requests/sec but tokens/sec per user
- **Idempotency keys**: at-most-once delivery for non-streaming requests
- **Retry semantics**: client retries shouldn't cause duplicate inference work
- **Prompt caching billing**: differentiate cached prefix tokens (cheap) from fresh tokens (expensive)

---

## 13. Observability

- Distributed tracing through full pipeline (request → router → scheduler → GPU → response)
- Latency percentiles: time to first token, inter-token latency, total latency
- GPU utilization, VRAM usage, KV cache hit rates
- Queue depth, wait times, batch sizes
- Eviction rates, cache miss rates, recompute costs
- Per-model, per-GPU, per-user breakdowns

---

## 14. Critical Test Scenarios

See [test-scenarios.md](test-scenarios.md) for full detail.

1. Cache-hit routing under load (prefer GPU with warm cache over idle GPU)
2. Affinity vs latency tradeoff (when to recompute vs wait for warm GPU)
3. Prefix deduplication (1000 requests, same system prompt, compute once per GPU)
4. Eviction under memory pressure (evict right cache, not the one needed in 500ms)
5. Cascading failure from bad eviction (dampening)
6. Session migration after GPU crash (no thundering herd)
7. Model swap with active KV caches (downstream cost accounting)
8. Streaming + client disconnect → GPU freed
9. Queue depth overflow → 429 with accurate retry-after
10. Priority preemption with fairness (high priority served first, low priority not starved)
11. Dynamic batching within time window
12. OOM retry across GPUs with eventual error
13. Speculative decoding verification failure → fallback to standard decoding
14. Request deduplication for identical simultaneous prompts
15. Cold start latency accounting after model load

---

## 15. Tech Stack

- **API layer**: TypeScript, NestJS 11 (runs locally / CPU-only servers)
- **GPU worker boundary**: gRPC over network — the real infrastructure seam. Proto-loader with keepCase, enums as strings.
- **GPU workers**: Python + transformers + grpcio (no vLLM — we control the full stack for learning). Written locally in `gpu-worker/`, rsynced to RunPod.
- **Testing**: Jest. All integration tests run against real GPU workers on RunPod, not mocks.

---

## 16. Simulation Setup

See [infrastructure.md](infrastructure.md) "Our Simulation Setup" for full detail.

**Local machine (MacBook)**: Control plane + data plane gateway + test suite. NestJS API, router, scheduler, KV cache manager, batch formation. All CPU-only logic.

**RunPod cluster** (ssh root@213.173.98.26 -p 13461): 2x NVIDIA RTX A4500 (20GB VRAM each), 48 CPU cores, 251GB RAM. Exposes gRPC endpoints for infer, cancel, health, cache_state, load_model, unload_model.

**Model Roster** (6 models across 5 modalities):

| Model | Type | VRAM (FP16) | Default GPU | HuggingFace ID |
|---|---|---|---|---|
| SmolLM2-135M-Instruct | Text gen (chat) | ~0.3GB | GPU-0 | HuggingFaceTB/SmolLM2-135M-Instruct |
| SmolLM2-360M-Instruct | Text gen (chat) | ~0.7GB | GPU-0 | HuggingFaceTB/SmolLM2-360M-Instruct |
| SmolLM2-1.7B-Instruct | Text gen (chat) | ~3.5GB | GPU-0 | HuggingFaceTB/SmolLM2-1.7B-Instruct |
| Qwen2.5-VL-3B | Vision-language | ~7GB | GPU-0 | Qwen/Qwen2.5-VL-3B-Instruct |
| Kokoro-82M | TTS / Audio | ~0.5GB | GPU-0 | hexgrad/Kokoro-82M |
| SD Turbo | Image gen | ~5-6GB | GPU-1 | stabilityai/sd-turbo |
| CogVideoX-2B | Video gen | ~6-8GB | GPU-1 | THUDM/CogVideoX-2b |

GPU-0 default load: SmolLM2-135M-Instruct + Qwen2.5-VL-3B + Kokoro-82M ≈ 8GB (headroom for KV cache + swaps)
GPU-1 default load: SD Turbo + CogVideoX-2B ≈ 12-14GB (tight — triggers real VRAM pressure)

**Simulation scenarios enabled**:
- Model loading/unloading: swap SmolLM2-135M-Instruct ↔ SmolLM2-360M-Instruct on GPU-0
- Cross-modality scheduling: text, vision, audio, image, video requests competing
- VRAM pressure: GPU-1 is tight enough to trigger real eviction decisions
- Latency diversity: text (ms) vs image (seconds) vs video (minutes)
- Batch incompatibility: can't batch text with image gen requests

**What it simulates**: Real network latency (Mac ↔ RunPod = real API-to-cluster hop), real GPU constraints (20GB VRAM), model heterogeneity (6 models, 5 modalities), independent workers, real VRAM pressure on GPU-1.

**What it doesn't simulate**: Multi-datacenter (can fake with artificial latency), tensor parallelism (models too small), scale (2 GPUs not 2000, but logic is identical), spot preemption (can simulate by killing workers).

---

## Changelog

- **2026-03-11**: Initial scope. Covers resource layers, KV cache routing/eviction, scheduling, batching, failure modes, streaming, pre/post inference pipeline, multi-GPU, API surface, observability, 15 critical test scenarios.
- **2026-03-11**: Added infrastructure layer. Physical topology (node → rack → cluster → DC), control plane / data plane separation, communication protocols (gRPC), load balancing layers (4-5 deep), scale examples (hyperscaler / mid / small player). Added simulation setup mapping local Mac to RunPod GPUs. Renumbered to 16 sections.
- **2026-03-11**: Added gRPC contract (grpc-contract.md). Full protobuf schema: InferenceWorker service with 8 RPCs, message schemas for inference (streaming), model management, state observation, cache management, health. 6 key design decisions documented.
- **2026-03-11**: Finalized multi-modal model roster. 6 models across 5 modalities: SmolLM2-135M/360M (text), Qwen2.5-VL-3B (vision), Kokoro-82M (TTS), SD Turbo (image gen), CogVideoX-2B (video gen). GPU-0 ≈ 8GB default, GPU-1 ≈ 12-14GB (tight, triggers real VRAM pressure).
- **2026-03-11**: Dropped vLLM in favor of raw transformers+grpcio for GPU workers (full control, learning exercise). Decided: no mocks — all tests against real GPU workers. Code written locally, rsynced to RunPod. NestJS gRPC client module complete (GpuWorkerModule + GpuWorkerService, 8 RPCs).
- **2026-03-11**: Completions resource built. OpenAI-compatible POST /v1/completions with streaming (SSE) and non-streaming. TypeORM + SQLite for persistence. 19 unit tests + 12 e2e tests. UI playground at public/index.html.
- **2026-03-11**: Started Worker Orchestrator layer — the "brain" between HTTP API and GPU workers. Three new services: WorkerRegistry (dynamic multi-worker gRPC clients via ClientProxyFactory), ModelManager (auto-load/unload, VRAM-aware placement, concurrent load coalescing), Router (model affinity → least loaded → trigger load). GpuWorkerService refactored from NestJS singleton to plain class instantiated per-worker by registry. GpuWorkerModule replaced by WorkerOrchestratorModule.
- **2026-03-11**: Completed Phases A-D (Scheduler + Batching + Cancel + Tokenization). New modules: TokenizerService (chars/4 approximation, context window validation), SchedulerService (per-user FIFO queues, round-robin within priority tiers, aging, queue depth + token budget → 429 + Retry-After, cancel for queued + active requests), BatchCollector (time-window accumulation, max batch size, compatibility filtering by token ratio, per-model buckets). CompletionsService rewired from Router to Scheduler, returns { promise, cancel } / { stream$, cancel }. Controller wires res.on('close') → cancel for both streaming and non-streaming. ModelManager caches ModelCapabilities from LoadModelResponse. 68 unit tests + 24 integration tests (T7-T10 scheduler fairness, T11/T13 batching, T16 cancel-disconnect). T12 (continuous batching) skipped — requires Python worker changes.
- **2026-03-11**: Connected Everything — wired BatchCollector into SchedulerService.tryDispatch(), added GET /v1/completions/stats endpoint (queueDepth, totalQueuedTokens, activeCount), updated UI playground with priority dropdown, user ID input, and live queue stats polling. Fixed 5 bugs discovered during E2E testing against real GPUs: (1) activeCount going negative from double-decrement in finishRequest() — made idempotent with Map.has() guard, (2) POST returning 201 instead of 200 — added @HttpCode(200), (3) response ID mismatch between scheduler internal ID and DB entity ID — fixed with spread override, (4) delete test assertion failing on empty object — switched to .toBeUndefined() check, (5) error model test too strict after scheduler refactoring. Added E2E tests for scheduler integration (priority/user fields, stats endpoint) and cancel-on-disconnect (AbortController + native fetch). Final count: 69 unit + 17 integration + 18 e2e = 104 tests passing.
- **2026-03-11**: Resolved 6 open questions through implementation: Q1 (three separate components: Router, SchedulerService, BatchCollector), Q2 (per-user FIFO queues + per-model batching buckets), Q7 (linear aging with configurable agingBoostPerSecond), Q8 (static batching window, adaptive deferred), Q16 (8 RPCs fully defined, 5 implemented in Python worker), Q17 (static config via WorkerRegistry). Updated test-scenarios.md with implementation status for all 22 scenarios (7 implemented, 1 skipped, 2 partial, 12 not started).
- **2026-03-11**: Approved plan for Multi-Worker + Full Multi-Modal (6 models, 5 modalities). Decided: separate OpenAI-compatible endpoints per modality (POST /v1/images/generations, POST /v1/audio/speech, POST /v1/video/generations). Proto extensions: image_data/image_mime_type on InferRequest, MediaOutput variant on InferResponse, model_type + modality flags on ModelCapabilities. Python worker refactored with pipeline abstraction (BasePipeline → text_gen, vision_language, tts, image_gen, video_gen). NestJS gateway gets model roster config, GPU affinity in ModelManager, and 3 new resource modules (images, audio, video). 5-phase implementation: multi-worker verification → proto changes → Python pipelines → NestJS endpoints → testing.
- **2026-03-12**: Completed Phases 1-5 (Multi-Worker + Full Multi-Modal). All 6 models across 5 modalities verified end-to-end against real GPUs on RunPod. Key implementation details and discoveries:
  - **Phase 1 — Multi-Worker**: Both GPU workers running simultaneously (worker-0 on GPU-0 port 50051, worker-1 on GPU-1 port 50052). Discovered CUDA_VISIBLE_DEVICES remapping: `CUDA_VISIBLE_DEVICES=1` maps physical GPU-1 to device index 0, so worker must use `--gpu-id 0`. Added `physical_gpu_id` to worker.py for correct pynvml indexing (pynvml is NOT affected by CUDA_VISIBLE_DEVICES).
  - **Phase 2 — Proto Changes**: Extended `inference_worker.proto` with `image_data`/`image_mime_type` on InferRequest, `MediaOutput` variant on InferResponse (`bytes data`, `string mime_type`, `bool is_final`), `model_type` on LoadModelRequest, and modality flags on ModelCapabilities (`supports_image_input/output`, `supports_audio_output`, `supports_video_output`). Regenerated Python stubs.
  - **Phase 3 — Python Pipeline Abstraction**: Refactored `gpu-worker/worker.py` with `MODEL_TYPE_REGISTRY` mapping model IDs → pipeline types. 5 pipeline implementations in `gpu-worker/pipelines/`: TextGenPipeline (AutoModelForCausalLM + TextIteratorStreamer), VisionLanguagePipeline (Qwen2VLForConditionalGeneration + AutoProcessor, decodes image_data bytes → PIL), TTSPipeline (KPipeline from kokoro, returns WAV bytes as MediaOutput), ImageGenPipeline (AutoPipelineForText2Image from diffusers/SD Turbo, returns PNG bytes), VideoGenPipeline (CogVideoXPipeline, returns MP4 bytes). Extended `InferenceResult` with `media_data`, `media_mime_type`, `is_media_final`. Server.py yields `MediaOutput` responses alongside `TokenChunk`.
  - **Phase 4 — NestJS Multi-Modal Endpoints**: Added `src/config/model-roster.ts` mapping all 6 models → `{ type, vramEstimate, defaultGpu }`. ModelManager updated with GPU affinity sorting (preferred worker first, then VRAM). Three new resource modules: ImagesModule (`POST /v1/images/generations` → `{ created, data: [{ b64_json }] }`), AudioModule (`POST /v1/audio/speech` → raw WAV binary), VideoModule (`POST /v1/video/generations` → raw MP4 binary). Vision support added to completions: `images?: string[]` on CreateCompletionDto, `image_data` passthrough in SchedulerService. UI playground updated with all 6 models, modality-aware output (image/audio/video display), image upload for vision.
  - **Phase 5 — Testing**: All 6 models verified via curl against real GPUs. Unit tests: 87/87 passing (15 suites). Integration tests: 48+ passing (7 suites). E2E tests: 20/20 passing (against real GPU workers). Total: 155+ tests. Test contention discovered: suites fail when run concurrently (GPU resource conflicts), all pass individually.
  - **Critical bugs found and fixed**: (1) CUDA_VISIBLE_DEVICES + pynvml mismatch → added `physical_gpu_id` resolution. (2) CogVideoX-2b OOM: `.to(device)` + `enable_model_cpu_offload()` compete for VRAM — removed `.to(device)`, use only CPU offload (peak dropped from ~19GB to ~5GB). (3) Vision model ignoring images: no DTO field or scheduler passthrough — added `images` to DTO and `image_data` Buffer passthrough. (4) Video export missing OpenCV: installed `imageio` + `imageio-ffmpeg` as recommended backend. (5) TTS OOM from stale GPU processes: killed stale processes, restarted workers on clean GPUs.
  - **Verified API endpoints**: SmolLM2-135M (text gen, ~0.3GB GPU-0), SmolLM2-360M (text gen, ~0.7GB GPU-0), Qwen2.5-VL-3B (vision-language, ~7GB GPU-0), Kokoro-82M (TTS/audio, ~0.5GB GPU-0), SD Turbo (image gen, ~5-6GB GPU-1), CogVideoX-2B (video gen, ~5GB with CPU offload GPU-1).
- **2026-03-12**: ClickHouse Metrics Pipeline + Load Test. Built full observability stack and ran first baseline measurements against real GPUs.
  - **MetricsModule** (`src/metrics/`): Global NestJS module with ClickHouseService (connection wrapper, graceful fallback if unavailable), MetricsService (recordInference fire-and-forget, getTps, getLatencyPercentiles, getBreakdown, getHistory), MetricsController (GET /v1/metrics/tps, /latency, /breakdown, /history). Uses `@clickhouse/client` (not TypeORM — ClickHouse is column-oriented OLAP, no UPDATE/DELETE).
  - **ClickHouse schema**: `inference.inference_metrics` table — 25 columns covering identity, GPU timing (prefill/decode/total), gateway timing (queue wait, routing, e2e, TTFT), tokens, computed TPS (decode_tps, prefill_tps), request context, media fields, error tracking. MergeTree engine, ordered by (model, timestamp), partitioned by month.
  - **Data loss fix**: Two points where GPU timing was being discarded: (1) `scheduler.service.ts` dispatchRequest() — now forwards full usage object + adds gateway timing (`_timing.queueWaitMs`, `_timing.routingTimeMs`), (2) all 5 service files (completions, images, audio, video) — now populate entity timing columns and call `metricsService.recordInference()`.
  - **Bug found**: `new Date().toISOString()` produces `2026-03-11T21:36:12.798Z` which ClickHouse DateTime64(3) can't parse in JSONEachRow format. Fixed to `Math.round(Date.now() / 1000)` (epoch seconds). All recordInference calls were silently failing (fire-and-forget .catch swallowed the parse error).
  - **Bug found**: `@clickhouse/client` warns "socket was closed or ended before response was fully read" on DDL queries. Fixed by using `result.text()` (fully drains stream) instead of `result.close()` (kills socket).
  - **Load test suite** (`test/load/load-test.spec.ts`): 6 scenarios against real GPUs, all tagged via `user` field for per-scenario ClickHouse queries. `npm run test:load` to run.
  - **Load test results** (SmolLM2-135M on RTX A4500, raw transformers, no vLLM):
    - **S1 Baseline**: ~42 decode TPS, ~130ms prefill, ~500ms decode, ~1s e2e for 30 tokens
    - **S2 Concurrency scaling**: TPS drops linearly (41→20→7→2 at c1→c2→c4→c8). No GPU-side batching benefit — expected with raw transformers (no continuous batching). E2E goes from 934ms to 10142ms at c8.
    - **S3 Prefix sharing waste**: 96.1% wasted prefill. 20 requests × 134-token shared prefix = 750ms total prefill, only 30ms needed with KV cache prefix sharing. 721ms wasted.
    - **S4 Multi-turn recompute**: 5-turn conversation growing 135→352 tokens. Prefill TPS actually increases with prompt length (249→8309 — GPU more efficient on larger batches). Cumulative recompute waste: 142ms. Decode dominates at ~500ms/turn regardless of prompt length.
    - **S5 Sustained load**: 20 requests @ 1rps, stable p50=944ms, 0 errors.
    - **S6 Cross-model contention**: Text+image concurrent shows <2% TPS impact — different GPUs, no interference as expected.
  - **Key finding for KV cache architecture**: Decode is the bottleneck (~500ms, 92% of GPU time), not prefill (~37ms, 5%). Prefix sharing saves 96% of prefill but prefill is small. The bigger win is **continuous batching** (letting multiple requests share GPU decode time). Under high concurrency (S2-c8), prefill grows to 511ms — prefix sharing becomes much more valuable under load.
  - **Test count**: 107 unit tests (16 suites) + 20 metrics-specific tests + 9 load test scenarios = 116+ tests.
- **2026-03-12**: GPU-Side Batch Inference — True tensor-level batching end-to-end, integrated with existing BatchCollector and SchedulerService.
  - **Proto**: Added `BatchInfer` RPC + `BatchInferRequest` (repeated InferRequest) to `inference_worker.proto`. Reuses `InferResponse` with `request_id` for demuxing per-request results. Total RPCs: 9.
  - **Python TextGenPipeline.infer_batch()**: Left-pads all prompts (`tokenizer.padding_side = 'left'`), single `model.generate()` call with batched input_ids + attention_mask. Splits outputs per request, removes trailing pad tokens. Non-streaming (complete text + InferComplete per request). Sequential fallback for non-text models.
  - **Python server.py**: `BatchInfer` RPC servicer validates same-model constraint, builds request dicts, yields tagged InferResponse messages per request.
  - **NestJS GpuWorkerService**: Added `batchInfer()` method calling gRPC `BatchInfer` RPC.
  - **NestJS BatchCollector**: Added `batchDispatch` callback + `context` field on BatchableRequest. When batch > 1 and callback set, calls batch handler instead of individual dispatch. Defaults: `enabled: true`, `maxBatchSize: 256`, `windowMs: 50`.
  - **NestJS SchedulerService**: Added `dispatchBatch()` — routes all requests to same worker, calls `batchInfer()`, demuxes results by `request_id` back to individual request promises. Single requests still use individual `Infer` RPC.
  - **Results**: Throughput now scales with concurrency. c=8: 2 TPS → 316 TPS (158x improvement). Per-request GPU time: 4.8s → 70ms. Validated via `npm run test:load` + ClickHouse comparison (183 pre-batching rows vs 92 post-batching rows).
  - **Bug**: Scheduler unit tests broke with batching enabled by default (`worker.batchInfer is not a function` — mock workers only had `infer`). Fix: `batchCollector.setConfig({ enabled: false })` in scheduler test beforeEach.
- **2026-03-12**: Instruct Model Integration — switched from base models to instruct-tuned variants with OpenAI-compatible chat messages support.
  - **Architecture decision**: Chat template application happens in Python worker (where tokenizer lives), not NestJS gateway. Gateway sends structured `messages: [{role, content}]` via gRPC `ChatMessage` proto type. Worker calls `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`. This is model-agnostic — same code path works for any model with a chat template in `tokenizer_config.json`.
  - **Proto**: Added `ChatMessage { string role, string content }` + `repeated ChatMessage messages = 9` on InferRequest. Both proto files updated (inference-api + gpu-worker), Python stubs regenerated.
  - **Python**: TextGenPipeline gained `_resolve_prompt()` method — if `messages` present, applies chat template; else falls back to raw `prompt`. Used in both `infer()` and `infer_batch()` (batch uses same `_resolve_prompt()` per request before left-padded tokenization). MODEL_TYPE_REGISTRY updated: removed SmolLM2-135M/360M, added SmolLM2-135M-Instruct/360M-Instruct/1.7B-Instruct.
  - **NestJS**: CreateCompletionDto accepts `messages?: Array<{role, content}>` alongside `prompt`. Controller validates either is present. SchedulerService passes messages through in `dispatchRequest()` and `dispatchBatch()`, estimates tokens from message content for queue budget. CompletionsService stores `JSON.stringify(messages)` in prompt DB column (no schema change needed).
  - **Model roster**: Removed 2 base models, added 3 instruct models (SmolLM2-135M-Instruct 0.3GB, SmolLM2-360M-Instruct 0.7GB, SmolLM2-1.7B-Instruct 3.5GB). Total: 7 models, 5 modalities.
  - **UI**: System prompt textarea above message input, instruct models in dropdown, auto-builds `messages` array. Model detection via ID containing "Instruct".
  - **Verified E2E**: SmolLM2-135M-Instruct loaded, received messages, applied chat template, responded conversationally (212ms warm, 12s cold including model load).
- **2026-03-12**: Per-Request Timeout + Stress Test Suite (S7-S12). Added `requestTimeoutMs` to SchedulerConfig (default 60s), timeout timer in dispatchRequest()/dispatchBatch(), cleared in finishRequest()/cancel(), `setRequestTimeout()` for test overrides. Extended load-test.spec.ts with 6 stress scenarios using realistic conversation-length prompts and extended concurrency ramp [1,2,4,8,16,32,64,128,256]:
  - **S7 Model Size Scaling**: All 3 text models peak at c=32. 135M→1,134 TPS, 360M→971 TPS, 1.7B→993 TPS. Surprise: 1.7B faster than 135M at low concurrency (53 vs 43 TPS at c=1).
  - **S8 Same-GPU Model Thrashing**: ~48ms swap cost (negligible — small models co-exist in VRAM). ~3% throughput degradation in interleaved mode. Model swap is nearly free for sub-1GB models.
  - **S9 Queue Overflow + Recovery**: 500 burst → all accepted, 0 rejections (batching absorbs everything). BatchCollector dispatches within 50ms windows with batch sizes up to 256. Drain in 503ms. Queue overflow would require sustained rate exceeding GPU throughput, not burst.
  - **S10 Mixed Modality (text + TTS on same GPU-0)**: 889% text latency increase when concurrent with TTS. This is the biggest real-world contention finding — text and TTS compete for same GPU compute.
  - **S11 Sustained Overload**: 6 rps × 30s = 180 requests. 174/180 succeeded, 0 zombies, 503ms drain, 891ms recovery latency. System handles 2x estimated capacity gracefully.
  - **S12 Request Timeout**: 5s timeout fires at 5.1s, activeCount returns to 0, recovery request succeeds. Timeout cleanup is clean.
  - **Key insight**: Batching is so effective that the system never produces 429s under burst load — BatchCollector absorbs bursts by dispatching large batches. The real bottleneck is mixed-modality contention on the same GPU (S10), not queue depth.
- **2026-03-12**: Disaggregated KV Cache — CPU DRAM Persistence. Implemented full disaggregated KV cache pipeline: CPU DRAM-backed store on GPU worker with LRU eviction, session routing from NestJS gateway, and S14-S16 stress tests proving the cost case.
  - **`gpu-worker/kv_cache_store.py`** (NEW): CPU DRAM-backed KV cache store. Thread-safe (threading.Lock + OrderedDict). Handles transformers v5.x (`DynamicCache.layers` with `.keys`/`.values` attributes), v4.x (`key_cache`/`value_cache`), and legacy tuple format. `save()` moves tensors to CPU, `load()` builds DynamicCache via `cache.update(k_gpu, v_gpu, layer_idx)` + `.to(device, non_blocking=True)` + `torch.cuda.synchronize()`. LRU eviction when over budget. Default 200GB budget (of 251GB DRAM). `set_max_bytes()` for testing.
  - **`gpu-worker/pipelines/base.py`** (MODIFIED): Added 5 cache fields to `InferenceResult`: `cache_hit`, `cache_load_ms`, `cache_save_ms`, `cache_size_bytes`, `cached_tokens`.
  - **`gpu-worker/pipelines/text_gen.py`** (MAJOR REWRITE): `infer()` now: (1) checks `request["cache_hint"]["session_id"]`, (2) on cache hit, tokenizes only new tokens from `request["new_prompt"]` + passes `past_key_values=restored_cache` to `model.generate()`, (3) uses `return_dict_in_generate=True` to extract past_key_values, (4) saves cache after generate. `infer_batch()` deliberately skips KV cache — can't batch per-request past_key_values with different sequence lengths.
  - **`gpu-worker/worker.py`** (MODIFIED): Creates `KVCacheStore` in `__init__()`, passes to TextGenPipeline. Reports cache stats (total_dram_bytes, hit_count, miss_count, hit_rate) in `get_worker_state()`.
  - **`gpu-worker/server.py`** (MODIFIED): Forwards `cache_hint` + `new_prompt` in `Infer()` and `BatchInfer()`. Populates real `CacheInfo` (cache_id, cached_tokens, new_tokens, cache_size_bytes) and `cache_load_ms`/`cache_save_ms` on `UsageStats` in `InferComplete`.
  - **Proto**: Added `float cache_load_ms = 7` and `float cache_save_ms = 8` on `UsageStats` in both `inference-api/proto/` and `gpu-worker/proto/`.
  - **NestJS Gateway**: `session_id` on CreateCompletionDto, auto-generated via `uuidv4()` in controller, returned in response. SchedulerService forwards `session_id` through `dispatchRequest()`/`dispatchBatch()` into `cache_hint`. Extracts `cache_load_ms`/`cache_save_ms` from worker response.
  - **S14 test result** (SmolLM2-1.7B-Instruct): 9/10 cache hits, 14.1% compute savings. Crossover at ~800 tokens. Cache load 0-33ms, save 45-263ms. At <2K tokens, savings are modest — real cost case emerges at longer contexts and with Anthropic's 0.1x cached pricing.
  - **Critical discovery**: transformers v5.3.0 on RunPod uses `DynamicCache.layers` (list of `DynamicLayer` with `.keys`/`.values`), NOT v4.x `.key_cache`/`.value_cache`. Both APIs handled in `kv_cache_store.py`.
  - **S14-S16 tests written** in `load-test.spec.ts`. S14 ran and verified. S15 (concurrent sessions) and S16 (eviction under budget) written but not yet run.
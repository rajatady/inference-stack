# Critical Test Scenarios — Detailed Breakdown

These are the architectural invariants the system must exhibit. Each is an integration or e2e test scenario targeting the hardest behaviors.

---

## Category: KV Cache Routing

### T1. Cache-hit routing under load
**Setup**: All GPUs busy. Request arrives with warm KV cache on GPU-2. GPU-2 finishes a request.
**Invariant**: System routes waiting request to GPU-2, even if GPU-0 freed up first.
**Why it's hard**: Naive load-balancing would pick the first free GPU.
**Status**: NOT STARTED — requires KV cache manager implementation.

### T2. Affinity vs latency tradeoff
**Setup**: Request has warm cache on GPU-2 (50 requests queued). GPU-0 is idle but has no cache.
**Invariant**: System correctly decides whether to wait (short history, fast recompute) or recompute (long history, long wait). Decision should be based on estimated wait time vs estimated recompute time.
**Why it's hard**: Requires real-time cost estimation for both paths.
**Status**: NOT STARTED — requires KV cache manager implementation.

### T3. Prefix deduplication
**Setup**: 1000 requests arrive with identical 2K-token system prompt within 1 second.
**Invariant**: System prompt KV cache is computed at most once per GPU. Requests sharing the prefix skip that prefill work.
**Why it's hard**: Prefix matching must be token-exact. Cache must be shared without corruption. Concurrent access to shared prefix cache.
**Status**: NOT STARTED — requires KV cache manager implementation.

### T4. Long context pinning and routing
**Setup**: User uploads 100K-token document. Follows up with 5 questions over 10 minutes.
**Invariant**: All 5 follow-up queries route to the GPU holding the document KV cache. Cache is not evicted between queries despite idle time.
**Why it's hard**: 10 minutes of idle time would normally trigger eviction. System must recognize the session pattern.
**Status**: NOT STARTED — requires KV cache manager implementation.

---

## Category: Eviction

### T5. Eviction under memory pressure
**Setup**: GPU is at 95% VRAM. New high-priority request arrives needing 8% VRAM for its KV cache.
**Invariant**: System evicts the cache with lowest `recompute_cost * reuse_probability / size`, not the most recent or largest.
**Why it's hard**: Requires maintaining accurate reuse probability estimates.
**Status**: NOT STARTED — requires KV cache manager implementation.

### T6. Cascading eviction prevention
**Setup**: GPU full. Evict cache A. 200ms later, request for A arrives. Recompute A would evict B. Request for B arrives 300ms later.
**Invariant**: System detects the cascade risk and either queues the recomputation or routes to a different GPU instead.
**Why it's hard**: Requires predictive awareness of near-future requests (or at minimum, a dampening mechanism).
**Status**: NOT STARTED — requires KV cache manager implementation.

---

## Category: Scheduling & Fairness

### T7. Priority preemption with fairness
**Setup**: Queue has [low, low, low, HIGH] priority requests. One GPU frees up.
**Invariant**: HIGH is served first. But if this pattern repeats 100 times, low-priority requests still make progress (aging).
**Why it's hard**: Pure priority → starvation. Pure fairness → priority is meaningless. Need balanced aging.
**Status**: IMPLEMENTED — SchedulerService uses Priority enum (HIGH=0, NORMAL=1, LOW=2) with linear aging (`agingBoostPerSecond`). `dequeueNext()` finds the best (lowest) effectivePriority across all queue heads, then round-robins among users whose heads are within 0.5 tolerance. Tested in `test/integration/scheduler-fairness.spec.ts` (3 tests: priority ordering, aging prevents starvation, round-robin within tier).

### T8. Per-user fairness under contention
**Setup**: User A sends 500 requests. User B sends 5 requests. Same priority tier.
**Invariant**: User B's requests are not stuck behind all 500 of User A's. System interleaves fairly.
**Why it's hard**: Per-user queuing vs global queue tradeoffs.
**Status**: IMPLEMENTED — SchedulerService uses per-user FIFO queues (`Map<userId, QueuedRequest[]>`) with round-robin dispatch. `dequeueNext()` iterates users starting from `rrIndex`, advancing each dispatch. User B's requests interleave with User A's regardless of User A's queue depth. Tested in `test/integration/scheduler-fairness.spec.ts` (per-user fairness test, round-robin interleaving test).

### T9. Queue depth overflow
**Setup**: Queue hard limit is 100. 101st request arrives.
**Invariant**: 101st request gets 429 with accurate `Retry-After` header. Already-queued requests are not affected.
**Why it's hard**: Retry-After must be a real estimate, not a fixed number.
**Status**: IMPLEMENTED — SchedulerService.admissionCheck() throws HttpException(429) when `totalQueued >= maxQueueDepth`. Response includes `retryAfter` estimated from `Math.max(1, Math.ceil(activeCount))`. Controller catches HttpException, sets `Retry-After` header, returns 429. Tested in `test/integration/scheduler-fairness.spec.ts` (queue depth overflow test). Also tested in E2E via `test/e2e/inference-api.e2e-spec.ts` (stats endpoint verifies queue metrics).

### T10. Token budget limiting
**Setup**: Queue has 50 requests, total queued tokens = 900K. Limit is 1M tokens. New request has 200K tokens.
**Invariant**: Request rejected (would exceed token budget), even though request count is under limit.
**Why it's hard**: Must track aggregate token count, not just request count.
**Status**: IMPLEMENTED — SchedulerService tracks `totalQueuedTokens` (incremented on enqueue, decremented on dequeue). `admissionCheck()` rejects when `totalQueuedTokens + estimatedTokens > maxQueuedTokens` with 429 + Retry-After. Token count estimated via TokenizerService (chars/4 heuristic). Tested in `test/integration/scheduler-fairness.spec.ts` (token budget overflow test).

---

## Category: Batching

### T11. Dynamic batching within time window
**Setup**: 3 requests arrive within 10ms batching window for same model on same GPU.
**Invariant**: Sent as one batch, not 3 individual inferences.
**Why it's hard**: Timing-sensitive. Window must close and dispatch correctly.
**Status**: IMPLEMENTED — BatchCollector accumulates requests in per-model buckets. Timer starts on first request (`windowMs`). When timer fires or `maxBatchSize` reached, all compatible requests in the bucket are dispatched together. Requests are compatibility-filtered by `maxSeqLengthRatio` (sequence length similarity). Tested in `test/integration/batching.spec.ts` (window-based batching test, immediate dispatch at max batch size test).

### T12. Continuous batching slot reclamation
**Setup**: Batch of 4 running. Request 2 finishes (short output). Request 5 is queued.
**Invariant**: Request 5 joins the batch in request 2's slot without waiting for requests 1, 3, 4 to finish.
**Why it's hard**: Mid-batch modification of batch composition. KV cache management during slot swap.
**Status**: PARTIALLY IMPLEMENTED — GPU-side static batching is implemented (BatchInfer RPC: left-padded tensor batching via `model.generate()`, all requests in batch start and finish together). True continuous batching (mid-batch slot reclamation) still requires Python worker to manage a batch slot table and accept new requests into running batches. Current batching gives 158x throughput improvement at c=8 (2→316 TPS) but wastes GPU cycles when short requests finish before long ones in the same batch. Test placeholder exists in `test/integration/batching.spec.ts` (skipped for continuous batching, static batching tested via load tests).

### T13. Batch compatibility filtering
**Setup**: 3 requests for same model. Two have 500-token inputs, one has 50K-token input.
**Invariant**: The two similar requests are batched together. The 50K request runs separately (padding waste would be too high).
**Why it's hard**: Sequence length similarity heuristic. Threshold tuning.
**Status**: IMPLEMENTED — BatchCollector uses `maxSeqLengthRatio` (configurable, e.g., 4.0) to filter compatible requests. When forming a batch, it checks if the ratio between max and min estimated token counts exceeds the threshold. Incompatible requests wait for the next batch window. Tested in `test/integration/batching.spec.ts` (compatibility filtering test — verifies small and large requests dispatch in separate batches).

---

## Category: Failure & Recovery

### T14. OOM retry across GPUs
**Setup**: GPU-0 OOMs mid-inference. GPU-1 is available.
**Invariant**: Request retried on GPU-1 with conservative VRAM reservation. After 2 failed retries → error to client.
**Why it's hard**: Retry state management. Don't retry on the same GPU that OOMed.
**Status**: NOT STARTED — requires OOM detection in worker error responses and retry logic in SchedulerService.dispatchRequest().

### T15. Session migration without thundering herd
**Setup**: GPU-2 crashes. It held KV caches for 15 active conversations.
**Invariant**: Recomputation of those 15 sessions is spread across GPUs with staggered timing. No single GPU gets overwhelmed.
**Why it's hard**: Coordination of staggered recomputation. Priority ordering of sessions.
**Status**: NOT STARTED — requires KV cache manager and session tracking implementation.

### T16. Streaming disconnect frees GPU
**Setup**: Client starts streaming request. Disconnects at token 50 of a 500-token generation.
**Invariant**: GPU inference is cancelled. KV cache VRAM is freed (or retained briefly for possible reconnect). GPU is available for next request within bounded time.
**Why it's hard**: Cancellation propagation through scheduler → GPU worker. Cleanup timing.
**Status**: IMPLEMENTED — Full cancel-on-disconnect pipeline: Controller wires `res.on('close')` → `cancel()` for both streaming and non-streaming. CompletionsService.create() and createStream() return `{ promise/stream$, cancel }`. `cancel()` calls `SchedulerService.cancel(requestId)` which: (1) for queued requests: removes from per-user queue, rejects promise, (2) for active requests: unsubscribes gRPC Observable (triggers gRPC context cancellation → Python worker receives cancellation signal), calls `finishRequest()` (idempotent, decrements activeCount, triggers tryDispatch). E2E tested in `test/e2e/inference-api.e2e-spec.ts` (AbortController abort → server stays healthy, activeCount >= 0). Integration tested in `test/integration/cancel-disconnect.spec.ts` (T16: streaming cancel, T16b: non-streaming cancel, queued request cancel, server health after cancel).

### T17. Model swap cost accounting
**Setup**: Requests queued for model B. Only available GPU holds model A with 10 active KV caches.
**Invariant**: System calculates cost of invalidating 10 sessions vs. benefit of serving model B requests. Makes the correct tradeoff.
**Why it's hard**: Multi-factor cost comparison across different dimensions (sessions lost vs requests served).
**Status**: NOT STARTED — requires KV cache manager and cost accounting in ModelManager. ModelManager currently does simple VRAM-aware placement with `tryEvictIdle()` but no session-aware cost analysis.

---

## Category: Pre/Post Inference

### T18. Request deduplication
**Setup**: 5 identical requests (same model, same prompt, same parameters) arrive within 100ms.
**Invariant**: Only one inference runs. All 5 clients receive the same result.
**Why it's hard**: Race condition window. Must handle streaming dedup (all 5 streams fed from one inference).
**Status**: NOT STARTED — requires deduplication layer in scheduler or pre-scheduler pipeline.

### T19. Context window rejection
**Setup**: Request with 150K tokens arrives for a model with 128K context window.
**Invariant**: Rejected at tokenization phase, before any GPU work. Clear error message with token count.
**Why it's hard**: Tokenization must be fast and happen before scheduling.
**Status**: PARTIALLY IMPLEMENTED — TokenizerService has `validateContextWindow(prompt, maxContext)` that estimates tokens (chars/4) and throws if over limit. ModelManager caches `ModelCapabilities.max_context_length` from LoadModelResponse. The validation is available but not yet wired into the scheduler admission check (currently only queue depth and token budget are checked at admission). Wiring the context window check into SchedulerService.enqueue() is straightforward. Tested in `test/integration/tokenization.spec.ts` (context window validation unit test).

### T20. Speculative decoding fallback
**Setup**: Draft model generates 5 candidate tokens. Target model rejects token 3.
**Invariant**: Tokens 1-2 are kept. Generation continues from token 3 via standard decoding. Client sees no disruption in stream.
**Why it's hard**: Seamless mid-generation switch between decoding strategies.
**Status**: NOT STARTED — requires speculative decoding implementation in Python worker and scheduler coordination for draft/target model pairing.

---

## Category: Observability (Testable)

### T21. End-to-end trace propagation
**Setup**: Request enters API. Flows through router → scheduler → GPU worker → response.
**Invariant**: A single trace ID is present at every stage. Latency breakdown is accurate.
**Why it's hard**: Trace context must cross async boundaries, queue waits, and potentially process boundaries.
**Status**: NOT STARTED — requires distributed tracing implementation (OpenTelemetry or similar).

### T22. Accurate retry-after under varying load
**Setup**: System under 50% load → 80% load → 95% load.
**Invariant**: Retry-After values in 429 responses scale proportionally with actual queue wait times. Measured error < 30%.
**Why it's hard**: Estimation accuracy under changing conditions.
**Status**: PARTIALLY IMPLEMENTED — SchedulerService.estimateRetryAfter() returns `Math.max(1, Math.ceil(activeCount))` as a rough estimate. This scales with active request count but doesn't yet account for average inference time or queue depth per model. The formula is `retry_after ≈ activeCount` (seconds), which is a reasonable first approximation. More accurate estimation (incorporating actual inference time history, per-model queue depth, and GPU count) is deferred.

---

## Implementation Summary

| Scenario | Status | Test File |
|---|---|---|
| T1-T4 | NOT STARTED | — (requires KV cache manager) |
| T5-T6 | NOT STARTED | — (requires KV cache manager) |
| T7 | IMPLEMENTED | test/integration/scheduler-fairness.spec.ts |
| T8 | IMPLEMENTED | test/integration/scheduler-fairness.spec.ts |
| T9 | IMPLEMENTED | test/integration/scheduler-fairness.spec.ts + test/e2e/inference-api.e2e-spec.ts |
| T10 | IMPLEMENTED | test/integration/scheduler-fairness.spec.ts |
| T11 | IMPLEMENTED | test/integration/batching.spec.ts |
| T12 | PARTIALLY IMPLEMENTED | test/integration/batching.spec.ts (static batching done, continuous batching deferred) |
| T13 | IMPLEMENTED | test/integration/batching.spec.ts |
| T14 | NOT STARTED | — (requires OOM retry logic) |
| T15 | NOT STARTED | — (requires KV cache manager + session tracking) |
| T16 | IMPLEMENTED | test/integration/cancel-disconnect.spec.ts + test/e2e/inference-api.e2e-spec.ts |
| T17 | NOT STARTED | — (requires KV cache manager + cost accounting) |
| T18 | NOT STARTED | — (requires deduplication layer) |
| T19 | PARTIALLY IMPLEMENTED | test/integration/tokenization.spec.ts (validation exists, not wired into admission) |
| T20 | NOT STARTED | — (requires speculative decoding) |
| T21 | NOT STARTED | — (requires distributed tracing) |
| T22 | PARTIALLY IMPLEMENTED | — (basic estimate exists, accuracy improvement deferred) |

**Score: 7 implemented + 0 skipped + 3 partial + 12 not started = 22 total scenarios**

---

## Category: Multi-Modal (Added 2026-03-12)

### T23. Multi-modal model loading with GPU affinity
**Setup**: 6 models across 2 GPUs. Each model has a preferred GPU (from model roster). Worker-0 has more VRAM available.
**Invariant**: Models load on their preferred GPU (e.g., SD Turbo → worker-1, Kokoro → worker-0), not just the GPU with most VRAM.
**Why it's hard**: Default VRAM-sorting would put everything on the emptiest GPU. Need affinity-aware sorting.
**Status**: IMPLEMENTED — ModelManager.loadOnBestWorker() sorts candidates by `getDefaultGpu(modelId)` preference first, then by VRAM. Verified via E2E: SD Turbo loads on worker-1, Kokoro on worker-0, matching model roster. Implemented in `src/worker-orchestrator/model-manager.ts` with `src/config/model-roster.ts`.

### T24. Vision model image passthrough
**Setup**: POST /v1/completions with `images: ["<base64>"]` and model=Qwen2.5-VL-3B.
**Invariant**: Image bytes flow through DTO → scheduler → gRPC image_data → Python worker → PIL Image → model. Model generates text describing the image.
**Why it's hard**: Base64 → Buffer → gRPC bytes → PIL conversion chain. Small images fail (10x10 PNG too small for PIL). Image must be large enough for the vision processor.
**Status**: IMPLEMENTED — CreateCompletionDto has `images?: string[]`. SchedulerService passes `Buffer.from(base64)` as `image_data` + `image_mime_type` to worker.infer(). VisionLanguagePipeline decodes bytes → PIL Image → AutoProcessor. Verified via curl with 256x256 PNG (Qwen2.5-VL correctly identified colors). Tested in E2E.

### T25. Media output pipeline (image/audio/video)
**Setup**: POST to /v1/images/generations, /v1/audio/speech, /v1/video/generations.
**Invariant**: Each returns the correct media type — PNG bytes (base64 in JSON), WAV binary, MP4 binary. MediaOutput flows from Python pipeline → gRPC → NestJS service → HTTP response.
**Why it's hard**: Large binary data through gRPC (video can be 45KB+). Different response formats per modality. gRPC max message size must be increased for video.
**Status**: IMPLEMENTED — Three NestJS resource modules (ImagesModule, AudioModule, VideoModule) each with controller, service, DTO. Python pipelines return `InferenceResult(media_data=bytes, media_mime_type=str, is_media_final=bool)`. Server.py yields `MediaOutput` proto messages. NestJS services collect media chunks, return formatted responses. grpc.max_send_message_length set to 100MB for video. Verified all 3 via curl against real GPUs.

### T26. CogVideoX CPU offloading under VRAM pressure
**Setup**: GPU-1 has ~20GB VRAM. CogVideoX-2B needs ~19GB peak without offloading.
**Invariant**: Video generation completes without OOM by using `enable_model_cpu_offload()` instead of `.to(device)`. Peak GPU VRAM stays under 6GB.
**Why it's hard**: `enable_model_cpu_offload()` and `.to(device)` conflict — calling both loads the full model to GPU then tries to manage it, exceeding VRAM. Must use only CPU offload. VAE slicing/tiling further reduce peak memory.
**Status**: IMPLEMENTED — VideoGenPipeline uses only `enable_model_cpu_offload(gpu_id=gpu_idx)` + `enable_vae_slicing()` + `enable_vae_tiling()`. No `.to(device)` call. Peak VRAM dropped from ~19GB to ~5GB. 50 diffusion steps complete in ~80 seconds, producing ~45KB MP4. Implemented in `gpu-worker/pipelines/video_gen.py`.

### T27. Cross-worker model routing
**Setup**: SmolLM2-135M on worker-0, SD Turbo on worker-1. Text request arrives, then image request.
**Invariant**: Text request routes to worker-0 (has text model), image request routes to worker-1 (has image model). No unnecessary model loading/unloading.
**Why it's hard**: Router must check model affinity across multiple workers before triggering a load.
**Status**: IMPLEMENTED — Router.route() checks all workers via WorkerRegistry.getSnapshots(), prefers worker already holding the model, falls back to ModelManager.loadOnBestWorker() with GPU affinity. Verified via E2E tests with concurrent text + image requests routing to correct workers.

---

## Observability & Performance (T28-T33) — IMPLEMENTED

### T28. ClickHouse metrics pipeline end-to-end
**Setup**: POST /v1/completions → ClickHouse should have a row with timing + token + TPS data.
**Invariant**: Every inference request produces exactly one row in `inference.inference_metrics` with non-zero GPU timing, gateway timing, token counts, and computed TPS.
**Status**: IMPLEMENTED — MetricsService.recordInference() inserts fire-and-forget into ClickHouse. All 5 service files (completions, images, audio, video + scheduler) call it. Bug found: ISO timestamps broke ClickHouse parsing — fixed with epoch seconds. 91 rows verified after load test.

### T29. Concurrency scaling measurement
**Setup**: Fire 8 requests at concurrency 1, 2, 4, 8 to SmolLM2-135M.
**Invariant**: With GPU-side batching, TPS should scale with concurrency. Without batching, TPS degrades linearly.
**Measured (before batching)**: c1=41 TPS/934ms, c2=20 TPS/1649ms, c4=7.2 TPS/3416ms, c8=2.2 TPS/10100ms. Linear degradation — each concurrent request queues behind others.
**Measured (after batching)**: c1=44 TPS, c8=316 TPS. TPS now scales linearly with concurrency. 158x improvement at c=8.
**Status**: IMPLEMENTED — `test/load/load-test.spec.ts` S2.

### T30. Prefix sharing waste quantification
**Setup**: 20 sequential requests with identical 134-token system prompt + unique 20-token query.
**Invariant**: Total prefill time should be ~20× single prefill. With KV cache prefix sharing, it would be ~1×.
**Measured**: 750ms total prefill, 30ms for first request. 96.1% wasted (721ms). With prefix cache: only 30ms needed.
**Status**: IMPLEMENTED — `test/load/load-test.spec.ts` S3.

### T31. Multi-turn recompute cost
**Setup**: 5-turn conversation, each turn sends full history (135→352 tokens growing).
**Invariant**: Prefill grows with context length. Decode stays constant (~40 TPS). Cumulative recompute waste measurable.
**Measured**: Prefill TPS increases with prompt length (249→8309 — GPU more efficient on larger batches). Decode stable at ~40 TPS/~500ms. Cumulative prefill waste: 142ms. Decode dominates (92% of GPU time).
**Status**: IMPLEMENTED — `test/load/load-test.spec.ts` S4.

### T32. Sustained load stability
**Setup**: 20 requests @ 1 rps to SmolLM2-135M.
**Invariant**: p95 latency should stay stable (no drift). Queue depth should not grow unboundedly. Zero errors under sustainable load.
**Measured**: Stable p50≈944ms, 0 errors, queue depth bounded.
**Status**: IMPLEMENTED — `test/load/load-test.spec.ts` S5.

### T33. Cross-model contention (text + image)
**Setup**: 3 text requests (GPU-0) + 2 image requests (GPU-1) fired concurrently.
**Invariant**: Text TPS should not degrade when image gen runs on a different GPU.
**Measured**: Text-only=11.5 TPS, text+image=11.7 TPS. <2% impact — different GPUs, no interference.
**Status**: IMPLEMENTED — `test/load/load-test.spec.ts` S6.

---

## Category: Disaggregated KV Cache (Added 2026-03-12)

### T34. Recompute vs Cache Transfer Cost (S14)
**Setup**: SmolLM2-1.7B-Instruct, 10-turn conversation. Phase A: no session_id (full recompute each turn). Phase B: with session_id (KV cache stored in CPU DRAM, loaded on subsequent turns).
**Invariant**: Phase B should show cache hits on turns 2-10 with measurable compute savings. Cache load (PCIe transfer) should be cheaper than recompute at sufficient context length.
**Measured**: 9/10 cache hits (Turn 1 always cold start). 14.1% average compute savings. Crossover at ~800 tokens — below this, recompute is cheaper than cache load overhead. Cache load: 0-33ms. Cache save: 45-263ms (post-generation, not user-facing).
**Status**: IMPLEMENTED — `test/load/load-test.spec.ts` S14. KVCacheStore in `gpu-worker/kv_cache_store.py`, DynamicCache integration in `gpu-worker/pipelines/text_gen.py`, session routing via `session_id` on DTO through NestJS scheduler to gRPC `cache_hint`.

### T35. Concurrent Session Caching (S15)
**Setup**: 5 sessions × 3 turns each, unique session_ids, SmolLM2-1.7B-Instruct. Real conversation data (different slices).
**Invariant**: Each session gets cache hits on turns 2-3. Total DRAM usage scales linearly with sessions. No cross-session cache corruption.
**Status**: IMPLEMENTED (test written) — `test/load/load-test.spec.ts` S15. Not yet run against GPUs.

### T36. Cache Eviction Under Budget Constraint (S16)
**Setup**: KVCacheStore set to 500MB budget (fits ~2-3 sessions of 1.7B at 1K tokens). Create 8 sessions → LRU eviction kicks in after 2-3.
**Invariant**: Revisiting evicted sessions shows cache miss + recompute. Non-evicted (recently used) sessions still have cache hits. LRU ordering is correct.
**Status**: IMPLEMENTED (test written) — `test/load/load-test.spec.ts` S16. Not yet run against GPUs.

---

## Implementation Summary

| Scenario | Status | Test File |
|---|---|---|
| T1-T4 | NOT STARTED | — (requires KV cache-aware routing in NestJS router) |
| T5-T6 | NOT STARTED | — (requires eviction cascading protection) |
| T7 | IMPLEMENTED | test/integration/scheduler-fairness.spec.ts |
| T8 | IMPLEMENTED | test/integration/scheduler-fairness.spec.ts |
| T9 | IMPLEMENTED | test/integration/scheduler-fairness.spec.ts + test/e2e/inference-api.e2e-spec.ts |
| T10 | IMPLEMENTED | test/integration/scheduler-fairness.spec.ts |
| T11 | IMPLEMENTED | test/integration/batching.spec.ts |
| T12 | PARTIALLY IMPLEMENTED | test/integration/batching.spec.ts (static batching done, continuous batching deferred) |
| T13 | IMPLEMENTED | test/integration/batching.spec.ts |
| T14 | NOT STARTED | — (requires OOM retry logic) |
| T15 | NOT STARTED | — (requires KV cache manager + session tracking) |
| T16 | IMPLEMENTED | test/integration/cancel-disconnect.spec.ts + test/e2e/inference-api.e2e-spec.ts |
| T17 | NOT STARTED | — (requires KV cache manager + cost accounting) |
| T18 | NOT STARTED | — (requires deduplication layer) |
| T19 | PARTIALLY IMPLEMENTED | test/integration/tokenization.spec.ts (validation exists, not wired into admission) |
| T20 | NOT STARTED | — (requires speculative decoding) |
| T21 | NOT STARTED | — (requires distributed tracing) |
| T22 | PARTIALLY IMPLEMENTED | — (basic estimate exists, accuracy improvement deferred) |
| T23-T27 | IMPLEMENTED | test/e2e/inference-api.e2e-spec.ts + integration tests |
| T28-T33 | IMPLEMENTED | test/load/load-test.spec.ts (S1-S6) |
| T34 | IMPLEMENTED | test/load/load-test.spec.ts (S14) |
| T35 | IMPLEMENTED (test written, not run) | test/load/load-test.spec.ts (S15) |
| T36 | IMPLEMENTED (test written, not run) | test/load/load-test.spec.ts (S16) |

**Score: 18 implemented + 2 written-not-run + 3 partial + 12 not started = 35 total scenarios**

---

## Changelog

- **2026-03-12**: Added T28-T33 (observability + performance scenarios). All implemented via ClickHouse metrics pipeline + load test suite. Key finding: decode is 92% of GPU time, prefix sharing saves 96% of prefill but prefill is only 5% of total. Continuous batching would be the bigger win over KV cache alone.
- **2026-03-12**: Added T34-T36 (disaggregated KV cache scenarios). T34 (recompute vs cache transfer) implemented and verified — 9/10 cache hits, 14.1% savings, crossover at ~800 tokens. T35 (concurrent sessions) and T36 (eviction under budget) tests written but not yet run against GPUs. KV cache stored in CPU DRAM on GPU worker (in-process Python memory), session routing via session_id through NestJS scheduler to gRPC cache_hint. Critical discovery: transformers v5.3.0 uses DynamicCache.layers (not key_cache/value_cache) — both APIs handled in kv_cache_store.py.

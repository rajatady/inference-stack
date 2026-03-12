# Open Questions — Unresolved Design Decisions

These need answers before or during implementation. Append decisions as they're made.

---

## Architecture

### Q1. Unified scheduler vs. separate router + scheduler + cache manager?
**Tradeoff**: Tight coupling = simpler, faster decisions. Loose coupling = testable in isolation, but coordination overhead.
**Leaning**: Three components with shared VRAM state view.
**Status**: DECIDED — Three separate components: Router (picks best worker per request, model affinity → least loaded → trigger load), SchedulerService (per-user FIFO queues, priority, fairness, backpressure, cancel), BatchCollector (time-window accumulation, max batch size, compatibility filtering, per-model buckets). Router and SchedulerService share state through WorkerRegistry snapshots. BatchCollector is wired into SchedulerService.tryDispatch(). All three are independently testable. Implemented in `src/worker-orchestrator/router.ts`, `src/scheduler/scheduler.service.ts`, `src/scheduler/batch-collector.ts`.

### Q2. Per-model queues vs. global queue?
**Tradeoff**: Per-model queues are simpler. Global queue allows cross-model fairness and priority.
**Status**: DECIDED — Hybrid approach. SchedulerService uses per-user FIFO queues (Map<userId, QueuedRequest[]>) with round-robin dispatch within priority tiers. This gives cross-model fairness (same user's requests for different models share a queue) and per-user fairness (round-robin prevents any single user from monopolizing). BatchCollector adds per-model bucket isolation on top (requests are grouped by model for batching). Implemented in `src/scheduler/scheduler.service.ts` (per-user queues) and `src/scheduler/batch-collector.ts` (per-model batching buckets).

### Q3. Cache migration: V1 feature or deferred?
**Tradeoff**: Migration avoids recompute for very large caches. But adds complexity.
**Leaning**: Defer to V2. Recompute is simpler and sufficient initially.
**Status**: DECIDED — deferred

---

## KV Cache

### Q4. How to estimate reuse probability for eviction scoring?
Options:
- Pure recency (LRU)
- Session activity heuristic (active session = high probability)
- Statistical model trained on access patterns
**Status**: OPEN

### Q5. Prefix cache sharing — how to handle concurrent writes?
When multiple requests try to create the same prefix cache simultaneously.
Options:
- Lock on prefix hash, first writer wins, others wait
- Optimistic: all compute, first to finish is kept, others discarded
**Status**: OPEN

### Q6. KV cache TTL — fixed or adaptive?
**Tradeoff**: Fixed TTL is simple but wasteful. Adaptive TTL based on load/VRAM pressure is smarter but complex.
**Status**: OPEN

---

## Scheduling

### Q7. Aging function for priority fairness?
Linear aging? Exponential? What's the time constant?
**Status**: DECIDED — Linear aging. Configurable `agingBoostPerSecond` (default from `DEFAULT_SCHEDULER_CONFIG`). Each second a request waits, its `effectivePriority` decreases (lower = higher priority) by `agingBoostPerSecond`. This means a LOW priority request waiting long enough will eventually outpriority a fresh HIGH request. Round-robin among users whose queue heads are within 0.5 tolerance of the best priority. Implemented in `src/scheduler/scheduler.service.ts` with aging timer (interval-based) and dequeueNext() tolerance check. Tested in `test/integration/scheduler-fairness.spec.ts` (T7).

### Q8. Batching window duration — static or adaptive?
**Tradeoff**: Static is predictable. Adaptive (wider window under load, narrower when idle) optimizes throughput/latency.
**Status**: DECIDED — Static for now. Configurable `windowMs` in BatchCollector (default from `DEFAULT_SCHEDULER_CONFIG`). Window starts on first request, dispatches when timer fires or `maxBatchSize` is reached (whichever comes first). Adaptive deferred to future iteration. Implemented in `src/scheduler/batch-collector.ts`. Tested in `test/integration/batching.spec.ts` (T11).

### Q9. Speculative decoding — opt-in per request or system-decided?
**Tradeoff**: Per-request gives users control. System-decided is simpler for clients.
**Status**: OPEN

---

## API Surface

### Q10. How to expose cache state to clients?
Should clients know if their request hit a cache? Response header? Billing metadata?
**Status**: PARTIALLY RESOLVED — `session_id` is auto-generated and returned in response body for client reuse. Worker returns `CacheInfo` (cached_tokens, new_tokens, cache_size_bytes) and `cache_load_ms`/`cache_save_ms` on `UsageStats`. Not yet surfaced to HTTP response headers or billing metadata. Gateway currently forwards `session_id` but not cache hit/miss status to client.

### Q11. Idempotency key format and TTL?
How long to remember idempotency keys? Client-provided vs server-generated?
**Status**: OPEN

### Q12. How to handle partial streaming failure for non-streaming clients?
Non-streaming request fails mid-generation. Return partial result with error flag? Or just error?
**Status**: OPEN

---

## Operations

### Q13. GPU health check protocol?
Heartbeat frequency? What counts as "unhealthy"? How long before marking a GPU as dead vs. degraded?
**Status**: OPEN

### Q14. Degraded mode triggers?
At what capacity threshold does the system enter degraded mode? Manual or automatic?
**Status**: OPEN

### Q15. Multi-node deployment model?
One NestJS API instance per node? Central API with distributed GPU workers? Service mesh?
**Status**: DECIDED — Central API (local Mac) with distributed GPU workers (RunPod). Mirrors real-world: CPU-only API servers, separate GPU clusters.

---

## Infrastructure

### Q16. gRPC service definition for GPU workers?
What RPCs does a GPU worker expose? Proposed: `Infer` (unary + server-streaming), `Cancel`, `Health`, `CacheState`, `LoadModel`, `UnloadModel`. Need to define proto schemas.
**Status**: DECIDED — Fully defined in `proto/inference_worker.proto` with 8 RPCs: `Infer` (server-streaming), `LoadModel`, `UnloadModel`, `GetWorkerState`, `WatchWorkerState`, `GetCacheEntries`, `EvictCache`, `Health`. No explicit `Cancel` RPC — cancellation is via gRPC context cancellation (standard pattern). Full message schemas documented in `grpc-contract.md`. Python GPU worker implements 5 of 8: Infer, LoadModel, UnloadModel, GetWorkerState, Health. NestJS GpuWorkerService wraps all 8 as methods. Cancellation via RPC context works end-to-end (tested in cancel-disconnect.spec.ts).

### Q17. How does the gateway discover GPU workers?
Static config? Service discovery (Consul, etcd)? For our simulation, static SSH tunnel or direct IP. In production, Kubernetes service discovery or custom registry.
**Status**: DECIDED — Static config via WorkerRegistry. Workers are defined in `src/worker-orchestrator/worker-orchestrator.module.ts` with env vars or defaults (worker-0 at localhost:50051, worker-1 at localhost:50052). WorkerRegistry instantiates a GpuWorkerService per worker config, manages connection lifecycle, provides snapshots (VRAM, loaded models, active inferences). In production this would be replaced with service discovery, but the WorkerRegistry interface stays the same. Implemented in `src/worker-orchestrator/worker-registry.ts`.

### Q18. Worker-side metrics push vs pull?
Workers push metrics to gateway (gRPC stream)? Or gateway polls workers (gRPC unary)?
Push = real-time but more complex. Pull = simpler but introduces staleness.
**Status**: OPEN

### Q19. How to handle network partition between gateway and GPU cluster?
Timeout thresholds? Retry policy? How quickly to mark workers as unreachable? What happens to in-flight streaming requests when connection drops?
**Status**: OPEN

### Q20. Multi-cluster routing for small players?
Small player with GPUs in datacenter A (5 GPUs) and datacenter B (10 GPUs). How does the global router decide? Latency to user? Available capacity? Model availability? Cost?
**Status**: OPEN

### Q21. Disaggregated prefill/decode — V1 or deferred?
Running prefill (compute-bound) and decode (memory-bound) on separate GPU pools. Up to 30x throughput on large models. But adds KV cache transfer complexity.
**Leaning**: Defer. Our models are tiny.
**Status**: OPEN

---

## Changelog

- **2026-03-11**: Initial 15 open questions from architecture review.
- **2026-03-11**: Added Q16-Q21 from infrastructure layer review. Resolved Q15 (central API with distributed workers).
- **2026-03-11**: Resolved Q1 (three separate components: Router, SchedulerService, BatchCollector), Q2 (per-user FIFO queues + per-model batching buckets), Q7 (linear aging with configurable agingBoostPerSecond), Q8 (static batching window, adaptive deferred), Q16 (8 RPCs fully defined in proto, 5 implemented in Python worker), Q17 (static config via WorkerRegistry, env vars or defaults). All resolved through implementation and testing against real GPUs.
- **2026-03-12**: Multi-modal implementation resolved additional design decisions: (1) GPU affinity — ModelManager sorts candidates by model roster's `defaultGpu` preference before VRAM, ensuring models land on intended GPUs. (2) CUDA_VISIBLE_DEVICES handling — worker.py resolves physical GPU ID for pynvml separately from CUDA device index. (3) CPU offloading strategy for large models — CogVideoX uses `enable_model_cpu_offload()` exclusively (no `.to(device)`), reducing peak VRAM from ~19GB to ~5GB. (4) Pipeline abstraction — Python worker uses MODEL_TYPE_REGISTRY mapping HuggingFace IDs → pipeline types, with BasePipeline ABC defining load/infer/unload/get_capabilities interface. (5) Multi-modal endpoints follow OpenAI API conventions: images return JSON with b64_json, audio returns raw WAV binary, video returns raw MP4 binary.
- **2026-03-12**: Disaggregated KV cache implementation partially resolved Q10 (cache state exposure) — session_id returned in response body, CacheInfo + cache timing returned from worker via gRPC. Not yet surfaced to client HTTP response headers or billing metadata.

# KV Cache Routing & Eviction — Detailed Breakdown

## Why KV Cache Routing Is the Hardest Problem

KV cache represents **computed work**. Every token in a conversation's history has been processed through every transformer layer, producing key and value tensors stored on GPU VRAM. Losing this cache means redoing all that computation (prefill), which:
- Increases time-to-first-token dramatically
- Wastes GPU compute that could serve other requests
- Creates unpredictable latency spikes for users

## Routing Scenarios

### A. Session Continuity (Multi-Turn Chat)

**Situation**: User sends turn N of a conversation. GPU-X holds the KV cache from turns 1 through N-1.

**Ideal**: Route to GPU-X. Prefill only needs to process the new user message, not the entire history.

**Complication**: GPU-X might be busy. The decision becomes:
- **Wait cost**: How long until GPU-X is free? (queue depth * avg inference time)
- **Recompute cost**: How many tokens in the history? (prefill time on a cold GPU)
- **Crossover point**: When wait_time > recompute_time, route elsewhere

The system needs a real-time estimate of both costs to make this decision.

### B. Prefix Sharing (Common System Prompts)

**Situation**: Many requests share the same system prompt (e.g., "You are a helpful assistant" — 500 tokens, or a complex agent prompt — 5000 tokens).

**Approach**: Compute the prefix KV cache once per GPU. All requests with that prefix on the same GPU skip prefix prefill.

**Implementation considerations**:
- Prefix tree (trie) mapping token sequences to GPU cache locations
- Prefix matching must be exact (token-level, not string-level)
- Partial prefix matches are valuable — if a request shares the first 3000 of 5000 prefix tokens, those 3000 can be reused
- Popular prefixes should be replicated across multiple GPUs to distribute load

### C. Long Context / Document QA

**Situation**: User uploads a 100K-token document. KV cache is enormous (potentially GBs).

**Approach**: Pin the cache. Route all follow-up queries to that GPU. Eviction cost is extremely high.

**Complications**:
- One document can consume a large fraction of a GPU's KV cache budget
- If the user stops querying, that VRAM is wasted
- TTL-based expiry with generous timeout for long-context caches
- May need to notify user that context was evicted (reupload needed)

## KV Cache Registry

Central data structure tracking all caches across all GPUs:

```
CacheEntry:
  id: string
  gpu_id: string
  model: string
  quantization: string
  token_hash: string          // hash of token sequence
  token_count: number         // for recompute cost estimation
  vram_bytes: number          // actual VRAM consumed
  created_at: timestamp
  last_accessed_at: timestamp
  access_count: number
  session_id: string | null   // null for shared prefix caches
  cache_type: session | prefix | document
  is_pinned: boolean          // prevents eviction
```

## Eviction Policy

### Weighted Score

Each cache gets an eviction score. Lowest score gets evicted first.

```
eviction_score = (recompute_cost * reuse_probability) / vram_consumed
```

Where:
- `recompute_cost` = f(token_count, model_size) — how expensive to recreate
- `reuse_probability` = f(recency, access_frequency, session_active, cache_type)
- `vram_consumed` = actual bytes on GPU

High score = keep. Low score = evict.

### Eviction Cascade Prevention

Problem: Evict cache A → request for A arrives → recompute A → A's prefill evicts cache B → request for B arrives → chain reaction.

Mitigations:
- **Eviction rate limiting**: max N evictions per time window per GPU
- **Grace period**: recently evicted cache IDs get a brief "cool-down" where new requests for them are routed elsewhere rather than triggering recompute on the same GPU
- **Prefill VRAM reservation**: before starting prefill, ensure enough VRAM exists for the new cache WITHOUT evicting anything. If not possible, queue the request instead.

## Cache Migration

Moving a KV cache between GPUs:
- Technically possible (GPU-to-GPU transfer over NVLink or PCIe)
- Expensive in bandwidth and latency
- Only worthwhile for very large caches (100K+ tokens) where recompute would be worse
- Not a V1 feature — recompute is simpler and sufficient initially

---

## Disaggregated KV Cache — CPU DRAM Persistence (IMPLEMENTED)

### Why Disaggregate to CPU DRAM

GPUs become stateless. KV cache stored in CPU RAM, loaded to any GPU on demand via PCIe.

- **VRAM is scarce** (~16GB free on RTX A4500) — holds ~40 sessions at 384MB each (1.7B model, 2K tokens)
- **DRAM is abundant** (~251GB on RunPod) — holds ~1,300 sessions
- **PCIe transfer** (~25ms for 384MB) is **7x cheaper** than recompute (~174ms for 2K tokens at 1.7B)
- Enables 1-hour TTL caching like Anthropic's prompt caching (0.1x price for cached tokens)

### KV Cache Math (1.7B model)

- **192 KB/token**: 24 layers × 32 KV heads × 64 head_dim × 2 bytes (FP16) × 2 (K+V)
- At 2K tokens: **384 MB** per session
- PCIe 4.0 transfer: ~15-22 GB/s → ~25ms to move 384MB
- Recompute 2K tokens: ~174ms (measured in S13d)
- **Crossover point**: ~800 tokens — below this, recompute is cheaper than cache load

### Implementation

**`gpu-worker/kv_cache_store.py`**: CPU DRAM-backed store with LRU eviction. Thread-safe (`threading.Lock` + `OrderedDict`). Handles:
- transformers v5.x: `DynamicCache.layers` → list of `DynamicLayer` objects with `.keys`/`.values` tensor attributes
- transformers v4.x: `DynamicCache.key_cache`/`.value_cache` lists
- Legacy tuple format: `((k, v), (k, v), ...)`

`save()`: Iterates layers, calls `.cpu()` on each K/V tensor, tracks total bytes. LRU eviction when over budget (default 200GB of 251GB).
`load()`: Builds `DynamicCache()`, calls `cache.update(k_gpu, v_gpu, layer_idx)` per layer, moves tensors to GPU with `.to(device, non_blocking=True)`, `torch.cuda.synchronize()`.

**`gpu-worker/pipelines/text_gen.py`**: `infer()` rewritten — before generate: check `cache_hint.session_id` → `kv_store.load()`. On cache hit, tokenize only new tokens (`request["new_prompt"]`), pass `past_key_values=restored_cache`. Uses `return_dict_in_generate=True` to extract `past_key_values` from output. After generate: `kv_store.save()`. `infer_batch()` deliberately skips KV cache — can't batch per-request past_key_values with different sequence lengths.

**NestJS Gateway**: `session_id` on `CreateCompletionDto`, auto-generated via `uuidv4()` in controller, forwarded through `SchedulerService.dispatchRequest()`/`dispatchBatch()` into gRPC `cache_hint`. `cache_load_ms`/`cache_save_ms` extracted from worker response.

### Measured Results (S14 — SmolLM2-1.7B-Instruct)

- 9/10 cache hits (Turn 1 always cold start)
- 14.1% average compute savings across 10 turns
- Cache load: 0-33ms, Cache save: 45-263ms (post-generation, not user-facing)
- At <2K tokens, savings modest — real cost case emerges with longer contexts and Anthropic's 0.1x cached pricing

### Remaining Work

- **Cache-aware routing** (T1-T2): NestJS router picks GPU with warm cache for session_id
- **Prefix sharing** (T3): Compute KV once for shared system prompts, reuse across requests
- **Eviction cascading protection** (T4-T6): Dampening, rate limiting, grace periods
- **Gateway-level cache registry**: NestJS tracks which sessions are cached on which worker, for routing decisions without round-tripping to GPU worker

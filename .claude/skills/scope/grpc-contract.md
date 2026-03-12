# gRPC Contract — Gateway ↔ GPU Worker Protocol

## Design Principles

1. **Workers are dumb**: They load models, run inference, report state. No routing, no scheduling, no business logic.
2. **Gateway is smart**: All decisions (routing, batching, eviction, priority) live in the gateway.
3. **Contract is the seam**: Everything the gateway needs to make decisions must be available through this contract. If the gateway can't observe it, it can't optimize for it.
4. **Streaming is first-class**: Token generation is inherently streaming. The contract must support it natively, not as an afterthought.

---

## Service Definition

```protobuf
syntax = "proto3";

package inference.worker.v1;

service InferenceWorker {
  // === Inference ===

  // Run inference. Returns a stream of tokens.
  // Gateway sends one request, worker streams back tokens as they're generated.
  // Client-side cancellation: gateway cancels the RPC context → worker stops generation.
  rpc Infer (InferRequest) returns (stream InferResponse);

  // Run batch inference. Multiple requests processed as single batched model.generate() call.
  // All requests must target the same model. Results tagged by request_id for demuxing.
  rpc BatchInfer (BatchInferRequest) returns (stream InferResponse);

  // === Model Management ===

  // Load a model onto this worker's GPU. Blocking — returns when model is ready.
  rpc LoadModel (LoadModelRequest) returns (LoadModelResponse);

  // Unload a model from this worker's GPU. Destroys all KV caches for that model.
  rpc UnloadModel (UnloadModelRequest) returns (UnloadModelResponse);

  // === State Observation ===

  // Get current worker state: loaded models, VRAM usage, active inferences.
  // Gateway polls this to build its routing/scheduling view.
  rpc GetWorkerState (GetWorkerStateRequest) returns (WorkerState);

  // Stream of state updates. Worker pushes whenever state changes materially.
  // Gateway subscribes on connect, uses this for real-time routing decisions.
  rpc WatchWorkerState (WatchWorkerStateRequest) returns (stream WorkerState);

  // Get KV cache entries on this worker. Gateway uses this for cache-aware routing.
  rpc GetCacheEntries (GetCacheEntriesRequest) returns (CacheEntriesResponse);

  // Explicitly evict a KV cache entry. Gateway decides what to evict, worker executes.
  rpc EvictCache (EvictCacheRequest) returns (EvictCacheResponse);

  // === Health ===

  // Standard gRPC health check. Used by gateway for liveness/readiness.
  rpc Health (HealthRequest) returns (HealthResponse);
}
```

---

## Message Schemas

### Inference

```protobuf
message InferRequest {
  string request_id = 1;           // Unique ID for tracing and cancellation
  string model_id = 2;             // Which model to run (must be loaded)

  // Input — exactly one of these must be set
  oneof input {
    repeated int32 token_ids = 3;  // Pre-tokenized input (gateway tokenizes)
    string prompt = 4;             // Raw text (worker tokenizes — fallback)
  }

  // Generation parameters
  GenerationParams params = 5;

  // KV cache hint — tells the worker to look for/create a cache with this key
  CacheHint cache_hint = 6;

  // Multi-modal input
  bytes image_data = 7;              // Raw image bytes for vision-language models
  string image_mime_type = 8;        // "image/jpeg", "image/png", etc.

  // Chat messages (instruct models) — worker applies chat template
  repeated ChatMessage messages = 9;
}

message ChatMessage {
  string role = 1;                   // "system", "user", "assistant"
  string content = 2;               // Message text
}

message GenerationParams {
  int32 max_tokens = 1;            // Max tokens to generate
  float temperature = 2;           // Sampling temperature
  float top_p = 3;                 // Nucleus sampling
  int32 top_k = 4;                 // Top-k sampling
  repeated string stop_sequences = 5; // Stop generation on these strings
  bool echo = 6;                   // Return input tokens in output

  // Structured output
  oneof constraint {
    string json_schema = 7;        // JSON mode — constrain output to this schema
    GrammarSpec grammar = 8;       // Grammar-constrained decoding
  }
}

message CacheHint {
  // Session-based caching: reuse KV cache from previous turns
  string session_id = 1;           // Empty = no session affinity

  // Prefix-based caching: reuse shared prefix KV cache
  string prefix_hash = 2;          // Hash of the prefix token sequence
  int32 prefix_length = 3;         // How many tokens the prefix covers
}

message GrammarSpec {
  string bnf = 1;                  // BNF grammar for constrained decoding
}

// Streamed back per token (or small batch of tokens)
message InferResponse {
  string request_id = 1;

  oneof payload {
    TokenChunk chunk = 2;          // Normal: generated tokens
    InferComplete complete = 3;    // Final message: generation finished
    InferError error = 4;          // Error: something went wrong mid-generation
  }
}

message TokenChunk {
  repeated int32 token_ids = 1;    // Generated token IDs
  string text = 2;                 // Detokenized text for this chunk

  // Optional per-token metadata
  repeated TokenLogprob logprobs = 3;
}

message TokenLogprob {
  int32 token_id = 1;
  float logprob = 2;
  repeated TopLogprob top_logprobs = 3; // Top-N alternatives
}

message TopLogprob {
  int32 token_id = 1;
  string token = 2;
  float logprob = 3;
}

message InferComplete {
  FinishReason finish_reason = 1;
  UsageStats usage = 2;
  CacheInfo cache_info = 3;        // What cache was created/reused
}

enum FinishReason {
  FINISH_REASON_UNSPECIFIED = 0;
  STOP = 1;                        // Hit stop sequence or EOS
  MAX_TOKENS = 2;                  // Hit max_tokens limit
  CONTENT_FILTER = 3;              // Post-inference safety filter triggered
}

message UsageStats {
  int32 prompt_tokens = 1;         // Input tokens processed
  int32 completion_tokens = 2;     // Output tokens generated
  int32 cached_tokens = 3;         // Tokens served from KV cache (not recomputed)
  float prefill_time_ms = 4;       // Time to process input (prefill phase)
  float decode_time_ms = 5;        // Time to generate output (decode phase)
  float total_time_ms = 6;         // Wall clock total
  float cache_load_ms = 7;         // CPU DRAM → GPU transfer time (disaggregated KV cache)
  float cache_save_ms = 8;         // GPU → CPU DRAM transfer time (disaggregated KV cache)
}

message CacheInfo {
  string cache_id = 1;             // ID of the KV cache created/reused
  int32 cached_tokens = 2;         // Tokens that were cache hits
  int32 new_tokens = 3;            // Tokens added to cache
  int64 cache_size_bytes = 4;      // VRAM used by this cache
}

message InferError {
  ErrorCode code = 1;
  string message = 2;
  bool retriable = 3;              // Gateway should retry on another worker?
}

enum ErrorCode {
  ERROR_CODE_UNSPECIFIED = 0;
  OOM = 1;                         // GPU out of memory
  MODEL_NOT_LOADED = 2;            // Requested model not on this GPU
  TIMEOUT = 3;                     // Generation exceeded time limit
  CANCELLED = 4;                   // Cancelled by gateway (client disconnect)
  INTERNAL = 5;                    // Worker internal error
  INVALID_INPUT = 6;               // Bad input (too many tokens, etc.)
}
```

### Model Management

```protobuf
message LoadModelRequest {
  string model_id = 1;             // Model identifier
  string model_path = 2;           // Path or HuggingFace repo
  string quantization = 3;         // "fp16", "int8", "int4", etc.

  // Resource hints
  int64 estimated_vram_bytes = 4;  // Gateway's estimate of VRAM needed
  int32 max_batch_size = 5;        // Max concurrent requests to support
}

message LoadModelResponse {
  bool success = 1;
  string error_message = 2;

  // Actual resource usage after loading
  int64 vram_used_bytes = 3;       // VRAM consumed by model weights
  int64 vram_available_bytes = 4;  // VRAM remaining for KV caches
  ModelCapabilities capabilities = 5;
}

message ModelCapabilities {
  int32 max_context_length = 1;    // Max tokens model supports
  int32 vocab_size = 2;
  bool supports_logprobs = 3;
  bool supports_json_mode = 4;
  bool supports_grammar = 5;
}

message UnloadModelRequest {
  string model_id = 1;
  bool force = 2;                  // Unload even if active inferences exist
                                   // (gateway should drain first, force is emergency)
}

message UnloadModelResponse {
  bool success = 1;
  string error_message = 2;
  int32 caches_destroyed = 3;      // Number of KV caches that were lost
  int64 vram_freed_bytes = 4;
}
```

### State Observation

```protobuf
message GetWorkerStateRequest {}

message WatchWorkerStateRequest {
  // Debounce: don't send updates more often than this
  int32 min_interval_ms = 1;       // Default: 100ms
}

message WorkerState {
  string worker_id = 1;
  int64 timestamp_ms = 2;

  // GPU info
  GpuInfo gpu = 3;

  // Loaded models
  repeated LoadedModel models = 4;

  // Active inference count
  int32 active_inferences = 5;
  int32 queued_inferences = 6;     // Worker-local queue (should be near 0
                                   // if gateway is scheduling properly)

  // KV cache summary (detailed entries via GetCacheEntries)
  CacheSummary cache_summary = 7;
}

message GpuInfo {
  string gpu_id = 1;               // e.g., "gpu-0"
  string gpu_model = 2;            // e.g., "NVIDIA RTX A4500"
  int64 vram_total_bytes = 3;
  int64 vram_used_bytes = 4;       // Weights + KV caches + other
  int64 vram_available_bytes = 5;
  float gpu_utilization = 6;       // 0.0 - 1.0
  float gpu_temperature_c = 7;
  bool healthy = 8;
}

message LoadedModel {
  string model_id = 1;
  string quantization = 2;
  int64 vram_used_bytes = 3;       // VRAM for weights only
  bool ready = 4;                  // True when model is ready for inference
}

message CacheSummary {
  int32 total_entries = 1;
  int64 total_vram_bytes = 2;      // Total VRAM used by all KV caches
  int32 session_caches = 3;        // Breakdown by type
  int32 prefix_caches = 4;
  int32 document_caches = 5;
}

message GetCacheEntriesRequest {
  string model_id = 1;             // Filter by model (empty = all)
  string session_id = 2;           // Filter by session (empty = all)
}

message CacheEntriesResponse {
  repeated CacheEntry entries = 1;
}

message CacheEntry {
  string cache_id = 1;
  string model_id = 2;
  string session_id = 3;           // Empty for prefix/shared caches
  string prefix_hash = 4;          // For prefix caches
  CacheType cache_type = 5;
  int32 token_count = 6;           // Tokens in this cache
  int64 vram_bytes = 7;
  int64 created_at_ms = 8;
  int64 last_accessed_at_ms = 9;
  int32 access_count = 10;
}

enum CacheType {
  CACHE_TYPE_UNSPECIFIED = 0;
  SESSION = 1;
  PREFIX = 2;
  DOCUMENT = 3;
}

message EvictCacheRequest {
  string cache_id = 1;             // Specific cache to evict
}

message EvictCacheResponse {
  bool success = 1;
  int64 vram_freed_bytes = 2;
}
```

### Health

```protobuf
message HealthRequest {}

message HealthResponse {
  HealthStatus status = 1;
  string message = 2;              // Human-readable status message
  int64 uptime_ms = 3;
  int64 total_inferences = 4;      // Lifetime inference count
}

enum HealthStatus {
  HEALTH_STATUS_UNSPECIFIED = 0;
  HEALTHY = 1;                     // Ready for inference
  DEGRADED = 2;                    // Running but impaired (high temp, near OOM)
  UNHEALTHY = 3;                   // Not accepting new requests
  LOADING = 4;                     // Starting up / loading model
}
```

---

## Key Design Decisions

### 1. Gateway tokenizes, not workers
The gateway does pre-flight tokenization (for validation, token counting, billing estimation). It sends `token_ids` to workers. Workers don't need a tokenizer — they just run the model. `prompt` field exists as a fallback but shouldn't be the normal path.

### 2. Eviction is gateway-directed
The gateway decides what to evict (it has the global view). Workers execute `EvictCache` on command. Workers never evict on their own except under OOM pressure (emergency self-preservation).

### 3. Cache hints, not cache commands
`InferRequest.cache_hint` tells the worker "look for this cache" — it's advisory. The worker may or may not find it. The response's `CacheInfo` tells the gateway what actually happened. The gateway uses this to update its routing tables.

### 4. Cancellation via RPC context
No explicit `Cancel` RPC needed. Gateway cancels the gRPC call context → worker receives cancellation signal → stops generation → frees resources. This is standard gRPC cancellation propagation.

### 5. WatchWorkerState for real-time routing
Polling `GetWorkerState` introduces staleness. `WatchWorkerState` gives the gateway a real-time stream of state changes. The `min_interval_ms` debounce prevents flooding during high activity.

### 6. UsageStats includes timing breakdown
`prefill_time_ms` and `decode_time_ms` are separate. This is critical for:
- Billing (cached vs uncached tokens)
- Performance monitoring (prefill bottleneck vs decode bottleneck)
- Scheduling decisions (how long is inference actually taking?)

---

## What This Contract Enables

| Gateway Capability | Contract Support |
|---|---|
| Cache-aware routing | `GetCacheEntries`, `CacheHint`, `CacheInfo` in response |
| VRAM-aware scheduling | `WorkerState.gpu.vram_available_bytes` |
| Priority-based eviction | `EvictCache` (gateway decides, worker executes) |
| Client disconnect → GPU freed | gRPC context cancellation |
| Health-based routing | `Health`, `WorkerState.gpu.healthy` |
| Model load/unload | `LoadModel`, `UnloadModel` |
| Real-time state for routing | `WatchWorkerState` stream |
| Billing with cache differentiation | `UsageStats.cached_tokens`, `CacheInfo` |
| Streaming token delivery | `stream InferResponse` |
| Structured output | `GenerationParams.json_schema/grammar` |

---

## What This Contract Does NOT Cover (Gateway's Job)

- Request queuing and priority
- Batch formation (gateway batches requests before sending)
- Session affinity decisions
- Prefix matching and deduplication
- Rate limiting
- Authentication
- API versioning
- Retry logic
- Fairness enforcement

---

## Multi-Modal Extensions (IMPLEMENTED)

The following proto changes are planned for multi-modal support (6 models, 5 modalities). All are backward-compatible in proto3 (new optional fields).

### InferRequest — image input for vision-language models

```protobuf
message InferRequest {
  // ... existing fields 1-6 ...
  bytes image_data = 7;           // Raw image bytes (JPEG/PNG) for vision-language models
  string image_mime_type = 8;     // "image/jpeg", "image/png", etc.
}
```

### InferResponse — media output variant for image/audio/video generation

```protobuf
message InferResponse {
  string request_id = 1;

  oneof payload {
    TokenChunk chunk = 2;          // Normal: generated tokens (text gen, vision-language)
    InferComplete complete = 3;    // Final message: generation finished
    InferError error = 4;          // Error: something went wrong mid-generation
    MediaOutput media = 5;         // NEW: generated media (image, audio, video)
  }
}

message MediaOutput {
  bytes data = 1;                  // Raw media bytes (PNG, WAV, MP4)
  string mime_type = 2;            // "image/png", "audio/wav", "video/mp4"
  bool is_final = 3;              // True if this is the final (or only) chunk of media
}
```

### LoadModelRequest — model type hint

```protobuf
message LoadModelRequest {
  // ... existing fields 1-5 ...
  string model_type = 6;           // "text_gen", "vision_language", "tts", "image_gen", "video_gen"
}
```

### ModelCapabilities — modality flags

```protobuf
message ModelCapabilities {
  // ... existing fields 1-5 ...
  string model_type = 6;           // Which pipeline type this model uses
  bool supports_image_input = 7;   // Accepts image_data in InferRequest (vision-language)
  bool supports_image_output = 8;  // Returns MediaOutput with image/png (image gen)
  bool supports_audio_output = 9;  // Returns MediaOutput with audio/wav (TTS)
  bool supports_video_output = 10; // Returns MediaOutput with video/mp4 (video gen)
}
```

### gRPC message size for large media

Video generation (CogVideoX-2B) can produce large MP4 files. `grpc.max_send_message_length` will be set to 100MB on the Python server side.

### Model type → pipeline mapping

| model_type | Pipeline Class | Input | Output | Proto Fields Used |
|---|---|---|---|---|
| text_gen | TextGenPipeline | prompt/token_ids | TokenChunk stream | chunk, complete |
| vision_language | VisionLanguagePipeline | prompt + image_data | TokenChunk stream | image_data, image_mime_type, chunk, complete |
| tts | TTSPipeline | prompt (text to speak) | MediaOutput (WAV) | media |
| image_gen | ImageGenPipeline | prompt (description) | MediaOutput (PNG) | media |
| video_gen | VideoGenPipeline | prompt (description) | MediaOutput (MP4) | media |

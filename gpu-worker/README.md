# gpu-worker

Python gRPC server that runs on GPU machines. One process per GPU (or one torchrun process spanning multiple GPUs for tensor parallelism). Handles model loading, inference, KV cache persistence, and multi-modal output — all the GPU-side logic.

## Architecture

```
gRPC Request (from gateway)
     │
     ▼
┌─────────────────────────────────────────────┐
│  server.py — InferenceWorkerServicer        │
│  (gRPC RPC handlers, proto ↔ Python dicts)  │
│                    │                        │
│                    ▼                        │
│  worker.py — GpuWorker                      │
│  (model lifecycle, pipeline dispatch,       │
│   VRAM tracking, thread-safe model dict)    │
│                    │                        │
│          ┌────────┼────────┐               │
│          ▼        ▼        ▼               │
│  ┌──────────┐ ┌────────┐ ┌──────────┐     │
│  │ text_gen │ │ vision │ │ tts/img/ │     │
│  │ pipeline │ │  lang  │ │  video   │     │
│  └────┬─────┘ └────────┘ └──────────┘     │
│       │                                    │
│       ▼                                    │
│  kv_cache_store.py                         │
│  (CPU DRAM, LRU eviction, 200GB budget)    │
└─────────────────────────────────────────────┘
     │
     ▼ (if TP mode)
┌─────────────────────────────────────────────┐
│  tp_coordinator.py                          │
│  rank 0 broadcasts commands to rank 1+      │
│  via torch.distributed; NCCL handles the    │
│  actual tensor ops during model.generate()  │
└─────────────────────────────────────────────┘
```

## Files

### `server.py`

gRPC entry point. Implements `InferenceWorkerServicer` with 9 RPCs:

| RPC | Type | What it does |
|-----|------|-------------|
| `Health` | Unary | Uptime, inference count, status |
| `LoadModel` | Unary | Load model into GPU VRAM, return capabilities |
| `UnloadModel` | Unary | Free model from VRAM |
| `GetWorkerState` | Unary | GPU metrics, loaded models, cache stats |
| `WatchWorkerState` | Server stream | Periodic state updates |
| `Infer` | Server stream | Single inference, streams TokenChunk/MediaOutput/InferComplete |
| `BatchInfer` | Server stream | Batched inference, demuxed by request_id |
| `GetCacheEntries` | Unary | List KV cache entries (stub) |
| `EvictCache` | Unary | Evict specific cache entry (stub) |

**Tensor parallelism mode**: Detects torchrun environment (`RANK`, `LOCAL_RANK` env vars). Rank 0 runs the gRPC server. Rank 1+ enters a command loop (`run_tp_follower`), participating in collective ops (model load, generate) via NCCL when rank 0 broadcasts commands through `TPCoordinator`.

### `worker.py`

GPU worker orchestration. Manages loaded models, dispatches inference to the right pipeline.

**`GpuWorker`** key methods:

| Method | What it does |
|--------|-------------|
| `load_model(model_id, ...)` | Detect model type from `MODEL_TYPE_REGISTRY`, instantiate pipeline, call `pipeline.load()`, measure VRAM delta, return capabilities |
| `unload_model(model_id)` | Call `pipeline.unload()`, measure VRAM freed |
| `infer(request)` | Dispatch to loaded pipeline's `infer()`, yield `InferenceResult` chunks |
| `infer_batch(requests)` | For text: `pipeline.infer_batch()` (single `model.generate()` with left-padded inputs). For other modalities: sequential fallback. |
| `get_worker_state()` | GPU metrics via pynvml, loaded models, active inferences, cache stats |
| `health()` | Status, uptime, total inferences |

**`MODEL_TYPE_REGISTRY`** maps HuggingFace model IDs to pipeline types:
```
SmolLM2-135M/360M/1.7B-Instruct → text_gen
Qwen2.5-VL-3B-Instruct          → vision_language
Kokoro-82M                       → tts
sd-turbo                         → image_gen
CogVideoX-2b                    → video_gen
Qwen3-14B                       → text_gen
```

**CUDA_VISIBLE_DEVICES handling**: When `CUDA_VISIBLE_DEVICES=1`, physical GPU 1 becomes device index 0. The worker resolves `physical_gpu_id` from the env var for correct pynvml indexing.

### `kv_cache_store.py`

CPU DRAM-backed KV cache with LRU eviction. Stores `past_key_values` from `model.generate()` in system memory (not VRAM) so subsequent turns in a conversation skip recomputing prior context.

| Method | What it does |
|--------|-------------|
| `save(session_id, past_key_values, seq_len, tp_size)` | Extract (key, value) tensors per layer, move to CPU, store. Evict oldest if over budget. |
| `load(session_id, device, tp_size)` | Lookup session, validate TP config match, move tensors back to GPU, reconstruct `DynamicCache`. |
| `evict(session_id)` | Remove specific entry. |
| `stats()` | Total bytes, entry count, hit/miss rates. |

**Why CPU DRAM, not VRAM**: VRAM is scarce (~16GB free after model weights). System DRAM is abundant (251GB on our RunPod machine). PCIe transfer (~25ms for 384MB) is 7x cheaper than recomputing (174ms for 2K tokens).

**Transformers compatibility**: Handles v5.x `DynamicCache.layers` (list of `DynamicLayer` with `.keys`/`.values`) and v4.x `key_cache`/`value_cache` list format.

Default budget: 200GB. Thread-safe via `threading.Lock` + `OrderedDict` for LRU ordering.

### `tp_coordinator.py`

Coordinates tensor parallelism across ranks when using `torchrun --nproc_per_node=2`.

Rank 0 (gRPC server) calls `broadcast_command()` to send load/unload/generate commands to rank 1+ via `torch.distributed.broadcast_object_list()`. Rank 1+ calls `recv_command()` in a loop, executes the command locally (NCCL handles the actual tensor ops during `model.generate()`), and discards output. Only rank 0 streams results back to the client.

### `pipelines/`

Each modality has a dedicated pipeline class inheriting from `BasePipeline`.

#### `base.py`

Abstract base class. Defines the interface:
- `load(model_path) → dict` — load model, return capabilities
- `infer(request) → Generator[InferenceResult]` — run inference, yield chunks
- `unload()` — free GPU resources
- `get_capabilities() → dict` — declare supported modalities

**`InferenceResult`** dataclass carries everything: text chunks (`chunk_text`, `is_thinking`), media bytes (`media_data`, `media_mime_type`), completion stats (`prompt_tokens`, `completion_tokens`, timing), and cache info (`cache_hit`, `cache_load_ms`, `cache_save_ms`).

#### `text_gen.py`

`AutoModelForCausalLM` + `TextIteratorStreamer`. The most complex pipeline.

**Single inference (`infer`)**:
1. Check KV cache for `session_id` → restore `past_key_values` if hit
2. Tokenize full prompt (miss) or only new tokens (hit)
3. `model.generate()` with `TextIteratorStreamer` in background thread
4. Parse `<think>...</think>` tags from stream (Qwen3-14B thinking mode)
5. Save updated KV cache to CPU DRAM
6. Yield text chunks + completion with timing and cache stats

**Batch inference (`infer_batch`)**:
1. Left-pad all prompts (`tokenizer.padding_side = 'left'`)
2. Single `model.generate()` call with batched inputs
3. Split outputs per request, strip padding
4. No KV cache (incompatible with variable-length past_key_values)

**TP mode**: Passes `tp_plan="auto"` to `AutoModelForCausalLM.from_pretrained()`. PyTorch DTensor handles NCCL distribution transparently during `generate()`.

#### `vision_language.py`

`Qwen2VLForConditionalGeneration` + `AutoProcessor`. Accepts image bytes + text prompt (or chat messages). Decodes image via PIL, builds multimodal input, applies chat template, streams text output.

#### `tts.py`

Kokoro-82M via `KPipeline`. Text → WAV audio. Concatenates audio segments from generator, writes to WAV buffer via soundfile. Returns complete audio as `MediaOutput` (no streaming).

#### `image_gen.py`

Stable Diffusion Turbo via `AutoPipelineForText2Image`. Text → PNG image. Single inference step (`num_inference_steps=1`, `guidance_scale=0.0` for turbo mode). Returns PNG bytes as `MediaOutput`.

#### `video_gen.py`

CogVideoX-2B via `CogVideoXPipeline`. Text → MP4 video. Uses `enable_model_cpu_offload()` (model too large for single GPU without offloading). VAE slicing/tiling for memory efficiency. Exports frames to MP4 via imageio at 8fps. Returns MP4 bytes as `MediaOutput`.

## Running

### Single GPU mode (2 workers)

```bash
# On GPU host:
cd /workspace/gpu-worker
pip install -r requirements.txt

# Worker 0 on GPU 0
CUDA_VISIBLE_DEVICES=0 python3 server.py --port 50051 --gpu-id 0 --worker-id worker-0

# Worker 1 on GPU 1 (separate terminal)
CUDA_VISIBLE_DEVICES=1 python3 server.py --port 50052 --gpu-id 0 --worker-id worker-1
```

Note: `--gpu-id 0` for both because `CUDA_VISIBLE_DEVICES` remaps the physical GPU to device index 0.

### Tensor parallel mode (1 worker, 2 GPUs)

```bash
torchrun --nproc_per_node=2 server.py --port 50051 --worker-id tp-worker-0 --tp
```

Both ranks initialize NCCL. Rank 0 starts gRPC server. Rank 1 enters command loop.

### CLI arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 50051 | gRPC listen port |
| `--gpu-id` | 0 | CUDA device index |
| `--worker-id` | worker-0 | Identifier reported to gateway |
| `--tp` | false | Enable tensor parallelism mode |

## Proto contract

Defined in `proto/inference_worker.proto`. Key message types:

- **`InferRequest`**: `request_id`, `model_id`, `prompt`, `messages` (chat), `params` (max_tokens, temperature, top_p, stop), `cache_hint` (session_id), `image_data` (bytes), `image_mime_type`
- **`InferResponse`** (oneof): `TokenChunk` (text + is_thinking), `MediaOutput` (bytes + mime_type), `InferComplete` (usage stats + cache info), `InferError`
- **`BatchInferRequest`**: repeated `InferRequest` (all must target same model)
- **`LoadModelRequest`**: `model_id`, `model_path`, `quantization`, `model_type`
- **`WorkerState`**: GPU metrics, loaded models with capabilities, active/queued inferences

## Not yet implemented

| Feature | What's missing | Why it matters |
|---------|---------------|---------------|
| **Prefix tree / shared prefixes** | Common system prompts computed once per GPU, reused across requests | Every request recomputes the same system prompt. With prefix sharing, only the first request pays the cost. |
| **Continuous batching** | In-flight batch joining — new requests start generating while earlier ones are still decoding | Current `infer_batch()` waits for all requests to finish before returning any. Continuous batching keeps the GPU saturated. |
| **Speculative decoding** | Small draft model + large verifier in tandem | Not implemented at pipeline level. Would require a second model loaded alongside the main one. |
| **KV cache for non-text pipelines** | Vision, TTS, image, video have no session caching | Only `TextGenPipeline` uses `KVCacheStore`. Multi-turn vision conversations recompute from scratch. |
| **Streaming media** | Audio/image/video are buffered entirely before sending | TTS could stream audio chunks as they're generated. Image gen could stream intermediate denoising steps. |
| **Quantization support** | INT8/INT4 model loading | `load()` accepts a `quantization` parameter but only FP16 is implemented. Would need `bitsandbytes` or GPTQ integration. |
| **GetCacheEntries / EvictCache RPCs** | gRPC stubs exist but return empty responses | Gateway can't inspect or manage individual cache entries on the worker. |
| **Graceful inference cancellation** | No way to abort a running `model.generate()` mid-stream | `context.is_active()` check prevents sending results after disconnect, but the GPU computation continues until completion. |
| **Multi-image vision** | Only single image per request | `VisionLanguagePipeline` processes one image. Qwen2.5-VL supports multiple images natively. |
| **Audio streaming (chunked TTS)** | Kokoro generates full audio before returning | Could yield audio segments as they're produced for lower time-to-first-byte. |

## Dependencies

Core: `torch`, `transformers`, `grpcio`, `pynvml`
Image gen: `diffusers`, `Pillow`
Video gen: `diffusers`, `imageio`, `imageio-ffmpeg`
TTS: `kokoro`
Vision: `qwen-vl-utils`, `Pillow`

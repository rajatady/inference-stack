# Inference Stack

A production-grade LLM inference API built from scratch in TypeScript/NestJS + Python, running against real GPUs. Built as a learning exercise to understand the same class of problems that OpenAI, Anthropic, and Google solve: GPU resource scheduling, KV cache-aware routing, streaming, dynamic batching, tensor parallelism, and multi-modal inference.

**This is not a wrapper around vLLM or TGI.** Every layer is built from raw `transformers` + `grpcio` to understand what happens under the hood.

## What it does

```
Your laptop (NestJS gateway)          Remote GPU cluster (Python workers)
┌─────────────────────────┐           ┌──────────────────────────────┐
│  HTTP API (OpenAI-compat)│           │  GPU-0: worker-0 (gRPC)     │
│  Scheduler + Batcher    │──gRPC────→│  GPU-1: worker-1 (gRPC)     │
│  Router + Model Manager │           │  or: TP worker (2 GPUs)     │
│  KV Cache Manager       │           │                              │
│  Metrics (ClickHouse)   │           │  8 models, 5 modalities     │
└─────────────────────────┘           └──────────────────────────────┘
```

- **8 models across 5 modalities**: text generation (SmolLM2 family + Qwen3-14B), vision-language (Qwen2.5-VL-3B), text-to-speech (Kokoro-82M), image generation (SD Turbo), video generation (CogVideoX-2B)
- **Tensor parallelism**: Qwen3-14B split across 2 GPUs via `tp_plan="auto"` + `torchrun`, with thinking mode (`<think>` tag parsing)
- **Dynamic batching**: 158x throughput improvement at concurrency 8 (2 TPS to 316 TPS)
- **KV cache persistence**: CPU DRAM-backed session cache with LRU eviction, 14% compute savings on multi-turn
- **Runtime mode switching**: Gateway SSHs to GPU host to switch between individual workers and tensor-parallel mode
- **Full observability**: ClickHouse metrics pipeline with TPS, latency percentiles, per-model breakdowns

## Architecture

Three separate planes, just like production inference systems:

| Plane | Where | What |
|-------|-------|------|
| **Gateway** | Your laptop | NestJS API, scheduler, router, KV cache manager, batch collector |
| **GPU Workers** | Remote cluster | Python gRPC servers, one per GPU, running raw transformers |
| **Metrics** | Your laptop | ClickHouse for inference analytics |

The API server **never** runs on GPU machines. Communication is via gRPC over an SSH tunnel.

## Key systems built

### Scheduler
Priority queue with per-user fairness, aging, backpressure (429 + Retry-After), and configurable timeouts. Integrates with BatchCollector for time-window batching.

### Batch Collector + GPU-Side Batching
Accumulates requests within a time window, groups by model, dispatches as a single `model.generate()` call with left-padded inputs. The throughput difference is dramatic:

| Concurrency | Without batching | With batching |
|-------------|-----------------|---------------|
| c=1 | 42 TPS | 43 TPS |
| c=8 | 2 TPS | 316 TPS |
| c=32 | OOM | 1,134 TPS |

### KV Cache (Disaggregated)
CPU DRAM-backed cache on the GPU host. On multi-turn conversations, restores `past_key_values` from CPU memory instead of recomputing from scratch. Handles transformers v4.x and v5.x cache formats.

### Model Manager
VRAM-aware model placement with auto-load/unload, GPU affinity, and concurrent load coalescing. Knows which models need tensor parallelism and triggers mode switches automatically.

### Router
Picks the best worker per request: model affinity (is the model already loaded?) > least loaded > trigger load on best candidate.

### Worker Registry
Manages N gRPC worker connections dynamically. Supports runtime mode switching (individual workers <-> tensor parallel) via SSH to the GPU host.

## Running it

### Prerequisites
- Node.js 18+, Python 3.10+
- A GPU machine accessible via SSH (RunPod, Lambda, etc.) with 2+ GPUs
- SSH tunnel to forward gRPC ports

### Setup

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USER/inference-stack.git
cd inference-stack

# 2. Gateway (NestJS)
cd inference-api
cp .env.example .env  # Edit with your GPU host details
npm install

# 3. GPU worker (Python) — rsync to your GPU host
rsync -avz gpu-worker/ $RUNPOD_SSH_USER@$RUNPOD_SSH_HOST:/workspace/gpu-worker/ -e 'ssh -p $RUNPOD_SSH_PORT'
# On GPU host: pip install -r requirements.txt

# 4. Start workers on GPU host
ssh $RUNPOD_SSH_USER@$RUNPOD_SSH_HOST -p $RUNPOD_SSH_PORT
cd /workspace/gpu-worker
CUDA_VISIBLE_DEVICES=0 python3 server.py --port 50051 --gpu-id 0 --worker-id worker-0 &
CUDA_VISIBLE_DEVICES=1 python3 server.py --port 50052 --gpu-id 0 --worker-id worker-1 &

# 5. SSH tunnel (from your laptop)
ssh -f -N -L 50051:localhost:50051 -L 50052:localhost:50052 $RUNPOD_SSH_USER@$RUNPOD_SSH_HOST -p $RUNPOD_SSH_PORT

# 6. Start gateway
npm run start:dev
# Open http://localhost:3000
```

### For tensor parallelism (Qwen3-14B)
```bash
# On GPU host:
torchrun --nproc_per_node=2 server.py --port 50051 --worker-id tp-worker-0 --tp

# Start gateway with:
WORKER_MODE=tensor-parallel npm run start:dev
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/completions` | POST | Text completion (streaming + non-streaming) |
| `/v1/images/generations` | POST | Image generation (SD Turbo) |
| `/v1/audio/speech` | POST | Text-to-speech (Kokoro) |
| `/v1/video/generations` | POST | Video generation (CogVideoX) |
| `/v1/completions/stats` | GET | Queue depth, active count |
| `/v1/metrics/tps` | GET | Tokens per second |
| `/v1/metrics/latency` | GET | Latency percentiles |

## Tests

```bash
# Unit tests (107 tests, no GPU needed)
cd inference-api && npx jest --testPathPattern='src/'

# Integration tests (requires GPU workers + SSH tunnel)
npm run test:e2e

# Load tests (18 scenarios against real GPUs)
npm run test:load
```

## What I learned building this

1. **Decode is the bottleneck, not prefill** — 92% of GPU time is decode. Prefix sharing saves 96% of prefill, but prefill is only 5% of total time at low concurrency.
2. **Batching is the single biggest lever** — 158x throughput from one change. This is why vLLM exists.
3. **VRAM is the constraint** — every decision (model placement, KV cache eviction, quantization) comes back to fitting things in GPU memory.
4. **`tp_plan="auto"` != `device_map="auto"`** — tensor parallelism splits matrix ops across GPUs for parallel speedup. Device map just puts different layers on different GPUs (pipeline parallelism, no speedup).
5. **The gap between "model runs" and "model serves"** — loading a model is 1% of the work. Scheduling, batching, caching, routing, backpressure, cancellation, timeout, observability is the other 99%.

## Project structure

```
inference-stack/
├── inference-api/              # NestJS gateway (runs on your laptop)
│   ├── src/
│   │   ├── worker-orchestrator/  # Registry, model manager, router
│   │   ├── scheduler/            # Priority queue, batch collector
│   │   ├── completions/          # OpenAI-compatible API
│   │   ├── images/audio/video/   # Multi-modal endpoints
│   │   ├── metrics/              # ClickHouse pipeline
│   │   └── config/model-roster.ts
│   ├── proto/inference_worker.proto
│   ├── public/index.html         # Playground UI
│   └── test/
├── gpu-worker/                 # Python GPU workers (runs on GPU host)
│   ├── server.py                 # gRPC server + torchrun TP support
│   ├── worker.py                 # GPU management, pipeline registry
│   ├── kv_cache_store.py         # CPU DRAM KV cache
│   └── pipelines/                # text_gen, vision, tts, image, video
└── .claude/skills/scope/       # Architecture documentation
```

## Not yet built (scope remains)

- Prefix sharing (shared system prompt KV cache across requests)
- Weighted KV cache eviction (recompute_cost x reuse_probability)
- Continuous batching (new requests joining in-flight batches)
- Speculative decoding
- Failure recovery (OOM retry, worker crash redistribution)
- Safety/content filtering pipeline
- Token-level rate limiting
- Quantization-aware routing

---

Built with Claude Code as a learning exercise. Every line was written by AI, directed by a human who wanted to understand how inference infrastructure works.

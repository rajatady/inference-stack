"""
GPU Worker gRPC Server

Entry point for the GPU worker process. Starts a gRPC server that
implements the InferenceWorker service from inference_worker.proto.

Usage (single GPU):
    python server.py --port 50051 --gpu-id 0

Usage (tensor parallelism across N GPUs):
    torchrun --nproc_per_node=2 server.py --port 50051 --tp
"""

import argparse
import logging
import os
import signal
import sys
import threading
import time
from concurrent import futures

import grpc

# Generated proto stubs (run generate_stubs.sh first)
sys.path.insert(0, ".")
from generated import inference_worker_pb2 as pb2
from generated import inference_worker_pb2_grpc as pb2_grpc

from worker import GpuWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# 100MB max message size for large media (video gen)
MAX_MESSAGE_LENGTH = 100 * 1024 * 1024


class InferenceWorkerServicer(pb2_grpc.InferenceWorkerServicer):
    """gRPC service implementation — delegates everything to GpuWorker."""

    def __init__(self, worker: GpuWorker, tp_coordinator=None):
        self.worker = worker
        self.tp = tp_coordinator  # TPCoordinator for rank coordination (None if single-GPU)

    def Health(self, request, context):
        result = self.worker.health()
        return pb2.HealthResponse(
            status=_health_status_to_enum(result["status"]),
            message=result["message"],
            uptime_ms=result["uptime_ms"],
            total_inferences=result["total_inferences"],
        )

    def GetWorkerState(self, request, context):
        state = self.worker.get_worker_state()
        return _build_worker_state(state)

    def WatchWorkerState(self, request, context):
        while context.is_active():
            state = self.worker.get_worker_state()
            yield _build_worker_state(state)
            time.sleep(max(request.min_interval_ms / 1000.0, 0.1))

    def LoadModel(self, request, context):
        # In TP mode, broadcast load command so all ranks load the model
        if self.tp:
            self.tp.broadcast_command({
                "action": "load_model",
                "model_id": request.model_id,
                "model_path": request.model_path,
                "quantization": request.quantization or "fp16",
                "model_type": request.model_type,
            })
        result = self.worker.load_model(
            model_id=request.model_id,
            model_path=request.model_path,
            quantization=request.quantization or "fp16",
            estimated_vram_bytes=request.estimated_vram_bytes,
            model_type=request.model_type,
        )

        capabilities = None
        if result.get("capabilities"):
            caps = result["capabilities"]
            capabilities = pb2.ModelCapabilities(
                max_context_length=caps.get("max_context_length", 0),
                vocab_size=caps.get("vocab_size", 0),
                supports_logprobs=caps.get("supports_logprobs", False),
                supports_json_mode=caps.get("supports_json_mode", False),
                supports_grammar=caps.get("supports_grammar", False),
                model_type=caps.get("model_type", ""),
                supports_image_input=caps.get("supports_image_input", False),
                supports_image_output=caps.get("supports_image_output", False),
                supports_audio_output=caps.get("supports_audio_output", False),
                supports_video_output=caps.get("supports_video_output", False),
            )

        return pb2.LoadModelResponse(
            success=result["success"],
            error_message=result.get("error_message", ""),
            vram_used_bytes=result.get("vram_used_bytes", 0),
            vram_available_bytes=result.get("vram_available_bytes", 0),
            capabilities=capabilities,
        )

    def UnloadModel(self, request, context):
        if self.tp:
            self.tp.broadcast_command({
                "action": "unload_model",
                "model_id": request.model_id,
            })
        result = self.worker.unload_model(
            model_id=request.model_id,
            force=request.force,
        )
        return pb2.UnloadModelResponse(
            success=result["success"],
            error_message=result.get("error_message", ""),
            caches_destroyed=result.get("caches_destroyed", 0),
            vram_freed_bytes=result.get("vram_freed_bytes", 0),
        )

    def Infer(self, request, context):
        # Check if model is loaded — return error via stream if not
        if request.model_id not in self.worker.loaded_models:
            yield pb2.InferResponse(
                request_id=request.request_id,
                error=pb2.InferError(
                    code=pb2.MODEL_NOT_LOADED,
                    message=f"Model {request.model_id} is not loaded on this worker",
                    retriable=False,
                ),
            )
            return

        # Build request dict for worker
        infer_request = {
            "model_id": request.model_id,
            "prompt": request.prompt,
            "token_ids": list(request.token_ids) if request.token_ids else [],
            "params": {
                "max_tokens": request.params.max_tokens if request.params else 50,
                "temperature": request.params.temperature if request.params else 1.0,
                "top_p": request.params.top_p if request.params else 1.0,
            } if request.params else {},
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "cache_hint": {
                "session_id": request.cache_hint.session_id if request.HasField("cache_hint") else "",
                "prefix_hash": request.cache_hint.prefix_hash if request.HasField("cache_hint") else "",
            },
        }

        # Forward new_prompt for cache-aware token alignment
        if request.HasField("cache_hint") and request.cache_hint.session_id:
            infer_request["new_prompt"] = request.prompt

        # Pass image data for vision-language models
        if request.image_data:
            infer_request["image_data"] = request.image_data
            infer_request["image_mime_type"] = request.image_mime_type

        # In TP mode, broadcast generate command so rank 1 calls model.generate() too
        if self.tp:
            self.tp.broadcast_command({
                "action": "generate",
                "model_id": request.model_id,
                "infer_request": infer_request,
            })

        for result in self.worker.infer(infer_request):
            # Check if client disconnected
            if not context.is_active():
                logger.info(f"Client disconnected for request {request.request_id}")
                return

            if result.is_complete:
                if result.finish_reason == "ERROR":
                    yield pb2.InferResponse(
                        request_id=request.request_id,
                        error=pb2.InferError(
                            code=pb2.INTERNAL,
                            message="Inference failed",
                            retriable=True,
                        ),
                    )
                else:
                    # Build cache info from result
                    cache_info = None
                    if result.cache_hit or result.cached_tokens > 0 or result.cache_size_bytes > 0:
                        session_id = (request.cache_hint.session_id
                                      if request.HasField("cache_hint") else "")
                        cache_info = pb2.CacheInfo(
                            cache_id=session_id,
                            cached_tokens=result.cached_tokens,
                            new_tokens=result.prompt_tokens - result.cached_tokens,
                            cache_size_bytes=result.cache_size_bytes,
                        )
                    yield pb2.InferResponse(
                        request_id=request.request_id,
                        complete=pb2.InferComplete(
                            finish_reason=_finish_reason_to_enum(result.finish_reason),
                            usage=pb2.UsageStats(
                                prompt_tokens=result.prompt_tokens,
                                completion_tokens=result.completion_tokens,
                                cached_tokens=result.cached_tokens,
                                prefill_time_ms=result.prefill_time_ms,
                                decode_time_ms=result.decode_time_ms,
                                total_time_ms=result.total_time_ms,
                                cache_load_ms=result.cache_load_ms,
                                cache_save_ms=result.cache_save_ms,
                            ),
                            cache_info=cache_info,
                        ),
                    )
            elif result.media_data is not None:
                # Media output (image, audio, video)
                yield pb2.InferResponse(
                    request_id=request.request_id,
                    media=pb2.MediaOutput(
                        data=result.media_data,
                        mime_type=result.media_mime_type or "",
                        is_final=result.is_media_final,
                    ),
                )
            else:
                # Text chunk (with thinking mode support)
                yield pb2.InferResponse(
                    request_id=request.request_id,
                    chunk=pb2.TokenChunk(
                        token_ids=result.chunk_token_ids or [],
                        text=result.chunk_text or "",
                        is_thinking=result.is_thinking,
                    ),
                )

    def BatchInfer(self, request, context):
        """Batch inference: process multiple requests in a single GPU batch."""
        if not request.requests:
            return

        # Validate all requests target the same model
        model_id = request.requests[0].model_id
        for req in request.requests:
            if req.model_id != model_id:
                for r in request.requests:
                    yield pb2.InferResponse(
                        request_id=r.request_id,
                        error=pb2.InferError(
                            code=pb2.INVALID_INPUT,
                            message="All requests in a batch must target the same model",
                            retriable=False,
                        ),
                    )
                return

        # Check model is loaded
        if model_id not in self.worker.loaded_models:
            for req in request.requests:
                yield pb2.InferResponse(
                    request_id=req.request_id,
                    error=pb2.InferError(
                        code=pb2.MODEL_NOT_LOADED,
                        message=f"Model {model_id} is not loaded on this worker",
                        retriable=False,
                    ),
                )
            return

        # Build request dicts
        infer_requests = []
        for req in request.requests:
            infer_requests.append({
                'request_id': req.request_id,
                'model_id': req.model_id,
                'prompt': req.prompt,
                'token_ids': list(req.token_ids) if req.token_ids else [],
                'params': {
                    'max_tokens': req.params.max_tokens if req.params else 50,
                    'temperature': req.params.temperature if req.params else 1.0,
                    'top_p': req.params.top_p if req.params else 1.0,
                } if req.params else {},
                'messages': [{'role': m.role, 'content': m.content} for m in req.messages],
                'cache_hint': {
                    'session_id': req.cache_hint.session_id if req.HasField('cache_hint') else '',
                },
            })

        try:
            batch_results = self.worker.infer_batch(infer_requests)
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            for req in request.requests:
                yield pb2.InferResponse(
                    request_id=req.request_id,
                    error=pb2.InferError(
                        code=pb2.INTERNAL,
                        message=f"Batch inference failed: {str(e)}",
                        retriable=True,
                    ),
                )
            return

        # Yield results for each request
        for i, (req, results) in enumerate(zip(request.requests, batch_results)):
            if not context.is_active():
                return

            for result in results:
                if result.is_complete:
                    if result.finish_reason == 'ERROR':
                        yield pb2.InferResponse(
                            request_id=req.request_id,
                            error=pb2.InferError(
                                code=pb2.INTERNAL,
                                message="Inference failed",
                                retriable=True,
                            ),
                        )
                    else:
                        yield pb2.InferResponse(
                            request_id=req.request_id,
                            complete=pb2.InferComplete(
                                finish_reason=_finish_reason_to_enum(result.finish_reason),
                                usage=pb2.UsageStats(
                                    prompt_tokens=result.prompt_tokens,
                                    completion_tokens=result.completion_tokens,
                                    cached_tokens=0,
                                    prefill_time_ms=result.prefill_time_ms,
                                    decode_time_ms=result.decode_time_ms,
                                    total_time_ms=result.total_time_ms,
                                ),
                            ),
                        )
                elif result.media_data is not None:
                    yield pb2.InferResponse(
                        request_id=req.request_id,
                        media=pb2.MediaOutput(
                            data=result.media_data,
                            mime_type=result.media_mime_type or "",
                            is_final=result.is_media_final,
                        ),
                    )
                else:
                    yield pb2.InferResponse(
                        request_id=req.request_id,
                        chunk=pb2.TokenChunk(
                            token_ids=result.chunk_token_ids or [],
                            text=result.chunk_text or "",
                        ),
                    )

    def GetCacheEntries(self, request, context):
        return pb2.CacheEntriesResponse(entries=[])

    def EvictCache(self, request, context):
        return pb2.EvictCacheResponse(success=False, vram_freed_bytes=0)


# ================================================================
# Helpers
# ================================================================

def _health_status_to_enum(status_str: str) -> int:
    mapping = {
        "HEALTHY": pb2.HEALTHY,
        "DEGRADED": pb2.DEGRADED,
        "UNHEALTHY": pb2.UNHEALTHY,
        "LOADING": pb2.LOADING,
    }
    return mapping.get(status_str, pb2.HEALTH_STATUS_UNSPECIFIED)


def _finish_reason_to_enum(reason_str: str) -> int:
    mapping = {
        "STOP": pb2.STOP,
        "MAX_TOKENS": pb2.MAX_TOKENS,
        "CONTENT_FILTER": pb2.CONTENT_FILTER,
    }
    return mapping.get(reason_str, pb2.FINISH_REASON_UNSPECIFIED)


def _build_worker_state(state: dict) -> pb2.WorkerState:
    gpu = state.get("gpu", {})
    models = [
        pb2.LoadedModel(
            model_id=m["model_id"],
            quantization=m.get("quantization", ""),
            vram_used_bytes=m.get("vram_used_bytes", 0),
            ready=m.get("ready", True),
        )
        for m in state.get("models", [])
    ]
    cache = state.get("cache_summary", {})

    return pb2.WorkerState(
        worker_id=state.get("worker_id", ""),
        timestamp_ms=state.get("timestamp_ms", 0),
        gpu=pb2.GpuInfo(
            gpu_id=gpu.get("gpu_id", ""),
            gpu_model=gpu.get("gpu_model", ""),
            vram_total_bytes=gpu.get("vram_total_bytes", 0),
            vram_used_bytes=gpu.get("vram_used_bytes", 0),
            vram_available_bytes=gpu.get("vram_available_bytes", 0),
            gpu_utilization=gpu.get("gpu_utilization", 0.0),
            gpu_temperature_c=gpu.get("gpu_temperature_c", 0.0),
            healthy=gpu.get("healthy", True),
        ),
        models=models,
        active_inferences=state.get("active_inferences", 0),
        queued_inferences=state.get("queued_inferences", 0),
        cache_summary=pb2.CacheSummary(
            total_entries=cache.get("total_entries", 0),
            total_vram_bytes=cache.get("total_vram_bytes", 0),
            session_caches=cache.get("session_caches", 0),
            prefix_caches=cache.get("prefix_caches", 0),
            document_caches=cache.get("document_caches", 0),
        ),
    )


# ================================================================
# Server
# ================================================================

def serve(port: int, gpu_id: int, worker_id: str,
          tp_mode: bool = False, world_size: int = 1,
          tp_coordinator=None):
    worker = GpuWorker(gpu_id=gpu_id, worker_id=worker_id,
                       tp_mode=tp_mode, world_size=world_size)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ],
    )
    pb2_grpc.add_InferenceWorkerServicer_to_server(
        InferenceWorkerServicer(worker, tp_coordinator=tp_coordinator), server
    )
    server.add_insecure_port(f"0.0.0.0:{port}")
    server.start()
    mode_str = f"TP mode ({world_size} GPUs)" if tp_mode else f"GPU {gpu_id}"
    logger.info(f"GPU Worker {worker_id} listening on port {port} ({mode_str})")

    # Graceful shutdown
    stop_event = threading.Event()

    def shutdown(signum, frame):
        logger.info("Shutting down...")
        if tp_coordinator:
            tp_coordinator.broadcast_command({"action": "shutdown"})
        stop_event.set()
        server.stop(grace=5)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    stop_event.wait()
    logger.info("Server stopped.")


def run_tp_follower(rank: int, local_rank: int, world_size: int):
    """
    Rank 1+ command loop: waits for commands from rank 0 and participates
    in collective operations (model loading, inference).

    With tp_plan="auto", all ranks must call from_pretrained() and
    model.generate() simultaneously for NCCL collectives to complete.
    """
    from tp_coordinator import TPCoordinator
    coordinator = TPCoordinator(rank=rank, world_size=world_size)
    worker = GpuWorker(gpu_id=local_rank, worker_id=f"tp-rank-{rank}",
                       tp_mode=True, world_size=world_size)

    logger.info(f"Rank {rank}: entering command loop")

    while True:
        try:
            cmd = coordinator.recv_command()
            if cmd is None or cmd.get("action") == "shutdown":
                logger.info(f"Rank {rank}: shutdown command received")
                break

            action = cmd["action"]

            if action == "load_model":
                logger.info(f"Rank {rank}: loading model {cmd['model_id']}")
                worker.load_model(
                    model_id=cmd["model_id"],
                    model_path=cmd.get("model_path", ""),
                    quantization=cmd.get("quantization", "fp16"),
                    model_type=cmd.get("model_type", ""),
                )
                logger.info(f"Rank {rank}: model {cmd['model_id']} loaded")

            elif action == "unload_model":
                logger.info(f"Rank {rank}: unloading model {cmd['model_id']}")
                worker.unload_model(model_id=cmd["model_id"])

            elif action == "generate":
                model_id = cmd["model_id"]
                if model_id not in worker.loaded_models:
                    logger.warning(f"Rank {rank}: model {model_id} not loaded, skipping generate")
                    continue
                # Rank 1 calls infer() on the same model with same inputs.
                # The DTensor NCCL ops will complete when both ranks participate.
                # We consume the generator but discard results (only rank 0 streams to client).
                infer_request = cmd["infer_request"]
                for _ in worker.infer(infer_request):
                    pass  # Participate in NCCL ops, discard output

        except Exception as e:
            logger.error(f"Rank {rank}: error in command loop: {e}", exc_info=True)

    logger.info(f"Rank {rank}: exiting")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Worker gRPC Server")
    parser.add_argument("--port", type=int, default=50051, help="Port to listen on")
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA device index")
    parser.add_argument("--worker-id", type=str, default="worker-0", help="Worker identifier")
    parser.add_argument("--tp", action="store_true", help="Enable tensor parallelism mode (use with torchrun)")
    args = parser.parse_args()

    # Detect torchrun environment
    is_tp = args.tp or os.environ.get("RANK") is not None
    rank = 0
    local_rank = args.gpu_id
    world_size = 1

    if is_tp:
        import torch
        import torch.distributed as dist

        # torchrun sets these env vars
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        logger.info(f"Tensor parallelism mode: rank={rank}, local_rank={local_rank}, world_size={world_size}")

        # Initialize distributed process group — blocks until all ranks join
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

        logger.info(f"Rank {rank}: process group initialized, device=cuda:{local_rank}")

    if rank == 0:
        tp_coordinator = None
        if is_tp:
            from tp_coordinator import TPCoordinator
            tp_coordinator = TPCoordinator(rank=0, world_size=world_size)

        serve(port=args.port, gpu_id=local_rank, worker_id=args.worker_id,
              tp_mode=is_tp, world_size=world_size, tp_coordinator=tp_coordinator)
    else:
        # Rank 1+: command loop — participates in collective model loading & inference
        run_tp_follower(rank=rank, local_rank=local_rank, world_size=world_size)

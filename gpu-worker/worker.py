"""
GPU Worker — manages one GPU: model loading, inference, state reporting.

This is the "dumb" worker. It does what the gateway tells it:
- Load/unload models
- Run inference (streaming tokens or media output)
- Report state (VRAM, loaded models, caches)

All scheduling, routing, and eviction decisions are made by the gateway.

Pipeline abstraction: each model type uses a dedicated pipeline class
(TextGenPipeline, VisionLanguagePipeline, TTSPipeline, ImageGenPipeline, VideoGenPipeline).
"""

import os
import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Generator

import torch

from pipelines.base import BasePipeline, InferenceResult
from pipelines.text_gen import TextGenPipeline
from pipelines.vision_language import VisionLanguagePipeline
from pipelines.tts import TTSPipeline
from pipelines.image_gen import ImageGenPipeline
from pipelines.video_gen import VideoGenPipeline
from kv_cache_store import KVCacheStore

logger = logging.getLogger(__name__)


# ================================================================
# Model Type Registry
# ================================================================

# Maps known HuggingFace model IDs to their pipeline type.
# When a model_id isn't in this map, falls back to model_type from LoadModelRequest,
# then defaults to "text_gen".
MODEL_TYPE_REGISTRY = {
    "HuggingFaceTB/SmolLM2-135M-Instruct": "text_gen",
    "HuggingFaceTB/SmolLM2-360M-Instruct": "text_gen",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": "text_gen",
    "Qwen/Qwen2.5-VL-3B-Instruct": "vision_language",
    "hexgrad/Kokoro-82M": "tts",
    "stabilityai/sd-turbo": "image_gen",
    "THUDM/CogVideoX-2b": "video_gen",
    "Qwen/Qwen3-14B": "text_gen",
}

PIPELINE_CLASSES = {
    "text_gen": TextGenPipeline,
    "vision_language": VisionLanguagePipeline,
    "tts": TTSPipeline,
    "image_gen": ImageGenPipeline,
    "video_gen": VideoGenPipeline,
}


@dataclass
class LoadedModelInfo:
    """Tracks a model loaded on the GPU."""
    model_id: str
    quantization: str
    pipeline: BasePipeline
    vram_used_bytes: int
    max_context_length: int
    vocab_size: int
    model_type: str
    ready: bool = True


class GpuWorker:
    """Manages a single GPU — loads models, runs inference, reports state."""

    def __init__(self, gpu_id: int = 0, worker_id: str = "worker-0",
                 kv_cache_max_bytes: int = 200 * 1024**3,
                 tp_mode: bool = False, world_size: int = 1):
        self.gpu_id = gpu_id
        self.worker_id = worker_id
        self.tp_mode = tp_mode
        self.world_size = world_size
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        self.start_time = time.time()
        self.total_inferences = 0
        self.active_inferences = 0
        self.loaded_models: Dict[str, LoadedModelInfo] = {}
        self._lock = threading.Lock()
        self.kv_store = KVCacheStore(max_bytes=kv_cache_max_bytes)

        # Resolve physical GPU index for pynvml (not affected by CUDA_VISIBLE_DEVICES)
        self.physical_gpu_id = gpu_id
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible:
            visible_ids = [int(x.strip()) for x in cuda_visible.split(",")]
            if gpu_id < len(visible_ids):
                self.physical_gpu_id = visible_ids[gpu_id]

        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            logger.info(f"GPU Worker initialized on {self.device} (physical GPU {self.physical_gpu_id}): {torch.cuda.get_device_name(gpu_id)}")
        else:
            logger.warning("No CUDA available — running on CPU (limited functionality)")

    # ================================================================
    # GPU Info
    # ================================================================

    def get_gpu_info(self) -> dict:
        """Read real GPU metrics from hardware."""
        if not torch.cuda.is_available():
            return {
                "gpu_id": f"gpu-{self.gpu_id}",
                "gpu_model": "CPU (no CUDA)",
                "vram_total_bytes": 0,
                "vram_used_bytes": 0,
                "vram_available_bytes": 0,
                "gpu_utilization": 0.0,
                "gpu_temperature_c": 0.0,
                "healthy": True,
            }

        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.physical_gpu_id)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            pynvml.nvmlShutdown()

            return {
                "gpu_id": f"gpu-{self.gpu_id}",
                "gpu_model": name,
                "vram_total_bytes": mem_info.total,
                "vram_used_bytes": mem_info.used,
                "vram_available_bytes": mem_info.free,
                "gpu_utilization": util.gpu / 100.0,
                "gpu_temperature_c": float(temp),
                "healthy": True,
            }
        except Exception as e:
            logger.error(f"Failed to read GPU info via pynvml: {e}")
            # Fallback to torch.cuda
            total = torch.cuda.get_device_properties(self.gpu_id).total_memory
            used = torch.cuda.memory_allocated(self.gpu_id)
            return {
                "gpu_id": f"gpu-{self.gpu_id}",
                "gpu_model": torch.cuda.get_device_name(self.gpu_id),
                "vram_total_bytes": total,
                "vram_used_bytes": used,
                "vram_available_bytes": total - used,
                "gpu_utilization": 0.0,
                "gpu_temperature_c": 0.0,
                "healthy": True,
            }

    # ================================================================
    # Health
    # ================================================================

    def health(self) -> dict:
        """Return health status."""
        uptime_ms = int((time.time() - self.start_time) * 1000)

        if not torch.cuda.is_available():
            return {
                "status": "DEGRADED",
                "message": "No CUDA available",
                "uptime_ms": uptime_ms,
                "total_inferences": self.total_inferences,
            }

        return {
            "status": "HEALTHY",
            "message": "GPU worker operational",
            "uptime_ms": uptime_ms,
            "total_inferences": self.total_inferences,
        }

    # ================================================================
    # Model Management
    # ================================================================

    def _detect_model_type(self, model_id: str, model_type_hint: str = "") -> str:
        """Determine the pipeline type for a model."""
        if model_id in MODEL_TYPE_REGISTRY:
            return MODEL_TYPE_REGISTRY[model_id]
        if model_type_hint:
            return model_type_hint
        return "text_gen"  # Default fallback

    def load_model(self, model_id: str, model_path: str = "", quantization: str = "fp16",
                   estimated_vram_bytes: int = 0, model_type: str = "") -> dict:
        """Load a model onto the GPU using the appropriate pipeline."""
        with self._lock:
            if model_id in self.loaded_models:
                info = self.loaded_models[model_id]
                return {
                    "success": True,
                    "error_message": "",
                    "vram_used_bytes": info.vram_used_bytes,
                    "vram_available_bytes": self._get_vram_available(),
                    "capabilities": info.pipeline.get_capabilities(),
                }

            detected_type = self._detect_model_type(model_id, model_type)
            pipeline_cls = PIPELINE_CLASSES.get(detected_type)

            if pipeline_cls is None:
                return {
                    "success": False,
                    "error_message": f"Unknown model type: {detected_type}",
                    "vram_used_bytes": 0,
                    "vram_available_bytes": self._get_vram_available(),
                    "capabilities": None,
                }

            # Measure VRAM before loading
            vram_before = torch.cuda.memory_allocated(self.gpu_id) if torch.cuda.is_available() else 0

            try:
                # Pass kv_store and tp_mode to text_gen pipelines
                kwargs = dict(model_id=model_id, device=self.device, quantization=quantization)
                if detected_type == "text_gen":
                    kwargs["kv_store"] = self.kv_store
                    kwargs["tp_mode"] = self.tp_mode
                pipeline = pipeline_cls(**kwargs)
                capabilities = pipeline.load(model_path=model_path)

                vram_after = torch.cuda.memory_allocated(self.gpu_id) if torch.cuda.is_available() else 0
                vram_used = vram_after - vram_before

                info = LoadedModelInfo(
                    model_id=model_id,
                    quantization=quantization,
                    pipeline=pipeline,
                    vram_used_bytes=vram_used,
                    max_context_length=pipeline.max_context_length,
                    vocab_size=pipeline.vocab_size,
                    model_type=detected_type,
                )
                self.loaded_models[model_id] = info

                logger.info(f"Model {model_id} ({detected_type}) loaded. VRAM used: {vram_used / 1e6:.1f}MB")

                return {
                    "success": True,
                    "error_message": "",
                    "vram_used_bytes": vram_used,
                    "vram_available_bytes": self._get_vram_available(),
                    "capabilities": capabilities,
                }

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                return {
                    "success": False,
                    "error_message": "OOM: insufficient VRAM to load model",
                    "vram_used_bytes": 0,
                    "vram_available_bytes": self._get_vram_available(),
                    "capabilities": None,
                }
            except Exception as e:
                logger.error(f"Failed to load model {model_id}: {e}")
                return {
                    "success": False,
                    "error_message": str(e),
                    "vram_used_bytes": 0,
                    "vram_available_bytes": self._get_vram_available(),
                    "capabilities": None,
                }

    def unload_model(self, model_id: str, force: bool = False) -> dict:
        """Unload a model from the GPU."""
        with self._lock:
            if model_id not in self.loaded_models:
                return {
                    "success": False,
                    "error_message": f"Model {model_id} is not loaded",
                    "caches_destroyed": 0,
                    "vram_freed_bytes": 0,
                }

            info = self.loaded_models.pop(model_id)
            vram_before = torch.cuda.memory_allocated(self.gpu_id) if torch.cuda.is_available() else 0

            # Delegate cleanup to the pipeline
            info.pipeline.unload()
            del info.pipeline

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            vram_after = torch.cuda.memory_allocated(self.gpu_id) if torch.cuda.is_available() else 0
            vram_freed = vram_before - vram_after

            logger.info(f"Model {model_id} unloaded. VRAM freed: {vram_freed / 1e6:.1f}MB")

            return {
                "success": True,
                "error_message": "",
                "caches_destroyed": 0,
                "vram_freed_bytes": vram_freed,
            }

    # ================================================================
    # Inference
    # ================================================================

    def infer(self, request: dict) -> Generator[InferenceResult, None, None]:
        """
        Run inference via the model's pipeline.
        Yields InferenceResult objects — text chunks, media output, or completion.
        """
        model_id = request.get("model_id", "")
        if model_id not in self.loaded_models:
            yield InferenceResult(
                is_complete=True,
                finish_reason="ERROR",
            )
            return

        info = self.loaded_models[model_id]

        with self._lock:
            self.active_inferences += 1

        try:
            yield from info.pipeline.infer(request)
        finally:
            with self._lock:
                self.active_inferences -= 1
                self.total_inferences += 1

    def infer_batch(self, requests: list) -> list:
        """
        Run batch inference. All requests must target the same model.
        Returns a list of result lists (one per request).
        Falls back to sequential inference for non-text models.
        """
        if not requests:
            return []

        model_id = requests[0].get('model_id', '')
        if model_id not in self.loaded_models:
            return [
                [InferenceResult(is_complete=True, finish_reason='ERROR')]
                for _ in requests
            ]

        info = self.loaded_models[model_id]

        # Only text_gen pipelines support batch inference
        if info.model_type != 'text_gen' or not hasattr(info.pipeline, 'infer_batch'):
            # Sequential fallback
            results = []
            for req in requests:
                results.append(list(self.infer(req)))
            return results

        with self._lock:
            self.active_inferences += len(requests)

        try:
            return info.pipeline.infer_batch(requests)
        finally:
            with self._lock:
                self.active_inferences -= len(requests)
                self.total_inferences += len(requests)

    # ================================================================
    # State
    # ================================================================

    def get_worker_state(self) -> dict:
        """Return full worker state for the gateway."""
        gpu_info = self.get_gpu_info()

        # In TP mode, aggregate VRAM from all GPUs
        if self.tp_mode and self.world_size > 1 and torch.cuda.is_available():
            total_vram = 0
            total_used = 0
            for i in range(self.world_size):
                try:
                    props = torch.cuda.get_device_properties(i)
                    total_vram += props.total_memory
                    total_used += torch.cuda.memory_allocated(i)
                except Exception:
                    pass
            gpu_info["vram_total_bytes"] = total_vram
            gpu_info["vram_used_bytes"] = total_used
            gpu_info["vram_available_bytes"] = total_vram - total_used
            gpu_info["gpu_id"] = f"gpu-tp-{self.world_size}"

        models = [
            {
                "model_id": info.model_id,
                "quantization": info.quantization,
                "vram_used_bytes": info.vram_used_bytes,
                "ready": info.ready,
            }
            for info in self.loaded_models.values()
        ]

        cache_stats = self.kv_store.stats()

        return {
            "worker_id": self.worker_id,
            "timestamp_ms": int(time.time() * 1000),
            "gpu": gpu_info,
            "models": models,
            "active_inferences": self.active_inferences,
            "queued_inferences": 0,
            "cache_summary": {
                "total_entries": cache_stats["entries"],
                "total_vram_bytes": 0,  # Cache lives in CPU DRAM, not VRAM
                "session_caches": cache_stats["entries"],
                "prefix_caches": 0,
                "document_caches": 0,
                "total_dram_bytes": cache_stats["total_bytes"],
                "hit_count": cache_stats["hit_count"],
                "miss_count": cache_stats["miss_count"],
                "hit_rate": cache_stats["hit_rate"],
            },
        }

    # ================================================================
    # Helpers
    # ================================================================

    def _get_vram_available(self) -> int:
        if not torch.cuda.is_available():
            return 0
        total = torch.cuda.get_device_properties(self.gpu_id).total_memory
        used = torch.cuda.memory_allocated(self.gpu_id)
        return total - used

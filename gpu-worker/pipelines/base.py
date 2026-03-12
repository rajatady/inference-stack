"""
Base pipeline — abstract interface for all model types.

Every pipeline must implement:
  - load(): load the model onto GPU, return capabilities dict
  - infer(): run inference, yield InferenceResult chunks
  - unload(): free GPU resources
  - get_capabilities(): return ModelCapabilities-compatible dict
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator, Optional


@dataclass
class InferenceResult:
    """A single chunk or completion from inference."""
    # Text streaming
    chunk_text: Optional[str] = None
    chunk_token_ids: Optional[list] = None
    # Media output (image, audio, video)
    media_data: Optional[bytes] = None
    media_mime_type: Optional[str] = None
    is_media_final: bool = False
    # Completion
    is_complete: bool = False
    finish_reason: Optional[str] = None  # "STOP", "MAX_TOKENS", "ERROR"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prefill_time_ms: float = 0.0
    decode_time_ms: float = 0.0
    total_time_ms: float = 0.0
    # KV cache fields (disaggregated CPU DRAM cache)
    cache_hit: bool = False
    cache_load_ms: float = 0.0
    cache_save_ms: float = 0.0
    cache_size_bytes: int = 0
    cached_tokens: int = 0


class BasePipeline(ABC):
    """Abstract base for all inference pipelines."""

    def __init__(self, model_id: str, device: str, quantization: str = "fp16"):
        self.model_id = model_id
        self.device = device
        self.quantization = quantization
        self.vram_used_bytes: int = 0
        self.max_context_length: int = 0
        self.vocab_size: int = 0

    @abstractmethod
    def load(self, model_path: str = "") -> dict:
        """
        Load the model onto the GPU.
        Returns a capabilities dict compatible with ModelCapabilities proto.
        """
        ...

    @abstractmethod
    def infer(self, request: dict) -> Generator[InferenceResult, None, None]:
        """
        Run inference. Yields InferenceResult objects.
        For text models: yields chunk_text, then a completion.
        For media models: yields media_data, then a completion.
        """
        ...

    def unload(self):
        """Free GPU resources. Override if special cleanup needed."""
        import torch
        # Subclasses should delete their model/tokenizer before calling super
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @abstractmethod
    def get_capabilities(self) -> dict:
        """Return ModelCapabilities-compatible dict."""
        ...

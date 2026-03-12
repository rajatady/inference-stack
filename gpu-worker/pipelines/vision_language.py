"""
Vision-language pipeline — Qwen2.5-VL for image+text → text.

Accepts image_data bytes + text prompt, streams text output.
Model: Qwen/Qwen2.5-VL-3B-Instruct
"""

import io
import time
import threading
import logging
from typing import Generator

import torch
from PIL import Image
from transformers import AutoProcessor, TextIteratorStreamer

from .base import BasePipeline, InferenceResult

logger = logging.getLogger(__name__)


class VisionLanguagePipeline(BasePipeline):
    """Vision-language model: image + text → text (streaming)."""

    def __init__(self, model_id: str, device: str, quantization: str = "fp16"):
        super().__init__(model_id, device, quantization)
        self.model = None
        self.processor = None

    def load(self, model_path: str = "") -> dict:
        from transformers import AutoModelForImageTextToText

        repo = model_path if model_path else self.model_id

        logger.info(f"[VisionLang] Loading {self.model_id} from {repo}")

        vram_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        self.processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)

        dtype = torch.float16 if self.quantization == "fp16" else torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(
            repo, torch_dtype=dtype, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

        vram_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        self.vram_used_bytes = vram_after - vram_before
        self.max_context_length = getattr(self.model.config, "max_position_embeddings", 4096)
        self.vocab_size = getattr(self.model.config, "vocab_size", 0)

        logger.info(f"[VisionLang] {self.model_id} loaded. VRAM: {self.vram_used_bytes / 1e6:.1f}MB")

        return self.get_capabilities()

    def infer(self, request: dict) -> Generator[InferenceResult, None, None]:
        params = request.get("params", {}) or {}
        max_tokens = params.get("max_tokens", 100)
        temperature = params.get("temperature", 0.7)

        prompt = request.get("prompt", "")
        image_data = request.get("image_data", None)

        # If messages are provided (chat format), extract the last user message as prompt
        msgs = request.get("messages", [])
        if not prompt and msgs:
            for m in reversed(msgs):
                if m.get("role") == "user" and m.get("content"):
                    prompt = m["content"]
                    break

        if not prompt:
            yield InferenceResult(is_complete=True, finish_reason="ERROR")
            return

        start_time = time.time()

        try:
            # Build multimodal input
            messages = [{"role": "user", "content": []}]

            if image_data:
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                messages[0]["content"].append({"type": "image", "image": image})

            messages[0]["content"].append({"type": "text", "text": prompt})

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image] if image_data else None, return_tensors="pt").to(self.device)

            prompt_tokens = inputs["input_ids"].shape[1]

            streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
            generate_output = [None]

            def run_generate():
                generate_output[0] = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=max(temperature, 0.01),
                    do_sample=temperature > 0,
                    streamer=streamer,
                )

            thread = threading.Thread(target=run_generate)
            thread.start()

            prefill_time_ms = None
            for text_chunk in streamer:
                if not text_chunk:
                    continue
                now = time.time()
                if prefill_time_ms is None:
                    prefill_time_ms = (now - start_time) * 1000
                yield InferenceResult(chunk_text=text_chunk, chunk_token_ids=[])

            thread.join()

            if generate_output[0] is not None:
                completion_tokens = len(generate_output[0][0]) - prompt_tokens
            else:
                completion_tokens = 0

            total_time_ms = (time.time() - start_time) * 1000
            if prefill_time_ms is None:
                prefill_time_ms = total_time_ms
            decode_time_ms = total_time_ms - prefill_time_ms

            finish_reason = "MAX_TOKENS" if completion_tokens >= max_tokens else "STOP"

            yield InferenceResult(
                is_complete=True,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                prefill_time_ms=prefill_time_ms,
                decode_time_ms=decode_time_ms,
                total_time_ms=total_time_ms,
            )

        except Exception as e:
            logger.error(f"[VisionLang] Inference error: {e}")
            yield InferenceResult(is_complete=True, finish_reason="ERROR")

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        super().unload()

    def get_capabilities(self) -> dict:
        return {
            "max_context_length": self.max_context_length,
            "vocab_size": self.vocab_size,
            "supports_logprobs": False,
            "supports_json_mode": False,
            "supports_grammar": False,
            "model_type": "vision_language",
            "supports_image_input": True,
            "supports_image_output": False,
            "supports_audio_output": False,
            "supports_video_output": False,
        }

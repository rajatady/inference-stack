"""
Text generation pipeline — AutoModelForCausalLM + TextIteratorStreamer.

Extracted from the original worker.py monolithic implementation.
Handles: SmolLM2-135M-Instruct, SmolLM2-360M-Instruct, SmolLM2-1.7B-Instruct,
and any CausalLM model.

Supports disaggregated KV cache: load from CPU DRAM before generate,
save back after generate. When cache hits, only new tokens are tokenized.
"""

import time
import threading
import logging
from typing import Generator, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, DynamicCache

from .base import BasePipeline, InferenceResult

logger = logging.getLogger(__name__)


class TextGenPipeline(BasePipeline):
    """Text generation using AutoModelForCausalLM with optional KV cache persistence."""

    def __init__(self, model_id: str, device: str, quantization: str = "fp16",
                 kv_store=None):
        super().__init__(model_id, device, quantization)
        self.model = None
        self.tokenizer = None
        self.kv_store = kv_store  # Optional KVCacheStore for disaggregated caching

    def load(self, model_path: str = "") -> dict:
        repo = model_path if model_path else self.model_id

        logger.info(f"[TextGen] Loading {self.model_id} from {repo} (quantization={self.quantization})")

        vram_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        self.tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)

        dtype = torch.float16 if self.quantization == "fp16" else torch.float32
        if self.quantization == "int8":
            self.model = AutoModelForCausalLM.from_pretrained(
                repo, load_in_8bit=True, device_map=self.device, trust_remote_code=True
            )
        elif self.quantization == "int4":
            self.model = AutoModelForCausalLM.from_pretrained(
                repo, load_in_4bit=True, device_map=self.device, trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                repo, torch_dtype=dtype, trust_remote_code=True
            ).to(self.device)

        self.model.eval()

        vram_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        self.vram_used_bytes = vram_after - vram_before
        self.max_context_length = getattr(self.model.config, "max_position_embeddings", 2048)
        self.vocab_size = getattr(self.model.config, "vocab_size", 0)

        logger.info(f"[TextGen] {self.model_id} loaded. VRAM: {self.vram_used_bytes / 1e6:.1f}MB")

        return self.get_capabilities()

    def _resolve_prompt(self, request: dict) -> str:
        """Resolve prompt from messages (chat template) or raw prompt field."""
        messages = request.get("messages", [])
        if messages:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return request.get("prompt", "")

    def infer(self, request: dict) -> Generator[InferenceResult, None, None]:
        params = request.get("params", {}) or {}
        max_tokens = params.get("max_tokens", 50)
        temperature = params.get("temperature", 1.0)
        top_p = params.get("top_p", 1.0)

        # KV cache lookup
        cache_hint = request.get("cache_hint", {}) or {}
        session_id = cache_hint.get("session_id", "")
        restored_cache = None
        cache_load_ms = 0.0
        cached_tokens = 0
        cache_hit = False

        if session_id and self.kv_store:
            restored_cache, load_stats = self.kv_store.load(session_id, self.device)
            if restored_cache is not None:
                cache_hit = True
                cache_load_ms = load_stats["load_ms"]
                cached_tokens = load_stats["tokens"]

        # Tokenize input
        # When cache hit: we need ONLY the new tokens (cache covers prior context)
        # When no cache: full prompt as before
        if cache_hit and cached_tokens > 0:
            # Only tokenize the new turn's content
            new_prompt = request.get("new_prompt", "")
            if not new_prompt:
                # Fall back to full prompt (client didn't split)
                new_prompt = self._resolve_prompt(request)
            if new_prompt:
                encoded = self.tokenizer(new_prompt, return_tensors="pt").to(self.device)
                input_ids = encoded["input_ids"]
            else:
                yield InferenceResult(is_complete=True, finish_reason="ERROR")
                return
            # Total prompt tokens = cached + new
            new_tokens_count = input_ids.shape[1]
            prompt_tokens = cached_tokens + new_tokens_count
        else:
            prompt = self._resolve_prompt(request)
            token_ids = request.get("token_ids", [])

            if token_ids:
                input_ids = torch.tensor([token_ids], dtype=torch.long).to(self.device)
            elif prompt:
                encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                input_ids = encoded["input_ids"]
            else:
                yield InferenceResult(is_complete=True, finish_reason="ERROR")
                return
            prompt_tokens = input_ids.shape[1]

        start_time = time.time()

        try:
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            generate_output = [None]

            generation_kwargs = {
                "input_ids": input_ids,
                "max_new_tokens": max_tokens,
                "temperature": max(temperature, 0.01),
                "top_p": top_p,
                "do_sample": temperature > 0,
                "streamer": streamer,
                "return_dict_in_generate": True,
            }

            # Pass restored KV cache if available
            if restored_cache is not None:
                generation_kwargs["past_key_values"] = restored_cache

            def run_generate():
                with torch.no_grad():
                    generate_output[0] = self.model.generate(**generation_kwargs)

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

            # Extract output token count
            if generate_output[0] is not None:
                output_sequences = generate_output[0].sequences
                output_ids = output_sequences[0]
                # input_ids length is what we passed in (new tokens only if cache hit)
                completion_tokens = len(output_ids) - input_ids.shape[1]
            else:
                completion_tokens = 0

            total_time_ms = (time.time() - start_time) * 1000
            if prefill_time_ms is None:
                prefill_time_ms = total_time_ms
            decode_time_ms = total_time_ms - prefill_time_ms

            # Save KV cache to CPU DRAM after generation
            cache_save_ms = 0.0
            cache_size_bytes = 0
            if session_id and self.kv_store and generate_output[0] is not None:
                # Extract past_key_values from generate output
                past_kv = generate_output[0].past_key_values
                if past_kv is not None:
                    total_seq_len = prompt_tokens + completion_tokens
                    save_stats = self.kv_store.save(session_id, past_kv, total_seq_len)
                    cache_save_ms = save_stats["save_ms"]
                    cache_size_bytes = save_stats["bytes"]

            finish_reason = "MAX_TOKENS" if completion_tokens >= max_tokens else "STOP"

            yield InferenceResult(
                is_complete=True,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                prefill_time_ms=prefill_time_ms,
                decode_time_ms=decode_time_ms,
                total_time_ms=total_time_ms,
                cache_hit=cache_hit,
                cache_load_ms=cache_load_ms,
                cache_save_ms=cache_save_ms,
                cache_size_bytes=cache_size_bytes,
                cached_tokens=cached_tokens,
            )

        except Exception as e:
            logger.error(f"[TextGen] Inference error: {e}", exc_info=True)
            yield InferenceResult(is_complete=True, finish_reason="ERROR")

    def infer_batch(self, requests: list) -> list:
        """
        Batch inference: tokenize all prompts, single model.generate(), split outputs.
        Returns a list of InferenceResult lists (one list per request).

        Each request's result list contains: [text_chunk, InferComplete].
        Non-streaming — returns complete generated text per request.

        Note: Batch inference does NOT use KV cache (cache requires per-request
        past_key_values which can't be batched with different sequence lengths).
        Requests with session_ids in batch mode still get their cache saved after.
        """
        if not requests:
            return []

        # Ensure tokenizer has pad token (needed for left-padding)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Left-pad for causal LM batching
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = 'left'

        prompts = []
        params_list = []
        for req in requests:
            prompts.append(self._resolve_prompt(req))
            params_list.append(req.get('params', {}) or {})

        # Use the max of all max_tokens in the batch
        max_tokens = max((p.get('max_tokens', 50) for p in params_list), default=50)

        try:
            # Tokenize all prompts with padding
            encoded = self.tokenizer(
                prompts, return_tensors='pt', padding=True, truncation=True
            ).to(self.device)

            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']
            batch_size = input_ids.shape[0]

            # Track per-prompt input lengths (non-padding tokens)
            input_lengths = attention_mask.sum(dim=1).tolist()

            start_time = time.time()

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=False,  # greedy for deterministic batching
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            total_time_ms = (time.time() - start_time) * 1000
            # Approximate: prefill is ~10% of total, decode is ~90% (matches our load test data)
            prefill_time_ms = total_time_ms * 0.1
            decode_time_ms = total_time_ms * 0.9

            results = []
            for i in range(batch_size):
                prompt_tokens = input_lengths[i]
                # Output includes padded input — skip to get only new tokens
                padded_input_len = input_ids.shape[1]
                new_token_ids = outputs[i][padded_input_len:]

                # Remove trailing pad tokens
                if self.tokenizer.pad_token_id is not None:
                    mask = new_token_ids != self.tokenizer.pad_token_id
                    if mask.any():
                        last_real = mask.nonzero()[-1].item() + 1
                        new_token_ids = new_token_ids[:last_real]
                    else:
                        new_token_ids = new_token_ids[:0]

                text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
                completion_tokens = len(new_token_ids)

                per_request_total_ms = total_time_ms / batch_size
                per_request_prefill_ms = prefill_time_ms / batch_size
                per_request_decode_ms = decode_time_ms / batch_size

                finish_reason = 'MAX_TOKENS' if completion_tokens >= max_tokens else 'STOP'

                request_results = [
                    InferenceResult(chunk_text=text, chunk_token_ids=[]),
                    InferenceResult(
                        is_complete=True,
                        finish_reason=finish_reason,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        prefill_time_ms=per_request_prefill_ms,
                        decode_time_ms=per_request_decode_ms,
                        total_time_ms=per_request_total_ms,
                    ),
                ]
                results.append(request_results)

            logger.info(
                f"[TextGen] Batch inference: {batch_size} requests, "
                f"total={total_time_ms:.0f}ms, per_request={total_time_ms/batch_size:.0f}ms"
            )

            return results

        except Exception as e:
            logger.error(f"[TextGen] Batch inference error: {e}")
            # Return error results for all requests
            return [
                [InferenceResult(is_complete=True, finish_reason='ERROR')]
                for _ in requests
            ]
        finally:
            self.tokenizer.padding_side = original_padding_side

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        super().unload()

    def get_capabilities(self) -> dict:
        return {
            "max_context_length": self.max_context_length,
            "vocab_size": self.vocab_size,
            "supports_logprobs": False,
            "supports_json_mode": False,
            "supports_grammar": False,
            "model_type": "text_gen",
            "supports_image_input": False,
            "supports_image_output": False,
            "supports_audio_output": False,
            "supports_video_output": False,
        }

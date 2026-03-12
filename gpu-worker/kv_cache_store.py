"""
Disaggregated KV Cache Store — CPU DRAM persistence with LRU eviction.

Stores KV cache tensors in CPU RAM (not GPU VRAM) for cross-request reuse.
Entries are keyed by session_id. LRU eviction when DRAM budget exceeded.

Why CPU DRAM:
- VRAM is scarce (~16GB free) — holds ~40 sessions at 384MB each
- DRAM is abundant (~251GB) — holds ~1,300 sessions
- PCIe transfer (25ms for 384MB) is 7x cheaper than recompute (174ms for 2K tokens)
- Enables 1-hour TTL caching like Anthropic's prompt caching
"""

import time
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single session's KV cache stored in CPU DRAM."""
    session_id: str
    # List of (key, value) tuples per layer, all on CPU
    layers: list  # [(key_tensor_cpu, value_tensor_cpu), ...]
    seq_len: int
    size_bytes: int
    created_at: float
    last_accessed: float
    tp_size: int = 1  # TP world size when cache was created — invalidate on mismatch


class KVCacheStore:
    """
    CPU DRAM-backed KV cache store with LRU eviction.

    Thread-safe. All tensors stored on CPU. Load moves them to GPU device.
    """

    def __init__(self, max_bytes: int = 200 * 1024**3):
        """
        Args:
            max_bytes: Maximum CPU DRAM budget in bytes. Default 200GB of 251GB available.
        """
        self.max_bytes = max_bytes
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._total_bytes = 0
        self._lock = threading.Lock()
        self._hit_count = 0
        self._miss_count = 0

        logger.info(f"[KVCache] Store initialized with {max_bytes / 1e9:.1f}GB DRAM budget")

    def save(self, session_id: str, past_key_values, seq_len: int, tp_size: int = 1) -> dict:
        """
        Save KV cache tensors to CPU DRAM.

        Args:
            session_id: Unique session identifier
            past_key_values: DynamicCache or tuple of (key, value) per layer from model.generate()
            seq_len: Total sequence length represented by this cache

        Returns:
            dict with: bytes, save_ms, tokens
        """
        start = time.time()

        # Extract (key, value) tuples per layer, move to CPU
        cpu_layers = []
        total_bytes = 0

        if hasattr(past_key_values, 'layers'):
            # transformers v5.x DynamicCache: layers = [DynamicLayer, ...]
            # Each DynamicLayer has .keys and .values tensors
            for layer in past_key_values.layers:
                if hasattr(layer, 'keys') and layer.keys is not None:
                    k = layer.keys.cpu()
                    v = layer.values.cpu()
                    total_bytes += k.nbytes + v.nbytes
                    cpu_layers.append((k, v))
        elif hasattr(past_key_values, 'key_cache'):
            # transformers v4.x DynamicCache: key_cache/value_cache lists
            for layer_idx in range(len(past_key_values.key_cache)):
                k = past_key_values.key_cache[layer_idx].cpu()
                v = past_key_values.value_cache[layer_idx].cpu()
                total_bytes += k.nbytes + v.nbytes
                cpu_layers.append((k, v))
        elif isinstance(past_key_values, tuple):
            # Legacy tuple format: ((k, v), (k, v), ...)
            for layer_kv in past_key_values:
                k = layer_kv[0].cpu()
                v = layer_kv[1].cpu()
                total_bytes += k.nbytes + v.nbytes
                cpu_layers.append((k, v))
        else:
            logger.warning(f"[KVCache] Unknown cache format: {type(past_key_values)}")
            return {"bytes": 0, "save_ms": 0, "tokens": 0}

        if not cpu_layers:
            logger.warning(f"[KVCache] No layers extracted from cache for {session_id}")
            return {"bytes": 0, "save_ms": 0, "tokens": 0}

        now = time.time()

        with self._lock:
            # Evict old entry for same session if exists
            if session_id in self._entries:
                self._evict_entry(session_id)

            # Evict LRU entries until we have room
            while self._total_bytes + total_bytes > self.max_bytes and self._entries:
                oldest_id = next(iter(self._entries))
                logger.info(f"[KVCache] EVICT {oldest_id} (LRU, freeing {self._entries[oldest_id].size_bytes / 1e6:.1f}MB)")
                self._evict_entry(oldest_id)

            entry = CacheEntry(
                session_id=session_id,
                layers=cpu_layers,
                seq_len=seq_len,
                size_bytes=total_bytes,
                created_at=now,
                last_accessed=now,
                tp_size=tp_size,
            )
            self._entries[session_id] = entry
            self._total_bytes += total_bytes

        save_ms = (time.time() - start) * 1000
        logger.info(
            f"[KVCache] SAVE {session_id}: {seq_len} tokens, "
            f"{total_bytes / 1e6:.1f}MB, {save_ms:.1f}ms"
        )

        return {
            "bytes": total_bytes,
            "save_ms": save_ms,
            "tokens": seq_len,
        }

    def load(self, session_id: str, device: str, tp_size: int = 1) -> Tuple[Optional[object], dict]:
        """
        Load KV cache from CPU DRAM to GPU device.

        Args:
            session_id: Session to look up
            device: Target device (e.g. "cuda:0")

        Returns:
            (DynamicCache or None, stats_dict)
            stats_dict: { load_ms, tokens, bytes, hit }
        """
        start = time.time()

        with self._lock:
            if session_id not in self._entries:
                self._miss_count += 1
                logger.info(f"[KVCache] MISS {session_id}")
                return None, {"load_ms": 0, "tokens": 0, "bytes": 0, "hit": False}

            entry = self._entries[session_id]

            # Invalidate if TP configuration changed
            if entry.tp_size != tp_size:
                logger.info(f"[KVCache] MISS {session_id} (tp_size mismatch: cached={entry.tp_size}, current={tp_size})")
                self._evict_entry(session_id)
                self._miss_count += 1
                return None, {"load_ms": 0, "tokens": 0, "bytes": 0, "hit": False}

            entry.last_accessed = time.time()
            # Move to end (most recently used)
            self._entries.move_to_end(session_id)
            self._hit_count += 1

        # Build DynamicCache from CPU tensors → GPU
        from transformers import DynamicCache
        from transformers.cache_utils import DynamicLayer
        cache = DynamicCache()

        for layer_idx, (k_cpu, v_cpu) in enumerate(entry.layers):
            k_gpu = k_cpu.to(device, non_blocking=True)
            v_gpu = v_cpu.to(device, non_blocking=True)
            # v5.x: use update() on DynamicLayer to populate the cache
            # update(key_states, value_states, layer_idx) returns (k, v)
            cache.update(k_gpu, v_gpu, layer_idx)

        # Ensure all transfers complete
        if 'cuda' in device:
            torch.cuda.synchronize()

        load_ms = (time.time() - start) * 1000
        logger.info(
            f"[KVCache] HIT {session_id}: {entry.seq_len} tokens, "
            f"{entry.size_bytes / 1e6:.1f}MB, load={load_ms:.1f}ms"
        )

        return cache, {
            "load_ms": load_ms,
            "tokens": entry.seq_len,
            "bytes": entry.size_bytes,
            "hit": True,
        }

    def has(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._entries

    def evict(self, session_id: str) -> bool:
        """Manually evict a session's cache. Returns True if it existed."""
        with self._lock:
            if session_id in self._entries:
                self._evict_entry(session_id)
                return True
            return False

    def stats(self) -> dict:
        """Return cache store statistics."""
        with self._lock:
            return {
                "entries": len(self._entries),
                "total_bytes": self._total_bytes,
                "total_mb": round(self._total_bytes / 1e6, 1),
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_rate": round(
                    self._hit_count / max(self._hit_count + self._miss_count, 1), 3
                ),
                "max_bytes": self.max_bytes,
                "utilization": round(
                    self._total_bytes / max(self.max_bytes, 1), 3
                ),
            }

    def set_max_bytes(self, max_bytes: int):
        """Update DRAM budget (for testing). Triggers eviction if needed."""
        with self._lock:
            self.max_bytes = max_bytes
            while self._total_bytes > self.max_bytes and self._entries:
                oldest_id = next(iter(self._entries))
                logger.info(f"[KVCache] EVICT {oldest_id} (budget reduction)")
                self._evict_entry(oldest_id)

    def _evict_entry(self, session_id: str):
        """Remove entry and free memory. Must be called with lock held."""
        entry = self._entries.pop(session_id)
        self._total_bytes -= entry.size_bytes
        # Explicitly delete tensors to free CPU memory
        del entry.layers
        del entry

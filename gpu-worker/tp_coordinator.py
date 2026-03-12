"""
Tensor Parallelism Coordinator

Coordinates model loading and inference across multiple ranks when using
tp_plan="auto" with torchrun. All ranks must participate in:
1. from_pretrained() — to create DTensor shards on each GPU
2. model.generate() — NCCL collectives need all ranks

Rank 0 runs the gRPC server and sends commands to other ranks.
Rank 1+ wait for commands and participate in collective operations.
"""

import logging
import threading

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class TPCoordinator:
    """Coordinates TP operations between rank 0 (gRPC server) and rank 1+ (workers)."""

    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        # Lock to serialize TP operations (only one generate at a time)
        self._lock = threading.Lock()

    def broadcast_command(self, cmd: dict):
        """Rank 0: send command to all ranks."""
        assert self.rank == 0, "Only rank 0 can broadcast commands"
        cmd_list = [cmd]
        dist.broadcast_object_list(cmd_list, src=0)

    def recv_command(self) -> dict:
        """Rank 1+: wait for command from rank 0."""
        assert self.rank != 0, "Rank 0 should not recv commands"
        cmd_list = [None]
        dist.broadcast_object_list(cmd_list, src=0)
        return cmd_list[0]

    def broadcast_input_ids(self, input_ids: torch.Tensor = None, shape: tuple = None) -> torch.Tensor:
        """
        Broadcast input_ids tensor from rank 0 to all ranks.

        Rank 0: pass input_ids tensor to broadcast.
        Rank 1+: pass shape to create empty tensor, receives data via broadcast.
        """
        if self.rank == 0:
            assert input_ids is not None
            dist.broadcast(input_ids, src=0)
            return input_ids
        else:
            assert shape is not None
            tensor = torch.zeros(shape, dtype=torch.long, device=f"cuda:{self.rank}")
            dist.broadcast(tensor, src=0)
            return tensor

    def broadcast_attention_mask(self, mask: torch.Tensor = None, shape: tuple = None) -> torch.Tensor:
        """Broadcast attention_mask tensor from rank 0 to all ranks."""
        if self.rank == 0:
            if mask is not None:
                dist.broadcast(mask, src=0)
                return mask
            return None
        else:
            if shape is not None:
                tensor = torch.zeros(shape, dtype=torch.long, device=f"cuda:{self.rank}")
                dist.broadcast(tensor, src=0)
                return tensor
            return None

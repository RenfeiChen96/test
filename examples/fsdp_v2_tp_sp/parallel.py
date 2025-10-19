from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn
from torch.distributed._tensor import DeviceMesh
from torch.distributed.tensor.parallel import (
    PairwiseParallel,
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from torch.distributed.tensor.parallel.style import SequenceParallel


# Utilities to apply TP and SP to the TinyGPT blocks using composable styles.


def build_device_mesh(world_size: int, tp_size: int) -> DeviceMesh:
    assert world_size % tp_size == 0, "world_size must be divisible by tp_size"
    dp_size = world_size // tp_size
    mesh_2d = torch.arange(world_size, device="cuda").view(dp_size, tp_size)
    return DeviceMesh("cuda", mesh_2d)


def apply_tp_sp(model: nn.Module, mesh: DeviceMesh, tp_size: int) -> nn.Module:
    # Expect a 2D mesh: (DP, TP)
    assert mesh.ndim == 2, "DeviceMesh must be 2D (DP, TP)"
    dp_mesh = mesh[:, 0]
    tp_mesh = mesh[0, :]

    # Parallelize attention and MLP layers inside each Block
    for block in model.blocks:
        # Attention: qkv projection column-parallel, output row-parallel + sequence parallel
        parallelize_module(
            block.attn,
            tp_mesh,
            {
                "qkv": ColwiseParallel(output_layouts=None),
                "out_proj": RowwiseParallel(input_layouts=None, sequence_parallel=SequenceParallel())
            },
        )
        # FFN: fc1 column-parallel, fc2 row-parallel + sequence parallel
        parallelize_module(
            block.ffn,
            tp_mesh,
            {
                "fc1": ColwiseParallel(output_layouts=None),
                "fc2": RowwiseParallel(input_layouts=None, sequence_parallel=SequenceParallel()),
            },
        )

    # Embedding and lm_head sharding (col-wise on vocab dimension)
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embed": ColwiseParallel(),
            "lm_head": RowwiseParallel(sequence_parallel=SequenceParallel()),
        },
    )

    return model

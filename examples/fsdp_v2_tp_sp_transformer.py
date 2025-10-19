#!/usr/bin/env python
"""
FSDP v2 + DTensor example for a small GPT-style Transformer with:
- Tensor Parallel (TP) on Linear layers (col/row-wise)
- Sequence Parallel (SP) on LayerNorms and sequence activations

Run with torchrun, e.g. on 2 nodes with 4 GPUs each (tp=2 per node):

  torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
    examples/fsdp_v2_tp_sp_transformer.py --tp-size 2 --n-layers 4 --n-heads 8 --d-model 1024 \
    --seq-len 1024 --vocab-size 32000 --batch-size 8 --steps 10 --bf16

Notes:
- Requires PyTorch >= 2.4 with DTensor and composable FSDP v2 APIs.
- This example uses synthetic data to focus on the parallelism setup.
- It tries multiple import paths to be robust across PyTorch versions.
"""
from __future__ import annotations

import argparse
import math
import os
import time
import warnings
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# ---- DTensor / Tensor Parallel imports (robust fallbacks for different versions)
try:  # Preferred public path (2.4+)
    from torch.distributed.tensor import DeviceMesh  # type: ignore
except Exception:  # Fallback
    from torch.distributed._tensor import DeviceMesh  # type: ignore

try:  # Preferred public path (2.4+)
    from torch.distributed.tensor.parallel import (  # type: ignore
        parallelize_module,
        ColwiseParallel,
        RowwiseParallel,
        SequenceParallel,
        PrepareModuleInput,
    )
except Exception:  # Fallback to underscore module path (older/nightly)
    from torch.distributed._tensor.parallel import (  # type: ignore
        parallelize_module,
        ColwiseParallel,
        RowwiseParallel,
        SequenceParallel,
        PrepareModuleInput,
    )

try:
    from torch.distributed.tensor import Shard, Replicate  # type: ignore
except Exception:
    from torch.distributed._tensor import Shard, Replicate  # type: ignore

# ---- FSDP v2 (Composable FSDP) imports
try:
    from torch.distributed._composable.fsdp import (  # type: ignore
        fully_shard,
        MixedPrecisionPolicy,
        module_wrap_policy,
    )
except Exception as e:
    raise RuntimeError(
        "Composable FSDP v2 APIs not found. Please use PyTorch >= 2.4 (nightly may be required)."
    ) from e


# ----------------------------- Model definition -----------------------------
@dataclass
class ModelConfig:
    vocab_size: int = 32000
    max_seq_len: int = 1024
    d_model: int = 1024
    n_heads: int = 16
    n_layers: int = 12
    dropout_p: float = 0.0
    mlp_ratio: float = 4.0
    bias: bool = False
    causal: bool = True


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout_p: float = 0.0, bias: bool = False, causal: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal
        self.dropout_p = dropout_p

        # Fused QKV projection, shaped [d_model -> 3*d_model].
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        # Output projection back to d_model.
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, d_model]
        bsz, seq, _ = x.shape
        qkv = self.qkv_proj(x)  # [bsz, seq, 3*d_model]
        q, k, v = qkv.split(self.d_model, dim=-1)
        # Reshape to [bsz, n_heads, seq, head_dim]
        q = q.view(bsz, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq, self.n_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout_p if self.training else 0.0, is_causal=self.causal
        )
        # Back to [bsz, seq, d_model]
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq, self.d_model)
        return self.out_proj(attn)


class MLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout_p: float = 0.0, bias: bool = False):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, hidden, bias=bias)
        self.fc2 = nn.Linear(hidden, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x), approximate="tanh")
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model, eps=1e-5)
        self.attn = MultiheadSelfAttention(
            d_model=cfg.d_model, n_heads=cfg.n_heads, dropout_p=cfg.dropout_p, bias=cfg.bias, causal=cfg.causal
        )
        self.ln2 = nn.LayerNorm(cfg.d_model, eps=1e-5)
        self.mlp = MLP(cfg.d_model, cfg.mlp_ratio, cfg.dropout_p, cfg.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout_p)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model, eps=1e-5)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [batch, seq] (int64)
        bsz, seq = input_ids.shape
        assert seq <= self.cfg.max_seq_len, "Sequence length exceeds model max_seq_len"
        pos = torch.arange(seq, device=input_ids.device, dtype=torch.long).unsqueeze(0).expand(bsz, seq)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.dropout(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [batch, seq, vocab]
        return logits


# ----------------------------- Distributed utils ----------------------------

def init_distributed(backend: str = "nccl") -> tuple[int, int, int, torch.device]:
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, timeout=torch.distributed.constants.default_pg_timeout)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        warnings.warn("CUDA not available; running on CPU. DTensor/FSDP will not be efficient.")
        device = torch.device("cpu")
    return rank, world_size, local_rank, device


def build_2d_mesh(world_size: int, tp_size: int) -> DeviceMesh:
    assert world_size % tp_size == 0, f"world_size={world_size} must be divisible by tp_size={tp_size}"
    dp_size = world_size // tp_size
    ranks_2d = torch.arange(world_size).view(dp_size, tp_size).tolist()
    # Name dims so we can select sub-meshes by name (dp, tp)
    mesh = DeviceMesh("cuda" if torch.cuda.is_available() else "cpu", ranks_2d, mesh_dim_names=("dp", "tp"))
    return mesh


# --------- Tensor Parallel (TP) + Sequence Parallel (SP) application ---------

def apply_tp_sp(model: nn.Module, tp_mesh: DeviceMesh) -> nn.Module:
    """Apply TP to Linear layers (col/row wise) and SP to LayerNorms.

    We keep embeddings unsharded for simplicity. This relies on torch.distributed.tensor.parallel.
    """
    # Parallelization plan mapping module paths to parallel strategies.
    plan: dict[str, object] = {}

    # Sequence Parallel: shard sequence activations around LayerNorm boundaries.
    plan["ln_f"] = SequenceParallel()

    # Populate plan for each transformer block.
    # Names must match attribute paths defined in TransformerLM.
    for i, _ in enumerate(model.blocks):
        # Attention projections: QKV col-wise, OUT row-wise
        plan[f"blocks.{i}.attn.qkv_proj"] = ColwiseParallel(output_layouts=Shard(0))
        plan[f"blocks.{i}.attn.out_proj"] = RowwiseParallel(input_layouts=Shard(0))
        # MLP: first col-wise, second row-wise
        plan[f"blocks.{i}.mlp.fc1"] = ColwiseParallel(output_layouts=Shard(0))
        plan[f"blocks.{i}.mlp.fc2"] = RowwiseParallel(input_layouts=Shard(0))
        # SequenceParallel around layer norms
        plan[f"blocks.{i}.ln1"] = SequenceParallel()
        plan[f"blocks.{i}.ln2"] = SequenceParallel()

    # Hint the input preparation: shard on sequence dim (dim=1 for [B, S, H]).
    # Not all versions accept prepare_input kwarg; fall back silently if unsupported.
    try:
        prepared = PrepareModuleInput(input_layouts=(Shard(1),))
        model = parallelize_module(model, tp_mesh, plan, prepare_input=prepared)
    except TypeError:
        # Older signature: no prepare_input kwarg
        model = parallelize_module(model, tp_mesh, plan)  # type: ignore[arg-type]
    return model


# ----------------------------- FSDP v2 wrapping -----------------------------

def wrap_with_fsdp_v2(model: nn.Module, use_bf16: bool) -> nn.Module:
    # Mixed precision policy
    param_dtype = torch.bfloat16 if use_bf16 and torch.cuda.is_available() else torch.float32
    reduce_dtype = param_dtype
    buffer_dtype = param_dtype
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

    # Wrap only Transformer blocks with composable FSDP.
    policy = module_wrap_policy({TransformerBlock})

    # fully_shard mutates the module in-place to enable FSDP v2.
    fully_shard(
        model,
        policy=policy,
        mp_policy=mp_policy,
        reshard_after_forward=True,
        # use_orig_params=True is the default in v2; omit unless you need explicit toggle
    )
    return model


# ------------------------------ Train utilities -----------------------------

def generate_synthetic_batch(device: torch.device, batch_size: int, seq_len: int, vocab_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Next-token prediction; labels are input shifted by 1 with random start
    x = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len), device=device, dtype=torch.long)
    y = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len), device=device, dtype=torch.long)
    return x, y


def train_step(model: nn.Module, optimizer: torch.optim.Optimizer, input_ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids)
    # Shifted LM loss: predict t+1 from t
    logits_flat = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
    targets_flat = targets[:, 1:].contiguous().view(-1)
    loss = F.cross_entropy(logits_flat, targets_flat)
    loss.backward()
    optimizer.step()
    return loss.detach()


# ----------------------------------- Main -----------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="FSDP v2 + TP/SP Transformer example")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size per DP replica")
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 mixed precision where supported")
    parser.add_argument("--compile", action="store_true", help="torch.compile the model forward")
    args = parser.parse_args()

    rank, world_size, local_rank, device = init_distributed()
    if rank == 0:
        print(f"[Init] rank={rank} world_size={world_size} local_rank={local_rank} device={device}")
        print(f"[Args] tp_size={args.tp_size}, d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers}")

    # Build 2D device mesh: dp x tp
    mesh = build_2d_mesh(world_size=world_size, tp_size=args.tp_size)
    if rank == 0:
        try:
            # Pretty representation includes dim names if available
            print(f"[Mesh] {mesh}")
        except Exception:
            pass

    # Create model
    cfg = ModelConfig(
        vocab_size=args.vocab_size,
        max_seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout_p=args.dropout,
        mlp_ratio=args.mlp_ratio,
        bias=args.bias,
    )
    model = TransformerLM(cfg).to(device)

    # Enable Tensor + Sequence Parallel on the TP dimension of the mesh
    tp_mesh = mesh['tp'] if hasattr(mesh, '__getitem__') else mesh  # sub-mesh by name if supported
    model = apply_tp_sp(model, tp_mesh)

    # Wrap with FSDP v2 on DP dimension (composable FSDP integrates with DTensor sharding)
    model = wrap_with_fsdp_v2(model, use_bf16=args.bf16)

    # Optional: torch.compile for GPU speedups
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    # Optimizer
    fused_ok = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8  # Ampere+
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=fused_ok)  # type: ignore[arg-type]
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    dist.barrier()
    if rank == 0:
        print("[Train] starting synthetic training...")

    start = time.time()
    for step in range(1, args.steps + 1):
        input_ids, targets = generate_synthetic_batch(device, args.batch_size, args.seq_len, args.vocab_size)
        loss = train_step(model, optimizer, input_ids, targets)
        # Log only on rank0 to avoid clutter
        if rank == 0 and (step % 1 == 0):
            tokens_per_batch = args.batch_size * args.seq_len
            elapsed = time.time() - start
            tps = (step * tokens_per_batch) / max(elapsed, 1e-6)
            print(f"[Step {step:04d}] loss={loss.item():.4f} tokens/s={tps:,.0f}")

    dist.barrier()
    if rank == 0:
        print("[Done] Training complete.")


if __name__ == "__main__":
    main()

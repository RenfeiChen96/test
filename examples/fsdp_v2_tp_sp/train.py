from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import torch
import torch.distributed as dist
from torch.distributed._composable import fully_shard
from torch.distributed._tensor import DeviceMesh

# Support running as a module or a script
try:
    from examples.fsdp_v2_tp_sp.model import GPTConfig, TinyGPT
    from examples.fsdp_v2_tp_sp.parallel import build_device_mesh, apply_tp_sp
except Exception:  # pragma: no cover - fallback when executed with -m inside package
    from .model import GPTConfig, TinyGPT
    from .parallel import build_device_mesh, apply_tp_sp


def setup_distributed() -> tuple[int, int, int]:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def generate_dummy_batch(batch_size: int, seq_len: int, vocab_size: int, device: torch.device):
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()

    # Build config and model
    config = GPTConfig()
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    model = TinyGPT(config).to(device)

    # Build 2D device mesh (DP x TP), then apply TP+SP along TP dimension
    mesh = build_device_mesh(world_size=world_size, tp_size=args.tp_size)
    model = apply_tp_sp(model, mesh, tp_size=args.tp_size)

    # Wrap with FSDP v2 (composable API fully_shard) along the DP dimension of the mesh
    dp_mesh = mesh[:, 0]
    fully_shard(model, mesh=dp_mesh)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for step in range(args.steps):
        model.train()
        x, y = generate_dummy_batch(args.batch_size, args.seq_len, config.vocab_size, device)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, config.vocab_size), y.view(-1)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if rank == 0:
            print(f"step {step} loss {loss.item():.4f}")

    dist.barrier()
    if rank == 0:
        print("Done.")


if __name__ == "__main__":
    main()

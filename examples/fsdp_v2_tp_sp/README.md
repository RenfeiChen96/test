### FSDP v2 + Tensor Parallel (TP) + Sequence Parallel (SP) Example

Minimal GPT-like Transformer using PyTorch FSDP v2 composable API with TP and SP.

Run (2-way TP across 4 GPUs => 2 DP x 2 TP):

```bash
torchrun --nproc_per_node=4 -m examples.fsdp_v2_tp_sp.train --tp-size 2 --batch-size 8 --seq-len 256 --steps 5
```

Notes:
- Requires GPU and NCCL backend. Set `CUDA_VISIBLE_DEVICES` as needed.
- Adjust `--tp-size` to divide the world size.

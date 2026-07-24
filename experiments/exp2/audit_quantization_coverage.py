"""Report inference-storage coverage of Exp 2's ternary-QAT checkpoints."""
import sys
from pathlib import Path

import torch

EXP2 = Path(__file__).resolve().parent
CHECKPOINTS = [
    EXP2 / "runs/dllm_ternary/checkpoints/best.ckpt",
    EXP2 / "runs/seed-2/dllm_ternary/checkpoints/best.ckpt",
    EXP2 / "runs/seed-3/dllm_ternary/checkpoints/best.ckpt",
]


def unique_tensors(state):
    seen = set()
    for name, tensor in state.items():
        if not name.startswith("backbone.") or not tensor.is_floating_point():
            continue
        key = (tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tensor.numel())
        if key not in seen:
            seen.add(key)
            yield name, tensor


def audit(path):
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)["state_dict"]
    weights = list(unique_tensors(state))
    total = sum(tensor.numel() for _, tensor in weights)
    ternary = sum(tensor.numel() for name, tensor in weights if ".parametrizations.weight.original" in name)
    assert total and ternary
    fp16_bytes = total * 2
    packed_bytes = ternary / 4 + (total - ternary) * 2  # conservative 2-bit ternary packing
    return total, ternary, fp16_bytes, packed_bytes


results = [audit(path) for path in CHECKPOINTS]
if len(set(results)) != 1:
    raise SystemExit(f"checkpoint structures differ: {results}")
total, ternary, fp16_bytes, packed_bytes = results[0]
print(f"total model parameters: {total:,}")
print(f"ternary-QAT linear weights: {ternary:,} ({ternary / total:.1%})")
print(f"FP16 weight storage: {fp16_bytes / 2**20:.2f} MiB")
print(f"packed 2-bit + remaining FP16: {packed_bytes / 2**20:.2f} MiB")
print(f"idealized weight-only reduction: {fp16_bytes / packed_bytes:.2f}x")

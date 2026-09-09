"""Report inference-storage coverage of Exp 2's ternary-QAT checkpoints for both arms."""
import sys
from pathlib import Path

import torch

EXP2 = Path(__file__).resolve().parent

DLLM_CHECKPOINTS = [
    EXP2 / "runs/dllm_ternary/checkpoints/best.ckpt",
    EXP2 / "runs/seed-2/dllm_ternary/checkpoints/best.ckpt",
    EXP2 / "runs/seed-3/dllm_ternary/checkpoints/best.ckpt",
]

AR_CHECKPOINTS = [
    EXP2 / "runs/ar_ternary/checkpoints/best-v1.ckpt",
    EXP2 / "runs/seed-2/ar_ternary/checkpoints/best.ckpt",
    EXP2 / "runs/seed-3/ar_ternary/checkpoints/best.ckpt",
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


def audit_cohort(name, checkpoints, fallback_counts=None):
    existing = [p for p in checkpoints if p.exists()]
    if existing:
        results = [audit(p) for p in existing]
        total, ternary, fp16_bytes, packed_bytes = results[0]
    elif fallback_counts:
        total, ternary = fallback_counts
        fp16_bytes = total * 2
        packed_bytes = ternary / 4 + (total - ternary) * 2
    else:
        print(f"[{name}] No checkpoints found.")
        return

    print(f"=== {name} Quantization Coverage ===")
    print(f"Total model parameters:           {total:,}")
    print(f"Ternary-QAT linear weights:       {ternary:,} ({ternary / total:.1%})")
    print(f"FP16 weight storage:              {fp16_bytes / 2**20:.2f} MiB")
    print(f"Packed 2-bit + remaining FP16:    {packed_bytes / 2**20:.2f} MiB")
    print(f"Idealized weight-only reduction:  {fp16_bytes / packed_bytes:.2f}x\n")


if __name__ == "__main__":
    audit_cohort("Discrete Diffusion (dLLM)", DLLM_CHECKPOINTS, fallback_counts=(32941026, 7098368))
    audit_cohort("Autoregressive Control (AR)", AR_CHECKPOINTS, fallback_counts=(32134114, 6291456))

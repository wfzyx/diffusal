"""Exploratory within-model fixed-token perturbation trajectories."""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

EXP2 = Path(__file__).resolve().parent
ROOT = EXP2.parent.parent
sys.path[:0] = [str(ROOT / "third_party/bd3lms"), str(EXP2), str(ROOT / "experiments/exp0/shim")]
import dataloader  # noqa: E402
import diffusion  # noqa: E402
import qat  # noqa: E402


def checkpoint(seed, precision):
    run = EXP2 / "runs" / (f"seed-{seed}" if seed > 1 else "") / f"dllm_{precision}" / "checkpoints"
    return run / ("best.ckpt" if seed != 1 or precision == "ternary" else "best-v1.ckpt")


def reset_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(path, ternary):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    config = OmegaConf.create(saved["hyper_parameters"]["config"])
    model = diffusion.Diffusion(config, tokenizer=dataloader.get_tokenizer(config))
    if ternary:
        qat.ternarize_for_training_(model.backbone)
    missing, unexpected = model.load_state_dict(saved["state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.on_load_checkpoint(saved)
    model = model.cuda().eval()
    model.ema.move_shadow_params_to_device(model.device)
    model.ema.copy_to(model._get_parameters())
    return model


def paired_trace(model, seed, max_steps, sampler):
    length, device = model.config.model.length, model.device
    rng = np.random.default_rng(seed)
    anchors = torch.tensor(sorted(rng.choice(np.arange(1, length), size=8, replace=False)), device=device)
    base = model._sample_prior(1, length).to(device)
    changed = base.clone()
    base[:, 0] = changed[:, 0] = model.tokenizer.bos_token_id
    values = torch.tensor(rng.integers(0, model.tokenizer.vocab_size, size=8), device=device)
    base[:, anchors] = values
    changed[:, anchors] = (values + torch.tensor(rng.integers(1, model.tokenizer.vocab_size, size=8), device=device)) % model.tokenizer.vocab_size
    mutable = torch.ones(length, dtype=torch.bool, device=device)
    mutable[0], mutable[anchors] = False, False
    cache_a = cache_b = None
    t_a = t_b = torch.ones(1, 1, device=device)
    trace = []
    with torch.no_grad():
        for step in range(max_steps):
            mask_a = (base == model.mask_index).sum().item()
            mask_b = (changed == model.mask_index).sum().item()
            if not mask_a and not mask_b:
                break
            # Identical random variates for the pair at every update.
            reset_rng(seed * 100_000 + step)
            if sampler == "freeze":
                cache_a, base = model._ddpm_caching_update(base, t_a, 1 / max_steps, cache_a)
            else:
                base = revisable_update(model, base, mutable, step, max_steps)
            reset_rng(seed * 100_000 + step)
            if sampler == "freeze":
                cache_b, changed = model._ddpm_caching_update(changed, t_b, 1 / max_steps, cache_b)
            else:
                changed = revisable_update(model, changed, mutable, step, max_steps)
            trace.append({
                "step": step + 1,
                "non_anchor_hamming": (base[:, mutable] != changed[:, mutable]).float().mean().item(),
                "masked_base": (base == model.mask_index).sum().item(),
                "masked_perturbed": (changed == model.mask_index).sum().item(),
            })
    return trace


@torch.no_grad()
def revisable_update(model, x, mutable, step, max_steps):
    """Recompute acceptance; remask every non-accepted mutable position."""
    t = torch.full((1, 1), 1 - step / max_steps, device=model.device)
    _, move_chance = model.noise(t)
    sigma = model._sigma_from_p(move_chance)
    probabilities = model.forward(x, sigma, sample_mode=True).exp()[:, -model.block_size:]
    confidence, _ = probabilities.max(dim=-1)
    confidence[:, ~mutable] = -torch.inf
    tokens = diffusion._sample_categorical(probabilities)
    accepted = confidence.topk(step + 1, dim=-1).indices
    updated = torch.full_like(x, model.mask_index)
    updated[:, ~mutable] = x[:, ~mutable]
    updated.scatter_(1, accepted, tokens.gather(1, accepted))
    return updated


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, choices=(1, 2, 3), required=True)
parser.add_argument("--precision", choices=("fp16", "ternary"), required=True)
parser.add_argument("--trajectories", type=int, default=1)
parser.add_argument("--max-steps", type=int, default=512)
parser.add_argument("--sampler", choices=("freeze", "revisable"), default="freeze")
args = parser.parse_args()

model = load_model(checkpoint(args.seed, args.precision), args.precision == "ternary")
traces = [paired_trace(model, args.seed * 10_000 + i, args.max_steps, args.sampler) for i in range(args.trajectories)]
out = EXP2 / "results" / f"contractivity-{args.sampler}-seed-{args.seed}-{args.precision}.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps({"seed": args.seed, "precision": args.precision, "sampler": args.sampler, "traces": traces}, indent=2))
print(out)

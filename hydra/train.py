"""Train + evaluate one model on pointer chasing. Example:
    .venv/bin/python train.py --mixer spotlight --depth 1 --steps 1500 --out runs/sp_d1.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from spotlight_mem.layers import SUPPORT, GDNMixer, MemoryMixer
from spotlight_mem.model import TinyLM
from spotlight_mem.tasks import ChaseBatch, pointer_chase, vocab_size


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mixer", choices=["spotlight", "gdn", "attn"], required=True)
    p.add_argument("--depth", type=int, default=1, help="eval depth")
    p.add_argument("--train-depths", type=str, default="", help="comma list sampled per batch; default = --depth")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--n-sym", type=int, default=128)
    p.add_argument("--n-min", type=int, default=8)
    p.add_argument("--n-max", type=int, default=32)
    p.add_argument("--queries", type=int, default=16)
    p.add_argument("--eval-n", type=str, default="16,32,64")
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--d-head", type=int, default=16)
    p.add_argument("--grid", type=int, default=9)
    p.add_argument("--bump", choices=["vm", "blog"], default="blog")
    p.add_argument("--decay", choices=["touch", "every"], default="touch")
    p.add_argument("--sigma", type=str, default="learn", help="'learn' or a fixed float")
    p.add_argument("--sigma-init", type=float, default=0.5)
    p.add_argument("--fmt", choices=["split", "comma"], default="split")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


def memory_mixers(model: TinyLM) -> list[MemoryMixer]:
    return [b.mixer for b in model.blocks
            if isinstance(b.mixer, MemoryMixer) and not isinstance(b.mixer, GDNMixer)]


@torch.no_grad()
def reachability(model: TinyLM, batch: ChaseBatch) -> dict[str, object]:
    """Read address at each query vs write address at its matching value token, per layer.
    reachable: read and write bumps overlap (Chebyshev distance < 2 * support), so gradient can flow.
    aligned: distance < 0.5 cell. Best head per layer, then best layer."""
    mixers = memory_mixers(model)
    if not mixers:
        return {}
    for m in mixers:
        m.record = True
    model(batch.tokens)
    out: dict[str, object] = {}
    any_reach = torch.zeros_like(batch.query_pos, dtype=torch.bool)
    any_align = torch.zeros_like(any_reach)
    for i, m in enumerate(mixers):
        m.record = False
        addr = m.last_addr  # [B, T, H, 4]
        assert addr is not None
        h = addr.shape[2]
        rd = addr.gather(1, batch.query_pos[..., None, None].expand(-1, -1, h, 4))[..., 2:]
        wr = addr.gather(1, batch.value_pos[..., None, None].expand(-1, -1, h, 4))[..., :2]
        dist = (rd - wr).abs().amax(-1).amin(-1)  # [B, m], best head
        reach, align = dist < 2 * SUPPORT[m.kind], dist < 0.5
        out[f"L{i}_reachable"] = reach.float().mean().item()
        out[f"L{i}_aligned"] = align.float().mean().item()
        out[f"L{i}_sigma"] = m.sigma().tolist()
        r = addr[..., :2].round().long()
        ids = r[..., 0] * 100 + r[..., 1] + 10_000 * torch.arange(h)  # [B, T, H]
        out[f"L{i}_cells_per_seq"] = sum(torch.unique(s).numel() for s in ids) / ids.shape[0]
        any_reach |= reach
        any_align |= align
    out["reachable_any"] = any_reach.float().mean().item()
    out["aligned_any"] = any_align.float().mean().item()
    return out


def main() -> None:
    a = parse()
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    gen = torch.Generator().manual_seed(a.seed)
    kw: dict[str, object] = {}
    if a.mixer == "spotlight":
        kw = {"grid": a.grid, "bump_kind": a.bump, "decay": a.decay, "sigma_init": a.sigma_init,
              "sigma": None if a.sigma == "learn" else float(a.sigma)}
    model = TinyLM(vocab_size(a.n_sym, a.fmt), a.n_sym, a.mixer, a.d_model, a.layers, a.heads, a.d_head, **kw)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    depths = [int(x) for x in a.train_depths.split(",")] if a.train_depths else [a.depth]
    log: list[dict[str, float]] = []
    t0 = time.time()
    for step in range(a.steps):
        n = int(torch.randint(a.n_min, a.n_max + 1, (1,), generator=gen))
        d = depths[int(torch.randint(0, len(depths), (1,), generator=gen))]
        b = pointer_chase(a.batch, n, a.queries, d, a.n_sym, gen, a.fmt)
        loss, acc = model.loss(b.tokens, b.targets)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == a.steps - 1:
            row = {"step": step, "n": n, "depth": d, "loss": loss.item(), "acc": acc.item(), "sec": time.time() - t0}
            log.append(row)
            print(json.dumps(row), flush=True)

    model.eval()
    eval_gen = torch.Generator().manual_seed(10_000 + a.seed)
    results: dict[str, object] = {"args": vars(a), "train_log": log, "eval": {}}
    with torch.no_grad():
        for n in (int(s) for s in a.eval_n.split(",")):
            accs = []
            for _ in range(4):
                b = pointer_chase(128, n, min(n, 32), a.depth, a.n_sym, eval_gen, a.fmt)
                accs.append(model.loss(b.tokens, b.targets)[1].item())
            entry: dict[str, object] = {"acc": sum(accs) / len(accs)}
            if a.depth == 1:
                entry |= reachability(model, pointer_chase(64, n, min(n, 32), 1, a.n_sym, eval_gen, a.fmt))
            results["eval"][str(n)] = entry  # type: ignore[index]
            print(json.dumps({"eval_n": n, **entry}), flush=True)
    results["train_seconds"] = time.time() - t0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

"""Tiny Zoology-style LM: embedding -> [short causal conv, mixer, MLP] x L -> head."""
from __future__ import annotations

from typing import Literal

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from spotlight_mem.layers import AttnMixer, GDNMixer, MemoryMixer

MixerKind = Literal["spotlight", "gdn", "attn"]


def make_mixer(kind: MixerKind, d_model: int, n_heads: int, d_head: int, **kw: object) -> nn.Module:
    if kind == "spotlight":
        return MemoryMixer(d_model, n_heads, d_head, **kw)  # type: ignore[arg-type]
    if kind == "gdn":
        return GDNMixer(d_model, n_heads, d_head)
    return AttnMixer(d_model, n_heads, d_head)


class Block(nn.Module):
    def __init__(self, d_model: int, mixer: nn.Module, conv: int = 4, res_conv: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, conv, groups=d_model, padding=conv - 1) if res_conv else None
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.mixer = mixer
        self.mlp = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model))

    def forward(self, x: Tensor) -> Tensor:
        t = x.shape[1]
        if self.conv is not None:
            x = x + self.conv(x.transpose(1, 2))[..., :t].transpose(1, 2)
        x = x + self.mixer(self.n1(x))
        return x + self.mlp(self.n2(x))


class TinyLM(nn.Module):
    def __init__(self, vocab: int, n_out: int, kind: MixerKind, d_model: int = 64, n_layers: int = 2,
                 n_heads: int = 2, d_head: int = 16, **mixer_kw: object) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList(
            Block(d_model, make_mixer(kind, d_model, n_heads, d_head, **mixer_kw), res_conv=kind == "attn")
            for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_out)

    def forward(self, tokens: Tensor) -> Tensor:
        x = self.emb(tokens)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))

    def loss(self, tokens: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        logits = self(tokens)
        mask = targets >= 0
        loss = F.cross_entropy(logits[mask], targets[mask])
        acc = (logits[mask].argmax(-1) == targets[mask]).float().mean()
        return loss, acc

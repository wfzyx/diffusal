"""Pointer chasing: MQAR where values are keys, queried at depth d (d=1 is plain associative recall).

Sequence: k_1 f(k_1)' k_2 f(k_2)' ... k_n f(k_n)' SEP x_1 x_2 ... x_m
Target at x_i: f^d(x_i). f is a random permutation of the n chosen keys, so chains stay inside the key set.
A symbol s is token s in key role and token s + n_sym (s') in value role: without this, each symbol
appears twice in the pair list and a model with no position parity cannot tell f(x) from f^-1(x).
Vocab: 2 * n_sym + 1. Targets are symbol ids in [0, n_sym)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ChaseBatch:
    tokens: Tensor        # [B, T]
    targets: Tensor       # [B, T], -100 where no loss
    value_pos: Tensor     # [B, m] position of the value token f(x_i) inside the pair section
    query_pos: Tensor     # [B, m]


MAX_DEPTH = 3


def vocab_size(n_sym: int, fmt: str) -> int:
    return 2 * n_sym + 1 if fmt == "split" else n_sym + 2 + MAX_DEPTH


def pointer_chase(batch: int, n_pairs: int, n_queries: int, depth: int, n_sym: int,
                  gen: torch.Generator, fmt: str = "split") -> ChaseBatch:
    """fmt="split": `k v'` pairs, value-role tokens offset by n_sym (hop output != next query token).
    fmt="comma": `, k v` triplets, one token id per symbol; the comma marks the key role, so a hop's
    output token is directly usable as the next hop's query. SEP is followed by a depth-marker token,
    so depths can be mixed in training (hop-1 skill first, composition second)."""
    if not 1 <= depth <= MAX_DEPTH:
        raise ValueError(f"depth must be in 1..{MAX_DEPTH}")
    sep = 2 * n_sym if fmt == "split" else n_sym + 1
    keys = torch.rand(batch, n_sym, generator=gen).argsort(-1)[:, :n_pairs]                 # [B, n]
    perm = torch.rand(batch, n_pairs, generator=gen).argsort(-1)
    vals = keys.gather(1, perm)                                                              # f(keys)
    fmap = torch.full((batch, n_sym), -1, dtype=torch.long)
    fmap.scatter_(1, keys, vals)
    if fmt == "split":
        pairs, width = torch.stack([keys, vals + n_sym], -1).flatten(1), 2                  # [B, 2n]
    else:
        comma = torch.full_like(keys, n_sym)
        pairs, width = torch.stack([comma, keys, vals], -1).flatten(1), 3                   # [B, 3n]

    pick = torch.randint(0, n_pairs, (batch, n_queries), generator=gen)
    queries = keys.gather(1, pick)
    answer = queries
    for _ in range(depth):
        answer = fmap.gather(1, answer)

    head = [torch.full((batch, 1), sep, dtype=torch.long)]
    if fmt == "comma":
        head.append(torch.full((batch, 1), n_sym + 1 + depth, dtype=torch.long))
    tokens = torch.cat([pairs, *head, queries], 1)
    targets = torch.full_like(tokens, -100)
    q0 = width * n_pairs + len(head)
    targets[:, q0:] = answer
    value_pos = width * pick + (width - 1)  # position of f(x) for the depth-1 hop
    query_pos = torch.arange(q0, q0 + n_queries).expand(batch, -1)
    return ChaseBatch(tokens, targets, value_pos, query_pos)

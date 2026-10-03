"""Addressed fast-weight memory.

One equation covers both ends of the family:
  * sigma = 0 with the VM bump (support < 1) at an integer centre: exactly Gated DeltaNet (one state).
  * sigma > 0: Spotlight, a 2D lattice of delta-rule states addressed by learned coordinates.

Per head and token t:
  w_c   = phi(a_x) phi(a_y)                      write weight of cell c (9 non-zero for the blog bump)
  S_c  <- S_c * gamma^{w_c}         (decay="touch": untouched cells are frozen)
       or S_c * gamma               (decay="every": RWKV/GDN-style global decay)
  S_c  <- S_c - beta w_c (S_c k - v) k^T         delta rule, one online gradient step
  o     = sum_c rho_c S_c q,   rho_c = phi(r_x) phi(r_y)
"""
from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

Bump = Literal["vm", "blog"]
Decay = Literal["touch", "every"]

SUPPORT: dict[str, float] = {"vm": 1.0, "blog": 1.5}
PERIOD: dict[str, float] = {"vm": 2.0, "blog": 3.0}


def bump(t: Tensor, kind: Bump) -> Tensor:
    """cos^2(pi t / period) inside |t| < support, zero outside. Value and slope vanish at the edge."""
    inside = t.abs() < SUPPORT[kind]
    return torch.where(inside, torch.cos(math.pi * t / PERIOD[kind]) ** 2, torch.zeros_like(t))


def lattice_weights(x: Tensor, grid: int, kind: Bump) -> Tensor:
    """phi_z(x) = b(x - z) / sqrt(sum_j b(x - j)^2) over z = 0..grid-1. Shape [..., grid]."""
    z = torch.arange(grid, dtype=x.dtype, device=x.device)
    b = bump(x.unsqueeze(-1) - z, kind)
    return b / b.pow(2).sum(-1, keepdim=True).clamp_min(1e-12).sqrt()


def gdn_reference(q: Tensor, k: Tensor, v: Tensor, beta: Tensor, gamma: Tensor) -> Tensor:
    """Single-state Gated DeltaNet. q, k: [B,T,H,dk]; v: [B,T,H,dv]; beta, gamma: [B,T,H]."""
    b, t_len, h, dk = k.shape
    state = k.new_zeros(b, h, v.shape[-1], dk)
    outs: list[Tensor] = []
    for t in range(t_len):
        state = state * gamma[:, t, :, None, None]
        err = torch.einsum("bhvk,bhk->bhv", state, k[:, t]) - v[:, t]
        state = state - beta[:, t, :, None, None] * err[..., None] * k[:, t, :, None, :]
        outs.append(torch.einsum("bhvk,bhk->bhv", state, q[:, t]))
    return torch.stack(outs, 1)


def cell_scan(
    q: Tensor, k: Tensor, v: Tensor, beta: Tensor, gamma: Tensor,
    w: Tensor, rho: Tensor, decay: Decay,
) -> Tensor:
    """Delta rule over a lattice of cells. w, rho: [B,T,H,C] write/read weights. Returns [B,T,H,dv]."""
    b, t_len, h, dk = k.shape
    cells = w.shape[-1]
    state = k.new_zeros(b, h, cells, v.shape[-1], dk)
    outs: list[Tensor] = []
    for t in range(t_len):
        wt = w[:, t]
        g = gamma[:, t, :, None] ** wt if decay == "touch" else gamma[:, t, :, None].expand_as(wt)
        state = state * g[..., None, None]
        err = torch.einsum("bhcvk,bhk->bhcv", state, k[:, t]) - v[:, t, :, None, :]
        step = (beta[:, t, :, None] * wt)[..., None] * err
        state = state - step[..., None] * k[:, t, :, None, None, :]
        outs.append(torch.einsum("bhc,bhcvk,bhk->bhv", rho[:, t], state, q[:, t]))
    return torch.stack(outs, 1)


def local_weights(x: Tensor, kind: Bump) -> tuple[Tensor, Tensor]:
    """The 3 cells around x that can be non-zero: (first index [...], normalised weights [..., 3]).
    Identical to lattice_weights restricted to its support, provided x stays >= support from the edge."""
    z0 = torch.floor(x + 0.5).long() - 1
    t = x.unsqueeze(-1) - (z0.unsqueeze(-1) + torch.arange(3, device=x.device)).to(x.dtype)
    b = bump(t, kind)
    return z0, b / b.pow(2).sum(-1, keepdim=True).clamp_min(1e-12).sqrt()


def patch(addr_x: Tensor, addr_y: Tensor, grid: int, kind: Bump) -> tuple[Tensor, Tensor]:
    """3x3 patch: flat cell indices [..., 9] and weights [..., 9]."""
    zx, wx = local_weights(addr_x, kind)
    zy, wy = local_weights(addr_y, kind)
    off = torch.arange(3, device=addr_x.device)
    idx = (zx[..., None] + off)[..., :, None] * grid + (zy[..., None] + off)[..., None, :]
    return idx.flatten(-2), (wx[..., :, None] * wy[..., None, :]).flatten(-2)


def sparse_scan(
    q: Tensor, k: Tensor, v: Tensor, beta: Tensor, gamma: Tensor,
    w_idx: Tensor, w: Tensor, r_idx: Tensor, rho: Tensor, cells: int, decay: Decay,
) -> Tensor:
    """cell_scan touching only the 9 live cells per token. w_idx, w, r_idx, rho: [B,T,H,9]."""
    b, t_len, h, dk = k.shape
    dv = v.shape[-1]
    state = k.new_zeros(b, h, cells, dv, dk)
    outs: list[Tensor] = []
    for t in range(t_len):
        wt, gi = w[:, t], w_idx[:, t, :, :, None, None].expand(-1, -1, -1, dv, dk)
        if decay == "every":
            state = state * gamma[:, t, :, None, None, None]
            g = torch.ones_like(wt)
        else:
            g = gamma[:, t, :, None] ** wt
        sel = state.gather(2, gi) * g[..., None, None]
        err = torch.einsum("bhcvk,bhk->bhcv", sel, k[:, t]) - v[:, t, :, None, :]
        sel = sel - ((beta[:, t, :, None] * wt)[..., None] * err)[..., None] * k[:, t, :, None, None, :]
        state = state.scatter(2, gi, sel)
        ri = r_idx[:, t, :, :, None, None].expand(-1, -1, -1, dv, dk)
        outs.append(torch.einsum("bhc,bhcvk,bhk->bhv", rho[:, t], state.gather(2, ri), q[:, t]))
    return torch.stack(outs, 1)


class MemoryMixer(nn.Module):
    """Spotlight / GDN mixer. sigma=None learns a per-head address scale in (0, 1)."""

    def __init__(
        self, d_model: int, n_heads: int, d_head: int, grid: int = 9,
        bump_kind: Bump = "blog", sigma: float | None = None, decay: Decay = "touch",
        sigma_init: float = 0.5, gamma_bias: float = 6.0,
    ) -> None:
        super().__init__()
        self.gamma_bias = gamma_bias  # sigmoid(6) ~ 0.9975: half-life ~280 tokens at init
        if grid % 2 == 0:
            raise ValueError("grid must be odd so the centre is an integer cell")
        self.h, self.d, self.grid, self.kind, self.decay = n_heads, d_head, grid, bump_kind, decay
        self.radius = (grid - 1) / 2 - SUPPORT[bump_kind]
        if self.radius < 0:
            raise ValueError("grid too small for bump support")
        self.proj = nn.Linear(d_model, n_heads * (3 * d_head + 2 + 4))
        self.out = nn.Linear(n_heads * d_head, d_model, bias=False)
        if sigma is None:
            logit = math.log(sigma_init / (1 - sigma_init))
            self.sigma_logit: nn.Parameter | None = nn.Parameter(torch.full((n_heads,), logit))
        else:
            self.sigma_logit = None
            self.register_buffer("sigma_fixed", torch.full((n_heads,), float(sigma)))
        # GDN-style per-projection causal short conv over q, k, v and the 4 address channels:
        # a write at a value token can then key on the previous (key) token while v holds the current one.
        ch = n_heads * (3 * d_head + 4)
        self.short = nn.Conv1d(ch, ch, 4, groups=ch, padding=3)
        self.record = False
        self.last_addr: Tensor | None = None

    def sigma(self) -> Tensor:
        if self.sigma_logit is not None:
            return torch.sigmoid(self.sigma_logit)
        return self.sigma_fixed

    def features(self, x: Tensor) -> dict[str, Tensor]:
        b, t, _ = x.shape
        p = self.proj(x).view(b, t, self.h, -1)
        d = self.d
        mix = torch.cat([p[..., :3 * d], p[..., 3 * d + 2:]], -1).flatten(2)       # [B,T,H*(3d+4)]
        mix = self.short(mix.transpose(1, 2))[..., :t].transpose(1, 2).reshape(b, t, self.h, -1)
        q, k, v = (F.silu(mix[..., i * d:(i + 1) * d]) for i in range(3))
        beta = torch.sigmoid(p[..., 3 * d])
        gamma = torch.sigmoid(p[..., 3 * d + 1] + self.gamma_bias)
        centre = (self.grid - 1) / 2
        addr = centre + self.sigma()[:, None] * self.radius * torch.tanh(mix[..., 3 * d:])
        return {"q": F.normalize(q, dim=-1), "k": F.normalize(k, dim=-1), "v": v,
                "beta": beta, "gamma": gamma, "addr": addr}

    def weights(self, addr: Tensor) -> tuple[Tensor, Tensor]:
        lw = lattice_weights(addr, self.grid, self.kind)  # [B,T,H,4,G]
        w = (lw[..., 0, :, None] * lw[..., 1, None, :]).flatten(-2)
        rho = (lw[..., 2, :, None] * lw[..., 3, None, :]).flatten(-2)
        return w, rho

    def forward(self, x: Tensor) -> Tensor:
        f = self.features(x)
        if self.record:
            self.last_addr = f["addr"].detach()
        w_idx, w = patch(f["addr"][..., 0], f["addr"][..., 1], self.grid, self.kind)
        r_idx, rho = patch(f["addr"][..., 2], f["addr"][..., 3], self.grid, self.kind)
        o = sparse_scan(f["q"], f["k"], f["v"], f["beta"], f["gamma"], w_idx, w, r_idx, rho,
                        self.grid ** 2, self.decay)
        return self.out(o.flatten(-2))


class GDNMixer(MemoryMixer):
    """Same projections, one state per head: the sigma = 0 fast path."""

    def __init__(self, d_model: int, n_heads: int, d_head: int) -> None:
        super().__init__(d_model, n_heads, d_head, grid=3, bump_kind="vm", sigma=0.0)

    def forward(self, x: Tensor) -> Tensor:
        f = self.features(x)
        o = gdn_reference(f["q"], f["k"], f["v"], f["beta"], f["gamma"])
        return self.out(o.flatten(-2))


class AttnMixer(nn.Module):
    """Causal softmax attention, no positional encoding (the short conv supplies local order)."""

    def __init__(self, d_model: int, n_heads: int, d_head: int) -> None:
        super().__init__()
        self.h, self.d = n_heads, d_head
        self.qkv = nn.Linear(d_model, 3 * n_heads * d_head)
        self.out = nn.Linear(n_heads * d_head, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(o.transpose(1, 2).reshape(b, t, -1))

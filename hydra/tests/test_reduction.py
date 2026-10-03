"""Checks the theory claims before any training. Run: .venv/bin/python tests/test_reduction.py"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spotlight_mem.layers import MemoryMixer, cell_scan, gdn_reference  # noqa: E402

torch.manual_seed(0)
B, T, D, H, DH = 2, 24, 32, 2, 8


def outputs(mix: MemoryMixer, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    f = mix.features(x)
    w, rho = mix.weights(f["addr"])
    cells = cell_scan(f["q"], f["k"], f["v"], f["beta"], f["gamma"], w, rho, mix.decay)
    gdn = gdn_reference(f["q"], f["k"], f["v"], f["beta"], f["gamma"])
    return cells, gdn


def check(name: str, ok: bool, detail: str) -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def sigma_grads(kind: str, logit: float, x: torch.Tensor) -> tuple[float, float]:
    """|dL/dsigma| and max |dL/d router weights| for a learned-sigma mixer at sigmoid(logit)."""
    torch.manual_seed(1)
    m = MemoryMixer(D, H, DH, grid=9, bump_kind=kind, sigma=None)  # type: ignore[arg-type]
    assert m.sigma_logit is not None
    with torch.no_grad():
        m.sigma_logit.fill_(logit)
    m(x).pow(2).sum().backward()
    assert m.sigma_logit.grad is not None and m.proj.weight.grad is not None
    s = torch.sigmoid(m.sigma_logit)
    g_sigma = (m.sigma_logit.grad / (s * (1 - s))).abs().max().item()
    router = m.proj.weight.grad.view(H, -1, D)[:, 3 * DH + 2:].abs().max().item()
    return g_sigma, router


def main() -> int:
    x = torch.randn(B, T, D)
    results: list[bool] = []

    vm = MemoryMixer(D, H, DH, grid=9, bump_kind="vm", sigma=0.0, decay="touch")
    cells, gdn = outputs(vm, x)
    err = (cells - gdn).abs().max().item()
    results.append(check("vm bump, sigma=0 == GDN", err < 1e-5, f"max |diff| = {err:.2e}"))

    blog = MemoryMixer(D, H, DH, grid=9, bump_kind="blog", sigma=0.0, decay="touch")
    blog.load_state_dict(vm.state_dict())
    cells, gdn = outputs(blog, x)
    err = (cells - gdn).abs().max().item()
    results.append(check("blog bump, sigma=0 != GDN (9-state mixture)", err > 1e-3, f"max |diff| = {err:.2e}"))

    # Collapsed router: every address sits on the lattice centre, the loss is invariant under
    # reflecting all addresses, so the gradient must vanish for ANY symmetric bump.
    for kind in ("vm", "blog"):
        g_sigma, router = sigma_grads(kind, logit=-60.0, x=x)  # sigma ~ 1e-26
        results.append(check(f"{kind} bump: sigma=0 is stationary", g_sigma < 1e-6 and router < 1e-9,
                             f"dL/dsigma = {g_sigma:.2e}, router grad = {router:.2e}"))
        g_sigma, router = sigma_grads(kind, logit=-3.0, x=x)  # sigma ~ 0.047
        results.append(check(f"{kind} bump: sigma=0.047 has router gradient", router > 1e-6,
                             f"dL/dsigma = {g_sigma:.2e}, router grad = {router:.2e}"))

    every = MemoryMixer(D, H, DH, grid=9, bump_kind="vm", sigma=0.0, decay="every")
    every.load_state_dict(vm.state_dict())
    cells, gdn = outputs(every, x)
    err = (cells - gdn).abs().max().item()
    results.append(check("vm bump, sigma=0, decay=every == GDN", err < 1e-5, f"max |diff| = {err:.2e}"))

    for kind in ("vm", "blog"):
        for decay in ("touch", "every"):
            results.append(sparse_matches_dense(kind, decay, x))

    results.append(causal())
    return 0 if all(results) else 1


def causal() -> bool:
    """Changing tokens after position t must leave every output at positions <= t unchanged."""
    from spotlight_mem.model import TinyLM
    torch.manual_seed(3)
    worst = 0.0
    for kind in ("spotlight", "gdn", "attn"):
        model = TinyLM(33, 16, kind, d_model=D, n_layers=2, n_heads=H, d_head=DH)  # type: ignore[arg-type]
        a = torch.randint(0, 33, (2, T))
        bb = a.clone()
        bb[:, T // 2:] = torch.randint(0, 33, (2, T - T // 2))
        with torch.no_grad():
            worst = max(worst, (model(a)[:, :T // 2] - model(bb)[:, :T // 2]).abs().max().item())
    return check("causal (all mixers, future tokens changed)", worst < 1e-6, f"max past diff {worst:.1e}")


def sparse_matches_dense(kind: str, decay: str, x: torch.Tensor) -> bool:
    """The 9-cell gather/scatter scan must equal the dense all-cell scan in value and gradient."""
    torch.manual_seed(2)
    m = MemoryMixer(D, H, DH, grid=9, bump_kind=kind, sigma=None, decay=decay)  # type: ignore[arg-type]
    xs = x.clone().requires_grad_(True)
    sparse = m(xs)
    gs = torch.autograd.grad(sparse.pow(2).sum(), [xs, m.proj.weight])
    xd = x.clone().requires_grad_(True)
    f = m.features(xd)
    w, rho = m.weights(f["addr"])
    dense = m.out(cell_scan(f["q"], f["k"], f["v"], f["beta"], f["gamma"], w, rho, m.decay).flatten(-2))
    gd = torch.autograd.grad(dense.pow(2).sum(), [xd, m.proj.weight])
    err = (sparse - dense).abs().max().item()
    gerr = max((a - b).abs().max().item() for a, b in zip(gs, gd))
    return check(f"sparse == dense ({kind}, {decay})", err < 1e-5 and gerr < 1e-4,
                 f"value {err:.1e}, grad {gerr:.1e}")


if __name__ == "__main__":
    raise SystemExit(main())

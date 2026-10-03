# Hydra: RWKV × Spotlight unified fast-weight memory

Research log, snapshot of 2026-10-03. Code and raw results live in [`hydra/`](hydra/).
It's a copy of `~/Code/personal/spotlight-mem` from the machine that was wiped. The venv and caches aren't included.

## Restore

```bash
cd hydra
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python numpy
.venv/bin/python tests/test_reduction.py      # expect: all 12 checks pass
./guard.sh && ./run_chase.sh                  # guard exits 2 if a train.py is already running
```

Original environment: Python 3.12, torch 2.14.1+cpu, numpy 2.5.3, i5-1135G7 in WSL (8 threads, 47 GB RAM, no GPU).

## Thesis

A Spotlight cell and an RWKV-7/GDN state are the same object: a d_k×d_v fast-weight matrix updated by the delta rule.
They differ in only two ways:

| | RWKV / GDN | Spotlight |
|---|---|---|
| Where a write goes | the one state matrix | 3×3 patch of lattice cells at a learned 2D address |
| When decay applies | every token | only when a token touches the cell ("decay-on-touch") |
| Capacity | fixed, O(d_k·d_v) | grows, O(N·d_k·d_v) |
| Per-token cost | O(d_k·d_v) | O(9·d_k·d_v), independent of N |

**The unifying knob is a learned per-head address scale σ.** σ = 0 collapses every token onto one cell (GDN/RWKV).
σ > 0 gives growing routed memory. Three tiers: slow weights θ (intelligence, fixed at inference) → RWKV state
(registers/working memory, decays every token) → Spotlight cells (RAM/disk, long-term, decays on touch).

Target is a workshop-scale paper, not SOTA. Prior work to position against: fast weights (Schmidhuber 1992,
Ba 2016, Schlag 2021), DeltaNet/GDN/GDN-2/RWKV-7, TTT/Titans, NTM/DNC, Kanerva SDM, product-key memory,
Mixture-of-Memories, log-linear attention, and Percepta's Spotlight posts (Oct 2026, HF `percepta-ai/spotlight-vm`).

### Per-head update (as implemented)

- Separate short causal Conv1d on q, k, v and the address channels (GDN-style). SiLU on q/k/v, tanh on the address.
  L2-normalised q and k.
- Write address from the conv'd k path, read address from the conv'd q path, each through its own linear layer,
  scaled by σ, giving 2D coordinates on a G×G lattice (default G = 9).
- Bump weights over the 3×3 neighbourhood:
  - `blog`: cos²(πt/3), |t| < 3/2 (Percepta LM post; 3 nonzero cells per axis, smooth edge)
  - `vm`: cos²(πt/2) (hand-built VM special case)
- Cell update: `S_c ← γ_c·S_c + β·w_c·k(v − kᵀS_c)` on the 9 touched cells only. Readout is the bump-weighted sum of
  `qᵀS_c` over the 9 read cells.
- Sparse 9-cell gather/scatter scan matches the dense 81-cell scan to 1e-8 (values) and 1e-6 (gradients).
  ~1.4 s/step for Spotlight vs ~0.3 s/step for GDN on CPU.

## Results so far

Tiny models throughout: d_model 64, 2 layers, 2 heads, d_head 16, 64 symbols, batch 32, lr 3e-3, 3,000 steps,
**one seed** (seed 0), CPU.

### Math checks: `tests/test_reduction.py`, all 12 pass

1. **VM bump at σ = 0 equals GDN exactly** (max diff ~1e-7).
2. **Blog bump at σ = 0 does not equal GDN.** You get a fixed mix of 9 GDN states with different effective
   update rates (diff 0.185). An earlier claim that it reduces to GDN was wrong.
3. **σ = 0 is a stationary point for any reflection-symmetric bump.** The router gradient is exactly zero there.
   Gradients appear around σ ≈ 0.047, so initialise σ > 0.05 (we use 0.5).
4. Sparse scan = dense scan (values and gradients).
5. Causality: changing tokens after t leaves outputs ≤ t unchanged for all three mixers (diff 0.0).

### Experiment 1: single-hop recall (MQAR-style pointer chase, depth 1)

Trained on n ∈ [2,16] pairs, evaluated up to 4× that range. Files: `runs/{attn,gdn,spotlight}_d1.*`.

| Pairs at test | 8 | 16 | 32 | 48 | 64 |
|---|---|---|---|---|---|
| Attention | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **Spotlight** | **1.000** | **1.000** | **0.9995** | **0.997** | **0.991** |
| GDN | 0.999 | 0.961 | 0.695 | 0.454 | 0.321 |

Same mixer code and parameter count: adding the learned 2D address takes GDN from 0.32 to 0.99 at 4× the training
range. The layer-0 router spread out and used ~34 of 81 cells per sequence, with 88% of reads landing within half a
cell of the matching write. Layer 1 stayed mostly unused.
**This is the paper's first solid result: learned addressing fixes recall for RWKV-style memory.**

### Experiment 2: two-hop chaining (pointer chase, depth 2)

Comma format, mixed depths {1,2} during training, evaluated at depth 2. Files: `runs/chase_{gdn,spotlight}.*`,
`runs/probe_attn_*`.

| Pairs at test | 8 | 16 | 32 |
|---|---|---|---|
| Attention | 1.000 | 1.000 | 1.000 |
| Spotlight | 0.202 | 0.092 | 0.042 |
| GDN | 0.203 | 0.094 | 0.041 |

Attention cracked depth 2 around step 1,300. Spotlight and GDN both stayed at chance. GDN briefly hit 0.64 on a
training batch at step 1600 but didn't generalise. More storage didn't make chaining any easier.
**The Reddit critique (NTM/DNC-style optimisation fragility) holds for chaining so far.**
"Recurrent models just need more steps" isn't ruled out yet.

### Diagnostics (GDN, depth 1, small problem: 32 symbols, n ≤ 8)

`runs/diag_*`: GDN learns small problems almost perfectly (0.99 at n ≤ 8, 0.95 at 2× range) across lr 1e-3/3e-3/1e-2
and d_head 32. The earlier GDN failure on 128 symbols with n ≤ 32 came from problem size and curriculum, not from
the architecture. That fits with Percepta needing a 7-stage MQAR curriculum.

## Bugs found and fixed (worth a paragraph in the paper)

1. **Task ambiguity:** each symbol appeared as both key and value, so "token after x" and "token before x" were
   indistinguishable. Attention was stuck at exactly 0.50. Fixed by giving keys and values separate roles
   (split vocab, later the comma format).
2. **GDN layout:** a shared residual-stream conv blended neighbouring tokens into q/k/v, which broke write/read key
   alignment at d_head 16. Fixed with per-projection short convs and separate write/read address heads.
3. **Curriculum:** sampling up to 32 pairs from step 0 drowned GDN's learning signal. Start small.
4. **Chaining learnability:** even attention needed (a) one token id per symbol with a comma marking keys and
   (b) mixed one- and two-hop training. Pure depth-2 training fails.
5. Ops: `pgrep -f` matches its own command line. Use `guard.sh` (anchored on argv[0] being python).

## Next steps (in order)

1. **Long depth-2 run, the decisive one:** GDN and Spotlight at 10,000 steps with the same config as `run_chase.sh`
   (`--steps 10000`), about 2.5 h on CPU overnight. This settles "slower to learn" vs "can't learn".
2. **3 seeds** for every table above (seeds 0,1,2). Report mean ± std.
3. **Dead-gradient test:** 33×33 grid (`--grid 33`). On 9×9 every key/query pair sits within bump reach, so the
   reachability metric reads 1.0. A bigger grid is needed to test the "gradients can't reach far cells" claim.
   Track the reachable fraction and dead-cell count over training.
4. **If chaining still fails:** try (a) a hybrid with 1 attention layer or a plain GDN layer stacked before Spotlight
   (does the second hop need a dense working-memory layer?), (b) a 3-layer Spotlight model, (c) a curriculum over
   depth, (d) σ schedule / larger bump support early in training.
5. **Remaining paper experiments:** retention curves (decay-on-touch vs decay-every, `--decay every`; half-life
   should grow linearly with N), per-head learned σ allocation in a mixed model, collision tolerance
   (keys ≫ cells), then a ~100M LM on 2–5B tokens comparing RWKV / Spotlight / hybrid. That last one needs a rented
   GPU and a Triton kernel; budget ~$100–600.

## Larger plan kept for later (deployment track, parked)

- Base: **RWKV-7 G1 0.4B**, swap 1 in 4 layers for Spotlight (Qwen3.5-style 3:1 layout), distil at 8K and then
  128K context. Not RWKV-8: ROSA overlaps with Spotlight and would confound the result. Keep ROSA as a baseline.
- Don't start at 7B. Kernel and router work is the same 2–6 weeks at any size, while compute costs ~17× more
  ($300–4K per run vs $30–200) and iteration takes days instead of hours. Order: 0.1B → 0.4B → 1.5B/2.9B → 7.2B.
- Estimate: 2–4 months solo, $1.5–5K compute. The Triton training kernel is the critical path.
- Local inference math (i5-1135G7, 40.7 GB/s measured, ~25 GB/s practical): ~70–110 tok/s at 0.4B Q4, ~28 at 1.5B,
  ~6 at 7.2B. Speed stays flat as memory grows.
- **RAM + NVMe cell tiering works** if cells only decay or change when touched (cold cells are immutable on disk).
  Measured on WSL ext4: 70K IOPS / 1.1 GB/s at 16K QD32. Packed 32×32 fp16 cells keep ~70 tok/s even with 100%
  misses.
- Gaps in the Percepta post to measure ourselves: cell size, cells allocated per token, memory per token, training
  throughput, multi-hop and multi-needle results.

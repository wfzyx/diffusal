# Roofline Analysis: Does Ternary Leave ALU Headroom for Block-Parallel Drafting?

**Status:** exploratory, analytic (no pre-registration). Not part of the Track A core.
**Script:** [`profile_roofline.py`](./profile_roofline.py)
**Upstream intensities:** [`profile_hardware.py`](./profile_hardware.py)

---

## 1. Why this analysis exists

`profile_hardware.py` reports a **1.91x DRAM traffic cut** for block-parallel
generation over autoregressive generation. That number is identical in FP32, INT4,
and ternary — because the reported ratio

```
(AR bytes) / (block-diffusion bytes) = (64 passes * active_params * bpp)
                                       -------------------------------
                                       (12 passes * total_params  * bpp)
```

cancels `bytes_per_param` exactly. The existing profiler is therefore **blind to
precision** and cannot answer whether block-parallel drafting still pays once the
weights are ternary.

The binding question is the roofline ridge point:

$$
\text{ridge} = \frac{\text{peak compute (FLOP/s)}}{\text{peak bandwidth (byte/s)}}
\qquad
\begin{cases}
I < \text{ridge} & \text{memory bound} \rightarrow \text{parallel drafting pays} \\
I \geq \text{ridge} & \text{compute bound} \rightarrow \text{parallel drafting is dead weight}
\end{cases}
$$

Ternary and speculative/block-parallel decoding attack **the same bottleneck**.
Ternary cuts bytes streamed per parameter; block-parallel decoding amortizes
weight streaming across more tokens. Composing them does not compose their wins.

## 2. Measured arithmetic intensities

From `profile_hardware.py` (19.6M total / 7.0M active params, top-2 of 8 experts,
64 tokens, block size 32, 6 denoise steps per block):

| precision | paradigm | passes | DRAM | GFLOPs | intensity |
|---|---|---:|---:|---:|---:|
| FP32 (4.00 B/param) | autoregressive | 64 | 1716.1 MB | 0.90 | 0.50 FLOP/B |
| FP32 | block-parallel | 12 | 897.8 MB | 5.40 | **5.73 FLOP/B** |
| INT4 (0.50 B/param) | autoregressive | 64 | 214.5 MB | 0.90 | 4.00 FLOP/B |
| INT4 | block-parallel | 12 | 112.2 MB | 5.40 | **45.88 FLOP/B** |
| Ternary (0.25 B/param) | autoregressive | 64 | 107.3 MB | 0.90 | 8.00 FLOP/B |
| Ternary | block-parallel | 12 | 56.1 MB | 5.40 | **91.75 FLOP/B** |

Block-parallel decoding raises arithmetic intensity by ~11.5x (1.91x fewer bytes
for 6.0x more arithmetic). Ternary raises it by a further 16x over FP32. Stacked,
intensity climbs from 0.50 to 91.75 FLOP/B — a **184x** move along the roofline.

## 3. Crossover by target

| target | ridge (FLOP/B) | FP32 | INT4 | Ternary |
|---|---:|:---:|:---:|:---:|
| RTX 2080 Super, fp32 CUDA core | 22.48 | PAYS | DEAD | DEAD |
| RTX 2080 Super, fp16 tensor core | 44.96 | PAYS | DEAD | DEAD |
| RTX 2080 Super, int8 tensor core | 179.84 | PAYS | PAYS | PAYS |
| Laptop CPU, AVX2 + DDR4 | 3.33 | DEAD | DEAD | DEAD |
| Apple M2 Max, unified memory | 34.00 | PAYS | DEAD | DEAD |
| H100 SXM, fp16 tensor core | 295.22 | PAYS | PAYS | PAYS |

`PAYS` = block-parallel drafting is still memory bound, so it amortizes weight
streaming. `DEAD` = ternary already freed the bus and the 6.0x arithmetic overhead
now binds.

## 4. Findings

1. **The kernel decides, not the bit width.** On an RTX 2080 Super the verdict flips
   entirely on which compute path the ternary GEMM lands in. Dequantize-to-fp16
   (ridge 44.96) puts ternary block-parallel decoding **2.0x over the ridge — dead**.
   A packed integer path using `dp4a`/INT8 tensor cores (ridge 179.84) leaves
   **2.0x headroom — alive**. Same weights, same model, opposite conclusion.

2. **INT4 is already past the crossover on consumer GPUs.** The block-parallel win
   dies at INT4 on the 2080 Super fp16 path (45.88 vs ridge 44.96) — ternary is not
   required to trigger the inversion. Any naive dequantizing low-bit kernel kills it.

3. **The bandwidth-starved-edge thesis inverts under ternary.** Laptop CPU
   (ridge 3.33) was the motivating target for block-parallel decoding in
   [`ROADMAP.md`](./ROADMAP.md) Phase 4. It is compute bound in **every** regime,
   including FP32. Once weights are ternary, block-parallel decoding runs 27.5x over
   ridge there. A ternary CPU deployment should decode autoregressively (8.00 FLOP/B,
   2.4x over ridge — marginal but far closer) and spend the saved bandwidth on more
   parameters, not on speculation.

4. **Server GPUs keep both wins.** H100 (ridge 295.22) has slack for ternary
   block-parallel decoding at 3.2x headroom. This matches the Uno paper's regime:
   their speedups hold at large batch sizes on H200 class hardware, where the
   arithmetic budget is not the constraint.

## 5. Consequence for the Uno / ternary composite

Uno's $\Psi$-Spec sampler does not change the weight representation, and ternary
does not change the sampling contract. They are independent axes:

```
sparse upcycle  -> more parameters, same active FLOPs    (capacity)
native ternary  -> add/sub GEMM, 1.58 bit/param          (footprint + bandwidth)
Uno Psi-Spec    -> B tokens per 2 forward passes         (latency, iff ALU is idle)
```

The third axis is conditional on the first two. Combining them on an RTX 2080 Super
requires a **packed integer kernel**; without one, the ternary model is already
compute bound and Uno's drafting plus tree verification only adds arithmetic.

Note also that Uno's per-request-optimal tree sampler `(B, K, V) = (16, 32, 32)`
costs considerably more arithmetic than the 6.0x modeled here for a linear block
sampler, which pushes the crossover further left.

## 6. Limitations

- Analytic roofline, not wall clock. No kernel launch overhead, no cache hierarchy,
  no occupancy, no KV-cache traffic (which grows with context and shifts AR toward
  memory bound at long sequence lengths).
- Peak vendor FLOP/s and bandwidth figures; achieved fractions are typically
  60-80% and differ between the compute and memory ceilings.
- Intensities come from the 19.6M-parameter toy MoE in `profile_hardware.py`.
  Intensity is roughly scale invariant for weight streaming at batch size 1, but
  expert-routing coverage assumptions (all 8 experts touched per 32-token block)
  are specific to top-2 of 8.
- Ternary is modeled at 2 bits/param packed (0.25 B/param), consistent with
  `QUANTIZATION-COVERAGE.md`, not the information-theoretic 1.58 bits.

## 7. Next step

Before any training run, settle finding 1 empirically: benchmark a packed 2-bit
expert GEMM against a dequantize-to-fp16 GEMM on the target device and measure
achieved FLOP/s. That single number determines whether the Uno layer belongs in
the architecture at all on consumer hardware.

## Reproduce

```bash
python experiments/part2-moe-diffusion/profile_hardware.py   # intensities
python experiments/part2-moe-diffusion/profile_roofline.py   # ridge points
```

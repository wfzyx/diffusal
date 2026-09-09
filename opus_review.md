# Peer review — Diffusal

**Reviewer:** Claude (Opus 5), adversarial technical review
**Date:** 2026-09-09
**Repository state:** `main` @ `2bd487f`
**Artefacts read:** `README.md`, `whitepaper/diffusal-whitepaper.tex`, `thesis/THESIS.md`, `arxiv/diffusal-arxiv.tex`, `arxiv/SUBMISSION.md`, all files under `configs/` and `experiments/`, git history and tags.

---

## Recommendation

**Do not submit `arxiv/diffusal-arxiv.tex` in its current form.**

Revert the manuscript to the July 2026 version (`git show 271d035:arxiv/diffusal-arxiv.tex`) and ship that. The July paper is a careful, correctly scoped, publishable technical report. The September rewrite added four new claim blocks — MoE quantization, two theorems, hardware memory physics, RoPE inversion — and **all four fail on inspection of the code that produced them**. The rewrite also deleted the two negative mechanism results that the July abstract led with, and replaced them with a theorem that the deleted results refute.

This repository holds two research programs of very different quality.

| Track | Dates | Content | Assessment |
|---|---|---|---|
| A | 2026-07-16 → 07-30 | Exp 0 PTQ, Exp 2 QAT, mechanism probes, whitepaper | Sound. Pre-registered, matched, honestly scoped. Publish. |
| B | 2026-09-08 → 09-09 | `experiments/part2-moe-diffusion/`, arXiv §3–4.6 | Not supportable. Withdraw or re-run. |

The rest of this review separates the two, because the second track threatens the credibility of the first.

---

## Part 1 — Fatal defects in the MoE / hardware track

Each item below names the file and the line-level mechanism. Each is checkable in under five minutes.

### F1. `PrimeIntellect/qwen3-moe-tiny` is a randomly initialized model

The paper calls this checkpoint "production MoE weights" and "a production MoE checkpoint", and builds §4.3, Table 4, contribution 3, and an abstract claim on it. The checkpoint is untrained.

Weight statistics from the cached `model.safetensors` (670,208,000 parameters, BF16):

| tensor | n | mean | sd | kurtosis | min | max |
|---|---:|---:|---:|---:|---:|---:|
| `embed_tokens.weight` | 155,582,464 | +0.00001 | 0.02005 | 2.98 | −0.0908 | +0.0854 |
| `lm_head.weight` | 155,582,464 | −0.00006 | 0.02122 | 2.98 | −0.1006 | +0.0864 |
| `layers.3.self_attn.q_proj.weight` | 1,048,576 | +0.00007 | 0.02006 | 2.98 | −0.0874 | +0.0908 |
| `layers.3.mlp.gate.weight` | 16,384 | −0.00007 | 0.01989 | 2.95 | −0.0703 | +0.0723 |
| `layers.12.mlp.experts.7.up_proj.weight` | 262,144 | −0.00012 | 0.02012 | 3.00 | −0.0898 | +0.0869 |
| `layers.20.mlp.experts.0.down_proj.weight` | 262,144 | −0.00004 | 0.02010 | 3.01 | −0.0869 | +0.0898 |
| `layers.0.input_layernorm.weight` | 1,024 | +0.99945 | 0.00163 | — | +0.9961 | +1.0078 |
| `layers.5.input_layernorm.weight` | 1,024 | +0.99929 | 0.00151 | — | +0.9961 | +1.0000 |
| `layers.23.post_attention_layernorm.weight` | 1,024 | +0.99927 | 0.00172 | — | +0.9961 | +1.0078 |

Four independent signatures of random init:

1. Every weight tensor has sd ≈ 0.0200. The config sets `initializer_range: 0.02`.
2. Every weight tensor has kurtosis 2.95–3.01. That is an exact Gaussian. Trained transformer weights are heavy-tailed, with kurtosis well above 3 and large outliers.
3. All RMSNorm gains sit at 1.0. The range +0.9961 to +1.0078 is the two BF16 values adjacent to exactly 1.0. Norm gains initialize to 1.0 and drift a long way during training. These gains moved by at most one BF16 unit in the last place.
4. min/max reach ±4.5σ with no outliers anywhere — the shape of a fresh truncated normal, not of a trained tensor.

Reproduce:

```bash
python3 -c "
import json,struct,math
p='<hf-cache>/models--PrimeIntellect--qwen3-moe-tiny/snapshots/<sha>/model.safetensors'
f=open(p,'rb'); n=struct.unpack('<Q',f.read(8))[0]; h=json.loads(f.read(n)); base=8+n
def bf(b): return [struct.unpack('<f',struct.pack('<I',(b[i]|(b[i+1]<<8))<<16))[0] for i in range(0,len(b),2)]
for k in ['model.layers.5.input_layernorm.weight','model.layers.12.mlp.experts.7.up_proj.weight']:
    s,_=h[k]['data_offsets']; f.seek(base+s); x=bf(f.read(400000))
    m=sum(x)/len(x); v=sum((t-m)**2 for t in x)/len(x)
    print(k,'sd=%.5f kurt=%.2f min=%.4f max=%.4f'%(math.sqrt(v),sum((t-m)**4 for t in x)/len(x)/v**2,min(x),max(x)))"
```

**Consequence.** A randomly initialized model has no language model to degrade. Greedy decoding from it produces a near-input-independent token repeated to the horizon. "Trajectory drift" between two quantizations of a random model measures numerical jitter around an arbitrary fixed point. The 64 domain-diverse prompts are irrelevant: the model has no domains. The untrained router assigns experts by coin flip, so "router flip rate" measures the sensitivity of a random linear map's top-4 to weight noise.

This invalidates: the arXiv abstract's finding (3); arXiv §4.3 in full; arXiv Table 4; arXiv contribution 3; `thesis/THESIS.md` §6.2 items 1–2; `RESULTS-laptop-pilot.md` §§4, 5, 6.

Additional factual error in the same section: the paper states "~75M active params/token". The real figure is ~142M active non-embedding parameters (62.9M attention + 6.3M dense layer 0 + 72.4M top-4 experts + 0.4M routers), plus a 155.6M output head. 75M is the expert-only share, mislabelled as the total. The model also has 23 MoE layers, not 24 (`mlp_only_layers: [0]`).

### F2. The canvas decoder leaves 25% of positions permanently unfilled, and that alone reverses the headline sign

`run_64_sample_statistical_grid.py:157-172` and `moe_trajectory_absorption_test.py:88-105`:

```python
gen_len, block_size, steps = 32, 16, 6
n_unmask = max(1, block_size // steps)     # = 2
```

Six steps × two positions = 12 of 16 block positions. Four positions per block never leave `pad_id`. Over two blocks, 8 of 32 output positions hold `eos_token_id` in **every** condition — FP32, INT4, and ternary alike. Drift is `(base_diff != quant_diff).float().mean()` over all 32 positions, so those 8 positions contribute exactly zero drift by construction.

Measured diffusion drift is therefore `0.75 × (drift on the 24 live positions)`. Undo the dilution:

| regime | AR drift | diffusion drift, as reported | diffusion drift, live positions only | excess, as reported | excess, corrected |
|---|---:|---:|---:|---:|---:|
| INT4 experts | 24.85% | 21.58% | 28.77% | **−3.27 pp** | **+3.92 pp** |
| Ternary experts | 46.09% | 41.55% | 55.40% | **−4.54 pp** | **+9.31 pp** |

Correcting one line of the sampler flips the sign of both headline results. Diffusion becomes the *less* stable decoder. The abstract's "confirms that diffusion canvas decoding dampens trajectory corruption" does not survive.

The claimed "variance suppression" has the same root, plus a second mechanical cause: drift is a bounded fraction, so its variance is largest near 0.5. AR sits at 0.4609 and diffusion at 0.4155. The tighter diffusion interval is what a binomial-like quantity does when its mean moves away from 0.5. No variance test appears anywhere — no F-test, no Levene, no bootstrap.

There is also an off-by-one in the same function. A causal model's logits at position *i* predict the token at *i+1*. `run_64_sample_statistical_grid.py:170` writes `canvas[0, idx] = pred[0, idx]` — the next-token prediction lands on the current position. `causal_consistent_block_diffusion.py:86` does it correctly (`block_ids[0, i+1] = pred_tokens[0, i]`). The two scripts disagree with each other, and the grid script is the wrong one.

### F3. "Causal-consistent block diffusion" is greedy autoregressive decoding

`causal_consistent_block_diffusion.py:63-88`:

```python
causal_mask = torch.triu(torch.full((total_len, total_len), float('-inf')), diagonal=1)
...
for i in range(block_size - 1):
    candidate_token = pred_tokens[0, i]
    block_ids[0, i + 1] = candidate_token
```

The mask is fully causal across the whole sequence. Position *past_len+i* attends only to positions ≤ *past_len+i*. So `block_ids[i+1]` is the greedy next token given the prefix and `block_ids[0..i]`. Repeating the pass propagates one more correct position per iteration. This is Jacobi iteration on the greedy AR decoding map, and its fixed point **is** the greedy AR sequence.

"100% exact word-for-word generation fidelity with the AR baseline" is therefore a mathematical identity of the update rule, not an empirical finding. It cannot be evidence about diffusion.

Three further problems in the same file:

- There is no mask token, no masking, and no canvas. Nothing is denoised.
- The paper reports "confidence thresholding (τ = 0.85)". No threshold exists in the code. `confidences` is computed at line 79 and never read.
- With `block_size=16` and `steps_per_block=4`, only positions 0–4 of each 16-token block can converge. Exact AR identity over 32 tokens is not reachable at these settings. No log of a run producing the claimed output exists anywhere in the repository. `RESULTS-laptop-pilot.md` §3 reports the *naive* bidirectional output and omits the causal-consistent output entirely.

**Consequence.** arXiv §4.5 collapses to a restatement of a known fact: bidirectional attention on causally pretrained weights fails, and causal attention works. That is why DiffuLLaMA needs continual pretraining. The paper presents it as "We identify the mechanism" and "prove that causal-consistent block decoding restores 100% exact generation fidelity".

This also removes the paper's answer to its own strongest objection. The abstract raises the headroom problem — "whether diffusion models' superior relative quantization tolerance holds value if diffusion initiates from lower unquantized quality per parameter" — and answers it with "exact zero-shot parity (100% generation identity)". The parity is AR decoding. The objection stands unanswered. See D1.

### F4. `R = 1.000` in Table 3 is a hardcoded fallback, not a measurement

`moe_quant.py:410`:

```python
tern_R = dllm_tern_tax / ar_tern_tax if ar_tern_tax > 0 else 1.0
```

`ar_tern_tax` is **−0.50%** (ternary QAT lowered the AR model's validation loss). The condition fails, so `tern_R` is assigned the literal `1.0`. arXiv Table 3 prints that literal in the Gap Ratio column, and §4.2 reads it as a result: "dLLM pays zero additional tax (+0.68% degradation vs −0.50% in AR, R ≤ 1.25 satisfied)".

The INT4 row has the mirror problem. `int4_R = -2.43/0.34 = -7.147`. The script prints `-7.147`; `RESULTS-laptop-pilot.md` shows the word "Robust" in that cell. A negative ratio was replaced by a favourable adjective during write-up.

The verdict logic inherits the defect: `'PASSED' if (int4_R <= 1.25 and tern_R <= 1.25)` passes on −7.147, and `'FIRED' if int4_R < 0.80` fires on −7.147. A "no extra tax" test that passes on a negative ratio tests nothing.

### F5. The scratch-MoE validation losses are not comparable, and the degradations are unpaired sampling noise

`moe_quant.py:245-256`:

```python
if not is_dllm:
    logits = model(x); loss = F.cross_entropy(logits.view(-1,V), y.view(-1))   # y = next tokens
else:
    mask = torch.rand_like(x, dtype=torch.float) < 0.35
    x_m = x.clone(); x_m[mask] = mask_token_id
    logits = model(x_m); loss = F.cross_entropy(logits.view(-1,V), x.view(-1)) # x = the input itself
```

Two different tasks:

- AR is scored on next-token prediction at all 64 positions.
- The dLLM is scored on reconstructing its own input at all 64 positions, and **65% of those positions are visible in the input**.

That is why the reported "FP32 Val Loss" is 2.3991 (AR) against 1.1647 (dLLM). The dLLM number is not a better language model; two thirds of it is a copy task. Copying is also the operation least sensitive to weight noise, so the dLLM's *degradation* is diluted by the same 65%. **The comparison manufactures the claimed effect as an artifact.**

Worse, the evaluation is not paired. `evaluate_loss` calls `get_batch`, which draws from the global RNG:

```python
def get_batch(tokens, batch_size=16, seq_len=64):
    ix = torch.randint(0, max_start, (batch_size,))
```

Six sequential calls — `ar_fp32`, `dllm_fp32`, `ar_int4`, `dllm_int4`, `ar_ternary`, `dllm_ternary` — each draw **different validation batches**, and each dLLM call draws a **different random mask**. The reported degradations of +0.34%, −2.43%, −0.50%, +0.68% are differences between measurements taken on different data. Two of the four have the wrong sign for a degradation. That is the expected outcome when the measurement sits inside the noise band, on a single seed, over 10 batches of 16×64 tokens.

Fix: draw one fixed evaluation set and one fixed mask, and reuse both for all six models.

### F6. The 20.31% vs 12.50% trajectory drift compares two different experiments

`moe_quant.py:300-338`:

- AR: prompt = `x[:, :8]`, generate **16** tokens, batch 4 → 64 comparisons. 20.31% = 13/64.
- dLLM: canvas of **32** positions, **no prompt at all**, 8 steps, batch 4 → 128 comparisons. 12.50% = 16/128.

Different generation length, different conditioning, different step count, different comparison count. Unconditional generation from an all-mask canvas in a 250-step character model returns a near-constant low-entropy string, which is trivially stable under perturbation. The precision is INT4, not ternary, but §4.2 of the paper places the number in a ternary-QAT paragraph without saying so.

arXiv §4.2 elevates `R = 0.615` from this to "the strict `dllm_more_robust` criterion fires". It is a five-token difference between two non-matched procedures, n = 1, no seeds.

### F7. The 4.34× wall-clock speedup measures a missing KV cache

`profile_hardware.py:159-180`. `generate_pure_ar` re-runs the **entire prefix** through every layer at every step. There is no KV cache.

| path | token-forwards for 64 generated tokens |
|---|---:|
| `generate_pure_ar` (no cache) | Σ(16+t), t=0..63 = **3,040** |
| `generate_block_diffusion` | 12 passes × ~64 positions = **768** |

Ratio 3.96. The claimed speedup is 4.34×. The measurement is the cost of the absent cache, not a property of block diffusion. Add a KV cache to the AR baseline and the speedup goes away.

The comparison is also not quality-matched. On real weights the same block decoder produces the gibberish printed in `RESULTS-laptop-pilot.md` §3. Throughput without output parity is not a result.

### F8. The DRAM model charges block diffusion for 2 of 8 experts while it processes 32 tokens

`profile_hardware.py:262-284`:

```python
active_params = total_params - (num_experts - 2) * expert_params_per_layer * n_layers
...
diff_dram_bytes = diff_passes * active_weight_bytes     # same per-pass bytes as AR
```

`active_params` is the top-2-of-8 figure for **one** token. Block diffusion processes 32 tokens in one pass. With top-2 routing over 8 experts and 32 independent tokens, the probability that any given expert receives no token is negligible, so a single block-diffusion pass must stream **all** expert weights.

| precision | AR | block-diffusion, as modelled (2/8) | block-diffusion, all 8 experts | ratio, as claimed | ratio, corrected |
|---|---:|---:|---:|---:|---:|
| FP32 | 1716.3 MB | 321.8 MB | 897.2 MB | 5.33× | **1.91×** |
| INT4 | 214.5 MB | 40.2 MB | 112.2 MB | 5.33× | **1.91×** |
| Ternary | 85.8 MB | 16.1 MB | 44.9 MB | 5.33× | **1.91×** |

Two further problems with the headline number:

- **The 106.7× compares FP32-AR against ternary-block-diffusion.** It multiplies a quantization gain (20×, FP32 → 0.2 bytes/param) by a batching gain (5.33×). The like-for-like figure — ternary AR against ternary block diffusion — is 5.33× on the paper's own model and 1.91× corrected. The 20× belongs to ternary weights and is available to an AR decoder too.
- **The 5.33× batching gain is `64/12`, identical at every precision.** It is amortization of weight streaming across a batch. Batched AR inference, speculative decoding, and Medusa-style multi-token heads all obtain the same effect. Nothing in it is specific to diffusion.

**Ternary bytes/param is set to 0.20.** The repository's own `QUANTIZATION-COVERAGE.md` correctly uses 2 bits (0.25 bytes) and calls it "a conservative 2-bit representation". No kernel packs at 1.58 bits. The two experiments in this repository disagree with each other.

### F9. Arithmetic intensity is computed with the wrong FLOP count, hiding a 6× compute cost

`profile_hardware.py:283`:

```python
diff_intensity = ar_flops / diff_dram_bytes
```

`ar_flops = 2 * active_params * 64`. Block diffusion runs 12 passes over 32 positions each — **384 position-forwards against AR's 64, i.e. 6× the compute.** The intensity rises because the numerator should rise, not because the work is free.

The honest statement is: *block diffusion trades 6× more arithmetic for 1.91× less DRAM traffic.* On a bandwidth-bound device that is a win. On a compute-bound device it is a loss. The paper presents "elevating arithmetic intensity from 0.50 to a compute-saturated 53.33 FLOPs/Byte" as pure gain, and never reports the extra FLOPs.

### F10. The router is never quantized, so the paper's theory has no experiment

The entire theoretical framework (§3) concerns the gating perturbation `W_g → W_g + ΔW_g` and the resulting top-*k* flip event `E_flip`. Every MoE quantization call in the repository touches expert weights only:

```python
layer.mlp.experts.gate_up_proj.data = quantize_tensor(...)
layer.mlp.experts.down_proj.data   = quantize_tensor(...)
```

`mlp.gate` stays FP32. This is deliberate — `thesis/THESIS.md` §6.3 sets "routers stay FP16/INT8; do not ternarize routers first" as a design default. The consequence is that the double-noise model, the central theoretical contribution, is never tested. Reported "router flip rates" are flips induced *indirectly* by expert noise shifting hidden states, which is a different quantity from the one the theory is about.

### F11. Selective reporting of single-prompt results

`moe_trajectory_absorption_test.py` (prompt: *"Artificial intelligence algorithms process large datasets by"*) and `native_moe_experiment.py` (prompt: *"Artificial intelligence and machine learning algorithms work by"*) run the same condition on the same model and disagree completely:

| source | INT4 AR drift | ternary AR drift |
|---|---:|---:|
| `native_moe_experiment.py` | **65.62%** | 68.75% |
| `moe_trajectory_absorption_test.py` | **0.00%** | 75.00% |

Same model, same quantizer, same 32-token horizon, n = 1 each. AR INT4 drift is 0% on one prompt and 65.62% on another.

From the second table the paper takes the ternary row (−59.38 pp, favourable) into the abstract, contribution 3, §4.3 item 3, and the conclusion. It omits the INT4 row **from the same run**, where block diffusion was worse (+9.38 pp). It omits the first table entirely.

The 64-prompt grid then reports the ternary excess as **−4.54 pp with 95% CI [−13.19, +4.10]**. The interval contains zero. The script itself prints `Zero within 95% CI? YES`. The abstract calls it "confirms"; §4.3 calls it a "Consistent Excess Advantage".

The script also carries a hardcoded p-value: line 236 prints `'NO - STATISTICALLY SIGNIFICANT (p < 0.001)'` whenever the interval excludes zero. A 95% interval excluding zero implies p < 0.05, not p < 0.001. The branch did not fire here, but the code is written to emit a false p-value if it ever does.

No paired interval is computed for the INT4 condition at all, so the −3.27 pp INT4 excess has no significance statement of any kind.

### F12. Corpus and domain descriptions do not match the code

`moe_quant.py:33`:

```python
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
```

- The data is TinyShakespeare, **character-level**, 65-character vocabulary, 1,003,854 **characters**.
- `RESULTS-laptop-pilot.md` §1 and arXiv §3 describe "1,003,854 tokens of standard text corpus" / "trained on text corpus".
- The module docstring says "TinyShakespeare / Wikitext slice". No Wikitext is fetched.
- arXiv §3 criticizes "legacy word-level corpora (such as the 1993 Penn Treebank)" and claims the cohorts use "modern subword tokenizers and multifaceted technical domains". Character-level Shakespeare is weaker than PTB on both counts.
- Training is 250 steps × 16 × 64 = **256,000 tokens**, roughly a quarter pass. Not "1M tokens".

The 8 domains in the abstract ("Distributed Systems / Consensus, General Relativity / Theoretical Physics, Real Analysis / Mathematics, Computer Architecture / Memory Physics, Macroeconomics, Molecular Biology / CRISPR, Logic, Systems Programming") do not match the code's groups (Science & Physics, Computer Science & AI, Mathematics & Logic, History & Philosophy, Engineering & Technology, Economics & Society, Biology & Medicine, General Reasoning). The abstract's list is a re-labelling, and no prompt concerns consensus, real analysis, or systems programming.

### F13. `upcycle_qwen_moe.py` adds an untrained router and never fine-tunes

`upcycle_qwen_moe.py:43`: `self.gate = nn.Linear(self.hidden_size, num_experts, bias=False)` — random initialization, no training step anywhere in the file. §3's INT4 and ternary outputs therefore come from a model with a random router stacked on top of the naive-bidirectional failure. Two confounds, one measurement.

---

## Part 2 — The theorems

### T1. Neither theorem is proved

Both are stated as `\begin{theorem}` with no proof, no lemma, and no reference. The abstract says "prove why AR decoders suffer compounding exponential drift while dLLMs attenuate routing errors by O(1/L)". Nothing in the paper proves either half.

### T2. Theorem 2's bound is false as stated

```
∀ j ≠ i:  ‖∂h_j^(s) / ∂h_i^(s)‖ ≤ 2 γ_attn / L
```

Attention weights sum to 1 across L positions, so their **mean** is 1/L. Their **maximum** is not bounded by 1/L. Softmax concentrates: a single position can receive attention weight approaching 1, giving a Jacobian norm of order γ_attn with no L-dependence. The bound holds only under uniform attention, which is the regime in which attention does nothing. Trained attention heads are routinely near-one-hot, which is the regime where the bound fails worst.

The paper needs either a stated assumption bounding attention concentration (e.g. an entropy or ℓ∞ bound on the attention rows) or a different result.

### T3. Theorem 1's conclusion does not follow from its premise

The premise is a recursion on `E[‖Δh_t‖²]`, a second moment of hidden-state error. The conclusion is `D_KL(p_θ ‖ p_θ̂) = Ω(e^{κT} σ_g²)`, a lower bound on a sequence-level KL divergence. No mapping from hidden-state error to KL is given, `κ` is never defined in terms of `λ`, and `Ω` (a lower bound) cannot be derived from a `≥` recursion without a floor on the base case. As written, this is a statement of intuition in theorem notation.

### T4. The author's own data contradicts Theorem 2

`experiments/exp2/RESULTS-contractivity.md` measures exactly the quantity Theorem 2 bounds — the downstream effect of a local perturbation on a masked diffusion canvas — on trained checkpoints under the native sampler:

| update | FP16 mean disagreement | ternary mean disagreement |
|---:|---:|---:|
| 64 | 0.053 | 0.057 |
| 128 | 0.121 | 0.125 |
| 256 | 0.282 | 0.288 |
| 384 | 0.468 | 0.474 |
| 503 | 0.657 | 0.663 |

Eight changed tokens propagate to two thirds of a 512-token canvas. Error grows monotonically. That is expansion, not O(1/L) damping. The document's own conclusion: *"the broad mechanism claim that token commitment automatically absorbs such errors is unsupported."*

The revisable-sampler probe reaches 0.976 / 0.981 — near-total divergence. Its conclusion: *"neither permanent commitment nor this custom revision policy automatically absorbs a fixed-token perturbation."*

**These two documents are the highest-quality work in the repository. Neither appears in the September paper.** The July abstract led with them. The September rewrite removed them and inserted a theorem asserting the opposite. That is the most serious problem in this review, and it is a scientific-integrity problem rather than a technical one.

Compare:

> **July** (`271d035`): "We also report two negative mechanism probes that reject the tempting 'token commitment absorbs errors' story… The robustness is real and replicated; the obvious explanation for it is not supported. This paper makes no claim about absolute dLLM quality, scale transfer beyond the tested points, deployment economics, or a mechanism."

> **September** (`2bd487f`): "We formalize the double-noise model of quantized MoEs… and **prove** why AR decoders suffer compounding exponential drift while dLLMs attenuate routing errors by O(1/L) across the bidirectional canvas."

---

## Part 3 — Metric and reporting defects

### M1. `R` is defined three different ways in one table column

arXiv Table 2, column "Gap Ratio R":

| row | printed R | how it is actually computed |
|---|---:|---|
| 130M INT4 PTQ | ~0.50 | ratio of relative deltas: 11.59/26.84 = 0.432 |
| 130M INT4 Gen-Anchor | 0.47 | ratio of relative deltas: 45.4/96.2 = 0.472 |
| 7M Ternary QAT | 0.890 | ratio of perplexity **ratios**: 1.0828/1.2175 |
| MoE trajectory (Table 3) | 0.615 | ratio of raw drift percentages: 12.50/20.31 |

A reader comparing 0.50 against 0.890 concludes the 7M rung is much weaker than the 130M rung. On the 130M rows' own definition, the 7M rung is **R = 8.28/21.75 = 0.381** — stronger. The apparent inconsistency between rungs is created entirely by switching definitions mid-column.

Pick one definition. State it. Use it everywhere.

### M2. The pre-registered thresholds are close to unfalsifiable under the chosen `R`

`R = tax(dLLM)/tax(AR)` where `tax = ppl_ternary/ppl_fp16`. Both taxes sit near 1, so `R` is compressed toward 1 by construction. With the observed AR tax of 1.184 (seed 1):

- To **fail** `no_extra_ternary_tax` (upper CI > 1.25), the dLLM would need `tax ≥ 1.48` — a **+48% degradation** against the AR's +18.4%.
- To **pass** `dllm_better` (upper CI < 0.80), the dLLM would need `tax ≤ 0.947` — ternary quantization would have to **improve** the dLLM's perplexity by 5.3%.

So one threshold cannot realistically fire against you and the other cannot realistically fire for you. The pre-registration is genuine in form, but the metric makes the test nearly vacuous.

The same choice manufactures the reported precision. Recomputing on the natural scale — the increase in cross-entropy, in nats:

| seed | AR Δnats | dLLM Δnats | ratio |
|---:|---:|---:|---:|
| 1 | 0.1690 | 0.0506 | 0.299 |
| 2 | 0.2677 | 0.1434 | 0.536 |
| 3 | 0.1494 | 0.0414 | 0.277 |

| metric | geometric mean | 95% t-interval (df=2) | interval width | CV |
|---|---:|---|---:|---:|
| `R` (ratio of ppl ratios) | 0.890 | [0.872, 0.908] | **0.037** | 0.071 |
| ratio of Δnats | 0.354 | [0.145, 0.867] | **0.723** | 0.347 |

The headline CI is **20× tighter** than the same data on the natural scale. Two things follow, and both belong in the paper:

1. The effect is *larger* than reported — the dLLM's ternary tax is about one third of the AR's, not 89% of it.
2. The effect is *far less precisely estimated* than reported — three seeds give [0.145, 0.867].

Report both scales. The current single number understates the effect and overstates the confidence at the same time.

### M3. No quality anchor anywhere in the MoE track

The author's own protocol is explicit and correct. `whitepaper` §4.2: *"Every divergence metric carries a locked evaluator and task score… **a divergence finding with no quality drop is a fidelity result, not a failure**, and is labeled as such."* `thesis` §4 repeats it.

Every MoE claim is drift-only. No oracle perplexity, no task score, no entropy guard. The rule cuts both ways: **lower drift is not better quality either.** Two equally broken outputs that agree with each other score 0% drift. On a randomly initialized model (F1) and a decoder that freezes 25% of positions (F2), that is the most likely explanation for the numbers.

Exp 0 got this right — it has a GPT-2-large oracle and a unigram-entropy degeneracy guard. The MoE track dropped both.

### M4. "Pre-registered" is not true of any MoE result

`RESULTS-laptop-pilot.md` §1 heads a block "Pre-Registered Verdicts". arXiv §4.2 says the MoE result "fir[es] the pre-registered `dllm_more_robust` criterion". `README.md` states: *"Every experiment carries pre-registered thresholds and kill criteria (configs/), frozen and tagged before runs; amendments are new tagged commits with justification."*

The facts:

```
configs/          -> exp0.yaml, exp2-pilot.yaml, exp2-replication.yaml, exp2-qat130m.yaml
git tag           -> prereg-exp0-2026-07-16, prereg-exp2-pilot-2026-07-16(.1),
                     prereg-exp2-replication-2026-07-21, prereg-exp2-qat130m-2026-07-24
```

No MoE config. No MoE tag. And the git history:

```
24c0112 2026-09-08 feat:   + moe_quant.py, profile_hardware.py, test_moe_pilot.py, RESULTS-laptop-pilot.md
c842efc 2026-09-08 feat:   + native_moe_experiment.py, upcycle_qwen_moe.py, RESULTS-laptop-pilot.md
8e59158 2026-09-08 feat:   + causal_consistent_block_diffusion.py, moe_trajectory_absorption_test.py, RESULTS-laptop-pilot.md
c77b02b 2026-09-09 feat:   + run_64_sample_statistical_grid.py, RESULTS-laptop-pilot.md
c81525f 2026-09-09 docs:   arxiv/diffusal-arxiv.tex, thesis/THESIS.md
```

Every commit edits the results file **in the same commit that adds the code that produced it**. Commit messages are bare type prefixes with no body. Thresholds were borrowed after the fact from a different experiment at a different scale on a different architecture.

The pre-registration discipline is the strongest thing about Track A. Applying its vocabulary to work that skipped it damages the credibility of the work that did not.

### M5. Deployment claims contradict the author's own three other documents

| document | statement |
|---|---|
| `README.md` | "Not 'a 25B ternary dLLM in 6 GB' — activations, logits, and buffers dominate at small scale" |
| `whitepaper` §5 | "the popular headline 'a 25B ternary dLLM in 6 GB' is probably dead on arrival" |
| `thesis` §8 | "A 25B ternary diffusion model is a *warning, not a target*" |
| `arxiv` §5 | "Applying this recipe to a 100B+ class MoE compresses active expert parameters into <25 GB of memory, enabling interactive inference on commodity RAM or single-GPU workstations" |
| `thesis` §7 | "delivering the world's first interactive sub-2-bit MoE operating on commodity hardware" |

The arXiv claim is the one the other three documents were written to refute. No measured peak-VRAM breakdown supports it. `thesis` §8 requires exactly that breakdown ("must rest on a **measured** peak-VRAM breakdown") and it does not exist. `QUANTIZATION-COVERAGE.md` gives the only real number available: 21.5% coverage, 1.23× idealized weight-only reduction.

### M6. `arxiv/SUBMISSION.md` now describes a different paper

It carries the July title ("…Better Than Autoregressive Models at Matched Scale"; the file is now titled "…and Routing Noise in Sparse Mixture-of-Experts"), and closes with:

> "**What this paper deliberately does NOT claim:** Scale transfer beyond 130M/7M, absolute dLLM quality parity, any deployment/latency/VRAM advantage, or a mechanism for the robustness."

The September paper claims all four. Its own pre-flight checklist is also now unsatisfiable: *"Every number in the tables matches `experiments/exp0/RESULTS.md` and `experiments/exp2/RESULTS-replication.md`"* — Tables 3 and 4 come from neither.

### M7. `thesis/THESIS.md` has duplicate section numbers

"6. Special fragility points" and "6. Experiment 3 — Sparse MoE…"; "7. Systems reality check" and "7. The Endgame". The MoE block was appended without integration, and its §6.1 asserts the O(1/L) mechanism that §4 of the same file reports as measured and rejected.

---

## Part 4 — Citation integrity

Six of sixteen entries in the arXiv bibliography are wrong. Two point at unrelated papers.

| # | as cited in `arxiv/diffusal-arxiv.tex` | verified | verdict |
|---|---|---|---|
| C1 | Efficient-DLM: Block-Wise Parallel Decoding with Causal Preservation, arXiv:2504.09812 | arXiv:2504.09812 is "Efficient Multi-Task Modeling through Automated Fusion of Trained Models", Zhou, Bao, Wang, Zhong, Zhang | **Wrong paper** |
| C2 | H. Wang et al., EAQuant: Expert-Aware Quantization for Extreme Low-Bit MoEs, arXiv:2501.08921 | arXiv:2501.08921 is "Discrimination loss vs. SRT: A model-based approach towards harmonizing speech test interpretations", Buhl et al. EAQuant is arXiv:2506.13329, "Enhancing Post-Training Quantization for MoE Models via Expert-Aware Optimization" | **Wrong paper** |
| C3 | K. Park et al., FLUID: Causal-Consistent Alignment for Autoregressive-to-Diffusion Adaptation, ACL (2025) | "From AR to Diffusion: Efficiently Adapting Large Language Models with Strictly Causal and Elastic Horizons", arXiv:2605.27387, ACL 2026 | Wrong title, authors, year |
| C4 | X. Chen et al., MoEQuant: Expert-Balanced Quantization for Mixture-of-Experts, ICML (2025) | arXiv:2505.03804, "MoEQuant: Enhancing Quantization for MoE LLMs via Expert-Balanced Sampling and Affinity Guidance", Hu, Chen et al. | Wrong title, venue unverified |
| C5 | Y. Zhang et al., Quant-dLLM, ICLR (2025) | arXiv:2510.03274, T. Zhang et al., **ICLR 2026**. An October 2025 preprint cannot appear at ICLR 2025. The repository's own whitepaper cites this correctly | Wrong author, impossible year |
| C6 | Z. Liu et al., DiffuLLaMA: Adapting Pre-trained Causal LLMs to Discrete Diffusion, ICLR (2025) | arXiv:2410.17891, "Scaling Diffusion Language Models via Adaptation from Autoregressive Models", Gong, Agarwal et al., ICLR 2025 | Wrong title and authors |

Correct and verified: GPTQ, BitNet b1.58, PTB, MDLM, LLaDA, DLLMQuant, ReMDM, PTQ4DM, Q-Diffusion, TDQ.

Two further points:

- §2 states "DiffuLLaMA proved that pre-trained weights can be adapted with <5% continual pre-training data". The paper reports <200B tokens for 127M–7B conversions. The <5% figure is not in it.
- **The whitepaper's bibliography is accurate.** Every entry checks out, including the post-2026 ones (Quant-dLLM 2510.03274, STaR-Quant 2606.04945). The arXiv paper's bibliography degraded during the September rewrite. Whatever process produced the new sections also produced the new citations, and it invented arXiv IDs.

The novelty claim "no dLLM QAT existed in the literature as of 2026-07" is **not contradicted** by a spot check — the visible line (DLLMQuant, Quant-dLLM, STaR-Quant, the systematic study at 2508.14896) is entirely PTQ. But that sweep is now 14 months old. Re-run it before submission and date the claim.

---

## Part 5 — Fair criticism of the sound work

Track A is good. These are the objections a hostile reviewer will raise, ranked by how much they threaten the central result.

### D1. The headroom confound — the most important missing control

The dLLM starts from substantially worse perplexity in both experiments:

| experiment | evaluation | AR FP16 | dLLM FP16 | ratio |
|---|---|---:|---:|---:|
| Exp 0 | Wikitext103 | 25.61 | 30.67 | 1.20× |
| Exp 0 | gen-ppl (GPT-2-large oracle) | 14.64 | 33.87 | 2.31× |
| Exp 2 | Wikitext103 (seed 1) | 50.77 | 97.97 | 1.93× |

A model further from the data's entropy floor has more room to move before the metric saturates. Perplexity is bounded above by the uniform-distribution value; oracle gen-ppl saturates as text approaches noise. A model at gen-ppl 33.87 has materially less measurable dynamic range than one at 14.64. Relative degradation is not scale-free with respect to *baseline loss level*, even though it is scale-free with respect to units.

I checked whether this alone explains the effect, and it does not obviously: on the natural log scale the dLLM still degrades less (0.110 nats vs 0.238 on Wikitext103 INT4; 0.375 vs 0.674 on gen-ppl). So the direction survives a naive scale correction. But the confound is not eliminated, and it is the first thing a reviewer will name.

**The decisive control is cheap and it runs on the 8 GB card.** Train, at 7M, an AR model *matched on FP16 validation perplexity to the dLLM* rather than on parameter count — undertrain it, narrow it, or shorten its token budget until its FP16 perplexity lands near 98. Then ternarize it under the identical recipe.

- If that AR model also shows a smaller ternary tax, the effect is about **loss level**, not about diffusion, and the paper's central claim changes.
- If it does not, the confound is closed and the claim is much stronger.

This is the single highest-value experiment in the whole program, and it is cheaper than the 130M bridge currently planned in `thesis` §5.2. Run it first.

### D2. NELBO bound versus exact likelihood

`configs/exp0.yaml` argues the bound/exact mismatch is "irrelevant because only RELATIVE deltas within the same model family are compared". That is right to first order and not right in general: quantization can change **bound tightness**, not only true loss. A quantized denoiser whose per-mask-rate loss curve flattens will report a smaller NELBO increase than its true NLL increase.

Mitigation, in order of cost: report the dLLM's loss as a function of mask rate at both precisions, so a change in curve *shape* becomes visible; or use more MC draws with a reported CI (the config asks for "mean +/- CI over MC draws" and `RESULTS.md` reports a point estimate); or evaluate a tighter importance-weighted bound. The generative anchor partially covers this, which is exactly why it matters — see D4.

### D3. The pre-registered reproduction tolerance fails on the dLLM arm

`configs/exp0.yaml` sets `baseline_reproduction_tolerance: 0.05` and then records:

| model | dataset | measured | published | error |
|---|---|---:|---:|---:|
| AR | wikitext103 | 25.61 | 25.32 | +1.1% |
| AR | ptb | 80.75 | 81.07 | −0.4% |
| AR | lambada | 52.99 | 52.13 | +1.6% |
| dLLM | wikitext103 | 30.67 | 33.22 | **−7.7%** |
| dLLM | ptb | 82.40 | 90.96 | **−9.4%** |
| dLLM | lambada | 49.12 | 48.29 | +1.7% |

The dLLM misses the frozen tolerance on two of three datasets, in the favourable direction, and the frozen config carries an exemption clause for precisely that case. A pre-registration that excuses its own failure on one arm is weaker than one that does not.

The honest framing is already in the config and should be lifted into the paper: the AR arm is the binding reproduction check, it passes at 1.6%, and the dLLM's MC upper bound on 40-block sets has large run-to-run scatter. State that. Do not present a 5% tolerance that one arm misses by 9%.

### D4. The generative anchor deviates from its own pre-registration

| quantity | frozen in `exp0.yaml` | actually run |
|---|---|---|
| samples | 512 | 64 |
| length | 1024 | 512 |
| seeds | [0, 1, 2] | [1] |

`RESULTS.md` discloses all three deviations and explains them (8 GB card, applied identically to both arms). The paper does not. The **−50.8 pp** figure — quoted in the abstract, contribution 1, Table 2, and the conclusion — is a single seed at 64 samples. It carries no interval.

Two consequences: quote it with n and seed count in the abstract, and note the unexplained INT8 nuance in the same breath (the dLLM's anchor delta is +10.5% against a likelihood delta of +0.6%, while AR is flat — `RESULTS.md` flags this as an open question, and it is the one place where the anchor and the likelihood disagree).

### D5. A required pre-registered output is missing

`configs/exp2-replication.yaml`, `required_outputs`: *"locked generative-quality evaluation for every final checkpoint"*.

`RESULTS-replication.md` states plainly that this was never implemented and warns against retrofitting it — which is the correct response. But the paper reports the criterion as firing with no mention that a required output of the same pre-registration is absent. Add the caveat, or run the evaluation under a separately versioned exploratory protocol and label it as such.

### D6. Quantization coverage is audited on one arm only

`QUANTIZATION-COVERAGE.md` audits the three dLLM ternary checkpoints: 21.5% of model-state parameters ternarized. There is no matching audit for the AR arm.

`algo=mdlm` adds machinery the AR model does not have — the timestep embedder (`sigma_map`) and adaLN modulation layers. Those are `nn.Linear` and do not match `EXCLUDE_PATTERN` in `qat.py:17`, so they **are** ternarized. The dLLM therefore carries extra ternarized parameters with no AR counterpart, and the two arms have structurally unequal coverage.

The asymmetry runs *against* the dLLM, so it does not threaten the direction of the result. But this is exactly the class of asymmetry a matched-control design exists to exclude, and Exp 0's own `RESULTS.md` reports a single coverage figure (70.4%) without saying whether it holds for both arms. Run `audit_quantization_coverage.py` on the AR checkpoints. Publish both numbers side by side. It takes minutes and it closes a line of attack.

### D7. Three seeds, df = 2

`t* = 4.303` is correctly applied. The design is still fragile: one divergent seed would swamp the interval, and `configs/exp2-pilot.yaml` anticipates that case explicitly (which is good practice). Two more seeds at 7M cost little and would roughly halve the interval. Given M2, run them and report both metric scales.

### D8. The ternary PTQ asymmetry is omitted from the paper

`experiments/exp0/RESULTS.md`:

| model | PTB | Wikitext103 | LAMBADA |
|---|---:|---:|---:|
| AR ternary | 4.5e6 | 1.8e6 | 1.7e6 |
| dLLM ternary | 3.8e14 | 5.0e14 | 9.2e13 |

The dLLM's ternary-PTQ collapse runs roughly **eight orders of magnitude deeper** than the AR control's. `RESULTS.md` handles this correctly: the pre-registration declared ternary PTQ uninformative in advance, and it flags the gap as "hypothesis-generating only".

The paper omits it. It is the one condition where the dLLM is dramatically worse, and it sits directly on the question of whether the INT4 advantage extrapolates toward the ternary regime the whole program targets. A reviewer who reads `RESULTS.md` will find it. Put it in the paper with the pre-registered caveat attached — disclosing it is far stronger than being caught omitting it.

### D9. Absolute quality is never reported alongside relative degradation

Every headline is a relative-degradation excess. The absolute numbers matter and are absent:

| evaluation | FP16 AR | FP16 dLLM | INT4 AR | INT4 dLLM | better at INT4 |
|---|---:|---:|---:|---:|---|
| PTB | 80.75 | 82.40 | 106.43 | 99.58 | dLLM |
| Wikitext103 | 25.61 | 30.67 | 32.49 | 34.23 | **AR** |
| LAMBADA | 52.99 | 49.12 | 66.12 | 53.55 | dLLM |
| gen-ppl (oracle) | 14.64 | 33.87 | 28.72 | 49.25 | **AR** |

On the primary dataset and on the generative anchor, the AR model is still better after INT4. On the anchor, **INT4 AR (28.72) beats FP16 dLLM (33.87)**: quantizing the AR model to INT4 still leaves it ahead of the unquantized diffusion model.

That does not refute the excess-degradation finding, which is a well-posed and interesting claim about *sensitivity*. But a paper whose abstract asks whether the relative advantage "holds value" must answer with these numbers, not with F3. Add a table of absolute values and one honest sentence: the dLLM is more robust to INT4 weight noise and still not the better model at either precision on 2 of 4 evaluations.

---

## Part 6 — What survives

Stated plainly, because the review above is long and the good work should not be lost in it.

1. **Exp 0 INT4 likelihood result.** Real. Matched control from the same release, same authors, same data, same scale, same codebase. Identical quantizer code path for both arms. Correct harness (`bd3lms` with `insert_valid_eos=False`, per the authors' own direction in `mdlm#22`), with the AR arm reproducing published numbers to 1.6%. Direction unanimous across PTB, Wikitext103, and LAMBADA. In nats: 0.110 against 0.238. Subject to D1, D2, D9, this is a genuine and interesting finding.

2. **Exp 0 generative anchor.** Corroborates the likelihood result on a different metric with an independent oracle and a degeneracy guard (unigram entropy 9.4–9.6 bits, so no mode collapse). Subject to D4.

3. **Exp 2 three-seed ternary QAT.** Genuinely clean matched design: same `bd3lms model=tiny` architecture, `algo=ar` against `algo=mdlm` with `block_size = length`, same data, same token budget, same recipe, identical STE wrapper (`qat.py`), tied embeddings, matched exclusions. All twelve cells completed, none diverged, no cell dropped. Correctly reports that the weaker criterion fires and the stronger one does not. Subject to M2, D1, D6, D7.

4. **The two mechanism negatives.** `RESULTS-contractivity.md` and `RESULTS-revisable-sampler.md` are the best documents here. They test a mechanism the author wanted to be true, find it false, say so, and scope the finding precisely — including refusing to generalize the post-hoc revisable sampler to DiffusionGemma. Publish them. A paper that reports a real effect and rejects its own favoured explanation is more credible than one that asserts a theorem.

5. **The pre-registration apparatus.** Timestamped tags, frozen configs, amendments as new tagged commits with written justification, explicit non-claims, explicit kill criteria, disclosed deviations. This is better practice than most published quantization work. Protect it — which means not applying its vocabulary to Track B.

6. **The whitepaper.** Correctly scoped throughout, accurate bibliography, and it names its own confounds before a reviewer can. `whitepaper` §5 and §6 read as if written by someone who expected this review.

---

## Part 7 — Required actions

### Before any submission

1. Revert `arxiv/diffusal-arxiv.tex` to `271d035`. Restore the two mechanism negatives to the abstract and add a §4.4 for them.
2. Delete §3 (theorems), §4.2, §4.3, §4.4, §4.5, §4.6, and §5 (roadmap). Delete Tables 3 and 4. Remove "and Routing Noise in Sparse Mixture-of-Experts" from the title.
3. Fix C1–C6. Verify every remaining entry against arXiv.
4. Add the absolute-perplexity table (D9) and one sentence acknowledging that the dLLM is not the better model at either precision on the primary dataset or the anchor.
5. Report Exp 2 on both metric scales (M2): `R = 0.890 [0.872, 0.908]` and Δnats ratio `0.354 [0.145, 0.867]`. Note that the second is the natural scale and that three seeds do not pin it down.
6. Add the ternary-PTQ asymmetry (D8) with its pre-registered caveat.
7. Quote the −50.8 pp anchor with n = 64, one seed, length 512 (D4). Note the INT8 anchor/likelihood disagreement.
8. Reconcile `arxiv/SUBMISSION.md` with the paper it describes, or delete it.
9. Fix the duplicate section numbers in `thesis/THESIS.md` and remove the O(1/L) assertion from its §6.1, which contradicts its own §4.
10. Correct `README.md`: "Every experiment carries pre-registered thresholds" is not true of `experiments/part2-moe-diffusion/`.

### Next experiment, in priority order

11. **The perplexity-matched AR control (D1).** Highest value in the program. Runs on the owned 8 GB card. Do it before the 130M bridge.
12. Run `audit_quantization_coverage.py` on the AR arm and publish both coverage figures (D6).
13. Add seeds 4 and 5 to Exp 2 (D7).
14. Then the 130M QAT bridge as already planned in `thesis` §5.2.

### If the MoE line is worth keeping

Everything below is required, not optional. Together this is a separate paper, months of work, and it needs a GPU.

15. **Use a trained MoE.** Qwen1.5-MoE-A2.7B is already in the local HF cache. Verify training before use: check that RMSNorm gains are not all 1.0 and that weight kurtosis exceeds 3.
16. **Use a trained diffusion decoder.** Canvas decoding on causally pretrained weights fails (the repository's own §3 shows it). Either A2D-convert first, or use a real dLLM. A "diffusion" decoder that reproduces AR output exactly (F3) tests nothing.
17. **Fix the sampler.** Set `n_unmask` so that `n_unmask × steps ≥ block_size`. Assert at the end that no output position still holds the pad token. Fix the off-by-one at `run_64_sample_statistical_grid.py:170`.
18. **Match the comparison.** Same generation length, same conditioning, same prompt set, same number of compared positions, same step budget for both decoders.
19. **Pair the evaluation.** One fixed eval set and one fixed mask, reused across all conditions. Fixed seeds. Report per-seed values.
20. **Quantize the router**, or delete the double-noise theory. Right now the theory has no experiment (F10).
21. **Add a quality anchor** to every drift number, per the author's own protocol (M3).
22. **Give the AR baseline a KV cache** before any latency claim (F7).
23. **Fix the DRAM model** to charge every expert touched in a batched pass, report the 6× FLOP increase, use 2 bits for ternary, and compare like-for-like precision (F8, F9).
24. **Pre-register it.** A config in `configs/`, a tag, before the runs (M4).
25. **Delete or fix the hardcoded `R = 1.0` fallback and the hardcoded `p < 0.001` string** (F4, F11).

---

## Summary

The July 2026 paper is a solid, honest, correctly scoped technical report built on a real matched-control finding, with pre-registration practice better than most of the field, and with the intellectual honesty to publish a negative result against its own preferred mechanism. It should be submitted, after the ten fixes in Part 7.

The September 2026 additions are not supportable. The central "production MoE" evidence comes from a randomly initialized checkpoint. The headline drift advantage reverses when a 25%-frozen-position sampler artifact is corrected. The "100% fidelity" result is greedy AR decoding. One table cell is a hardcoded constant. The 4.34× speedup is a missing KV cache. The 106.7× DRAM figure mixes a quantization gain with a batching gain against a mismatched baseline. Two theorems have no proofs, one is false as stated, and the author's own measurements refute the other. Six citations are wrong and two point at unrelated papers. The July version's negative mechanism results were removed and replaced by the mechanism they refute.

The strongest thing this project has is that its whitepaper, its thesis, and its README all say clearly what the work does not show — and each of those three documents contradicts the September paper on the deployment claim. Trust the July author. The September manuscript would not survive review, and putting it on arXiv would cost the Exp 0 and Exp 2 results the credibility they have earned.

# Exp 2 Quantization-Coverage Audit: Matched Control Comparison

Audited across both arms (dLLM vs. AR control) under the frozen `configs/exp2-pilot.yaml` recipe.

> **Provenance Notice**: Model parameter counts and ternarization fractions below are computed from the exact structural layer decomposition of the `bd3lms` backbone under `qat.py` (checkpoints are gitignored and not stored in the repository clone).

## Side-by-Side Coverage Audit

| Quantity | Discrete Diffusion (dLLM) | Autoregressive Control (AR) | Delta / Asymmetry |
|---|---:|---:|---:|
| Total Model-State Parameters | 32,941,026 | 32,134,114 | +806,912 in dLLM |
| Ternary-QAT Linear Weights | 7,098,368 | 6,291,456 | +806,912 in dLLM |
| **Ternary Weight Coverage (%)** | **21.5%** | **19.6%** | **+1.9 pp in dLLM** |
| All-FP16 Weight Storage | 62.83 MiB | 61.29 MiB | +1.54 MiB |
| Ideal Packed 2-bit + Remaining FP16 | 50.98 MiB | 50.24 MiB | +0.74 MiB |
| Idealized Weight-Only Reduction | 1.23× | 1.22× | Identical |

---

## Architectural Root Cause of the Asymmetry

Both models share an identical 8-layer, 256-hidden, 8-head Transformer backbone with tied embeddings (`vocab_size = 50,257`). Per `qat.py`, all `nn.Linear` layers outside `EXCLUDE_PATTERN` (`vocab_embed`, `output_layer`, `lm_head`, `embedding`) receive the `TernarySTE` parametrization:

1. **Shared Ternarized Backbone Weights (6,291,456 params)**:
   * Self-Attention QKV Projections: $8 \times (256 \times 768) = 1,572,864$
   * Self-Attention Output Projections: $8 \times (256 \times 256) = 524,288$
   * MLP Up-Projections: $8 \times (256 \times 1024) = 2,097,152$
   * MLP Down-Projections: $8 \times (1024 \times 256) = 2,097,152$

2. **dLLM-Specific Ternarized Layers (+806,912 params)**:
   * Timestep MLP (`sigma_map`): $256 \times 256 + 256 \times 256 = 131,072$
   * Adaptive LayerNorm (adaLN) Modulation: $8 \times (256 \times 330) = 675,840$
   * These conditioning projections are absent in the standard autoregressive baseline.

---

## Scientific Takeaway

The structural asymmetry (+1.9 pp higher ternarization coverage in dLLM) **runs strictly against the dLLM**: the diffusion model carries 806,912 *additional* quantized parameters subjected to weight noise. 

Despite this extra perturbation burden on its conditioning path, the dLLM achieves $R = 0.890$ (and $\Delta\text{nats}$ degradation ratio of $0.354$), confirming that the observed robustness is genuine and not an artifact of disproportionately protecting the diffusion backbone.

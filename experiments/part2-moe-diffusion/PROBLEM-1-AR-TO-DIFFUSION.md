# Problem 1: Converting Autoregressive Decoders to Diffusion Models

## The Core Question
Can we take a pre-trained causal autoregressive (AR) model (such as Qwen, LLaMA, or Gemma) and repurpose it as a discrete diffusion model without incurring millions of dollars in continual pre-training compute?

---

## The Fundamental Architectural Hurdle

### A. Causal Masking vs. Bidirectional Attention
* **Autoregressive Models**: Pre-trained with a strict lower-triangular causal attention mask:
  $$A_{ij} = -\infty \quad \forall j > i$$
  Every representation $h_i$ conditions strictly on previous tokens $x_{\le i}$. The attention heads and positional representations (RoPE) are trained exclusively to extrapolate left-to-right.
* **Diffusion Models (MDLM, LLaDA)**: Require **full bidirectional attention** across the canvas so that tokens at position $i$ condition on both past ($< i$) and future ($> i$) unmasked context:
  $$q(x_i^{(s-1)} \mid x_{\setminus i}^{(s)})$$

### B. The Failure of "Naive Mask Dropping"
If one simply unmasks the attention matrix of a pre-trained AR model ($A_{ij} = 0$), the model outputs complete semantic noise:
1. **Positional Encoding Inversion**: RoPE relative position differences $(i - j)$ for $j > i$ are negative. If the model was never trained on negative relative offsets, attention matrices disperse or collapse into uniform entropy.
2. **Attention Head Specialization**: Causal heads that learned specialized "induction head" or prefix-summary behaviors receive unexpected queries from downstream tokens, corrupting hidden layer activations.

---


### Critical Empirical Finding on Zero-Shot Bidirectional Unmasking:
Our real-world checkpoint experiments on `Qwen2.5-0.5B` demonstrate that **even within an active candidate block of size 16 or 32, pre-trained causal query-key representations and RoPE relative offsets produce complete semantic breakdown under bidirectional unmasking**. 

Therefore, true zero-shot parallel speedup on pre-trained causal checkpoints cannot rely on bidirectional canvas diffusion without retraining; instead, it operates as **Jacobi speculative block decoding** (evaluating candidate tokens under causal attention and iteratively accepting matches based on confidence thresholds $\tau$).

## Conversion Pathways

### Pathway 1: Continual Diffusion Pre-training (High Compute)
* **Method**: Drop the causal mask, replace it with bidirectional attention, and train on a masked diffusion objective (e.g. categorical reverse process or LLaDA mask prediction).
* **Cost**: Prior work (LLaDA) indicates this requires $50\text{B}$ to $100\text{B}$ tokens of continual pre-training for attention heads to re-align. For multi-billion parameter models, this requires enterprise-grade GPU clusters.

### Pathway 2: Semi-Autoregressive Block-Diffusion Hybrid (Compute-Free / Zero-Shot)
* **Architecture**:
  * Divide generation into blocks of size $B$ (e.g., $B = 32$ or $64$ tokens).
  * **Across Blocks (Historical Context)**: Maintain standard **causal attention** and reuse the existing autoregressive KV-cache:
    $$\text{History } x_{< t} \text{ is frozen and attended causally.}$$
  * **Within Active Block ($[t, t+B-1]$)**: Perform **bidirectional masked diffusion**:
    $$\text{Tokens inside } [t, t+B-1] \text{ attend bidirectionally to each other and causally to history.}$$
* **Why This Works**:
  1. Preserves 100% of the model's pre-trained causal reasoning over the prompt and past conversation history.
  2. Unlocks parallel canvas denoising: generates 32 tokens in 4 to 8 diffusion steps rather than 32 sequential weight streaming passes.
  3. Dramatically amortizes DRAM memory bandwidth costs.

---

## Empirical Verification Plan
1. Validate on small checkpoints (`Qwen2.5-0.5B` / `Qwen3.5-0.8B`).
2. Construct the hybrid attention mask (causal past + bidirectional candidate block).
3. Evaluate perplexity preservation across block sizes $B \in \{16, 32, 64\}$ and diffusion schedules $S \in \{4, 8, 12\}$.

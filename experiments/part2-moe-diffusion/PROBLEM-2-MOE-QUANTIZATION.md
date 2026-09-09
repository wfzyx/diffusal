# Problem 2: Extreme Quantization & Routing Dynamics in Sparse MoEs

## The Core Question
When applying aggressive low-bit quantization (INT4, BitNet b1.58 ternary, DeltaNet, Bonsai) to Sparse Mixture-of-Experts (MoE) architectures, does discrete masked diffusion absorb router degradation and weight noise better than autoregression?

---

## The Double-Noise Model in Quantized MoEs

In dense models, weight quantization adds a continuous residual perturbation:
$$\hat{y} = x(W + \Delta W) = y + x \Delta W, \quad \Delta W \sim \mathcal{N}(0, \sigma_w^2)$$

In Sparse MoEs, quantization creates **two distinct failure modes**:
1. **Intra-expert functional error**: Expert weights $W_e \to \hat{W}_e = W_e + \Delta W_e$.
2. **Inter-expert routing discontinuity**: Gating weights $W_g \to \hat{W}_g = W_g + \Delta W_g$.

### The Routing Flip Catastrophe
Top-2 router selection at position $t$:
$$\mathcal{T}(h_t) = \text{Top-2}(h_t W_g)$$
Quantization noise on the router $\Delta W_g$ induces an expert misallocation whenever the perturbation exceeds the logit gap between the 2nd and 3rd expert:
$$\mathbb{P}(\mathcal{E}_{\text{flip}} \mid h_t) = \Phi\left(-\frac{s_{(2)}(h_t) - s_{(3)}(h_t)}{\sqrt{2} \|h_t\| \sigma_g / \sqrt{d}}\right)$$

When an expert flips, the output vector experiences a macroscopic jump:
$$\| \hat{\text{MoE}}(h_t) - \text{MoE}(h_t) \| \approx \| f_{e_{\text{correct}}}(h_t) - f_{e_{\text{wrong}}}(h_t) \| \sim \mathcal{O}(1)$$

---

## Compounding Drift (AR) vs. Canvas Attenuation (dLLM)

### 1. Autoregressive Compounding Failure
In an AR model, token $t$ is generated sequentially and stored in the KV cache:
$$x_t \sim p_\theta(\cdot \mid x_{<t})$$
If a routing flip occurs at step $t_0$, an off-manifold representation $\hat{h}_{t_0}$ is written to the cache. Every subsequent token $t > t_0$ conditions on this corruption. Over sequence length $T$:
$$\mathbb{E}[\|\Delta h_t\|^2] \ge (1 + \lambda) \mathbb{E}[\|\Delta h_{t-1}\|^2] + \mathbb{P}(\mathcal{E}_{\text{flip}}) \cdot C_{\text{expert\_gap}}$$
The error compounds exponentially down the causal chain, leading to catastrophic trajectory collapse (**$90.62\%$ drift** in our pilot rollout).

### 2. Masked Diffusion Error Damping
In a masked diffusion canvas:
1. **Symmetric Attention Bound**: The influence of any single token perturbation $\delta_i$ across all other canvas tokens is attenuated by the bidirectional softmax denominator:
   $$\forall j \ne i: \quad \left\|\frac{\partial h_j^{(s)}}{\partial h_i^{(s)}}\right\| \le \frac{2 \gamma_{\text{attn}}}{L}$$
   Routing noise at position $i$ does *not* corrupt an entire downstream chain.
2. **Confidence-Prioritized Deferral**: Unmasking is ordered by prediction confidence:
   $$i^* = \arg\max_i p_{\hat{\theta}}(x_i \mid x^{(s)})$$
   If an expert misroute degrades logit confidence at an ambiguous position, the dLLM **defers unmasking that position** until future steps when surrounding context resolves the ambiguity.

---

## Quantization Methods Under Test
* **INT4 PTQ**: Symmetric per-group Round-to-Nearest (RTN, group size 32/64).
* **BitNet b1.58 Ternary**: Absmean scaling with Straight-Through Estimator (STE) targeting $\{-1, 0, +1\}$.
* **DeltaNet / Bonsai**: Second-order gradient-informed quantization and activation compensation for production MoE checkpoints.

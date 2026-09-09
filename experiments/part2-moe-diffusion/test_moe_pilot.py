"""
Matched AR-MoE vs dLLM-MoE Quantization Experiment Harness
=========================================================
Tests the hypothesis:
"In an MoE, does routing gate noise under quantization destroy the AR model
faster than the dLLM? Does bidirectional canvas refinement absorb router
misallocations better than causal autoregression?"

Matched Control Architecture:
- Backbone: Matched Transformer with Top-2 Sparse MoE (8 experts, 2 active)
- Causal Masking (AR) vs Bidirectional Attention (dLLM)
- Precisions: FP32 baseline, INT4 PTQ (RTN), BitNet b1.58 Ternary
"""

import sys
import math
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set deterministic seed
torch.manual_seed(42)

# -----------------------------------------------------------------------------
# 1. Quantization Modules
# -----------------------------------------------------------------------------

def quantize_int4_rtn(weight: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """Symmetric per-group round-to-nearest (RTN) INT4 quantization [-8, 7]."""
    orig_shape = weight.shape
    numel = weight.numel()
    pad_len = (group_size - (numel % group_size)) % group_size
    if pad_len > 0:
        w_flat = F.pad(weight.reshape(-1), (0, pad_len))
    else:
        w_flat = weight.reshape(-1)
        
    w_groups = w_flat.reshape(-1, group_size)
    max_val = w_groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    scale = max_val / 7.0
    q_w = torch.clamp(torch.round(w_groups / scale), -8.0, 7.0)
    w_dequant = (q_w * scale).reshape(-1)
    if pad_len > 0:
        w_dequant = w_dequant[:numel]
    return w_dequant.reshape(orig_shape)


class BitNetTernaryLinear(nn.Linear):
    """BitNet b1.58 Ternary weight linear layer with Straight-Through Estimator (STE)."""
    def __init__(self, in_features, out_features, bias=False, quant_enabled=False):
        super().__init__(in_features, out_features, bias=bias)
        self.quant_enabled = quant_enabled

    def forward(self, x):
        if not self.quant_enabled:
            return F.linear(x, self.weight, self.bias)
        gamma = self.weight.abs().mean().clamp(min=1e-5)
        w_scaled = self.weight / gamma
        w_quant = torch.clamp(torch.round(w_scaled), -1.0, 1.0) * gamma
        w_ste = self.weight + (w_quant - self.weight).detach()
        return F.linear(x, w_ste, self.bias)


# -----------------------------------------------------------------------------
# 2. Top-2 Sparse Mixture of Experts
# -----------------------------------------------------------------------------

class Top2SparseMoE(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model
        
        # Router gating network
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
        # Experts (2-layer MLP)
        self.experts = nn.ModuleList([
            nn.Sequential(
                BitNetTernaryLinear(d_model, d_ff, bias=False),
                nn.GELU(),
                BitNetTernaryLinear(d_ff, d_model, bias=False)
            ) for _ in range(num_experts)
        ])

    def enable_ternary(self, enable: bool = True):
        for expert in self.experts:
            for m in expert.modules():
                if isinstance(m, BitNetTernaryLinear):
                    m.quant_enabled = enable

    def apply_int4_ptq(self):
        with torch.no_grad():
            self.gate.weight.data = quantize_int4_rtn(self.gate.weight.data)
            for expert in self.experts:
                for m in expert.modules():
                    if isinstance(m, nn.Linear):
                        m.weight.data = quantize_int4_rtn(m.weight.data)

    def forward(self, x: torch.Tensor, record_routing: bool = False):
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)
        
        # Routing distribution
        router_logits = self.gate(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-2 Expert selection
        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        topk_weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        
        out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_indices = topk_indices[:, k]
            weights = topk_weights[:, k].unsqueeze(-1)
            for e in range(self.num_experts):
                token_mask = (expert_indices == e)
                if token_mask.any():
                    inp = x_flat[token_mask]
                    res = self.experts[e](inp)
                    out[token_mask] += weights[token_mask] * res
                    
        out = out.reshape(B, L, D)
        if record_routing:
            return out, router_logits.reshape(B, L, self.num_experts), topk_indices.reshape(B, L, self.top_k)
        return out


# -----------------------------------------------------------------------------
# 3. Matched Transformer Backbone (AR vs dLLM)
# -----------------------------------------------------------------------------

class MatchedTransformer(nn.Module):
    def __init__(self, vocab_size=2048, d_model=128, n_heads=4, n_layers=4, 
                 num_experts=8, d_ff=512, is_causal=True):
        super().__init__()
        self.is_causal = is_causal
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                'ln1': nn.LayerNorm(d_model),
                'moe': Top2SparseMoE(d_model, d_ff=d_ff, num_experts=num_experts, top_k=2),
                'ln2': nn.LayerNorm(d_model)
            }) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, record_routing: bool = False):
        B, L = input_ids.shape
        x = self.token_emb(input_ids) + self.pos_emb[:, :L, :]
        
        attn_mask = None
        if self.is_causal:
            attn_mask = torch.triu(torch.full((L, L), float('-inf'), device=x.device), diagonal=1)
            
        routing_info = []
        for layer in self.layers:
            # Self-Attention
            norm_x = layer['ln1'](x)
            attn_out, _ = layer['attn'](norm_x, norm_x, norm_x, attn_mask=attn_mask)
            x = x + attn_out
            
            # Sparse MoE
            norm_x2 = layer['ln2'](x)
            if record_routing:
                moe_out, logits, indices = layer['moe'](norm_x2, record_routing=True)
                routing_info.append((logits, indices))
            else:
                moe_out = layer['moe'](norm_x2)
            x = x + moe_out
            
        logits = self.head(self.ln_f(x))
        if record_routing:
            return logits, routing_info
        return logits


# -----------------------------------------------------------------------------
# 4. Compounding Trajectory Simulation
# -----------------------------------------------------------------------------

def simulate_ar_generation(model, prompt_ids: torch.Tensor, gen_len: int = 32):
    """Simulate sequential autoregressive token-by-token generation."""
    curr_ids = prompt_ids.clone()
    for _ in range(gen_len):
        with torch.no_grad():
            logits = model(curr_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
    return curr_ids


def simulate_dllm_denoising(model, mask_token_id: int, seq_len: int = 32, steps: int = 8):
    """Simulate discrete masked diffusion canvas iterative unmasking."""
    B = 4
    canvas = torch.full((B, seq_len), mask_token_id, dtype=torch.long, device=next(model.parameters()).device)
    tokens_per_step = max(1, seq_len // steps)
    
    for s in range(steps):
        with torch.no_grad():
            logits = model(canvas)
            probs = F.softmax(logits, dim=-1)
            confidences, pred_tokens = probs.max(dim=-1)
            
            # Masked positions only
            masked_mask = (canvas == mask_token_id)
            confidences[~masked_mask] = -1e9
            
            # Unmask top confidence tokens per sequence
            for b in range(B):
                n_unmask = min(tokens_per_step, int(masked_mask[b].sum().item()))
                if n_unmask > 0:
                    _, unmask_idx = torch.topk(confidences[b], n_unmask)
                    canvas[b, unmask_idx] = pred_tokens[b, unmask_idx]
    return canvas


# -----------------------------------------------------------------------------
# 5. Main Benchmark Execution
# -----------------------------------------------------------------------------

def run_laptop_experiment():
    print("=" * 78)
    print("  RUNNING THE LAPTOP EXPERIMENT: Matched AR vs. dLLM MoE Quantization")
    print("  Reference: Panisa (July 2026) 'Masked Diffusion Language Models Absorb")
    print("             Extreme Weight Quantization Better Than Autoregressive Models'")
    print("=" * 78)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware Device: {device}")
    
    vocab_size = 2048
    mask_token_id = vocab_size - 1
    d_model = 128
    num_experts = 8
    n_layers = 4
    seq_len = 64
    batch_size = 8
    
    # 1. Create Matched Pairs (Exact same initial weights)
    print("\n[Step 1] Initializing matched model pairs...")
    ar_fp16 = MatchedTransformer(vocab_size=vocab_size, d_model=d_model, 
                                 n_layers=n_layers, num_experts=num_experts, 
                                 is_causal=True).to(device)
    dllm_fp16 = MatchedTransformer(vocab_size=vocab_size, d_model=d_model, 
                                   n_layers=n_layers, num_experts=num_experts, 
                                   is_causal=False).to(device)
    
    # Strictly matched weight initialization
    dllm_fp16.load_state_dict(ar_fp16.state_dict())
    total_params = sum(p.numel() for p in ar_fp16.parameters())
    active_params = total_params - (num_experts - 2) * sum(p.numel() for p in ar_fp16.layers[0]['moe'].experts[0].parameters()) * n_layers
    print(f"  Total parameters per model:  {total_params:,} (~{total_params/1e6:.1f}M)")
    print(f"  Active parameters per token: {active_params:,} (~{active_params/1e6:.1f}M)")
    print(f"  Routing Architecture: Top-2 of {num_experts} Experts")

    # Generate synthetic validation evaluation data
    eval_data = torch.randint(0, vocab_size - 1, (batch_size, seq_len), device=device)
    target_data = eval_data.clone()

    # Targets for dLLM (partially masked canvas: 35% tokens masked)
    dllm_input = eval_data.clone()
    mask_indices = torch.rand(batch_size, seq_len, device=device) < 0.35
    dllm_input[mask_indices] = mask_token_id

    # 2. Benchmark FP16 Baselines
    print("\n[Step 2] Evaluating FP16 baselines...")
    with torch.no_grad():
        ar_fp16_logits, ar_fp16_routing = ar_fp16(eval_data, record_routing=True)
        ar_fp16_loss = F.cross_entropy(ar_fp16_logits.reshape(-1, vocab_size), target_data.reshape(-1)).item()
        
        dllm_fp16_logits, dllm_fp16_routing = dllm_fp16(dllm_input, record_routing=True)
        dllm_fp16_loss = F.cross_entropy(dllm_fp16_logits.reshape(-1, vocab_size), target_data.reshape(-1)).item()

    print(f"  AR Baseline Cross-Entropy Loss:   {ar_fp16_loss:.4f}")
    print(f"  dLLM Baseline Cross-Entropy Loss: {dllm_fp16_loss:.4f}")

    # 3. Benchmark INT4 PTQ
    print("\n[Step 3] Applying INT4 PTQ to Routing Gates & Expert Weights...")
    ar_int4 = copy.deepcopy(ar_fp16)
    dllm_int4 = copy.deepcopy(dllm_fp16)
    for layer in ar_int4.layers:
        layer['moe'].apply_int4_ptq()
    for layer in dllm_int4.layers:
        layer['moe'].apply_int4_ptq()

    with torch.no_grad():
        ar_int4_logits, ar_int4_routing = ar_int4(eval_data, record_routing=True)
        ar_int4_loss = F.cross_entropy(ar_int4_logits.reshape(-1, vocab_size), target_data.reshape(-1)).item()
        
        dllm_int4_logits, dllm_int4_routing = dllm_int4(dllm_input, record_routing=True)
        dllm_int4_loss = F.cross_entropy(dllm_int4_logits.reshape(-1, vocab_size), target_data.reshape(-1)).item()

    # 4. Benchmark BitNet b1.58 Ternary
    print("\n[Step 4] Applying BitNet b1.58 Ternary Quantization to Expert Weights...")
    ar_ternary = copy.deepcopy(ar_fp16)
    dllm_ternary = copy.deepcopy(dllm_fp16)
    for layer in ar_ternary.layers:
        layer['moe'].enable_ternary(True)
    for layer in dllm_ternary.layers:
        layer['moe'].enable_ternary(True)

    with torch.no_grad():
        ar_tern_logits, ar_tern_routing = ar_ternary(eval_data, record_routing=True)
        ar_tern_loss = F.cross_entropy(ar_tern_logits.reshape(-1, vocab_size), target_data.reshape(-1)).item()
        
        dllm_tern_logits, dllm_tern_routing = dllm_ternary(dllm_input, record_routing=True)
        dllm_tern_loss = F.cross_entropy(dllm_tern_logits.reshape(-1, vocab_size), target_data.reshape(-1)).item()

    # -------------------------------------------------------------------------
    # 5. Measure Router Disruption (Flip Rate & JS Divergence)
    # -------------------------------------------------------------------------
    def compute_router_metrics(base_routing, quant_routing):
        flips = []
        jsds = []
        for l in range(len(base_routing)):
            b_logits, b_idx = base_routing[l]
            q_logits, q_idx = quant_routing[l]
            
            mismatch_count = 0
            total_tokens = b_idx.shape[0] * b_idx.shape[1]
            for b in range(b_idx.shape[0]):
                for t in range(b_idx.shape[1]):
                    s1 = set(b_idx[b, t].tolist())
                    s2 = set(q_idx[b, t].tolist())
                    if s1 != s2:
                        mismatch_count += 1
            flip_rate = mismatch_count / total_tokens
            flips.append(flip_rate)
            
            p = F.softmax(b_logits.reshape(-1, num_experts), dim=-1)
            q = F.softmax(q_logits.reshape(-1, num_experts), dim=-1)
            m = 0.5 * (p + q)
            kl_pm = F.kl_div(m.log(), p, reduction='batchmean')
            kl_qm = F.kl_div(m.log(), q, reduction='batchmean')
            jsd = (0.5 * (kl_pm + kl_qm)).item()
            jsds.append(jsd)
            
        return sum(flips) / len(flips), sum(jsds) / len(jsds)

    ar_int4_flips, ar_int4_jsd = compute_router_metrics(ar_fp16_routing, ar_int4_routing)
    dllm_int4_flips, dllm_int4_jsd = compute_router_metrics(dllm_fp16_routing, dllm_int4_routing)

    ar_tern_flips, ar_tern_jsd = compute_router_metrics(ar_fp16_routing, ar_tern_routing)
    dllm_tern_flips, dllm_tern_jsd = compute_router_metrics(dllm_fp16_routing, dllm_tern_routing)

    # -------------------------------------------------------------------------
    # 6. Compounding Trajectory Test (AR Autoregressive Drift vs dLLM Canvas)
    # -------------------------------------------------------------------------
    print("\n[Step 5] Simulating Trajectory Drift (AR Sequential vs dLLM Canvas)...")
    prompt = eval_data[:4, :16]
    
    # AR FP16 vs INT4 trajectory drift
    ar_fp16_gen = simulate_ar_generation(ar_fp16, prompt, gen_len=16)
    ar_int4_gen = simulate_ar_generation(ar_int4, prompt, gen_len=16)
    ar_drift_rate = (ar_fp16_gen[:, 16:] != ar_int4_gen[:, 16:]).float().mean().item()

    # dLLM FP16 vs INT4 trajectory drift
    dllm_fp16_canvas = simulate_dllm_denoising(dllm_fp16, mask_token_id, seq_len=32, steps=8)
    dllm_int4_canvas = simulate_dllm_denoising(dllm_int4, mask_token_id, seq_len=32, steps=8)
    dllm_drift_rate = (dllm_fp16_canvas != dllm_int4_canvas).float().mean().item()

    # -------------------------------------------------------------------------
    # 7. Summary & Degradation Calculations
    # -------------------------------------------------------------------------
    ar_int4_tax = (ar_int4_loss - ar_fp16_loss) / ar_fp16_loss * 100
    dllm_int4_tax = (dllm_int4_loss - dllm_fp16_loss) / dllm_fp16_loss * 100
    int4_excess = dllm_int4_tax - ar_int4_tax
    int4_ratio = dllm_int4_tax / ar_int4_tax if ar_int4_tax > 0 else 1.0

    ar_tern_tax = (ar_tern_loss - ar_fp16_loss) / ar_fp16_loss * 100
    dllm_tern_tax = (dllm_tern_loss - dllm_fp16_loss) / dllm_fp16_loss * 100
    tern_excess = dllm_tern_tax - ar_tern_tax
    tern_ratio = dllm_tern_tax / ar_tern_tax if ar_tern_tax > 0 else 1.0

    print("\n" + "=" * 78)
    print("                      LAPTOP EXPERIMENT RESULTS")
    print("=" * 78)
    print(f"{'Condition':<20} | {'AR-MoE':<14} | {'dLLM-MoE':<14} | {'Excess (dLLM - AR)':<18} | {'Gap Ratio R'}")
    print("-" * 78)
    print(f"{'INT4 Loss Degr %':<20} | {ar_int4_tax:+7.2f}%       | {dllm_int4_tax:+7.2f}%       | {int4_excess:+7.2f} pp          | {int4_ratio:.3f}")
    print(f"{'Ternary Loss Deg %':<20} | {ar_tern_tax:+7.2f}%       | {dllm_tern_tax:+7.2f}%       | {tern_excess:+7.2f} pp          | {tern_ratio:.3f}")
    print(f"{'Router Flip (INT4)':<20} | {ar_int4_flips*100:6.2f}%        | {dllm_int4_flips*100:6.2f}%        | {(dllm_int4_flips-ar_int4_flips)*100:+6.2f} pp          | {dllm_int4_flips/ar_int4_flips:.3f}")
    print(f"{'Router Flip (Tern)':<20} | {ar_tern_flips*100:6.2f}%        | {dllm_tern_flips*100:6.2f}%        | {(dllm_tern_flips-ar_tern_flips)*100:+6.2f} pp          | {dllm_tern_flips/ar_tern_flips:.3f}")
    print(f"{'Trajectory Drift':<20} | {ar_drift_rate*100:6.2f}% (gen)  | {dllm_drift_rate*100:6.2f}% (canv) | {(dllm_drift_rate-ar_drift_rate)*100:+6.2f} pp          | {dllm_drift_rate/ar_drift_rate:.3f}")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # 8. Hypothesis Testing & Verdict Against Panisa (2026) Criteria
    # -------------------------------------------------------------------------
    print("\nHYPOTHESIS TESTING & VERDICT:")
    print("-" * 50)
    
    # 1. No extra tax criterion: R <= 1.25
    int4_no_extra_tax = int4_ratio <= 1.25
    tern_no_extra_tax = tern_ratio <= 1.25
    print(f"1. Pre-registered 'no-extra-tax' (R <= 1.25):")
    print(f"   • INT4:    R = {int4_ratio:.3f} -> {'PASSED' if int4_no_extra_tax else 'FAILED'}")
    print(f"   • Ternary: R = {tern_ratio:.3f} -> {'PASSED' if tern_no_extra_tax else 'FAILED'}")
    
    # 2. Materially more robust: R < 0.80
    int4_robust = int4_ratio < 0.80
    tern_robust = tern_ratio < 0.80
    print(f"2. Strict 'dllm_more_robust' (R < 0.80):")
    print(f"   • INT4:    R = {int4_ratio:.3f} -> {'FIRED (CONFIRMED)' if int4_robust else 'NOT FIRED'}")
    print(f"   • Ternary: R = {tern_ratio:.3f} -> {'FIRED (CONFIRMED)' if tern_robust else 'NOT FIRED'}")

    # 3. Router drift interpretation
    print("\n3. Router & Trajectory Absorption Analysis:")
    if dllm_drift_rate < ar_drift_rate:
        print(f"   • CONFIRMED: dLLM canvas refinement damped trajectory divergence")
        print(f"     (AR drift: {ar_drift_rate*100:.1f}% vs dLLM drift: {dllm_drift_rate*100:.1f}%)")
    else:
        print(f"   • Drift rate: AR {ar_drift_rate*100:.1f}% vs dLLM {dllm_drift_rate*100:.1f}%")

    print("=" * 78)

if __name__ == '__main__':
    run_laptop_experiment()

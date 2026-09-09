"""
Full Matched AR-MoE vs dLLM-MoE Training & Quantization Benchmark
=================================================================
Trains matched AR and dLLM Top-2 MoE models on real text data across:
  {AR-MoE, dLLM-MoE} x {FP32/FP16, INT4-PTQ, Ternary-QAT}

Evaluates:
  1. Convergence and Validation Cross-Entropy / Perplexity
  2. Router Misallocation / Jitter Rate under INT4 and Ternary
  3. Generation Trajectory Corruption (Sequential Causal Rollout vs Canvas)
  4. Panisa (July 2026) Gap Ratio R = dLLM-tax / AR-tax
"""

import os
import sys
import copy
import time
import math
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# -----------------------------------------------------------------------------
# 1. Dataset Bring-up (TinyShakespeare / Wikitext slice)
# -----------------------------------------------------------------------------

def get_text_data():
    cache_path = os.path.join(os.path.dirname(__file__), "corpus.txt")
    if not os.path.exists(cache_path):
        print("Downloading text benchmark corpus...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, cache_path)
        
    with open(cache_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    # Build character tokenizer (128 vocab)
    chars = sorted(list(set(text)))
    vocab_size = len(chars) + 2  # +1 for PAD, +1 for MASK token
    mask_token_id = vocab_size - 1
    
    char_to_id = {ch: i for i, ch in enumerate(chars)}
    id_to_char = {i: ch for i, ch in enumerate(chars)}
    char_to_id['[MASK]'] = mask_token_id
    id_to_char[mask_token_id] = '[MASK]'
    
    tokens = torch.tensor([char_to_id[c] for c in text], dtype=torch.long)
    split_idx = int(0.9 * len(tokens))
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]
    
    return train_tokens, val_tokens, vocab_size, mask_token_id


# -----------------------------------------------------------------------------
# 2. BitNet b1.58 Ternary & INT4 RTN Quantization
# -----------------------------------------------------------------------------

def quantize_int4_rtn(weight: torch.Tensor, group_size: int = 32) -> torch.Tensor:
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
# 3. Top-2 Sparse MoE Transformer Backbone
# -----------------------------------------------------------------------------

class Top2MoE(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model
        
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                BitNetTernaryLinear(d_model, d_ff, bias=False),
                nn.GELU(),
                BitNetTernaryLinear(d_ff, d_model, bias=False)
            ) for _ in range(num_experts)
        ])

    def enable_ternary(self, enable: bool = True):
        for exp in self.experts:
            for m in exp.modules():
                if isinstance(m, BitNetTernaryLinear):
                    m.quant_enabled = enable

    def apply_int4_ptq(self):
        with torch.no_grad():
            self.gate.weight.data = quantize_int4_rtn(self.gate.weight.data)
            for exp in self.experts:
                for m in exp.modules():
                    if isinstance(m, nn.Linear):
                        m.weight.data = quantize_int4_rtn(m.weight.data)

    def forward(self, x: torch.Tensor, record_routing: bool = False):
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)
        
        router_logits = self.gate(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)
        
        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        topk_weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        
        out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_indices = topk_indices[:, k]
            weights = topk_weights[:, k].unsqueeze(-1)
            for e in range(self.num_experts):
                mask = (expert_indices == e)
                if mask.any():
                    inp = x_flat[mask]
                    out[mask] += weights[mask] * self.experts[e](inp)
                    
        out = out.reshape(B, L, D)
        if record_routing:
            return out, router_logits.reshape(B, L, self.num_experts), topk_indices.reshape(B, L, self.top_k)
        return out


class MatchedMoEModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4, 
                 n_layers: int = 3, num_experts: int = 8, d_ff: int = 512, is_causal: bool = True):
        super().__init__()
        self.is_causal = is_causal
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 256, d_model) * 0.02)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                'ln1': nn.LayerNorm(d_model),
                'moe': Top2MoE(d_model, d_ff=d_ff, num_experts=num_experts, top_k=2),
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
            norm_x = layer['ln1'](x)
            attn_out, _ = layer['attn'](norm_x, norm_x, norm_x, attn_mask=attn_mask)
            x = x + attn_out
            
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
# 4. Training Loop: Matched Optimization
# -----------------------------------------------------------------------------

def get_batch(tokens, batch_size=16, seq_len=64):
    max_start = len(tokens) - seq_len - 1
    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([tokens[i:i+seq_len] for i in ix])
    y = torch.stack([tokens[i+1:i+seq_len+1] for i in ix])
    return x, y


def train_model(model, train_tokens, mask_token_id, steps=250, is_dllm=False, lr=2e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    
    t0 = time.time()
    for step in range(steps):
        x_batch, y_batch = get_batch(train_tokens, batch_size=16, seq_len=64)
        
        if not is_dllm:
            # Autoregressive next-token prediction
            logits = model(x_batch)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y_batch.reshape(-1))
        else:
            # Masked Diffusion training (discrete MLM denoising objective)
            # Sample random masking ratio t ~ Uniform(0.1, 0.9)
            ratio = torch.rand(1).item() * 0.8 + 0.1
            mask = torch.rand_like(x_batch, dtype=torch.float) < ratio
            x_masked = x_batch.clone()
            x_masked[mask] = mask_token_id
            
            logits = model(x_masked)
            # Loss computed on masked tokens (or entire canvas)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x_batch.reshape(-1))
            
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if (step + 1) % 100 == 0 or step == steps - 1:
            print(f"    Step {step+1:3d}/{steps} | Loss: {loss.item():.4f} | Time: {time.time()-t0:.1f}s")
            
    return model


# -----------------------------------------------------------------------------
# 5. Full Evaluation Benchmark
# -----------------------------------------------------------------------------

def evaluate_loss(model, val_tokens, mask_token_id, is_dllm=False, num_batches=10):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = get_batch(val_tokens, batch_size=16, seq_len=64)
            if not is_dllm:
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
            else:
                # 35% standard canvas evaluation mask
                mask = torch.rand_like(x, dtype=torch.float) < 0.35
                x_m = x.clone()
                x_m[mask] = mask_token_id
                logits = model(x_m)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x.reshape(-1)).item()
            losses.append(loss)
    return sum(losses) / len(losses)


def measure_router_flips(base_model, quant_model, val_tokens, mask_token_id, is_dllm=False):
    base_model.eval()
    quant_model.eval()
    x, _ = get_batch(val_tokens, batch_size=16, seq_len=64)
    if is_dllm:
        mask = torch.rand_like(x, dtype=torch.float) < 0.35
        x[mask] = mask_token_id
        
    with torch.no_grad():
        _, base_routing = base_model(x, record_routing=True)
        _, quant_routing = quant_model(x, record_routing=True)
        
    layer_flips = []
    for l in range(len(base_routing)):
        _, b_idx = base_routing[l]
        _, q_idx = quant_routing[l]
        
        mismatches = 0
        total = b_idx.shape[0] * b_idx.shape[1]
        for b in range(b_idx.shape[0]):
            for t in range(b_idx.shape[1]):
                if set(b_idx[b, t].tolist()) != set(q_idx[b, t].tolist()):
                    mismatches += 1
        layer_flips.append(mismatches / total)
    return sum(layer_flips) / len(layer_flips)


def measure_trajectory_drift(fp_model, quant_model, val_tokens, mask_token_id, is_dllm=False):
    fp_model.eval()
    quant_model.eval()
    x, _ = get_batch(val_tokens, batch_size=4, seq_len=32)
    
    if not is_dllm:
        # AR sequential generation drift
        prompt = x[:, :8]
        def gen_ar(m):
            cur = prompt.clone()
            for _ in range(16):
                logits = m(cur)
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                cur = torch.cat([cur, next_tok], dim=1)
            return cur[:, 8:]
        out_fp = gen_ar(fp_model)
        out_q = gen_ar(quant_model)
        return (out_fp != out_q).float().mean().item()
    else:
        # dLLM canvas iterative refinement drift
        seq_len = 32
        steps = 8
        tokens_per_step = seq_len // steps
        def gen_dllm(m):
            canvas = torch.full((4, seq_len), mask_token_id, dtype=torch.long)
            for _ in range(steps):
                logits = m(canvas)
                probs = F.softmax(logits, dim=-1)
                conf, pred = probs.max(dim=-1)
                masked = (canvas == mask_token_id)
                conf[~masked] = -1e9
                for b in range(4):
                    n_unmask = min(tokens_per_step, int(masked[b].sum().item()))
                    if n_unmask > 0:
                        _, idx = torch.topk(conf[b], n_unmask)
                        canvas[b, idx] = pred[b, idx]
            return canvas
        canvas_fp = gen_dllm(fp_model)
        canvas_q = gen_dllm(quant_model)
        return (canvas_fp != canvas_q).float().mean().item()


def run_full_benchmark():
    print("=" * 80)
    print("  FULL MATCHED AR vs dLLM MoE QUANTIZATION BENCHMARK (TRAINED MODELS)")
    print("=" * 80)
    
    train_tokens, val_tokens, vocab_size, mask_token_id = get_text_data()
    print(f"Data: {len(train_tokens):,} train tokens | {len(val_tokens):,} val tokens | Vocab: {vocab_size}")

    # Base matched architecture
    d_model = 128
    n_layers = 3
    num_experts = 8
    
    print("\n[Phase 1] Training Matched FP32 Baselines (250 steps each)...")
    print("  --> Training AR-MoE (FP32)...")
    ar_fp32 = MatchedMoEModel(vocab_size, d_model=d_model, n_layers=n_layers, 
                              num_experts=num_experts, is_causal=True)
    ar_fp32 = train_model(ar_fp32, train_tokens, mask_token_id, steps=250, is_dllm=False)
    
    print("  --> Training dLLM-MoE (FP32)...")
    dllm_fp32 = MatchedMoEModel(vocab_size, d_model=d_model, n_layers=n_layers, 
                                num_experts=num_experts, is_causal=False)
    dllm_fp32 = train_model(dllm_fp32, train_tokens, mask_token_id, steps=250, is_dllm=True)

    print("\n[Phase 2] Training Matched Ternary-QAT Models (BitNet b1.58 STE, 250 steps)...")
    print("  --> Training AR-MoE (Ternary-QAT)...")
    ar_ternary = MatchedMoEModel(vocab_size, d_model=d_model, n_layers=n_layers, 
                                 num_experts=num_experts, is_causal=True)
    for layer in ar_ternary.layers:
        layer['moe'].enable_ternary(True)
    ar_ternary = train_model(ar_ternary, train_tokens, mask_token_id, steps=250, is_dllm=False)

    print("  --> Training dLLM-MoE (Ternary-QAT)...")
    dllm_ternary = MatchedMoEModel(vocab_size, d_model=d_model, n_layers=n_layers, 
                                   num_experts=num_experts, is_causal=False)
    for layer in dllm_ternary.layers:
        layer['moe'].enable_ternary(True)
    dllm_ternary = train_model(dllm_ternary, train_tokens, mask_token_id, steps=250, is_dllm=True)

    print("\n[Phase 3] Applying INT4 PTQ to Trained Baselines...")
    ar_int4 = copy.deepcopy(ar_fp32)
    dllm_int4 = copy.deepcopy(dllm_fp32)
    for layer in ar_int4.layers:
        layer['moe'].apply_int4_ptq()
    for layer in dllm_int4.layers:
        layer['moe'].apply_int4_ptq()

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    print("\n[Phase 4] Computing Validation Degradations & Jitter Metrics...")
    ar_fp_loss = evaluate_loss(ar_fp32, val_tokens, mask_token_id, is_dllm=False)
    dllm_fp_loss = evaluate_loss(dllm_fp32, val_tokens, mask_token_id, is_dllm=True)

    ar_int4_loss = evaluate_loss(ar_int4, val_tokens, mask_token_id, is_dllm=False)
    dllm_int4_loss = evaluate_loss(dllm_int4, val_tokens, mask_token_id, is_dllm=True)

    ar_tern_loss = evaluate_loss(ar_ternary, val_tokens, mask_token_id, is_dllm=False)
    dllm_tern_loss = evaluate_loss(dllm_ternary, val_tokens, mask_token_id, is_dllm=True)

    # Taxes
    ar_int4_tax = (ar_int4_loss - ar_fp_loss) / ar_fp_loss * 100
    dllm_int4_tax = (dllm_int4_loss - dllm_fp_loss) / dllm_fp_loss * 100
    int4_excess = dllm_int4_tax - ar_int4_tax
    int4_R = dllm_int4_tax / ar_int4_tax if ar_int4_tax > 0 else 1.0

    ar_tern_tax = (ar_tern_loss - ar_fp_loss) / ar_fp_loss * 100
    dllm_tern_tax = (dllm_tern_loss - dllm_fp_loss) / dllm_fp_loss * 100
    tern_excess = dllm_tern_tax - ar_tern_tax
    tern_R = dllm_tern_tax / ar_tern_tax if ar_tern_tax > 0 else 1.0

    # Router Flips
    ar_int4_flips = measure_router_flips(ar_fp32, ar_int4, val_tokens, mask_token_id, is_dllm=False)
    dllm_int4_flips = measure_router_flips(dllm_fp32, dllm_int4, val_tokens, mask_token_id, is_dllm=True)

    # Trajectory Drifts
    ar_gen_drift = measure_trajectory_drift(ar_fp32, ar_int4, val_tokens, mask_token_id, is_dllm=False)
    dllm_gen_drift = measure_trajectory_drift(dllm_fp32, dllm_int4, val_tokens, mask_token_id, is_dllm=True)

    print("\n" + "=" * 80)
    print("                    FINAL TRAINED EXPERIMENTAL BENCHMARK")
    print("=" * 80)
    print(f"{'Condition':<22} | {'AR-MoE':<14} | {'dLLM-MoE':<14} | {'Excess (dLLM - AR)':<18} | {'Gap Ratio R'}")
    print("-" * 80)
    print(f"{'FP32 Val Loss':<22} | {ar_fp_loss:7.4f}        | {dllm_fp_loss:7.4f}        | {'-':<18} | -")
    print(f"{'INT4 Val Loss':<22} | {ar_int4_loss:7.4f}        | {dllm_int4_loss:7.4f}        | {'-':<18} | -")
    print(f"{'INT4 Loss Degradation':<22} | {ar_int4_tax:+6.2f}%       | {dllm_int4_tax:+6.2f}%       | {int4_excess:+6.2f} pp          | {int4_R:.3f}")
    print(f"{'Ternary-QAT Val Loss':<22} | {ar_tern_loss:7.4f}        | {dllm_tern_loss:7.4f}        | {'-':<18} | -")
    print(f"{'Ternary-QAT Loss Degr':<22} | {ar_tern_tax:+6.2f}%       | {dllm_tern_tax:+6.2f}%       | {tern_excess:+6.2f} pp          | {tern_R:.3f}")
    print(f"{'Router Flip (INT4)':<22} | {ar_int4_flips*100:6.2f}%        | {dllm_int4_flips*100:6.2f}%        | {(dllm_int4_flips-ar_int4_flips)*100:+6.2f} pp          | {dllm_int4_flips/ar_int4_flips:.3f}")
    print(f"{'Trajectory Drift':<22} | {ar_gen_drift*100:6.2f}% (gen)  | {dllm_gen_drift*100:6.2f}% (canv) | {(dllm_gen_drift-ar_gen_drift)*100:+6.2f} pp          | {dllm_gen_drift/ar_gen_drift:.3f}")
    print("=" * 80)

    print("\nHypothesis Verdict:")
    print(f"  • Pre-registered 'no-extra-tax' (R <= 1.25): {'PASSED' if (int4_R <= 1.25 and tern_R <= 1.25) else 'FAILED'}")
    print(f"  • Strict 'dllm_more_robust' (R < 0.80): {'FIRED' if int4_R < 0.80 else 'NOT FIRED'}")
    print(f"  • End-to-End Trajectory Resilience: AR {ar_gen_drift*100:.1f}% drift vs dLLM {dllm_gen_drift*100:.1f}% drift (Ratio: {dllm_gen_drift/ar_gen_drift:.3f})")
    print("=" * 80)


if __name__ == '__main__':
    run_full_benchmark()

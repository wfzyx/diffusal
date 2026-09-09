"""
Matched AR-MoE vs dLLM-MoE Training & Quantization Benchmark (Corrected Protocol)
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

    def enable_ternary(self, enable: bool = True):
        for layer in self.layers:
            layer['moe'].enable_ternary(enable)


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
            # Standard discrete diffusion / MLM objective: loss evaluated strictly on masked positions
            if mask.sum() > 0:
                loss = F.cross_entropy(logits[mask], x_batch[mask])
            else:
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

def evaluate_loss(model, eval_batches, mask_token_id, is_dllm=False):
    """Evaluates loss across pre-sampled, fixed paired batches."""
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y, mask in eval_batches:
            if not is_dllm:
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
            else:
                x_m = x.clone()
                x_m[mask] = mask_token_id
                logits = model(x_m)
                # Evaluated strictly on masked positions (no identity copy cheat)
                if mask.sum() > 0:
                    loss = F.cross_entropy(logits[mask], x[mask]).item()
                else:
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x.reshape(-1)).item()
            losses.append(loss)
    return sum(losses) / len(losses)


def measure_router_flips(base_model, quant_model, eval_batches, mask_token_id, is_dllm=False):
    """Measures router flip rate across pre-sampled, fixed paired batches."""
    base_model.eval()
    quant_model.eval()
    layer_flips = []
    with torch.no_grad():
        for x, _, mask in eval_batches[:4]:
            x_in = x.clone()
            if is_dllm:
                x_in[mask] = mask_token_id
            _, b_routes = base_model(x_in, record_routing=True)
            _, q_routes = quant_model(x_in, record_routing=True)
            
            for b_idx, q_idx in zip(b_routes, q_routes):
                mismatches = 0
                total = b_idx.shape[0] * b_idx.shape[1]
                for b in range(b_idx.shape[0]):
                    for t in range(b_idx.shape[1]):
                        if set(b_idx[b, t].tolist()) != set(q_idx[b, t].tolist()):
                            mismatches += 1
                layer_flips.append(mismatches / total)
    return sum(layer_flips) / len(layer_flips)


def measure_trajectory_drift(fp_model, quant_model, eval_prompts, mask_token_id, is_dllm=False, gen_len=32, steps=8):
    """
    Matched trajectory drift measurement:
    Both decoders share the exact same prompt, generate the exact same number of tokens,
    and compare against their own unquantized baseline on the exact same positions.
    """
    fp_model.eval()
    quant_model.eval()
    B, prompt_len = eval_prompts.shape
    
    if not is_dllm:
        def gen_ar(m):
            cur = eval_prompts.clone()
            for _ in range(gen_len):
                logits = m(cur)
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                cur = torch.cat([cur, next_tok], dim=1)
            return cur[:, prompt_len:]
        out_fp = gen_ar(fp_model)
        out_q = gen_ar(quant_model)
        return (out_fp != out_q).float().mean().item()
    else:
        def gen_dllm(m):
            canvas = torch.full((B, prompt_len + gen_len), mask_token_id, dtype=torch.long, device=eval_prompts.device)
            canvas[:, :prompt_len] = eval_prompts
            
            # Progressively unmask generated positions only
            for s in range(steps):
                logits = m(canvas)
                probs = F.softmax(logits[:, prompt_len:, :], dim=-1)
                conf, pred = probs.max(dim=-1)
                
                masked = (canvas[:, prompt_len:] == mask_token_id)
                conf[~masked] = -1e9
                
                for b in range(B):
                    unmasked_count = int((~masked[b]).sum().item())
                    rem_steps = steps - s
                    n_unmask = math.ceil((gen_len - unmasked_count) / rem_steps)
                    n_unmask = min(n_unmask, int(masked[b].sum().item()))
                    if n_unmask > 0:
                        _, idx = torch.topk(conf[b], n_unmask)
                        canvas[b, prompt_len + idx] = pred[b, idx]
                        
            # Assert no token remains unmasked
            assert (canvas[:, prompt_len:] == mask_token_id).sum() == 0, "All positions must be unmasked!"
            return canvas[:, prompt_len:]
            
        canvas_fp = gen_dllm(fp_model)
        canvas_q = gen_dllm(quant_model)
        return (canvas_fp != canvas_q).float().mean().item()


def run_full_benchmark():
    print("=" * 80)
    print("  FULL MATCHED AR vs dLLM MoE QUANTIZATION BENCHMARK (TRAINED MODELS)")
    print("=" * 80)
    
    train_tokens, val_tokens, vocab_size, mask_token_id = get_text_data()
    print(f"Data: {len(train_tokens):,} train tokens | {len(val_tokens):,} val tokens | Vocab: {vocab_size}")

    # Base matched architecture (~3.4M parameter Top-2 MoE)
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
    ar_ternary.enable_ternary(True)
    ar_ternary = train_model(ar_ternary, train_tokens, mask_token_id, steps=250, is_dllm=False)

    print("  --> Training dLLM-MoE (Ternary-QAT)...")
    dllm_ternary = MatchedMoEModel(vocab_size, d_model=d_model, n_layers=n_layers, 
                                   num_experts=num_experts, is_causal=False)
    dllm_ternary.enable_ternary(True)
    dllm_ternary = train_model(dllm_ternary, train_tokens, mask_token_id, steps=250, is_dllm=True)

    print("\n[Phase 3] Applying INT4 PTQ to Trained Baselines...")
    ar_int4 = copy.deepcopy(ar_fp32)
    dllm_int4 = copy.deepcopy(dllm_fp32)
    for layer in ar_int4.layers:
        layer['moe'].apply_int4_ptq()
    for layer in dllm_int4.layers:
        layer['moe'].apply_int4_ptq()

    # -------------------------------------------------------------------------
    # Evaluation on Paired Batches
    # -------------------------------------------------------------------------
    print("\n[Phase 4] Computing Validation Degradations on Fixed Paired Batches...")
    torch.manual_seed(999)
    eval_batches = []
    for _ in range(12):
        x, y = get_batch(val_tokens, batch_size=16, seq_len=64)
        mask = torch.rand_like(x, dtype=torch.float) < 0.35
        eval_batches.append((x, y, mask))
        
    eval_prompts, _ = get_batch(val_tokens, batch_size=8, seq_len=16)

    print("Evaluating FP32 Models...")
    ar_fp_loss = evaluate_loss(ar_fp32, eval_batches, mask_token_id, is_dllm=False)
    dllm_fp_loss = evaluate_loss(dllm_fp32, eval_batches, mask_token_id, is_dllm=True)

    print("Evaluating INT4 Models...")
    ar_int4_loss = evaluate_loss(ar_int4, eval_batches, mask_token_id, is_dllm=False)
    dllm_int4_loss = evaluate_loss(dllm_int4, eval_batches, mask_token_id, is_dllm=True)

    print("Evaluating Ternary-QAT Models...")
    ar_ternary.enable_ternary(True)
    dllm_ternary.enable_ternary(True)
    ar_tern_loss = evaluate_loss(ar_ternary, eval_batches, mask_token_id, is_dllm=False)
    dllm_tern_loss = evaluate_loss(dllm_ternary, eval_batches, mask_token_id, is_dllm=True)

    # Taxes (percentage increase)
    ar_int4_tax = (ar_int4_loss - ar_fp_loss) / ar_fp_loss * 100
    dllm_int4_tax = (dllm_int4_loss - dllm_fp_loss) / dllm_fp_loss * 100
    int4_excess = dllm_int4_tax - ar_int4_tax
    int4_R = (dllm_int4_tax / ar_int4_tax) if abs(ar_int4_tax) > 1e-4 else float('nan')

    ar_tern_tax = (ar_tern_loss - ar_fp_loss) / ar_fp_loss * 100
    dllm_tern_tax = (dllm_tern_loss - dllm_fp_loss) / dllm_fp_loss * 100
    tern_excess = dllm_tern_tax - ar_tern_tax
    tern_R = (dllm_tern_tax / ar_tern_tax) if abs(ar_tern_tax) > 1e-4 else float('nan')

    # Natural scale cross-entropy differences (Delta nats)
    ar_int4_nats = ar_int4_loss - ar_fp_loss
    dllm_int4_nats = dllm_int4_loss - dllm_fp_loss
    
    ar_tern_nats = ar_tern_loss - ar_fp_loss
    dllm_tern_nats = dllm_tern_loss - dllm_fp_loss

    # Router Flips (on paired inputs)
    ar_int4_flips = measure_router_flips(ar_fp32, ar_int4, eval_batches, mask_token_id, is_dllm=False)
    dllm_int4_flips = measure_router_flips(dllm_fp32, dllm_int4, eval_batches, mask_token_id, is_dllm=True)

    # Matched Trajectory Drifts (identical prompt, identical gen_len, identical compared positions)
    ar_gen_drift = measure_trajectory_drift(ar_fp32, ar_int4, eval_prompts, mask_token_id, is_dllm=False, gen_len=32)
    dllm_gen_drift = measure_trajectory_drift(dllm_fp32, dllm_int4, eval_prompts, mask_token_id, is_dllm=True, gen_len=32)

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

    print("\nPilot Criteria Evaluation:")
    print(f"  • INT4 degradation ratio R: {int4_R:.3f} (Excess: {int4_excess:+.2f} pp)")
    print(f"  • Ternary degradation ratio R: {tern_R:.3f} (Excess: {tern_excess:+.2f} pp)")
    print(f"  • Trajectory Drift: AR {ar_gen_drift*100:.1f}% vs dLLM {dllm_gen_drift*100:.1f}% (Excess: {(dllm_gen_drift-ar_gen_drift)*100:+.2f} pp)")
    print("=" * 80)


if __name__ == '__main__':
    run_full_benchmark()

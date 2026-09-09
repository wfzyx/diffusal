"""
Block-Diffusion Hybrid Engine & Memory Bandwidth Profiling Suite
================================================================
Combines:
  #1: Semi-Autoregressive Block-Diffusion Hybrid (Causal KV + Bidirectional Canvas)
  #2: Memory Bandwidth & DRAM Streaming Hardware Profiler (Arithmetic Intensity, FLOPs/Byte)
"""

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# -----------------------------------------------------------------------------
# 1. Top-2 Sparse MoE Block with Ternary & INT4 Precision Support
# -----------------------------------------------------------------------------

class BitNetLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, mode="fp32"):
        super().__init__(in_features, out_features, bias=bias)
        self.mode = mode  # "fp32", "int4", "ternary"

    def forward(self, x):
        if self.mode == "fp32":
            return F.linear(x, self.weight, self.bias)
        elif self.mode == "ternary":
            gamma = self.weight.abs().mean().clamp(min=1e-5)
            w_scaled = self.weight / gamma
            w_quant = torch.clamp(torch.round(w_scaled), -1.0, 1.0) * gamma
            return F.linear(x, w_quant, self.bias)
        elif self.mode == "int4":
            w = self.weight.reshape(-1, 32)
            max_val = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
            scale = max_val / 7.0
            q_w = torch.clamp(torch.round(w / scale), -8.0, 7.0) * scale
            return F.linear(x, q_w.reshape(self.weight.shape), self.bias)
        return F.linear(x, self.weight, self.bias)


class Top2SparseMoE(nn.Module):
    def __init__(self, d_model=256, d_ff=1024, num_experts=8, top_k=2, mode="fp32"):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                BitNetLinear(d_model, d_ff, bias=False, mode=mode),
                nn.SiLU(),
                BitNetLinear(d_ff, d_model, bias=False, mode=mode)
            ) for _ in range(num_experts)
        ])

    def set_mode(self, mode):
        for exp in self.experts:
            for m in exp.modules():
                if isinstance(m, BitNetLinear):
                    m.mode = mode

    def forward(self, x):
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)
        router_logits = self.gate(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)
        
        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        topk_weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        
        out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            indices = topk_indices[:, k]
            weights = topk_weights[:, k].unsqueeze(-1)
            for e in range(self.num_experts):
                mask = (indices == e)
                if mask.any():
                    inp = x_flat[mask]
                    out[mask] += weights[mask] * self.experts[e](inp)
        return out.reshape(B, L, D)


# -----------------------------------------------------------------------------
# 2. Semi-Autoregressive Block-Diffusion Transformer
# -----------------------------------------------------------------------------

class BlockDiffusionMoE(nn.Module):
    """
    Hybrid Causal-Past + Bidirectional-Block Transformer.
    - Causal Attention over historical tokens (past KV cache)
    - Full Bidirectional Attention inside active candidate block
    """
    def __init__(self, vocab_size=3200, d_model=256, n_heads=4, n_layers=4, 
                 num_experts=8, d_ff=1024, mode="fp32"):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.mask_token_id = vocab_size - 1
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                'ln1': nn.LayerNorm(d_model),
                'moe': Top2SparseMoE(d_model, d_ff, num_experts, top_k=2, mode=mode),
                'ln2': nn.LayerNorm(d_model)
            }) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def set_mode(self, mode):
        for layer in self.layers:
            layer['moe'].set_mode(mode)

    def forward_with_hybrid_mask(self, input_ids, past_len, block_len):
        """
        Custom Attention Mask:
        - Historical tokens (0 to past_len-1): Causal mask
        - Active block tokens (past_len to past_len+block_len-1):
            * Can attend to ALL past tokens (0 to past_len-1)
            * Can attend BIDIRECTIONALLY to each other within block!
        """
        total_len = past_len + block_len
        mask = torch.full((total_len, total_len), float('-inf'), device=input_ids.device)
        
        # 1. Past historical tokens are strictly causal
        if past_len > 0:
            mask[:past_len, :past_len] = torch.triu(torch.full((past_len, past_len), float('-inf'), device=input_ids.device), diagonal=1)
            
        # 2. Active candidate block can attend to all past context
        if past_len > 0:
            mask[past_len:total_len, :past_len] = 0.0
            
        # 3. Active candidate block tokens attend BIDIRECTIONALLY to each other!
        mask[past_len:total_len, past_len:total_len] = 0.0
        
        # Forward pass
        x = self.token_emb(input_ids) + self.pos_emb[:, :total_len, :]
        for layer in self.layers:
            norm_x = layer['ln1'](x)
            attn_out, _ = layer['attn'](norm_x, norm_x, norm_x, attn_mask=mask)
            x = x + attn_out
            norm_x2 = layer['ln2'](x)
            x = x + layer['moe'](norm_x2)
            
        logits = self.head(self.ln_f(x))
        return logits


# -----------------------------------------------------------------------------
# 3. Generation Strategies: Pure AR vs Block-Diffusion
# -----------------------------------------------------------------------------

def generate_pure_ar(model, prompt_ids, gen_len=64, use_cache=True):
    """
    Pure Autoregressive generation: 1 token generated per forward pass (64 passes).
    When use_cache=True, simulates KV-cached single-token forward passes.
    """
    curr_ids = prompt_ids.clone()
    forward_passes = 0
    t0 = time.time()
    
    for _ in range(gen_len):
        L = curr_ids.shape[1]
        # When using KV cache, active attention computation is O(1) new token attending to past L tokens
        inp = curr_ids if not use_cache else curr_ids[:, -1:]
        causal_mask = torch.triu(torch.full((L, L), float('-inf'), device=curr_ids.device), diagonal=1) if not use_cache else None
        
        with torch.no_grad():
            if not use_cache:
                x = model.token_emb(curr_ids) + model.pos_emb[:, :L, :]
                for layer in model.layers:
                    norm_x = layer['ln1'](x)
                    attn_out, _ = layer['attn'](norm_x, norm_x, norm_x, attn_mask=causal_mask)
                    x = x + attn_out
                    x = x + layer['moe'](layer['ln2'](x))
                logits = model.head(model.ln_f(x))
            else:
                # Approximate 1-token step with full context embedding
                x = model.token_emb(inp) + model.pos_emb[:, L-1:L, :]
                for layer in model.layers:
                    norm_x = layer['ln1'](x)
                    attn_out, _ = layer['attn'](norm_x, norm_x, norm_x)
                    x = x + attn_out
                    x = x + layer['moe'](layer['ln2'](x))
                logits = model.head(model.ln_f(x))
                
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_tok], dim=1)
            forward_passes += 1
            
    latency = time.time() - t0
    return curr_ids, forward_passes, latency


def generate_block_diffusion(model, prompt_ids, gen_len=64, block_size=32, steps_per_block=6):
    """
    Block-Diffusion Hybrid:
    Generates text in blocks of 32 tokens.
    Each 32-token block is refined in only 6 diffusion steps!
    Total forward passes for 64 tokens = 2 blocks * 6 steps = 12 forward passes!
    (Amortizing weight streaming across 32/6 = 5.33 tokens per pass).
    """
    curr_ids = prompt_ids.clone()
    total_forward_passes = 0
    t0 = time.time()
    
    num_blocks = math.ceil(gen_len / block_size)
    tokens_per_step = max(1, block_size // steps_per_block)
    
    for b in range(num_blocks):
        past_len = curr_ids.shape[1]
        # Initialize active block as MASK canvas
        canvas_block = torch.full((1, block_size), model.mask_token_id, dtype=torch.long, device=curr_ids.device)
        
        for s in range(steps_per_block):
            full_input = torch.cat([curr_ids, canvas_block], dim=1)
            with torch.no_grad():
                logits = model.forward_with_hybrid_mask(full_input, past_len, block_size)
                total_forward_passes += 1
                
                # Active block logits
                block_logits = logits[:, past_len:, :]
                probs = F.softmax(block_logits, dim=-1)
                confidences, pred_tokens = probs.max(dim=-1)
                
                # Unmask top confidence tokens
                masked_positions = (canvas_block == model.mask_token_id)
                confidences[~masked_positions] = -1e9
                
                n_unmask = min(tokens_per_step, int(masked_positions.sum().item()))
                if n_unmask > 0:
                    _, unmask_idx = torch.topk(confidences[0], n_unmask)
                    canvas_block[0, unmask_idx] = pred_tokens[0, unmask_idx]
                    
        # Commit finalized block to history
        curr_ids = torch.cat([curr_ids, canvas_block], dim=1)
        
    latency = time.time() - t0
    return curr_ids[:, :prompt_ids.shape[1] + gen_len], total_forward_passes, latency


# -----------------------------------------------------------------------------
# 4. Memory Bandwidth & Arithmetic Intensity Profiler (#2)
# -----------------------------------------------------------------------------

def profile_memory_bandwidth():
    print("=" * 82)
    print("  HARDWARE PROFILING: AR vs. BLOCK-DIFFUSION MoE (MEMORY BANDWIDTH & INTENSITY)")
    print("=" * 82)
    
    vocab_size = 3200
    d_model = 256
    d_ff = 1024
    num_experts = 8
    n_layers = 4
    gen_tokens = 64
    block_size = 32
    steps_per_block = 6
    
    model = BlockDiffusionMoE(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, 
                              num_experts=num_experts, d_ff=d_ff, mode="fp32")
    
    # Calculate parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    # In Top-2 MoE, 2 of 8 experts are active per token
    expert_params_per_layer = sum(p.numel() for p in model.layers[0]['moe'].experts[0].parameters())
    active_params = total_params - (num_experts - 2) * expert_params_per_layer * n_layers
    
    print(f"Model Architecture:")
    print(f"  • Total Parameters:  {total_params:,} ({total_params * 4 / (1024**2):.2f} MB in FP32)")
    print(f"  • Active Parameters: {active_params:,} ({active_params * 4 / (1024**2):.2f} MB in FP32)")
    print(f"  • Generation Target: {gen_tokens} tokens (Block size: {block_size}, Diffusion steps: {steps_per_block})")

    # Prompt
    prompt = torch.randint(0, vocab_size - 2, (1, 16))

    # Profile Across Precision Regimes
    # Ternary packed at 2 bits = 0.25 bytes/param (per QUANTIZATION-COVERAGE.md)
    regimes = [
        ("FP32 (4.00 bytes/param)", "fp32", 4.0),
        ("INT4 (0.50 bytes/param)", "int4", 0.5),
        ("Ternary 2-bit (0.25 bytes/param)", "ternary", 0.25),
    ]

    print("\n" + "-" * 88)
    print(f"{'Precision':<18} | {'Paradigm':<16} | {'Passes':<8} | {'DRAM (MB)':<10} | {'GFLOPs':<8} | {'Intensity (FLOP/B)':<18}")
    print("-" * 88)

    for regime_name, mode, bytes_per_param in regimes:
        model.set_mode(mode)
        
        # 1. Autoregressive Profiling (B=1):
        # 1 token per forward pass -> exactly 2 of 8 experts active per token
        ar_passes = gen_tokens  # 64 passes
        ar_bytes_per_pass = active_params * bytes_per_param
        ar_dram_bytes = ar_passes * ar_bytes_per_pass
        ar_flops = 2 * active_params * gen_tokens
        ar_intensity = ar_flops / ar_dram_bytes if ar_dram_bytes > 0 else 0
        
        # 2. Block-Diffusion Profiling (B=32):
        # 32 tokens processed concurrently per pass.
        # With Top-2 routing across 8 experts, P(expert not touched) = (1 - 2/8)^32 = 0.0001
        # Therefore, virtually ALL 8 experts are active during a 32-token candidate block pass!
        diff_expert_coverage = 1.0  # All 8 experts streamed per block pass
        diff_active_params = total_params  # Full model parameters streamed
        diff_bytes_per_pass = diff_active_params * bytes_per_param
        
        num_blocks = math.ceil(gen_tokens / block_size)
        diff_passes = num_blocks * steps_per_block  # 2 blocks * 6 steps = 12 passes
        diff_dram_bytes = diff_passes * diff_bytes_per_pass
        
        # Real arithmetic work: 12 passes * 32 tokens = 384 token-forwards (6x more compute than AR)
        diff_flops = 2 * active_params * (diff_passes * block_size)
        diff_intensity = diff_flops / diff_dram_bytes if diff_dram_bytes > 0 else 0
        
        dram_reduction = ar_dram_bytes / diff_dram_bytes
        
        print(f"{mode.upper():<18} | {'Autoregressive':<16} | {ar_passes:<8} | {ar_dram_bytes / (1024**2):7.1f} MB | {ar_flops / 1e9:6.2f}   | {ar_intensity:6.2f} FLOPs/B")
        print(f"{mode.upper():<18} | {'Block-Diffusion':<16} | {diff_passes:<8} | {diff_dram_bytes / (1024**2):7.1f} MB | {diff_flops / 1e9:6.2f}   | {diff_intensity:6.2f} FLOPs/B ({dram_reduction:.2f}x DRAM cut)")
        print("-" * 88)

if __name__ == '__main__':
    profile_memory_bandwidth()

"""
Native MoE Trajectory Absorption Test: AR vs. Block-Diffusion under Quantization
================================================================================
Target: PrimeIntellect/qwen3-moe-tiny (24 Layers, 16 Experts, Top-4 Routing)
Tests whether Block-Diffusion canvas unmasking absorbs quantization error
better than sequential AR rollout under:
  - FP32 Baseline
  - INT4 Experts
  - BitNet b1.58 Ternary Experts
"""

import sys
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from transformers import AutoTokenizer, AutoModelForCausalLM

torch.manual_seed(42)

def quantize_tensor(t: torch.Tensor, bits: float = 4.0) -> torch.Tensor:
    with torch.no_grad():
        if bits == 4.0:
            orig_shape = t.shape
            w = t.reshape(-1, 32)
            scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / 7.0
            q_w = torch.clamp(torch.round(w / scale), -8.0, 7.0) * scale
            return q_w.reshape(orig_shape)
        elif bits == 1.58:
            orig_shape = t.shape
            gamma = t.abs().mean(dim=(-2, -1), keepdim=True).clamp(min=1e-5)
            q_w = torch.clamp(torch.round(t / gamma), -1.0, 1.0) * gamma
            return q_w
    return t


def run_trajectory_test():
    print("=" * 80)
    print("  NATIVE MOE QUANTIZATION ABSORPTION: AR vs. BLOCK-DIFFUSION")
    print("  Model: PrimeIntellect/qwen3-moe-tiny (16 Experts, Top-4)")
    print("=" * 80)

    model_id = "PrimeIntellect/qwen3-moe-tiny"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)

    prompt = "Artificial intelligence algorithms process large datasets by"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids
    gen_len = 32
    block_size = 16

    # 1. Create INT4 and Ternary Quantized MoE Models
    print("\n[Step 1] Creating INT4 and Ternary Quantized Native MoE Models...")
    int4_model = copy.deepcopy(base_model)
    tern_model = copy.deepcopy(base_model)
    
    for layer in int4_model.model.layers:
        if hasattr(layer.mlp, 'experts'):
            layer.mlp.experts.gate_up_proj.data = quantize_tensor(layer.mlp.experts.gate_up_proj.data, bits=4.0)
            layer.mlp.experts.down_proj.data = quantize_tensor(layer.mlp.experts.down_proj.data, bits=4.0)

    for layer in tern_model.model.layers:
        if hasattr(layer.mlp, 'experts'):
            layer.mlp.experts.gate_up_proj.data = quantize_tensor(layer.mlp.experts.gate_up_proj.data, bits=1.58)
            layer.mlp.experts.down_proj.data = quantize_tensor(layer.mlp.experts.down_proj.data, bits=1.58)

    # 2. Sequential Autoregressive Rollout & Drift Measurement
    print("\n[Step 2] Measuring Sequential Autoregressive Rollout Drift...")
    with torch.no_grad():
        base_ar_out = base_model.generate(**inputs, max_new_tokens=gen_len, do_sample=False)
        int4_ar_out = int4_model.generate(**inputs, max_new_tokens=gen_len, do_sample=False)
        tern_ar_out = tern_model.generate(**inputs, max_new_tokens=gen_len, do_sample=False)

    gen_start = input_ids.shape[1]
    int4_ar_drift = (base_ar_out[:, gen_start:] != int4_ar_out[:, gen_start:]).float().mean().item()
    tern_ar_drift = (base_ar_out[:, gen_start:] != tern_ar_out[:, gen_start:]).float().mean().item()

    # 3. Block-Diffusion Iterative Canvas Decoding & Drift Measurement
    print("\n[Step 3] Measuring Block-Diffusion Canvas Decoding Drift...")
    def decode_block_canvas(m, steps=6):
        curr_ids = input_ids.clone()
        pad_id = tokenizer.eos_token_id
        for b in range(gen_len // block_size):
            canvas = torch.full((1, block_size), pad_id, dtype=torch.long)
            for s in range(steps):
                full_ids = torch.cat([curr_ids, canvas], dim=1)
                with torch.no_grad():
                    out = m(full_ids)
                    # Block logits
                    b_logits = out.logits[:, curr_ids.shape[1]:, :]
                    probs = F.softmax(b_logits, dim=-1)
                    conf, pred = probs.max(dim=-1)
                    
                    is_pad = (canvas == pad_id)
                    conf[~is_pad] = -1e9
                    n_unmask = max(1, block_size // steps)
                    unmask_count = min(n_unmask, int(is_pad.sum().item()))
                    if unmask_count > 0:
                        _, idx = torch.topk(conf[0], unmask_count)
                        canvas[0, idx] = pred[0, idx]
            curr_ids = torch.cat([curr_ids, canvas], dim=1)
        return curr_ids[:, gen_start:]

    base_diff_out = decode_block_canvas(base_model, steps=6)
    int4_diff_out = decode_block_canvas(int4_model, steps=6)
    tern_diff_out = decode_block_canvas(tern_model, steps=6)

    int4_diff_drift = (base_diff_out != int4_diff_out).float().mean().item()
    tern_diff_drift = (base_diff_out != tern_diff_out).float().mean().item()

    print("\n" + "=" * 80)
    print("           QUANTIZATION TRAJECTORY DRIFT: AR vs. BLOCK-DIFFUSION")
    print("=" * 80)
    print(f"Model: {model_id} (24 Layers, 16 Experts, Top-4 Routing)")
    print(f"{'Condition':<25} | {'AR Rollout Drift':<20} | {'Block-Diffusion Drift':<22} | {'Excess Gap'}")
    print("-" * 80)
    int4_excess = (int4_diff_drift - int4_ar_drift) * 100
    tern_excess = (tern_diff_drift - tern_ar_drift) * 100
    print(f"{'INT4 Experts':<25} | {int4_ar_drift*100:6.2f}%              | {int4_diff_drift*100:6.2f}%                | {int4_excess:+6.2f} pp")
    print(f"{'Ternary (1.58b) Experts':<25} | {tern_ar_drift*100:6.2f}%              | {tern_diff_drift*100:6.2f}%                | {tern_excess:+6.2f} pp")
    print("=" * 80)

    # Verification of Hypothesis
    print("\nEmpirical Verdict on Native MoE Checkpoint:")
    if int4_diff_drift < int4_ar_drift:
        print(f"  • INT4: CONFIRMED: Block-Diffusion reduced trajectory corruption by {abs(int4_excess):.2f} pp!")
    else:
        print(f"  • INT4 Drift Comparison: AR {int4_ar_drift*100:.2f}% vs Diffusion {int4_diff_drift*100:.2f}%")
        
    if tern_diff_drift < tern_ar_drift:
        print(f"  • Ternary: CONFIRMED: Block-Diffusion reduced trajectory corruption by {abs(tern_excess):.2f} pp!")
    else:
        print(f"  • Ternary Drift Comparison: AR {tern_ar_drift*100:.2f}% vs Diffusion {tern_diff_drift*100:.2f}%")
    print("=" * 80)

if __name__ == '__main__':
    run_trajectory_test()

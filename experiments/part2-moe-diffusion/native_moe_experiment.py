"""
Experiment 2: Native Small MoE Routing & Quantization Profiling
===============================================================
Target Model: PrimeIntellect/qwen3-moe-tiny (Structural Topology Testbed - Untrained Random Weights)
- 24 layers (Layer 0 dense, Layers 1-23 Qwen3MoeSparseMoeBlock)
- 16 experts per MoE layer, Top-4 active routing
- Tests:
  1. Top-4 router selection flip rate under INT4 and Ternary expert weights
  2. Router logit perturbation
  3. Output generation stability
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
            # BitNet b1.58 absmean per expert slice
            orig_shape = t.shape
            gamma = t.abs().mean(dim=(-2, -1), keepdim=True).clamp(min=1e-5)
            q_w = torch.clamp(torch.round(t / gamma), -1.0, 1.0) * gamma
            return q_w
    return t


def run_native_moe_experiment():
    print("=" * 80)
    print("  EXPERIMENT 2: Native Small MoE Routing & Quantization Profiling")
    print("  Model: PrimeIntellect/qwen3-moe-tiny (16 Experts, Top-4)")
    print("=" * 80)

    model_id = "PrimeIntellect/qwen3-moe-tiny"
    print(f"\n[Step 1] Loading tokenizer and model from local cache: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    print("  Model loaded successfully!")

    prompt = "Artificial intelligence and machine learning algorithms work by"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 1. Baseline AR Generation
    print(f"\n[Step 2] Generating Baseline Text on: \"{prompt}\"...")
    t0 = time.time()
    with torch.no_grad():
        base_out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        base_outputs = model(**inputs, output_router_logits=True)
    base_time = time.time() - t0
    base_text = tokenizer.decode(base_out[0], skip_special_tokens=True)
    base_router_logits = base_outputs.router_logits  # 23 layers of [seq_len, 16]
    print(f"  FP32 Baseline Generation (32 tokens in {base_time:.2f}s):\n    \"{base_text}\"")
    print(f"  Recorded routing logits across {len(base_router_logits)} MoE layers.")

    # 2. INT4 Quantization of Native MoE Experts
    print("\n[Step 3] Quantizing Native MoE Experts to INT4 (RTN)...")
    model_int4 = copy.deepcopy(model)
    for layer in model_int4.model.layers:
        if hasattr(layer.mlp, 'experts'):
            layer.mlp.experts.gate_up_proj.data = quantize_tensor(layer.mlp.experts.gate_up_proj.data, bits=4.0)
            layer.mlp.experts.down_proj.data = quantize_tensor(layer.mlp.experts.down_proj.data, bits=4.0)

    # 3. Ternary Quantization of Native MoE Experts
    print("\n[Step 4] Quantizing Native MoE Experts to BitNet b1.58 Ternary...")
    model_tern = copy.deepcopy(model)
    for layer in model_tern.model.layers:
        if hasattr(layer.mlp, 'experts'):
            layer.mlp.experts.gate_up_proj.data = quantize_tensor(layer.mlp.experts.gate_up_proj.data, bits=1.58)
            layer.mlp.experts.down_proj.data = quantize_tensor(layer.mlp.experts.down_proj.data, bits=1.58)

    # 4. Measure Router Flip Rate & Trajectory Drift under INT4
    print("\n[Step 5] Evaluating Router Flips & Rollout Stability under INT4...")
    with torch.no_grad():
        int4_out = model_int4.generate(**inputs, max_new_tokens=32, do_sample=False)
        int4_outputs = model_int4(**inputs, output_router_logits=True)
    int4_text = tokenizer.decode(int4_out[0], skip_special_tokens=True)
    int4_router_logits = int4_outputs.router_logits
    
    # Calculate router flip rate between base and quantized models
    def calc_flips(r_base, r_quant, top_k=4):
        total_flips = 0
        total_checks = 0
        for l in range(len(r_base)):
            # Each tensor is [seq_len, 16]
            _, base_idx = torch.topk(r_base[l], k=top_k, dim=-1)
            _, quant_idx = torch.topk(r_quant[l], k=top_k, dim=-1)
            for t in range(base_idx.shape[0]):
                s1 = set(base_idx[t].tolist())
                s2 = set(quant_idx[t].tolist())
                if s1 != s2:
                    total_flips += 1
                total_checks += 1
        return total_flips / total_checks

    int4_flips = calc_flips(base_router_logits, int4_router_logits, top_k=4)
    int4_drift = (base_out[:, inputs.input_ids.shape[1]:] != int4_out[:, inputs.input_ids.shape[1]:]).float().mean().item()

    # 5. Measure Router Flip Rate & Trajectory Drift under Ternary
    print("\n[Step 6] Evaluating Router Flips & Rollout Stability under Ternary...")
    with torch.no_grad():
        tern_out = model_tern.generate(**inputs, max_new_tokens=32, do_sample=False)
        tern_outputs = model_tern(**inputs, output_router_logits=True)
    tern_text = tokenizer.decode(tern_out[0], skip_special_tokens=True)
    tern_router_logits = tern_outputs.router_logits
    
    tern_flips = calc_flips(base_router_logits, tern_router_logits, top_k=4)
    tern_drift = (base_out[:, inputs.input_ids.shape[1]:] != tern_out[:, inputs.input_ids.shape[1]:]).float().mean().item()

    print("\n" + "=" * 80)
    print("               NATIVE PRODUCTION MOE QUANTIZATION BENCHMARK")
    print("=" * 80)
    print(f"Model: {model_id} (24 Layers, 16 Experts, Top-4 Routing)")
    print(f"{'Metric':<25} | {'INT4 Experts':<18} | {'Ternary (1.58b) Experts'}")
    print("-" * 80)
    print(f"{'Top-4 Router Flip Rate':<25} | {int4_flips*100:6.2f}%            | {tern_flips*100:6.2f}%")
    print(f"{'AR Rollout Drift (32 tok)':<25} | {int4_drift*100:6.2f}%            | {tern_drift*100:6.2f}%")
    print("=" * 80)
    print(f"\nText Outputs:")
    print(f"  • FP32 Baseline: \"{base_text}\"")
    print(f"  • INT4 Experts:  \"{int4_text}\"")
    print(f"  • Ternary Exp:   \"{tern_text}\"")
    print("=" * 80)

if __name__ == '__main__':
    run_native_moe_experiment()

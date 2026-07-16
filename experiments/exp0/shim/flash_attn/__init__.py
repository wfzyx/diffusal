"""Drop-in stand-in for the two flash_attn entry points used by third_party/mdlm,
implemented with torch.nn.functional.scaled_dot_product_attention so the code
runs on pre-Ampere GPUs (sm75). Exposed via PYTHONPATH, never installed.

Registered in configs/exp0.yaml as a platform deviation: FP16 baselines must
reproduce published perplexities before any quantized run.
"""
from . import flash_attn_interface, layers  # noqa: F401

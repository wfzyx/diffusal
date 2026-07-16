"""Weight-only round-to-nearest fake quantization for Exp 0 (configs/exp0.yaml).

Pure functions over tensors plus one in-place module walker. Fake-quant:
weights are quantized then dequantized back into the fp tensor, so quality
degradation is measured with unchanged eval code. Per the frozen config:
symmetric RTN; embeddings, the vocab head, layernorms, and biases stay fp;
the identical code path serves both the dLLM and the AR control.
"""
import re

import torch

EXCLUDE_PATTERN = re.compile(r'(^|\.)(vocab_embed|output_layer|lm_head|embedding)(\.|$)')


def rtn_int8(w):
  """Symmetric per-output-channel absmax, levels [-127, 127]."""
  scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
  return (w / scale).round().clamp(-127, 127) * scale


def rtn_int4_g128(w, group_size=128):
  """Symmetric absmax per (output channel x input group of 128), levels [-7, 7]."""
  out_dim, in_dim = w.shape
  pad = (-in_dim) % group_size
  if pad:
    w = torch.nn.functional.pad(w, (0, pad))
  g = w.view(out_dim, -1, group_size)
  scale = g.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / 7.0
  q = (g / scale).round().clamp(-7, 7) * scale
  return q.view(out_dim, -1)[:, :in_dim]


def rtn_ternary(w):
  """BitNet b1.58 absmean: per-tensor scale = mean|W|, levels {-1, 0, +1}."""
  scale = w.abs().mean().clamp(min=1e-8)
  return (w / scale).round().clamp(-1, 1) * scale


SCHEMES = {'int8': rtn_int8, 'int4': rtn_int4_g128, 'ternary': rtn_ternary}


@torch.no_grad()
def quantize_module_(module, scheme):
  """Fake-quantize every nn.Linear weight in `module` in place, except heads/
  embeddings (EXCLUDE_PATTERN). Biases untouched. Returns an audit list of
  (qualified_name, shape, quantized_fraction_of_params)."""
  fn = SCHEMES[scheme]
  report = []
  total, quantized = 0, 0
  for name, mod in module.named_modules():
    if not isinstance(mod, torch.nn.Linear):
      continue
    n = mod.weight.numel()
    total += n
    if EXCLUDE_PATTERN.search(name):
      report.append((name, tuple(mod.weight.shape), 'EXCLUDED'))
      continue
    mod.weight.copy_(fn(mod.weight.float()).to(mod.weight.dtype))
    quantized += n
    report.append((name, tuple(mod.weight.shape), scheme))
  report.append(('__summary__', (quantized, total),
                 f'{quantized/max(total,1):.1%} of linear params quantized'))
  return report

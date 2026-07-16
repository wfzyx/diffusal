"""Ternary quantization-aware training via straight-through estimator.

Implemented as a torch parametrization over nn.Linear weights, so the
unmodified bd3lms model code trains with ternary forward weights while
gradients flow to the latent fp weights. The identical wrapper serves the
dLLM and the AR control (configs/exp2-pilot.yaml).

BitNet b1.58 style: per-tensor absmean scale, levels {-1, 0, +1}.
Weights only — activations stay fp (isolates the variable Exp 0 measured).
Excluded modules follow the same pattern as Exp 0 (embeddings, vocab head).
"""
import re

import torch
import torch.nn.utils.parametrize as parametrize

EXCLUDE_PATTERN = re.compile(r'(^|\.)(vocab_embed|output_layer|lm_head|embedding)(\.|$)')


class TernarySTE(torch.nn.Module):
  """weight -> ternarized(weight) in forward; identity gradient in backward."""

  def forward(self, w):
    scale = w.abs().mean().clamp(min=1e-8)
    q = (w / scale).round().clamp(-1, 1) * scale
    return w + (q - w).detach()


def ternarize_for_training_(module):
  """Register the STE parametrization on every non-excluded nn.Linear.

  Returns an audit list of (name, shape, status).
  """
  report = []
  for name, mod in module.named_modules():
    if not isinstance(mod, torch.nn.Linear):
      continue
    if EXCLUDE_PATTERN.search(name):
      report.append((name, tuple(mod.weight.shape), 'EXCLUDED'))
      continue
    parametrize.register_parametrization(mod, 'weight', TernarySTE())
    report.append((name, tuple(mod.weight.shape), 'ternary-qat'))
  return report

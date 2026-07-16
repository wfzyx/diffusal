"""Sanity checks for quantize.py (CPU, no model needed)."""
import torch

from quantize import rtn_int8, rtn_int4_g128, rtn_ternary, quantize_module_

torch.manual_seed(0)


def levels(w, q, per='row'):
  """Distinct quantized levels per scale group must respect the bit width."""
  if per == 'row':
    return max(torch.unique(qr).numel() for qr in q)
  return torch.unique(q).numel()


def test_int8():
  w = torch.randn(16, 64)
  q = rtn_int8(w)
  assert q.shape == w.shape
  assert (q - w).abs().max() < w.abs().max() / 127 + 1e-6, 'error exceeds one step'
  scale = w.abs().amax(1, keepdim=True) / 127
  assert torch.allclose((q / scale).round(), q / scale, atol=1e-4), 'not on grid'
  print('int8: OK')


def test_int4():
  w = torch.randn(8, 300)                       # non-multiple of 128 exercises padding
  q = rtn_int4_g128(w)
  assert q.shape == w.shape
  g = torch.nn.functional.pad(w, (0, 84)).view(8, -1, 128)
  qg = torch.nn.functional.pad(q, (0, 84)).view(8, -1, 128)
  scale = g.abs().amax(2, keepdim=True) / 7
  ints = qg / scale
  assert torch.allclose(ints.round(), ints, atol=1e-4), 'not on int4 grid'
  assert ints.abs().max() <= 7 + 1e-4, 'exceeds int4 range'
  print('int4-g128: OK')


def test_ternary():
  w = torch.randn(32, 32)
  q = rtn_ternary(w)
  s = w.abs().mean()
  assert set(torch.unique((q / s).round()).tolist()) <= {-1.0, 0.0, 1.0}
  assert (q != 0).any() and (q == 0).any(), 'degenerate ternarization'
  print('ternary: OK')


def test_module_walker():
  class Toy(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.attn_qkv = torch.nn.Linear(8, 24)
      self.output_layer = torch.nn.Sequential(torch.nn.Linear(8, 100))
      self.norm = torch.nn.LayerNorm(8)

  m = Toy()
  head_before = m.output_layer[0].weight.clone()
  bias_before = m.attn_qkv.bias.clone()
  report = quantize_module_(m, 'ternary')
  assert torch.equal(m.output_layer[0].weight, head_before), 'head must stay fp'
  assert torch.equal(m.attn_qkv.bias, bias_before), 'bias must stay fp'
  statuses = {name: s for name, _, s in report}
  assert statuses['attn_qkv'] == 'ternary'
  assert statuses['output_layer.0'] == 'EXCLUDED'
  print('module walker: OK')


if __name__ == '__main__':
  test_int8()
  test_int4()
  test_ternary()
  test_module_walker()
  print('quantize sanity: ALL OK')

"""CPU sanity checks for the ternary-QAT parametrization."""
import torch

from qat import TernarySTE, ternarize_for_training_


def test_forward_is_ternary():
  w = torch.randn(32, 16)
  q = TernarySTE()(w)
  s = w.abs().mean()
  assert set(torch.unique((q / s).round()).tolist()) <= {-1.0, 0.0, 1.0}
  print('forward ternary: OK')


def test_ste_gradient_is_identity():
  w = torch.randn(8, 8, requires_grad=True)
  TernarySTE()(w).sum().backward()
  assert torch.allclose(w.grad, torch.ones_like(w)), 'STE must pass gradient through'
  print('STE gradient: OK')


def test_training_moves_latent_weights():
  m = torch.nn.Linear(16, 16)
  ternarize_for_training_(torch.nn.ModuleDict({'attn_qkv': m}))
  latent_before = m.parametrizations.weight.original.clone()
  opt = torch.optim.SGD(m.parameters(), lr=0.1)
  x = torch.randn(4, 16)
  for _ in range(5):
    opt.zero_grad()
    (m(x) ** 2).mean().backward()
    opt.step()
  latent = m.parametrizations.weight.original
  assert not torch.equal(latent, latent_before), 'latent weights must update'
  s = latent.abs().mean()
  assert set(torch.unique((m.weight / s.clamp(min=1e-8)).round()).tolist()) \
    <= {-1.0, 0.0, 1.0}, 'effective weight must stay ternary'
  print('training loop: OK')


def test_exclusion():
  mod = torch.nn.ModuleDict({
    'attn_out': torch.nn.Linear(8, 8),
    'output_layer': torch.nn.ModuleDict({'linear': torch.nn.Linear(8, 100)}),
  })
  report = ternarize_for_training_(mod)
  statuses = {n: s for n, _, s in report}
  assert statuses['attn_out'] == 'ternary-qat'
  assert statuses['output_layer.linear'] == 'EXCLUDED'
  print('exclusion: OK')


if __name__ == '__main__':
  test_forward_is_ternary()
  test_ste_gradient_is_identity()
  test_training_moves_latent_weights()
  test_exclusion()
  print('qat sanity: ALL OK')

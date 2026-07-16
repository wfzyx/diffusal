"""Numerical sanity checks for the flash_attn shim (run before any model eval).

Validates on CPU in fp32 against straightforward reference implementations:
1. rotary: matches applying dit.py's own cos/sin math to q and k.
2. attention: matches explicit softmax(QK^T/sqrt(d))V, causal and non-causal.
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shim'))

import torch
import flash_attn
import flash_attn.layers.rotary

torch.manual_seed(0)
B, S, H, D = 2, 16, 4, 32


def ref_rotary(x, cos, sin):
  # non-interleaved rotary, full head dim; cos/sin: (S, D/2)
  cos2 = torch.cat((cos, cos), dim=-1)[None, :, None, :]
  sin2 = torch.cat((sin, sin), dim=-1)[None, :, None, :]
  x1, x2 = x.chunk(2, dim=-1)
  return x * cos2 + torch.cat((-x2, x1), dim=-1) * sin2


def test_rotary():
  inv_freq = 1.0 / (10_000 ** (torch.arange(0, D, 2).float() / D))
  t = torch.arange(S).float()
  freqs = torch.einsum('i,j->ij', t, inv_freq)
  cos, sin = freqs.cos(), freqs.sin()          # (S, D/2)

  qkv = torch.randn(B, S, 3, H, D)
  expect_q = ref_rotary(qkv[:, :, 0], cos, sin)
  expect_k = ref_rotary(qkv[:, :, 1], cos, sin)
  expect_v = qkv[:, :, 2].clone()

  out = flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv.clone(), cos, sin)
  assert torch.allclose(out[:, :, 0], expect_q, atol=1e-6), 'q rotary mismatch'
  assert torch.allclose(out[:, :, 1], expect_k, atol=1e-6), 'k rotary mismatch'
  assert torch.allclose(out[:, :, 2], expect_v), 'v must pass through untouched'
  print('rotary: OK')


def ref_attention(qkv_bshd, causal):
  q, k, v = qkv_bshd.permute(2, 0, 3, 1, 4).unbind(0)     # (B, H, S, D)
  scores = q @ k.transpose(-1, -2) / math.sqrt(D)
  if causal:
    scores = scores.masked_fill(
      torch.triu(torch.ones(S, S, dtype=torch.bool), 1), float('-inf'))
  return (scores.softmax(-1) @ v).permute(0, 2, 1, 3).reshape(B * S, H, D)


def test_attention():
  qkv = torch.randn(B, S, 3, H, D)
  cu = torch.arange(0, (B + 1) * S, S, dtype=torch.int32)
  packed = qkv.reshape(B * S, 3, H, D)
  for causal in (False, True):
    out = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
      packed, cu, S, 0.0, causal=causal)
    assert torch.allclose(out, ref_attention(qkv, causal), atol=1e-5), \
      f'attention mismatch (causal={causal})'
  print('attention (causal and non-causal): OK')


if __name__ == '__main__':
  test_rotary()
  test_attention()
  print('shim sanity: ALL OK')

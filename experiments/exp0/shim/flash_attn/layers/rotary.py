import torch


def _rotate_half(x):
  x1, x2 = x.chunk(2, dim=-1)
  return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb_qkv_(qkv, cos, sin, interleaved=False):
  """Non-interleaved rotary on packed qkv, matching flash_attn semantics.

  qkv: (batch, seqlen, 3, n_heads, head_dim)
  cos/sin: (seqlen, rotary_dim / 2) — mdlm/bd3lms pass rotary_dim == head_dim.
  Rotary applies to q and k only; v passes through.

  Unlike the real kernel (in-place with a custom autograd backward), this
  builds a NEW tensor: every caller in mdlm/bd3lms uses the return value, and
  a functional implementation keeps autograd correct during QAT training.
  """
  assert not interleaved, 'shim implements the non-interleaved layout only'
  rotary_dim = cos.shape[-1] * 2
  assert rotary_dim == qkv.shape[-1], 'shim assumes full-head-dim rotary'
  cos = torch.cat((cos, cos), dim=-1)[None, :, None, None, :].to(qkv.dtype)
  sin = torch.cat((sin, sin), dim=-1)[None, :, None, None, :].to(qkv.dtype)
  qk = qkv[:, :, :2]
  qk = qk * cos + _rotate_half(qk) * sin
  return torch.cat([qk, qkv[:, :, 2:]], dim=2)

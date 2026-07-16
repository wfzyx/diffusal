import torch


def _rotate_half(x):
  x1, x2 = x.chunk(2, dim=-1)
  return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb_qkv_(qkv, cos, sin, interleaved=False):
  """Non-interleaved rotary on packed qkv, matching flash_attn semantics.

  qkv: (batch, seqlen, 3, n_heads, head_dim)
  cos/sin: (seqlen, rotary_dim / 2) — mdlm passes rotary_dim == head_dim.
  Rotary is applied to q and k only; v passes through. flash_attn mutates
  in place; doing the same keeps callers that rely on either behavior correct.
  """
  assert not interleaved, 'shim implements the non-interleaved layout only'
  rotary_dim = cos.shape[-1] * 2
  cos = torch.cat((cos, cos), dim=-1)[None, :, None, :].to(qkv.dtype)  # (1, s, 1, ro)
  sin = torch.cat((sin, sin), dim=-1)[None, :, None, :].to(qkv.dtype)
  qk = qkv[:, :, :2, :, :rotary_dim]
  qkv[:, :, :2, :, :rotary_dim] = (
    qk * cos.unsqueeze(2) + _rotate_half(qk) * sin.unsqueeze(2))
  return qkv

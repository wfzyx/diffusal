import torch
import torch.nn.functional as F


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen,
                                     dropout_p=0.0, softmax_scale=None,
                                     causal=False):
  """SDPA-backed replacement for the varlen packed-QKV kernel.

  mdlm only ever calls this with uniform sequence lengths (seqlens=None in
  DDiTBlock.forward / autoregressive.py), so cu_seqlens is always the uniform
  arange grid and the varlen problem reduces to a plain batched one.
  """
  total, three, n_heads, head_dim = qkv.shape
  assert three == 3
  batch_size = cu_seqlens.numel() - 1
  seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
  if not bool((seqlens == max_seqlen).all()):
    raise NotImplementedError(
      'shim only supports uniform sequence lengths; got ' + str(seqlens.tolist()))

  # (b*s, 3, h, d) -> 3 x (b, h, s, d)
  q, k, v = qkv.view(batch_size, max_seqlen, 3, n_heads, head_dim) \
               .permute(2, 0, 3, 1, 4).unbind(0)
  out = F.scaled_dot_product_attention(
    q, k, v, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)
  # (b, h, s, d) -> (b*s, h, d)
  return out.permute(0, 2, 1, 3).reshape(total, n_heads, head_dim)

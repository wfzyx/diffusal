"""One-time sanitizer for the MDLM release checkpoints (user-approved trusted load).

Loads the original pickle-bearing lightning checkpoint with weights_only=False,
keeps only the keys diffusion.Diffusion actually needs at eval time
(state_dict, ema, loops progress counters, version/step metadata) as plain
dicts/tensors, and re-saves so every future load passes torch's safe
weights-only unpickler. Optimizer state is dropped (~4x size reduction).

Usage: .venv/bin/python sanitize_ckpt.py checkpoints/ar.ckpt [checkpoints/mdlm.ckpt ...]
"""
import sys
import torch


def plain(obj):
  """Recursively convert containers to plain dict/list; keep tensors/scalars."""
  if isinstance(obj, dict) or type(obj).__name__ in ('DictConfig',):
    return {k: plain(v) for k, v in dict(obj).items()}
  if isinstance(obj, (list, tuple)) or type(obj).__name__ in ('ListConfig',):
    return [plain(v) for v in obj]
  return obj


def sanitize(path):
  ckpt = torch.load(path, map_location='cpu', weights_only=False)
  clean = {
    'state_dict': ckpt['state_dict'],
    'pytorch-lightning_version': str(ckpt.get('pytorch-lightning_version', '2.2.1')),
    'epoch': int(ckpt.get('epoch', 0)),
    'global_step': int(ckpt.get('global_step', 0)),
  }
  if 'ema' in ckpt:                      # EMA weights: what published evals use
    clean['ema'] = plain(ckpt['ema'])
  if 'loops' in ckpt:                    # on_load_checkpoint reads fit_loop progress
    clean['loops'] = plain(ckpt['loops'])
  out = path.replace('.ckpt', '-clean.ckpt')
  torch.save(clean, out)
  reloaded = torch.load(out, weights_only=True, map_location='cpu')
  assert set(clean) == set(reloaded)
  print(f'{path} -> {out}: {len(reloaded["state_dict"])} tensors, '
        f'ema={"ema" in reloaded}, step {reloaded["global_step"]}')


if __name__ == '__main__':
  for p in sys.argv[1:]:
    sanitize(p)

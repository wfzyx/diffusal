"""Exp 0 quantized-eval driver (configs/exp0.yaml, frozen prereg-exp0-2026-07-16).

Runs the unmodified bd3lms ppl_eval, but patches Diffusion.on_validation_epoch_start
to fake-quantize the backbone IN PLACE right after EMA weights are swapped in —
so the quantizer sees exactly the weights that would otherwise be evaluated.
The identical code path serves the dLLM and the AR control.

Usage:
  run_quant_eval.py <ar|dllm> <fp16|int8|int4|ternary> <dataset> [extra hydra overrides...]

Writes the quantization audit (module list + peak VRAM) to stderr; metrics are
printed by the harness as usual.
"""
import os
import sys

EXP0 = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(EXP0))
BD3LMS = os.path.join(ROOT, 'third_party', 'bd3lms')
sys.path.insert(0, BD3LMS)
sys.path.insert(0, EXP0)          # for quantize.py
sys.path.insert(0, os.path.join(EXP0, 'shim'))

model, precision, dataset, *extra = sys.argv[1:]
assert model in ('ar', 'dllm') and precision in ('fp16', 'int8', 'int4', 'ternary')

import torch  # noqa: E402
import quantize  # noqa: E402
import diffusion  # noqa: E402

_orig = diffusion.Diffusion.on_validation_epoch_start


def patched(self):
  _orig(self)                      # applies EMA weights to the backbone
  if precision != 'fp16':
    report = quantize.quantize_module_(self.backbone, precision)
    for name, shape, status in report:
      print(f'[quant] {name} {shape} {status}', file=sys.stderr)
  torch.cuda.reset_peak_memory_stats()


diffusion.Diffusion.on_validation_epoch_start = patched

_orig_end = diffusion.Diffusion.on_validation_epoch_end


def patched_end(self):
  peak = torch.cuda.max_memory_allocated() / 2**20
  print(f'[vram] peak allocated during validation: {peak:.0f} MiB', file=sys.stderr)
  _orig_end(self)


diffusion.Diffusion.on_validation_epoch_end = patched_end

if model == 'ar':
  model_args = ['algo=ar',
                f'eval.checkpoint_path={EXP0}/checkpoints/ar-clean.ckpt']
else:
  model_args = ['algo=mdlm', 'algo.backbone=hf_dit',
                'eval.checkpoint_path=kuleshov-group/mdlm-owt']

sys.argv = ['main.py', 'mode=ppl_eval', 'loader.eval_batch_size=2',
            'model=small', f'data={dataset}',
            f'data.cache_dir={os.path.expanduser("~/.cache/mdlm_data")}',
            '+data.insert_valid_eos=False', 'model.length=1024',
            'wandb=null', *model_args, *extra]

os.environ.setdefault('WANDB_MODE', 'offline')
os.chdir(BD3LMS)
# Execute as a script so hydra resolves config_path relative to main.py;
# the diffusion module is already imported (and patched) above, so the
# monkeypatch persists into this run.
import runpy  # noqa: E402

sys.argv[0] = os.path.join(BD3LMS, 'main.py')
runpy.run_path(sys.argv[0], run_name='__main__')

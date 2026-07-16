"""Exp 0 quantized-eval driver (configs/exp0.yaml, frozen prereg-exp0-2026-07-16).

Runs the unmodified bd3lms ppl_eval, but patches Diffusion.on_validation_epoch_start
to fake-quantize the backbone IN PLACE right after EMA weights are swapped in —
so the quantizer sees exactly the weights that would otherwise be evaluated.
The identical code path serves the dLLM and the AR control.

Usage:
  run_quant_eval.py <ar|dllm> <fp16|int8|int4|ternary> <dataset|gen> [extra hydra overrides...]

The third argument selects likelihood eval on a dataset, or 'gen' for the
generative anchor: unconditional sampling (nucleus 0.9, per the bd3lms
gen_ppl protocol) scored with generative perplexity under gpt2-large.
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
gen_mode = dataset == 'gen'

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

# Generative-anchor hook: restore_model_and_sample copies EMA weights in,
# then calls _sample; quantizing at first _sample entry therefore hits
# exactly the weights that generate.
_orig_sample = diffusion.Diffusion._sample
_quantized_for_sampling = []


def patched_sample(self, *args, **kwargs):
  if precision != 'fp16' and not _quantized_for_sampling:
    report = quantize.quantize_module_(self.backbone, precision)
    print(f'[quant] {report[-1]}', file=sys.stderr)
    _quantized_for_sampling.append(True)
  return _orig_sample(self, *args, **kwargs)


diffusion.Diffusion._sample = patched_sample

if model == 'ar':
  model_args = ['algo=ar',
                f'eval.checkpoint_path={EXP0}/checkpoints/ar-clean.ckpt']
else:
  model_args = ['algo=mdlm', 'algo.backbone=hf_dit',
                'eval.checkpoint_path=kuleshov-group/mdlm-owt']

if gen_mode:
  # inline oracle kept small (gpt2) so sampler + oracle fit in 8 GB;
  # the official pre-registered oracle (gpt2-large) scores the saved
  # samples offline via score_samples.py.
  mode_args = ['mode=sample_eval', 'loader.eval_batch_size=1',
               'eval.gen_ppl_eval_model_name_or_path=gpt2',
               'sampling.num_sample_batches=64', 'sampling.nucleus_p=0.9',
               f'sampling.logdir={EXP0}/results/anchor/samples_{model}_{precision}',
               'seed=1'] + (['algo.T=1000'] if model == 'dllm' else [])
else:
  mode_args = ['mode=ppl_eval', 'loader.eval_batch_size=2',
               '+data.insert_valid_eos=False']

sys.argv = ['main.py', 'model=small',
            f'data={"wikitext103" if gen_mode else dataset}',
            f'data.cache_dir={os.path.expanduser("~/.cache/mdlm_data")}',
            'model.length=1024', 'wandb=null',
            *mode_args, *model_args, *extra]

os.environ.setdefault('WANDB_MODE', 'offline')
os.chdir(BD3LMS)
# Execute as a script so hydra resolves config_path relative to main.py;
# the diffusion module is already imported (and patched) above, so the
# monkeypatch persists into this run.
import runpy  # noqa: E402

sys.argv[0] = os.path.join(BD3LMS, 'main.py')
runpy.run_path(sys.argv[0], run_name='__main__')

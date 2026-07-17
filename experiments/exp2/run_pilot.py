"""Exp 2 pilot driver: train tiny {AR, dLLM} x {fp16, ternary-QAT} on wikitext103.

Same runpy pattern as exp0/run_quant_eval.py: the unmodified bd3lms trainer
runs; for ternary runs, Diffusion.__init__ is wrapped to register the STE
parametrization on the backbone right after construction (EMA keeps tracking
the latent fp weights — the same Parameter objects — so validation applies
EMA then ternarizes in forward, which is the correct QAT eval semantics).

Usage: run_pilot.py <ar|dllm> <fp16|ternary> [extra hydra overrides...]
"""
import os
import sys

EXP2 = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(EXP2))
BD3LMS = os.path.join(ROOT, 'third_party', 'bd3lms')
EXP0 = os.path.join(ROOT, 'experiments', 'exp0')
sys.path.insert(0, BD3LMS)
sys.path.insert(0, EXP2)                       # for qat.py
sys.path.insert(0, os.path.join(EXP0, 'shim'))

model, precision, *extra = sys.argv[1:]
assert model in ('ar', 'dllm') and precision in ('fp16', 'ternary')

import torch  # noqa: E402

# Resuming loads OUR OWN lightning checkpoints (written by this driver into
# experiments/exp2/runs/). Their hyper_parameters embed the hydra config as
# omegaconf containers, which torch's restricted unpickler structurally cannot
# rebuild (defaultdict SETITEM is unsupported even when allowlisted). Full
# loading is therefore enabled STRICTLY for files under our own runs dir —
# provenance is this driver itself; all other torch.load calls keep the safe
# weights-only default.
_OWN_RUNS_DIR = os.path.join(EXP2, 'runs') + os.sep
_orig_torch_load = torch.load


def _torch_load_own_ckpts(f, *args, **kwargs):
  # lightning opens checkpoints through fsspec, so f may be a file object;
  # its .name carries the path.
  if isinstance(f, (str, os.PathLike)):
    path = os.fspath(f)
  else:
    path = str(getattr(f, 'name', '') or getattr(f, 'path', '') or '')
  if path and os.path.abspath(path).startswith(_OWN_RUNS_DIR):
    kwargs['weights_only'] = False
  return _orig_torch_load(f, *args, **kwargs)


torch.load = _torch_load_own_ckpts

import diffusion  # noqa: E402
import qat  # noqa: E402

# NOTE: wrapping __init__ breaks lightning's save_hyperparameters() frame
# introspection; the setup() hook runs once per fit stage after construction,
# before any forward. EMA (created in __init__) keeps tracking the same latent
# Parameter objects, so EMA-then-ternarize eval semantics are preserved.
_orig_setup = diffusion.Diffusion.setup


def patched_setup(self, stage=None):
  _orig_setup(self, stage)
  if precision == 'ternary' and not getattr(self, '_qat_applied', False):
    self._qat_applied = True
    report = qat.ternarize_for_training_(self.backbone)
    for name, shape, status in report:
      print(f'[qat] {name} {shape} {status}', file=sys.stderr)
    if self.ema is not None:
      # registering parametrizations changes parameters() iteration order,
      # which would misalign the EMA shadow list built in __init__; rebuild
      # it (same as __init__: shadow = current init weights).
      self.ema = type(self.ema)(
        self._get_parameters(), decay=self.config.training.ema)


diffusion.Diffusion.setup = patched_setup

run_dir = os.path.join(EXP2, 'runs', f'{model}_{precision}')
os.makedirs(run_dir, exist_ok=True)

algo_args = ['algo=ar'] if model == 'ar' else ['algo=mdlm']

# Budget per configs/exp2-pilot.yaml: 6000 steps x 64 x 512 tokens ~= 196M tokens.
sys.argv = ['main.py', 'mode=train', 'model=tiny', 'model.length=512',
            'data=wikitext103',
            f'data.cache_dir={os.path.expanduser("~/.cache/mdlm_data")}',
            'loader.global_batch_size=64', 'loader.batch_size=2',
            'loader.eval_batch_size=2', 'loader.num_workers=4',
            'trainer.max_steps=6000', 'trainer.precision=16-mixed',
            'trainer.val_check_interval=1000',
            'trainer.limit_val_batches=64',
            # ~730 MB/ckpt at 500-step cadence filled the disk (100%) and
            # killed the first grid pass; 2000-step cadence still bounds
            # resume loss to ~1h of compute.
            'callbacks.checkpoint_every_n_steps.every_n_train_steps=2000',
            'optim.lr=3e-4', 'lr_scheduler.num_warmup_steps=250',
            f'checkpointing.save_dir={run_dir}',
            'wandb=null', 'seed=1', *algo_args, *extra]

os.environ.setdefault('WANDB_MODE', 'offline')
os.chdir(BD3LMS)
import runpy  # noqa: E402

sys.argv[0] = os.path.join(BD3LMS, 'main.py')
runpy.run_path(sys.argv[0], run_name='__main__')

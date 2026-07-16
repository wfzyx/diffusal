# Exp 0 preparation status

Pre-registration config: [`configs/exp0.yaml`](../../configs/exp0.yaml) (DRAFT until tagged).

## Platform facts (verified 2026-07-16)

- GPU: RTX 2080 Super, 8 GB, **sm75 (Turing)** — sufficient for 130M-param models in fp16.
- MDLM codebase cloned at `third_party/mdlm` (kuleshov-group/mdlm, shallow).
- dLLM checkpoint: `kuleshov-group/mdlm-owt` on HF (not yet downloaded).
- AR control: `ar.ckpt` from the MDLM release Google Drive folder (needs `gdown`; not yet downloaded).

## Known blockers and their fixes

1. **flash-attn incompatibility.** `models/dit.py`, `models/autoregressive.py`, and
   `dataloader.py` hard-import `flash_attn`; FlashAttention-2 requires sm80+ (Ampere),
   the 2080 Super is sm75. Fix: patch attention to `torch.nn.functional.scaled_dot_product_attention`
   and replace `flash_attn.layers.rotary` with a plain-torch rotary embedding.
   Applied identically to both models; FP16 baselines must reproduce published
   perplexities within the tolerance in exp0.yaml before the prereg tag is cut.
2. **Disk.** Single 222 GB disk at 98% (~6 GB free). Needed: ~6 GB (uv env with torch)
   + ~1 GB (checkpoints) + ~3 GB (gpt2-large oracle) + <1 GB (wikitext103/ptb/lambada).
   ~22 GB of stale models sit in `~/.cache/huggingface/hub` (FastContext-1.0-4B-RL,
   Qwen2.5-Coder-3B, Qwen2.5-3B-Instruct, Qwen2.5-1.5B) — clearing them is the plan.
3. **OpenWebText eval dropped.** The natural in-distribution eval set is a ~55 GB
   download; infeasible here. exp0.yaml pins the small zero-shot sets instead
   (wikitext103 primary). Only relative deltas are compared, so this is acceptable
   and is recorded as a deviation in the config.
4. **Environment.** Repo ships a conda `requirements.yaml` (python 3.9, torch+cu121,
   lightning 2.2.1, hydra). Plan: `uv venv` + pip-install the subset needed for eval
   (skip mamba-ssm/causal-conv1d unless the checkpoint config requires them; MDLM-owt
   is a DiT, not Mamba).

## Environment pins discovered during bring-up (2026-07-16)

- `setuptools<81` (83 removed pkg_resources; lightning 2.2.1 needs it)
- `transformers==4.38.2` (mdlm-owt remote code breaks on modern transformers)
- `numpy<2` (datasets 2.18 torch formatter uses `np.array(..., copy=False)`)
- `wandb` required even offline (`WANDB_MODE=offline`); `data.cache_dir` must
  override the authors' hard-coded cluster path
- batch size 2 max: fp32 logits over the 50k vocab at length 1024 OOM an 8 GB
  card at batch 4 — the §5 "logits dominate" effect, visible already at 130M
- shim validated: rotary + SDPA match references numerically (test_shim.py);
  **FP16 mdlm-owt wikitext103 ppl = 34.03 vs ~32.8 published (+3.7%, within
  the 5% tolerance)**

## Platform switch: mdlm repo -> bd3lms (2026-07-16)

The AR control failed to reproduce Table 3 through the mdlm repo (wikitext103
34.82 vs 25.75 published; ptb 148 vs 82; lambada fine). Root cause: the paper's
zero-shot protocol evaluates with `insert_valid_eos=False` (no EOS wrapped
between validation documents), a dataloader feature that exists only in the
authors' bd3lms repo — and the authors themselves direct Table 3 reproduction
there (mdlm issue #22). With bd3lms + `+data.insert_valid_eos=False`, AR
wikitext103 = 25.61 vs 25.75 published (-0.5%). Diagnostics that were ruled
out along the way: EMA weights (applied correctly; raw weights slightly worse),
the SDPA shim (identical code path reproduced the dLLM), input/target shift
(correct in source).

Additional bd3lms bring-up facts:
- lightning upgraded 2.2.1 -> 2.6.5: lightning 2.2.1 monkeypatches
  `torch.compile` with a wrapper that breaks decorator-form compile under
  torch 2.13 (bd3lms models/dit.py:76). Fixed upstream in newer lightning.
- the lightning upgrade drags numpy back to 2.x; re-pin `numpy<2` after.
- checkpoints sanitized to weights-only via sanitize_ckpt.py (keeps
  state_dict + ema + loops; drops optimizer state and pickled DictConfig).
- run_fp16_baselines.sh is superseded by the bd3lms invocation (results in
  results/fp16-bd3lms/); kept only as a record of the mdlm-repo attempt.

## Order of work

1. Free disk (user-approved cache cleanup).
2. `uv` env; install torch + eval deps.
3. Patch flash-attn → SDPA; run FP16 likelihood eval for `mdlm-owt` on wikitext103;
   compare against published numbers.
4. Download `ar.ckpt`; same FP16 reproduction for the AR control.
5. If both reproduce within tolerance → freeze `configs/exp0.yaml` (fill final
   sampler/step defaults from the repo config), commit, tag `prereg-exp0-<date>`.
6. Only then: implement the RTN quantizer (int8/int4/ternary) and run the grid.

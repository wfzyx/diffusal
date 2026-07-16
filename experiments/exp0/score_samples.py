"""Offline generative-anchor scorer (configs/exp0.yaml): generative perplexity
under gpt2-large plus unigram entropy, computed from the sample CSVs written by
run_quant_eval.py gen mode (results/anchor/samples_<model>_<precision>/*.csv).

Usage: score_samples.py <csv-or-dir> [more...]
"""
import csv
import math
import os
import sys
from collections import Counter

import torch
import transformers

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_samples(path):
  files = []
  if os.path.isdir(path):
    files = [os.path.join(path, f) for f in sorted(os.listdir(path))
             if f.endswith('.csv')]
  else:
    files = [path]
  texts = []
  csv.field_size_limit(10**8)
  for f in files:
    with open(f) as fh:
      for row in csv.reader(fh):
        # headerless CSV from bd3lms utils.update_and_save_csv:
        # gen_ppl, nfes, entropy, length, samples(list-as-string), seed
        if not row or row[0].startswith('gen_ppl'):
          continue
        s = row[-2].strip().strip('[]').strip('"\' ')
        if s:
          texts.append(s)
  return texts


@torch.no_grad()
def gen_ppl(texts, tokenizer, oracle, ctx=1024):
  total_nll, total_tokens = 0.0, 0
  for t in texts:
    ids = tokenizer(t, return_tensors='pt').input_ids[:, :ctx].to(DEVICE)
    if ids.numel() < 2:
      continue
    out = oracle(ids, labels=ids)
    n = ids.numel() - 1
    total_nll += out.loss.item() * n
    total_tokens += n
  return math.exp(total_nll / max(total_tokens, 1)), total_tokens


def unigram_entropy(texts, tokenizer):
  counts = Counter()
  for t in texts:
    counts.update(tokenizer(t).input_ids)
  n = sum(counts.values())
  return -sum((c / n) * math.log2(c / n) for c in counts.values())


if __name__ == '__main__':
  tok = transformers.AutoTokenizer.from_pretrained('gpt2-large')
  oracle = transformers.AutoModelForCausalLM.from_pretrained(
    'gpt2-large', torch_dtype=torch.float16).to(DEVICE).eval()
  for path in sys.argv[1:]:
    texts = load_samples(path)
    if not texts:
      print(f'{path}: NO SAMPLES')
      continue
    ppl, ntok = gen_ppl(texts, tok, oracle)
    ent = unigram_entropy(texts, tok)
    print(f'{os.path.basename(path.rstrip("/"))}: n={len(texts)} '
          f'gen_ppl(gpt2-large)={ppl:.2f} unigram_entropy={ent:.2f} bits '
          f'({ntok} tokens scored)')

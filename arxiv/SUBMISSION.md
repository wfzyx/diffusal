# arXiv submission — checklist and metadata

Source of truth for content: [../ARXIV-PAPER.md](../ARXIV-PAPER.md). This directory holds the LaTeX build that goes to arXiv.

## Files in this submission

- `diffusal-arxiv.tex` — the paper (self-contained: manual `thebibliography`, no external `.bib`, no figures).

That single `.tex` is the entire upload. No `.bbl` needed (bibliography is inline), no image files, no style files beyond arXiv's stock TeX Live.

## Compile locally before uploading (recommended)

No TeX engine is installed on this machine. Compile with any of:

```bash
# TeX Live (standard)
cd arxiv && pdflatex diffusal-arxiv.tex && pdflatex diffusal-arxiv.tex   # run twice for refs

# or tectonic (single pass, self-fetches packages)
cd arxiv && tectonic diffusal-arxiv.tex

# or via docker if nothing is installed locally
docker run --rm -v "$PWD":/work -w /work texlive/texlive:latest \
  sh -c 'cd arxiv && pdflatex -interaction=nonstopmode diffusal-arxiv.tex && pdflatex -interaction=nonstopmode diffusal-arxiv.tex'
```

Confirm the PDF renders both tables and the abstract with no overfull-box disasters, then upload the `.tex` to arXiv (arXiv recompiles server-side; it does not accept a PDF-only submission for new LaTeX papers unless you deliberately opt into PDF upload).

## Submission metadata (fill into the arXiv form)

- **Title:** Masked Diffusion Language Models Absorb Extreme Weight Quantization Better Than Autoregressive Models at Matched Scale
- **Authors:** Victor Panisa
- **Primary category:** `cs.LG` (Machine Learning)
- **Cross-list:** `cs.CL` (Computation and Language)
- **License:** recommend CC BY 4.0 (or arXiv's non-exclusive default if you prefer).
- **Comments field (suggested):** "Technical report. Matched-control study of quantization robustness in masked diffusion vs autoregressive LMs; code, checkpoint hashes, and pre-registrations released."
- **Abstract:** paste the abstract from `diffusal-arxiv.tex` as plain text (strip the LaTeX macros: `\%`→`%`, `\times`→"×", `\{-1,0,+1\}`→"{-1, 0, +1}", `$-11$`→"-11", etc.).

## Pre-flight content checks (do NOT skip before adversarial venues)

- [ ] Every number in the tables matches `experiments/exp0/RESULTS.md` and `experiments/exp2/RESULTS-replication.md`.
- [ ] Pre-registration release tags exist and are public: `prereg-exp0-2026-07-16`, `prereg-exp2-pilot-2026-07-16.1`.
- [ ] The reproducibility release (configs, checkpoint SHA-256, shim) is pushed and linked, or a URL placeholder is replaced before upload.
- [ ] Limitations section still lists all three QAT confounds (sub-3B scale, 21.5% coverage, NELBO bound).
- [ ] Author name / email / affiliation correct.

## Version plan

- **v1 (now):** PTQ spine (130M) + 7M QAT + mechanism negatives. Ship to arXiv; do **not** push to HN/Reddit yet.
- **v2 (before HN/Reddit):** add the 130M native-QAT twin — a new row in the §4.3 table plus one paragraph. This is the specific rebuttal to "the only from-scratch ternary result is a 7M toy." Requires a rented 24–40GB GPU (does not fit the local 8GB card). See [../PHD-THESIS.md](../PHD-THESIS.md) §5.2.

## What this paper deliberately does NOT claim

Scale transfer beyond 130M/7M, absolute dLLM quality parity, any deployment/latency/VRAM advantage, or a mechanism for the robustness. Those belong to the program document ([../PHD-THESIS.md](../PHD-THESIS.md)), not this paper.

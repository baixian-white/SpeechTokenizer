# Conversion Notes

## Source and Output

- Authoritative Chinese source: `output/doc/paper_drafts/scit_speech_method_cn_draft_20260622.md`
- English LaTeX project: `output/doc/paper_drafts/scit_speech_latex_en_20260629/`
- Main TeX: `output/doc/paper_drafts/scit_speech_latex_en_20260629/main.tex`
- Bibliography: `output/doc/paper_drafts/scit_speech_latex_en_20260629/references.bib`
- Citation audit: `output/doc/paper_drafts/scit_speech_latex_en_20260629/citation_audit.md`
- Figures: `output/doc/paper_drafts/scit_speech_latex_en_20260629/figures/`

Only the current source draft above was used as the manuscript authority. Files named `before_*.bak`, `before_5_7_edit.md`, and other historical backups were not used as content sources.

Update on 2026-07-01: `main.tex` was resynchronized from the current Chinese source draft rather than only recompiling the old English text. The resync added the latest source-draft details for entropy/packetization accounting, experiment setup, same-rate Codec2/ViSQOL interpretation, clean ASR table, three-user router figure, prototype jitter/drop fields, LCA marginal-significance table, and the updated limitations/conclusion wording.

## Writing Decisions

- The English manuscript keeps the core framing: SCIT-Speech is a 500--1500 bps ultra-low-bitrate speech communication system based on shared RVQ codebook index transmission.
- The phrase "general framework" is avoided.
- NAS is presented as transmitter-side efficiency evidence only, not as proof that the NAS encoder matches the hand-designed encoder in final reconstruction quality.
- HuBERT distillation is kept as part of the Base training objective, not restored as an independent ablation contribution.
- The ablation section contains one theme only: V0--V4 factorized LCA components. No 6.1/6.2 structure is used.
- Chinese AISHELL / Chinese zero-shot experiments were not restored.
- The three-user demo is kept as system prototype validation only: single-machine TCP loopback, one router plus three clients, router forwards index packets only, each client sends one stream and receives/decodes two streams, 60 s per `(device, L)` setting, CPU/GPU RTF < 1, and per-end total bandwidth 11.5--14.5 kbps.

## Figures

English-safe figures included in the LaTeX project:

- `figures/fig1_system_overview.pdf`: redrawn in English using `make_english_figures.py`.
- `figures/fig2_rate_quality_tradeoff.pdf`: copied from the existing English asset `assets/fig2_rate_quality_tradeoff.pdf`; caption corrected to match the actual panels (STOI, PESQ-WB, WER).
- `figures/fig3_visqol.pdf`: copied from the existing English asset `assets/fig_visqol.pdf`.
- `figures/fig4_perturbed_asr_wer.pdf`: regenerated in English from `output/experiments/exp11_perturbed_asr_wer_20260609/.../perturbed_asr_wer_results.csv`.
- `figures/fig5_perturbation_robustness.pdf`: copied from the existing English asset `assets/fig3_perturbation_robustness.pdf`.
- `figures/fig6_lca_ablation.pdf`: regenerated in English from `output/experiments/exp5_lca_component_factorial_20260603_seed42/reports/statistical_tests_20260610/factorial_variant_summary.csv`.
- `figures/fig7_three_user_router_architecture.png`: copied from the current Chinese draft figure `output/doc/paper_drafts/figures/scit_speech_three_user_router_architecture_v10_top_text_fixed_20260630.png`.

No Chinese filename or Chinese figure path is referenced in `main.tex`.

## Citation Verification

See `citation_audit.md` for per-entry status. Conservative policy used:

- Verified sources include arXiv, ACL Anthology, RFC Editor, ITU recommendation pages, DOI/IEEE metadata, official GitHub, or publication pages found during the pass.
- If a DOI, arXiv ID, or final venue could not be confirmed, it was not fabricated.
- `tian2025largesc` and `han2025packetloss` have verified arXiv-safe metadata.

## Compilation

Preferred command:

```powershell
latexmk -pdf main.tex
```

Fallback command:

```powershell
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Actual compile commands used:

```powershell
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Current result after the 2026-07-01 resynchronization: `main.pdf` was regenerated as an 11-page PDF (`546050` bytes). A fresh log check found no unresolved citations, undefined references, BibTeX warnings, LaTeX errors, emergency stops, or fatal errors. MiKTeX may still emit a local maintenance warning about update checks, and LaTeX may emit non-blocking underfull/overfull typography notices.

# IEEEtran Conversion Notes

## Source and Output

- Source English project: `output/doc/paper_drafts/scit_speech_latex_en_20260629/`
- Authoritative Chinese draft remains: `output/doc/paper_drafts/scit_speech_method_cn_draft_20260622.md`
- IEEEtran project: `output/doc/paper_drafts/scit_speech_latex_en_ieeetran_20260630/`
- Main TeX: `main.tex`
- Bibliography: `references.bib`
- PDF: `main.pdf` after compilation

The previous generic `article` version was not modified. This directory is a separate IEEEtran-format project.

## Template Changes

- Document class changed from `article` to:

```latex
\documentclass[conference]{IEEEtran}
```

- Bibliography style changed from `plain` to:

```latex
\bibliographystyle{IEEEtran}
```

- Added IEEE-style anonymous author block and `IEEEkeywords`.
- Replaced generic "Figure" text references with IEEE-style `Fig.~`.
- Kept the same figures, citation keys, and citation audit from the checked English project.
- Kept the same content boundaries: no Chinese AISHELL/zero-shot restoration, no 6.1/6.2 ablation split, Limitations remains four items, and NAS remains transmitter-side efficiency evidence only.
- Update on 2026-07-01: the IEEEtran `main.tex` body was rebuilt from the resynchronized English project after that project was updated against the current Chinese source draft. It now includes the latest entropy/packetization accounting, experiment setup, same-rate Codec2/ViSQOL interpretation, clean ASR table, three-user router figure, prototype jitter/drop fields, LCA marginal-significance table, and updated limitations/conclusion wording.

## Notes

- This is a standard `IEEEtran` conference-style conversion, not a specific ICASSP/Interspeech/IEEE journal official package.
- If targeting a specific IEEE venue, replace the anonymous author block and add any venue-specific copyright, page-limit, or camera-ready commands required by that venue.

## Compilation

Preferred manual command sequence on this machine:

```powershell
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Current result after the 2026-07-01 resynchronization: `main.pdf` was regenerated
as an 11-page PDF (`582593` bytes). A fresh log check found no unresolved
citations, undefined references, BibTeX warnings, LaTeX errors, emergency stops,
or fatal errors. The remaining acceptable log messages are non-fatal typography
notices from IEEEtran/MiKTeX and the standard IEEEtran last-page column reminder.

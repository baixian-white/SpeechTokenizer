# Results and Supplementary Methods LaTeX bundle

This bundle contains the latest standalone Results v3 PDF available in the
repository on 2026-10-10 and the explicitly requested S8/S9-reflowed
Supplementary Methods PDF. Each directory includes the LaTeX source and every
figure/table asset directly referenced by that source.

## Contents

- `results_v3/results_v3.pdf`: Results PDF (6 pages).
- `results_v3/results_v3.tex`: standalone LaTeX entry point.
- `supplementary_methods_v3/supplementary_methods_v3_s8_s9_reflow_20261010.pdf`:
  reflowed Supplementary Methods PDF (20 pages).
- `supplementary_methods_v3/supplementary_methods_v3.tex`: standalone LaTeX
  entry point.

Both entries use `elsarticle` and standard TeX packages, which must be
installed separately. From either document's directory, run:

```text
pdflatex -interaction=nonstopmode -halt-on-error <entry-point>.tex
pdflatex -interaction=nonstopmode -halt-on-error <entry-point>.tex
```

The Supplementary Methods entry point outputs `supplementary_methods_v3.pdf`
when compiled; its supplied PDF uses the longer filename to distinguish the
S8/S9 layout revision from earlier PDFs.

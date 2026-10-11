# Methods Main-Text / Supplement Revision

Date: 2026-09-18.

## Scope and Deliverables

Only the Methods range of `manuscript.tex` was condensed. The Results text,
three Results tables, Introduction, Discussion, abstract, title and
bibliography were not rewritten. The main component figure was retained.
No experiment was run and no reported estimate was changed.

| File | Purpose |
| --- | --- |
| `manuscript.tex` / `manuscript.pdf` | Main manuscript with concise Methods |
| `supplementary_methods.tex` / `supplementary_methods.pdf` | Standalone complete methods and reproducibility record |
| `supplementary_methods_body.tex` | Original complete Methods, with S1--S9 section headings |
| `supplementary_methods_references.tex` | Bibliography for the standalone supplement |
| `methods_main_zh.md` | Translation of the concise main Methods |
| `methods_zh.md` | Unchanged original full-length translation |
| `archive/20260918_before_methods_split/` | Original English source, PDF and full Chinese Methods |

## Length

Counts include section headings and prose numerals, but exclude display
math, inline math, figure/table environments, captions within those
environments, comments, citation keys and reference labels. These are
consistent whitespace-token estimates, not a journal's official word count.

| Section | Before | After | Change |
| --- | ---: | ---: | ---: |
| Results | 1,138 | 1,138 | 0 |
| Methods | 3,246 | 1,521 | -1,725 (-53.1%) |
| Results + Methods | 4,384 | 2,659 | -1,725 |

Counting alphabetic English words only gives 1,086 for Results and
3,064 to 1,454 for Methods. The two conventions must not be mixed.
The main Methods retain nine subsections and five numbered equation
environments. The supplement preserves all 16 original numbered equations.

The full main-manuscript PDF is six pages, down from eight; this includes
the existing title, abstract, Introduction, Discussion and references.
The standalone supplement is six pages. Both retain the existing 10-pt
two-column template.

This is not a claim that the final Results/Methods figure plan is complete:
the current main source contains one component diagram and three tables.
Additional result figures and the unfinished six-user experiment require
a fresh layout check against the requested six-to-seven-page allowance
for Results and Methods. Do not add filler merely to occupy that allowance.

## Content Allocation

| Main subsection | Main-text function retained | Complete detail in supplement |
| --- | --- | --- |
| 4.1 Representation and transmission | Fixed L=3 path, codebook lookup, nominal rate versus packet overhead | S1: residual quantization equations, lower-layer rates, byte packing and metadata |
| 4.2 NAS and profiling | Staged selection, chosen architecture, reused proxy training subsets, encoder-only RTF boundary | S2: search space, all operators, proxy objectives, ranking weights and profiling settings |
| 4.3 Codec training | Data split, objective, frozen teacher, optimization and actual checkpoint selection | S3: decoder configuration, spectral settings, discriminator losses, EMA codebook updates |
| 4.4 Channel adaptation | Sampled perturbations, objective, gradient routing and robustness-selected checkpoint | S4: full fine-tuning and validation conditions |
| 4.5 Reconstruction evaluation | Distinct 300/2,620 populations, rate-matched baselines, metrics, bootstrap and paired masks | S5: exact spectral/resampling settings and token-domain perturbation construction |
| 4.6 Speaker classifier | Frozen codec and ECAPA backbone, two branches, gate and classification loss | S6: dimensions, temporal blocks, pooling and pretrained checkpoint |
| 4.7 Speaker protocol | Fixed split, three seeds, validation-only selection, calibration, uncertainty and closed-set boundary | S7: augmentation, padding, optimizer settings, software versions and bootstrap hierarchy |
| 4.8 Inference | Buffered rather than sample-causal operation, paired per-stream inputs, unmeasured system endpoints | S8: block/padding rules, calibration formula and bundle/buffer interfaces |
| 4.9 Six-user integration | Visible unfinished-experiment placeholder | S9: same placeholder; no fabricated result |

The supplement deliberately retains the full documented method, including
some overlap with the concise main text, so it is readable independently.
Main-text detail was relocated, not erased from the reproducibility record.
No new claim or numerical result was appended to Results.

## Verification

- Exact normalized-text comparison: everything before and after the Methods
  range is unchanged.
- The complete original Methods matches the supplementary body after only
  removal of the outer Methods heading/label and promotion of subsection
  headings to S-numbered sections.
- Main Figure 1 source is unchanged.
- All main and supplementary citation keys and figure references resolve.
- Both PDFs were compiled repeatedly; final logs have no undefined
  references/citations, overfull boxes, LaTeX warnings or package warnings.
- Residual underfull-line messages are recorded for visual inspection, not
  misreported as a warning-free build.
- All six main-manuscript pages and all six supplementary pages were rendered
  and visually inspected. No overlapping text, clipped tables, overflowing
  equations or unresolved citation markers were found. The final reference
  pages retain the template's partial-column layout.

## Unresolved Evidence

1. The executed speaker-bootstrap replicate count and random seed remain
   missing. No helper default was substituted for evidence.
2. Six-user throughput, end-to-end latency and identity performance remain
   unmeasured.
3. The unchanged title and abstract retain the broader overall-study framing.
   They require a separate evidence-alignment pass when integration results
   are available.

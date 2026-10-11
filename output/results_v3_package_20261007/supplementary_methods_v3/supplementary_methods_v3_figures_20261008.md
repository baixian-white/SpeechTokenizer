# Supplementary Methods v3: figure update

Date: 2026-10-08.

This update adds three data figures to the existing supplement without changing the
underlying experiments, model checkpoints, tables, or main Results. The English
LaTeX source and Chinese counterpart now contain five figures in source order:

| Figure | Content | Retained source |
| --- | --- | --- |
| S1 | Eight final NAS candidates: proxy mel-L1/MACs and latent SmoothL1/parameters | `output/experiments/exp1_nas_distill_run1_seed42/metrics/stage4_final.csv` |
| S2 | Index replacement and grouped-mask protocol (previously S1) | Existing protocol diagram |
| S3 | Paired SCIT-LCA minus SCIT changes in five channel conditions | `output/experiments/exp8_librispeech_test_full_perturb_20260606/eval_perturb_full_lca_nosave/test-clean/metrics/base_vs_lca_perturb_results.csv`; `output/experiments/exp9_librispeech_test_full_packet_burst_20260608/eval_packet_burst_full_lca_nosave/test-clean/metrics/packet_burst_results.csv` |
| S4 | Per-speaker Top-1 accuracy for formal seeds 41, 42 and 43 | `D:/tmp/speechtokenizer-exp23/tmp/exp23_formal/seed*/metrics/test_metrics.json` |
| S5 | Receiver pairing protocol (previously S2) | Existing receiver diagram |

The new figures are 300-dpi PNGs with generation scripts in
`supplementary_methods_v3_assets/`; they do not have separate PDF exports.
The three formal per-seed speaker-metric JSON records are copied byte-for-byte
under `supplementary_methods_v3_assets/speaker_formal_metrics/` so that the
speaker plot can be regenerated from the delivered bundle. Their hashes match
the formal-source hashes retained in the Results v3 verification record.
S1 shows search-stage proxy measurements, not final codec quality. S3 uses
2,620 matched utterances per condition and retains negative mean PESQ-WB
differences for clean, packet and burst conditions; its violin display trims
the outer 1% at each tail. S4 uses the fixed 110-speaker closed-set test split
with 15 utterance groups per speaker. The figure captions state these limits.

The complete English supplement was compiled twice with local pdfLaTeX and
exported to `../output/pdf/supplementary_methods_v3.pdf`. It is now 16 pages,
with 9 sections, 48 subsections, 11 tables, 5 figures and one algorithm.
The latest structured checks are in
`supplementary_methods_v3_assets/verification.json`. The 2026-09-22 revision
note remains a historical record of the earlier 14-page version.

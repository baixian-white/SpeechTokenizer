# CPU evaluations of 19 October 2026 (main-figure plan, stage 0)

All runs are CPU-only (`CUDA_VISIBLE_DEVICES=''`), L=3, on the 300 test-clean utterances of the external-codec comparison
and the first 300 entries of the test-other manifest (exp12 originals; 6 and 4 speakers; the paper reports exp12 on
test-clean only). Every model encodes each utterance itself; each
(utterance, condition) gets one perturbation from a CPU generator seeded with crc32(split|utterance|condition), shared by
all models (the per-row frame and erasure/substitution counts confirm this).

- `eval_n3_n4.py`: the evaluation (models: SCIT, SCIT-LCA, factorial V0-V3 at 30,000 steps, exp13 at its validation-selected
  checkpoint and at step 100,000). `run_missing_shards.sh`, `run_factorial_shards.sh`: memory-guarded shard runners (the
  host is shared; at most 4 processes, none started below 16 GB available).
- `n3_n4_rows/`, `n3_factorial_rows/`, `n4b_rows/`: per-utterance rows.
- `analyze_n3_n4.py` -> `n3_n4_means.csv`, `n3_n4_paired.csv`, `n3_n4_summary.md`: paired differences with utterance and
  speaker-cluster bootstrap intervals (10,000 resamples, seed 42).
- `analyze_factorial.py` -> `factorial_means.csv`, `factorial_contrasts.csv`, `factorial_summary.md`: component contrasts.
- `sixuser_source_data.py` -> `sixuser_source/`: decision-lag ECDFs and quantiles, capture-to-decode components (means that
  sum exactly), 1-min stability, identification on/off pairs, from `runs/`.
  Per-run rows also carry the maximum decision lag and `foreign_gpu_probes` (1-s worker probes with another process on
  the worker's GPU, `render_paper.foreign_gpu`, the S10 definition); the six-user figure greys runs and pairs from it.
- `speaker_source_data.py` -> `speaker_source/`: per-speaker and per-seed accuracy from the formal exp23 records.
- `fig4_source_data.py` -> `fig4_dose.csv` (SCIT-LCA minus SCIT at L=3 for every operator and dose on all 2,620 test-clean
  and 2,939 test-other utterances, from the exp8/exp9 records, with utterance and speaker-cluster bootstrap intervals) and
  `fig4_control.csv` (V0 and SCIT-LCA against SCIT from the CPU rows above).

Since 20 October 2026 (stage 2), `fig4_*.csv` and `sixuser_source/` feed the new Results figures (robustness and six-user)
and the numbers of Results 2.4 and 2.6, through `paper/render_paper.py`. The other files are analyses for the figure plan,
not paper text. CPU results differ from the GPU records of the same model and utterance by about 1e-3 in mel-L1 on
average (at most 0.03); compare models only within these files.

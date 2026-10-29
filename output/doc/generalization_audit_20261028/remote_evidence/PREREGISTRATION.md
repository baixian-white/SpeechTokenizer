# Six-user rerun on the RTX 4090 host — pre-registered protocol and acceptance criteria

Frozen before any formal run. Do not edit after the first formal run starts. Any later
change goes into a dated addendum at the end, and that addendum gets its own git commit.

## Deployment (both systems)

- Host: Linux, Intel Core i9-14900K (32 logical CPUs), 125 GiB RAM, two RTX 4090s.
  Every run uses GPU 1 only (`CUDA_VISIBLE_DEVICES=1`). GPU 0 drives the desktop and
  runs other users' work.
- Software: Python 3.10.20, torch 2.10.0+cu128, torchaudio 2.10.0+cu128, SpeechBrain
  1.1.0 (fixed system only), pystoi 0.4.1, pesq 0.0.4. These differ from the original
  runs (Windows, GTX 1650, torch 2.11.0+cu128) and are reported as a different deployment.
- Numerics: fp32 with TF32 disabled for matmul and cuDNN; the effective flags are logged.
- Topology: one host, TCP loopback, one router, six client processes, one shared GPU
  worker process. Playback off; acoustic output latency not measured.
- Frozen models: LCA checkpoint sha256 `af60223b…`; NAS encoder config resolved from the
  file (the run aborts if ops would be inferred); Exp23 bundle `classifier.pt` sha256
  `e497e2e2…`, 110-class output, temperature 1.01877, no candidate mask.
- Frozen material: the 90 locked test utterances of p225–p230 (15 each). Source FLAC
  files are verified against the handoff SHA-256 list.
- Observation deadlines (unchanged): block 10 s after source block end; prediction 15 s
  after source window end; receive queue 64 blocks.
- Cadence (unchanged): sequential and concurrent use one decision per 3-s utterance per
  directed link. Switch and stability use a 3-s window with a 1-s hop.

## Systems

- **Baseline**: the delivered code. Single-queue, single-thread worker; Exp23 on CPU
  (TorchScript); predict called synchronously in each client's decode thread. Only the
  portability fixes are applied (Windows paths, NAS config path, TF32 pinned off, GPU probe).
- **Fixed**: two-lane worker (codec lane at high CUDA-stream priority, identification
  lane batching up to 32 contexts); Exp23 on CUDA through a strict state-dict transplant
  of the frozen bundle; an asynchronous identification thread per client; one worker
  connection per client thread; warm-up before the start gate. Codec weights, classifier
  weights, scoring and windows are unchanged.

Equivalence gate (passed before this file was written): on the 90 contexts built through
the system path, the CPU bundle and the GPU transplant give 90/90 identical labels (max
|Δlogit| 2.3e-4). The labels also match the original GTX 1650 sequential run 90/90.

## Runs

| Order | System | Condition / scenario | Repeats |
|---|---|---|---|
| 1 | Fixed | D sequential, D concurrent, C concurrent (identification off) | 3 |
| 2 | Fixed | D switch | 3 |
| 3 | Fixed | D stability, 1,800 s | 1 (more if time allows; every run is reported) |
| 4 | Baseline | D sequential, D concurrent | 1 |
| 5 | Baseline | D stability, 300 s (diagnostic, not the 1,800-s protocol) | 1 |

No run is excluded. A failed or aborted run is reported with its reason.

## Acceptance criteria (fixed system; evaluated per run)

A condition may be called real-time on this deployment only if **every** run of it meets
**every** applicable criterion.

- AC-COV: coverage `n_predicted / n_expected` = 100% for sequential, concurrent and
  switch; ≥ 99.9% for stability.
- AC-BLK: `deadline_failure_blocks` = 0, `receiver_queue_drops` = 0,
  `router_queue_drops` = 0, every expected block received and decoded on time.
- AC-C2D: capture-to-decode P95 ≤ 1.00 s and P99 ≤ 1.20 s. This includes 0.5 s of
  block accumulation. For stability, also P95 ≤ 1.00 s in every 5-min bin.
- AC-LAG: decision lag (`output_ns − source_window_end_ns`) P95 ≤ 1.00 s and max ≤ 3.00 s.
  For stability, also P95 ≤ 1.00 s in every 5-min bin.
- AC-INT (1-s hop scenarios): per-link identity output interval median 1.00 ± 0.05 s and
  P95 ≤ 1.25 s.
- AC-LBL: utterance-aligned labels in sequential and concurrent equal the reference
  labels from the equivalence gate, so Top-1 = 77/90 per receiver copy.
- AC-Q: every reconstruction is complete (`missing_blocks` = 0) with 0 PESQ failures.
  Quality is reported over all reconstructions.
- AC-DEC: decoupling control. Mean over repeats of D-concurrent c2d P95 minus
  C-concurrent c2d P95 ≤ 0.05 s.
- AC-STAB: 0 crashes, disconnects, worker errors, timeouts and `queue_full` responses.

## Reporting rules

- Every condition reports coverage, end-to-end correct rate and valid-prediction Top-1
  side by side, with N; quality over all reconstructions, with failure counts.
- Latency is reported as capture-to-decode and decision lag (P50/P95/max). The old
  "identification RTF" (request round trip / 3 s) is not used as a real-time criterion.
- The baseline-on-4090 runs are reported as a hardware control: they show what changes
  with the faster host alone.
- The original GTX 1650 / CPU runs remain the record of the earlier deployment. They are
  not mixed with these runs in one table without labelling.

## Addendum 1 (2026-10-12, 14:3x +0800): replication of the short conditions on an unshared GPU

Written and committed before the replication runs start.

- **Reason.** The worker's once-per-second GPU probe shows a second process on the worker's GPU (GPU 1). It was another training job of the authors', not part of this test or its analysis.
  - 09:23:43–09:30:09: about 2.4 GiB, at times near 100% utilization.
  - 09:34:15–09:35:23: about 1.4 GiB.
  - Runs it overlapped: C sequential R1 (sessions 3–6), C concurrent R1 (first 26 s of source time), D sequential R2 (sessions 1–3), 2 s of D concurrent R2, and C sequential R2 (session 1).
  - Every registered criterion was still met in all these runs. The C concurrent R1 capture-to-decode P95, 0.731 s against 0.579 and 0.578 s in R2 and R3, coincides with this overlap.
  - The switching runs, the second and third 30-min runs and the hardware-control runs show no foreign use of GPU 1.
  - The first 30-min run overlaps the separately recorded 35-s offline job.
  - The other job now runs on GPU 0, and GPU 1 is free.
- **Replication.** Run the short suite again unchanged on GPU 1, with the same code, plan generator and commit: `run_suite.sh fixed_short`, i.e. D and C, sequential and concurrent, 3 repeats.
  - The worker probe records any foreign use; any overlap that occurs is reported in the same way.
- **Reporting rules, fixed before seeing the replication.**
  1. All registered runs and all replication runs are reported in full in S10, each with its own acceptance evaluation.
  2. The main-text table keeps the registered runs.
  3. The main-text statement on the effect of identification on decoding uses only simultaneous/control pairs whose runs had no foreign use of the worker's GPU: registered R2 and R3 and the three replication repeats. The registered R1 pair is shown in S10 with the reason.
  4. If a replication run fails a criterion, that failure is reported in the main text.
- **Thresholds.** None changed.

# Six-User System Measurement Protocol

The `run_metrics_experiment.py` entry point implements the measurement protocol in
`six_user_demo_metrics_requirements_20260924.md`. It reuses the existing LCA codec,
RVQ bit packing, demonstration XOR transport, Router, and the seed42 Exp23 bundle.

## Conditions and Scenarios

| Condition | Connected clients | Recognition | Transport |
| --- | --- | --- | --- |
| A | One local process at a time, covering all six sources | Exp23 | Local paired encode/decode |
| B | ABC and DEF in separate three-process jobs | Exp23 | TCP loopback |
| C | Six processes | Disabled | TCP loopback |
| D | Six processes | Exp23 | TCP loopback |

`sequential` runs one active source per network job. `concurrent` opens a common
start barrier for all sources in the connected group. The Router must confirm
every registration before the barrier opens. A is a single-process local
baseline and does not simulate six competing processes.

`switch` uses one stable routing ID while cycling through the six true identities
in six-second dwell intervals. The routing packet contains no identity label.
`stability` continuously cycles the chosen source list for at least the requested
duration, rounded up to a multiple of three seconds; the default is 1,800 seconds.

## Frozen Inputs and Timing

- Development: one existing validation utterance per source; test: 15 existing
  held-out utterances per source. Selection is independent of observed accuracy.
- Prepare 16 kHz mono FLOAT WAVs using mean channel mixing, polyphase resampling,
  center crop or end padding to 48,000 samples. Original and prepared hashes are
  stored in the audio manifest.
- Encode 8,000 samples per block, three RVQ layers and 10 bits per index. Samples
  are released at their scheduled block-tail time, reproducing block collection
  waiting during WAV replay. Slow processing is recorded as lag.
- Independent utterance trials produce one prediction per three-second segment.
  Continuous switch/stability trials use a three-second window and one-second
  source-time hop. All windows are planned before reception.
- Pair complete source-aligned blocks and all their actual RVQ frames. Missing
  blocks prevent a valid paired window. Reconstructed WAVs retain block positions
  with zero fill; failed fragments are never removed from the evaluation length.
- Cross-identity windows are recorded as transitions, with the latest sample's
  identity as the target. They are excluded from steady accuracy and retained in
  switch response records. Initial three-second context is included in response
  time. First correct response means one on-time correct output; continuous-flow
  response success also requires output before that identity's dwell ends.
  A later correct output is retained as `correct_after_identity_left` and counted
  as a response failure. Stability's initial response searches the full identity
  dwell, not just its first three-second segment.
- Default block deadline: ten seconds after the scheduled block tail. Prediction
  deadline: fifteen seconds after its scheduled window end. These are fixed
  engineering observation deadlines, not latency acceptance targets. Deadline
  expiry and late output remain visible in the data.
- This implementation is single-host only. `time.perf_counter_ns()` is shared
  across these Python 3.10+ Windows processes; timestamps are nanoseconds. It does
  not compare independent machines' unsynchronized clocks.

## Measurement Boundaries

`capture_to_decode_ms` starts at the scheduled WAV block-first sample and ends at
actual decoder completion. `block_tail_to_decode_ms` uses the scheduled block
tail. Each is computed from that block's paired timestamps. `transport_ms` runs
from client socket-send start to receiving the complete message and includes
serialization and Router scheduling. Local condition A has no network latency.

With `--playback off`, playback timing is missing. With `--playback device`, each
client mixes decoded streams for a sounddevice output callback. Callback timing,
queue drops, and buffer underruns are recorded; actual acoustic device output
latency remains unmeasured. Neither mode reports a configuration budget as a
measured end-to-end acoustic call delay. Clients use WAV input rather than a mic.

Traffic counters cover `codes_v1` application messages. Handshake, TCP/IP headers,
retransmissions and link traffic are outside this boundary. Payload and header
bytes, client transmit/receive and Router ingress/egress are separate. Per-source
bitrate uses planned source audio duration. The packet's source sample range is
included in the measured application header overhead.

## Scoring and Reporting

Top-1 = correct / valid on-time predictions; coverage = valid on-time predictions
/ planned steady windows; end-to-end correct output rate = correct / planned
steady windows. Empty denominators produce missing values. Late predictions keep
their raw 110-class label but count as no on-time output. Main Macro-F1 uses the
six true classes with other-class predictions and no output counted as false
negatives. Confusion data preserves six rows and eight display columns and raw
110-class labels in predictions.csv.

STOI uses `extended=False`; PESQ uses wideband 16 kHz. Mel-L1 reuses the repository
`mel_loss` with the frozen codec configuration. Each score covers the full
prepared source length, after per-block output trim/end padding, with no
cross-correlation time shifting. Failed metric computations retain their reason.
Quality summaries first average receiver/replay copies per source utterance,
then average source utterances equally.

Repeat summaries report per-system-run mean and sample SD; SD is missing for
one run. These are system restart repeats, not training seeds. Receiver copies,
overlapping windows and repeated playback are technical repetitions. No
confidence interval is computed under an independence assumption.

CPU and RSS plus machine CPU/memory and queue length are sampled every second
from loading to exit. GPU fields are missing with `not_used_cpu_run`. Automatic
reconnect is not enabled; disconnects and failures are retained. No six-process
real-time capability is inferred from a single decoder's RTF.

## Artifacts and Recalculation

The output directory contains frozen `experiment_plan.json`, `run_manifest.json`,
`audio_manifest.csv`, plus `jobs/<id>/job.json`, `receipt.json`, append-only
`events/*.jsonl`, full logs, and per-receiver pre-mix `audio/`. Top-level normalized
files are `blocks.csv`, `predictions.csv`, `audio_metrics.csv`, `resources.csv`,
`events.csv`, `per_stream_metrics.csv`, `summary.csv`, `repeat_summary.csv`,
`per_speaker_metrics.csv`, `confusion_matrix.csv`, `traffic.csv`, and
`identity_responses.csv`. Analysis status, failure records and diagnostic PNG/PDF
figures are retained alongside their CSV sources.

Execute `summarize_metrics_experiment.py <run-directory>` to recompute from frozen
plans and raw events. `--no-quality` and `--no-plots` support a quick raw-data
audit and explicitly mark unavailable scores. Old Stage 1 result files use their
own schema and should be evaluated by their original scripts.

## Prerequisites and Validation Scope

Python `st_audio_py310` runs the codec and Exp23. It is installed with
`torch==2.11.0+cu128` and `torchaudio==2.11.0+cu128`; CUDA is available to
explicit GPU probes. The measurement client currently keeps the system replay
on CPU so its published baseline remains comparable. `pystoi`, `pesq` and
matplotlib are installed in the same environment and the post-run scoring worker
uses the same `--quality-python` interpreter by default. A GPU run must be labeled
as a separate condition after changing the client device policy and recording
GPU memory/utilization.

Before startup, free-memory requirements are 2 GiB for A, 4 GiB for B and 8 GiB
for C/D. Available disk space is reported; long audio collection can consume
several GiB. A failed startup or replay keeps its folder and all planned
denominators. Inspect failures and quality status independently of process exit.

The `--fixture` flag on the measurement client is reserved for transport tests.
It does not load model weights, and its parent manifest must be marked fixture.
Fixture recordings are excluded from audio-quality scoring and are not model
performance measurements.

# Audio sample appendix (Exp4)

All wav files are 16 kHz mono. Paths are relative to the run directory.
Each sample has 23 versions: 1 original + 6 SCIT (Base/LCA x L=1/2/3) + 1 PCM + 5 Opus + 4 EnCodec + 7 DAC.

## Sample `1455-138263-0040`

| Method | Setting | Path | WER vs GT |
|---|---|---|---:|
| (original) | -- | `samples/original/1455-138263-0040.wav` | -- |
| scit_base | L=1 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_base/L1/1455-138263-0040.wav` | 0.366 |
| scit_base | L=2 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_base/L2/1455-138263-0040.wav` | 0.122 |
| scit_base | L=3 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_base/L3/1455-138263-0040.wav` | 0.122 |
| scit_lca | L=1 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_lca/L1/1455-138263-0040.wav` | 0.366 |
| scit_lca | L=2 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_lca/L2/1455-138263-0040.wav` | 0.171 |
| scit_lca | L=3 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_lca/L3/1455-138263-0040.wav` | 0.122 |
| pcm | 16bit_16khz_passthrough | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/pcm/1455-138263-0040.wav` | 0.073 |
| encodec | bw1.5kbps_n_cb2 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/encodec/bw1.5kbps/1455-138263-0040.wav` | 0.098 |
| encodec | bw3.0kbps_n_cb4 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/encodec/bw3.0kbps/1455-138263-0040.wav` | 0.146 |
| encodec | bw6.0kbps_n_cb8 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/encodec/bw6.0kbps/1455-138263-0040.wav` | 0.146 |
| encodec | bw12.0kbps_n_cb16 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/encodec/bw12.0kbps/1455-138263-0040.wav` | 0.098 |
| dac | n_q_1 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q1/1455-138263-0040.wav` | 0.927 |
| dac | n_q_2 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q2/1455-138263-0040.wav` | 0.317 |
| dac | n_q_3 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q3/1455-138263-0040.wav` | 0.244 |
| dac | n_q_4 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q4/1455-138263-0040.wav` | 0.146 |
| dac | n_q_6 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q6/1455-138263-0040.wav` | 0.171 |
| dac | n_q_9 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q9/1455-138263-0040.wav` | 0.073 |
| dac | n_q_12 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q12/1455-138263-0040.wav` | 0.098 |
| opus | opus_6000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br6000/1455-138263-0040.wav` | 0.220 |
| opus | opus_8000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br8000/1455-138263-0040.wav` | 0.098 |
| opus | opus_12000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br12000/1455-138263-0040.wav` | 0.122 |
| opus | opus_16000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br16000/1455-138263-0040.wav` | 0.073 |
| opus | opus_24000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br24000/1455-138263-0040.wav` | 0.098 |

## Sample `1737-148989-0006`

| Method | Setting | Path | WER vs GT |
|---|---|---|---:|
| (original) | -- | `samples/original/1737-148989-0006.wav` | -- |
| scit_base | L=1 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_base/L1/1737-148989-0006.wav` | 0.516 |
| scit_base | L=2 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_base/L2/1737-148989-0006.wav` | 0.258 |
| scit_base | L=3 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_base/L3/1737-148989-0006.wav` | 0.258 |
| scit_lca | L=1 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_lca/L1/1737-148989-0006.wav` | 0.419 |
| scit_lca | L=2 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_lca/L2/1737-148989-0006.wav` | 0.161 |
| scit_lca | L=3 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_lca/L3/1737-148989-0006.wav` | 0.065 |
| pcm | 16bit_16khz_passthrough | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/pcm/1737-148989-0006.wav` | 0.000 |
| encodec | bw1.5kbps_n_cb2 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/encodec/bw1.5kbps/1737-148989-0006.wav` | 0.065 |
| encodec | bw3.0kbps_n_cb4 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/encodec/bw3.0kbps/1737-148989-0006.wav` | 0.000 |
| encodec | bw6.0kbps_n_cb8 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/encodec/bw6.0kbps/1737-148989-0006.wav` | 0.000 |
| encodec | bw12.0kbps_n_cb16 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/encodec/bw12.0kbps/1737-148989-0006.wav` | 0.000 |
| dac | n_q_1 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q1/1737-148989-0006.wav` | 0.968 |
| dac | n_q_2 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q2/1737-148989-0006.wav` | 0.452 |
| dac | n_q_3 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q3/1737-148989-0006.wav` | 0.097 |
| dac | n_q_4 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q4/1737-148989-0006.wav` | 0.000 |
| dac | n_q_6 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q6/1737-148989-0006.wav` | 0.032 |
| dac | n_q_9 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q9/1737-148989-0006.wav` | 0.000 |
| dac | n_q_12 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q12/1737-148989-0006.wav` | 0.000 |
| opus | opus_6000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br6000/1737-148989-0006.wav` | 0.065 |
| opus | opus_8000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br8000/1737-148989-0006.wav` | 0.000 |
| opus | opus_12000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br12000/1737-148989-0006.wav` | 0.032 |
| opus | opus_16000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br16000/1737-148989-0006.wav` | 0.032 |
| opus | opus_24000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br24000/1737-148989-0006.wav` | 0.032 |

## Sample `311-124404-0119`

| Method | Setting | Path | WER vs GT |
|---|---|---|---:|
| (original) | -- | `samples/original/311-124404-0119.wav` | -- |
| scit_base | L=1 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_base/L1/311-124404-0119.wav` | 0.778 |
| scit_base | L=2 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_base/L2/311-124404-0119.wav` | 0.444 |
| scit_base | L=3 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_base/L3/311-124404-0119.wav` | 0.167 |
| scit_lca | L=1 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_lca/L1/311-124404-0119.wav` | 0.556 |
| scit_lca | L=2 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_lca/L2/311-124404-0119.wav` | 0.167 |
| scit_lca | L=3 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_lca/L3/311-124404-0119.wav` | 0.056 |
| pcm | 16bit_16khz_passthrough | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/pcm/311-124404-0119.wav` | 0.000 |
| encodec | bw1.5kbps_n_cb2 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/encodec/bw1.5kbps/311-124404-0119.wav` | 0.444 |
| encodec | bw3.0kbps_n_cb4 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/encodec/bw3.0kbps/311-124404-0119.wav` | 0.000 |
| encodec | bw6.0kbps_n_cb8 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/encodec/bw6.0kbps/311-124404-0119.wav` | 0.000 |
| encodec | bw12.0kbps_n_cb16 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/encodec/bw12.0kbps/311-124404-0119.wav` | 0.000 |
| dac | n_q_1 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q1/311-124404-0119.wav` | 1.167 |
| dac | n_q_2 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q2/311-124404-0119.wav` | 0.167 |
| dac | n_q_3 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q3/311-124404-0119.wav` | 0.056 |
| dac | n_q_4 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q4/311-124404-0119.wav` | 0.000 |
| dac | n_q_6 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q6/311-124404-0119.wav` | 0.000 |
| dac | n_q_9 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q9/311-124404-0119.wav` | 0.000 |
| dac | n_q_12 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q12/311-124404-0119.wav` | 0.000 |
| opus | opus_6000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br6000/311-124404-0119.wav` | 0.000 |
| opus | opus_8000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br8000/311-124404-0119.wav` | 0.000 |
| opus | opus_12000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br12000/311-124404-0119.wav` | 0.000 |
| opus | opus_16000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br16000/311-124404-0119.wav` | 0.000 |
| opus | opus_24000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br24000/311-124404-0119.wav` | 0.000 |

## Sample `4018-103416-0033`

| Method | Setting | Path | WER vs GT |
|---|---|---|---:|
| (original) | -- | `samples/original/4018-103416-0033.wav` | -- |
| scit_base | L=1 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_base/L1/4018-103416-0033.wav` | 0.400 |
| scit_base | L=2 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_base/L2/4018-103416-0033.wav` | 0.156 |
| scit_base | L=3 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_base/L3/4018-103416-0033.wav` | 0.089 |
| scit_lca | L=1 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_lca/L1/4018-103416-0033.wav` | 0.533 |
| scit_lca | L=2 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_lca/L2/4018-103416-0033.wav` | 0.289 |
| scit_lca | L=3 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_lca/L3/4018-103416-0033.wav` | 0.067 |
| pcm | 16bit_16khz_passthrough | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/pcm/4018-103416-0033.wav` | 0.089 |
| encodec | bw1.5kbps_n_cb2 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/encodec/bw1.5kbps/4018-103416-0033.wav` | 0.356 |
| encodec | bw3.0kbps_n_cb4 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/encodec/bw3.0kbps/4018-103416-0033.wav` | 0.133 |
| encodec | bw6.0kbps_n_cb8 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/encodec/bw6.0kbps/4018-103416-0033.wav` | 0.133 |
| encodec | bw12.0kbps_n_cb16 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/encodec/bw12.0kbps/4018-103416-0033.wav` | 0.067 |
| dac | n_q_1 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q1/4018-103416-0033.wav` | 0.978 |
| dac | n_q_2 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q2/4018-103416-0033.wav` | 0.311 |
| dac | n_q_3 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q3/4018-103416-0033.wav` | 0.133 |
| dac | n_q_4 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q4/4018-103416-0033.wav` | 0.156 |
| dac | n_q_6 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q6/4018-103416-0033.wav` | 0.133 |
| dac | n_q_9 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q9/4018-103416-0033.wav` | 0.067 |
| dac | n_q_12 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q12/4018-103416-0033.wav` | 0.044 |
| opus | opus_6000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br6000/4018-103416-0033.wav` | 0.133 |
| opus | opus_8000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br8000/4018-103416-0033.wav` | 0.067 |
| opus | opus_12000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br12000/4018-103416-0033.wav` | 0.067 |
| opus | opus_16000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br16000/4018-103416-0033.wav` | 0.089 |
| opus | opus_24000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br24000/4018-103416-0033.wav` | 0.089 |

## Sample `4680-16026-0019`

| Method | Setting | Path | WER vs GT |
|---|---|---|---:|
| (original) | -- | `samples/original/4680-16026-0019.wav` | -- |
| scit_base | L=1 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_base/L1/4680-16026-0019.wav` | 0.429 |
| scit_base | L=2 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_base/L2/4680-16026-0019.wav` | 0.286 |
| scit_base | L=3 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_base/L3/4680-16026-0019.wav` | 0.286 |
| scit_lca | L=1 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_lca/L1/4680-16026-0019.wav` | 0.857 |
| scit_lca | L=2 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_lca/L2/4680-16026-0019.wav` | 0.286 |
| scit_lca | L=3 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_lca/L3/4680-16026-0019.wav` | 0.429 |
| pcm | 16bit_16khz_passthrough | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/pcm/4680-16026-0019.wav` | 0.286 |
| encodec | bw1.5kbps_n_cb2 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/encodec/bw1.5kbps/4680-16026-0019.wav` | 1.000 |
| encodec | bw3.0kbps_n_cb4 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/encodec/bw3.0kbps/4680-16026-0019.wav` | 0.286 |
| encodec | bw6.0kbps_n_cb8 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/encodec/bw6.0kbps/4680-16026-0019.wav` | 0.286 |
| encodec | bw12.0kbps_n_cb16 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/encodec/bw12.0kbps/4680-16026-0019.wav` | 0.286 |
| dac | n_q_1 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q1/4680-16026-0019.wav` | 0.714 |
| dac | n_q_2 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q2/4680-16026-0019.wav` | 0.429 |
| dac | n_q_3 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q3/4680-16026-0019.wav` | 0.286 |
| dac | n_q_4 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q4/4680-16026-0019.wav` | 0.286 |
| dac | n_q_6 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q6/4680-16026-0019.wav` | 0.286 |
| dac | n_q_9 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q9/4680-16026-0019.wav` | 0.286 |
| dac | n_q_12 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q12/4680-16026-0019.wav` | 0.286 |
| opus | opus_6000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br6000/4680-16026-0019.wav` | 0.429 |
| opus | opus_8000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br8000/4680-16026-0019.wav` | 0.286 |
| opus | opus_12000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br12000/4680-16026-0019.wav` | 0.286 |
| opus | opus_16000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br16000/4680-16026-0019.wav` | 0.286 |
| opus | opus_24000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br24000/4680-16026-0019.wav` | 0.286 |

## Sample `5390-30102-0015`

| Method | Setting | Path | WER vs GT |
|---|---|---|---:|
| (original) | -- | `samples/original/5390-30102-0015.wav` | -- |
| scit_base | L=1 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_base/L1/5390-30102-0015.wav` | 0.083 |
| scit_base | L=2 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_base/L2/5390-30102-0015.wav` | 0.028 |
| scit_base | L=3 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_base/L3/5390-30102-0015.wav` | 0.028 |
| scit_lca | L=1 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_lca/L1/5390-30102-0015.wav` | 0.028 |
| scit_lca | L=2 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_lca/L2/5390-30102-0015.wav` | 0.000 |
| scit_lca | L=3 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_lca/L3/5390-30102-0015.wav` | 0.000 |
| pcm | 16bit_16khz_passthrough | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/pcm/5390-30102-0015.wav` | 0.000 |
| encodec | bw1.5kbps_n_cb2 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/encodec/bw1.5kbps/5390-30102-0015.wav` | 0.000 |
| encodec | bw3.0kbps_n_cb4 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/encodec/bw3.0kbps/5390-30102-0015.wav` | 0.000 |
| encodec | bw6.0kbps_n_cb8 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/encodec/bw6.0kbps/5390-30102-0015.wav` | 0.000 |
| encodec | bw12.0kbps_n_cb16 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/encodec/bw12.0kbps/5390-30102-0015.wav` | 0.000 |
| dac | n_q_1 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q1/5390-30102-0015.wav` | 0.944 |
| dac | n_q_2 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q2/5390-30102-0015.wav` | 0.028 |
| dac | n_q_3 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q3/5390-30102-0015.wav` | 0.000 |
| dac | n_q_4 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q4/5390-30102-0015.wav` | 0.000 |
| dac | n_q_6 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q6/5390-30102-0015.wav` | 0.000 |
| dac | n_q_9 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q9/5390-30102-0015.wav` | 0.000 |
| dac | n_q_12 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q12/5390-30102-0015.wav` | 0.000 |
| opus | opus_6000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br6000/5390-30102-0015.wav` | 0.000 |
| opus | opus_8000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br8000/5390-30102-0015.wav` | 0.000 |
| opus | opus_12000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br12000/5390-30102-0015.wav` | 0.000 |
| opus | opus_16000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br16000/5390-30102-0015.wav` | 0.000 |
| opus | opus_24000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br24000/5390-30102-0015.wav` | 0.000 |

## Sample `8098-278278-0035`

| Method | Setting | Path | WER vs GT |
|---|---|---|---:|
| (original) | -- | `samples/original/8098-278278-0035.wav` | -- |
| scit_base | L=1 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_base/L1/8098-278278-0035.wav` | 0.462 |
| scit_base | L=2 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_base/L2/8098-278278-0035.wav` | 0.103 |
| scit_base | L=3 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_base/L3/8098-278278-0035.wav` | 0.128 |
| scit_lca | L=1 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_lca/L1/8098-278278-0035.wav` | 0.385 |
| scit_lca | L=2 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_lca/L2/8098-278278-0035.wav` | 0.154 |
| scit_lca | L=3 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_lca/L3/8098-278278-0035.wav` | 0.103 |
| pcm | 16bit_16khz_passthrough | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/pcm/8098-278278-0035.wav` | 0.154 |
| encodec | bw1.5kbps_n_cb2 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/encodec/bw1.5kbps/8098-278278-0035.wav` | 0.205 |
| encodec | bw3.0kbps_n_cb4 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/encodec/bw3.0kbps/8098-278278-0035.wav` | 0.128 |
| encodec | bw6.0kbps_n_cb8 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/encodec/bw6.0kbps/8098-278278-0035.wav` | 0.128 |
| encodec | bw12.0kbps_n_cb16 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/encodec/bw12.0kbps/8098-278278-0035.wav` | 0.103 |
| dac | n_q_1 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q1/8098-278278-0035.wav` | 0.769 |
| dac | n_q_2 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q2/8098-278278-0035.wav` | 0.308 |
| dac | n_q_3 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q3/8098-278278-0035.wav` | 0.154 |
| dac | n_q_4 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q4/8098-278278-0035.wav` | 0.154 |
| dac | n_q_6 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q6/8098-278278-0035.wav` | 0.103 |
| dac | n_q_9 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q9/8098-278278-0035.wav` | 0.103 |
| dac | n_q_12 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q12/8098-278278-0035.wav` | 0.154 |
| opus | opus_6000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br6000/8098-278278-0035.wav` | 0.128 |
| opus | opus_8000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br8000/8098-278278-0035.wav` | 0.128 |
| opus | opus_12000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br12000/8098-278278-0035.wav` | 0.128 |
| opus | opus_16000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br16000/8098-278278-0035.wav` | 0.103 |
| opus | opus_24000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br24000/8098-278278-0035.wav` | 0.103 |

## Sample `8838-298545-0037`

| Method | Setting | Path | WER vs GT |
|---|---|---|---:|
| (original) | -- | `samples/original/8838-298545-0037.wav` | -- |
| scit_base | L=1 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_base/L1/8838-298545-0037.wav` | 0.545 |
| scit_base | L=2 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_base/L2/8838-298545-0037.wav` | 0.455 |
| scit_base | L=3 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_base/L3/8838-298545-0037.wav` | 0.182 |
| scit_lca | L=1 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_lca/L1/8838-298545-0037.wav` | 0.545 |
| scit_lca | L=2 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_lca/L2/8838-298545-0037.wav` | 0.273 |
| scit_lca | L=3 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/scit_lca/L3/8838-298545-0037.wav` | 0.295 |
| pcm | 16bit_16khz_passthrough | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/pcm/8838-298545-0037.wav` | 0.000 |
| encodec | bw1.5kbps_n_cb2 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/encodec/bw1.5kbps/8838-298545-0037.wav` | 0.250 |
| encodec | bw3.0kbps_n_cb4 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/encodec/bw3.0kbps/8838-298545-0037.wav` | 0.045 |
| encodec | bw6.0kbps_n_cb8 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/encodec/bw6.0kbps/8838-298545-0037.wav` | 0.045 |
| encodec | bw12.0kbps_n_cb16 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/encodec/bw12.0kbps/8838-298545-0037.wav` | 0.000 |
| dac | n_q_1 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q1/8838-298545-0037.wav` | 0.909 |
| dac | n_q_2 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q2/8838-298545-0037.wav` | 0.432 |
| dac | n_q_3 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q3/8838-298545-0037.wav` | 0.250 |
| dac | n_q_4 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q4/8838-298545-0037.wav` | 0.205 |
| dac | n_q_6 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q6/8838-298545-0037.wav` | 0.136 |
| dac | n_q_9 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q9/8838-298545-0037.wav` | 0.114 |
| dac | n_q_12 | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/dac/n_q12/8838-298545-0037.wav` | 0.023 |
| opus | opus_6000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br6000/8838-298545-0037.wav` | 0.205 |
| opus | opus_8000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br8000/8838-298545-0037.wav` | 0.023 |
| opus | opus_12000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br12000/8838-298545-0037.wav` | 0.182 |
| opus | opus_16000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br16000/8838-298545-0037.wav` | 0.023 |
| opus | opus_24000bps | `output/experiments/exp4_baseline_comparison_20260531_seed42/samples/opus/br24000/8838-298545-0037.wav` | 0.000 |

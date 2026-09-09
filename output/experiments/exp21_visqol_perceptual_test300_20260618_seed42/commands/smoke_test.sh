#!/usr/bin/env bash
set -uo pipefail
cd "$HOME/visqol"
VB=./bazel-bin/visqol
EX12=/mnt/h/H-CODE/speechtokenizer/output/experiments/exp12_baseline_comparison_test300_20260610_seed42/runs/test-clean_300/samples
REF="$EX12/original/1089-134686-0000.wav"

echo "REF exists: $([ -f "$REF" ] && echo yes || echo NO)"
echo "=== PCM passthrough (expect ~4.6-5.0) ==="
"$VB" --reference_file "$REF" --degraded_file "$EX12/pcm/1089-134686-0000.wav" --use_speech_mode 2>&1 | grep -iE "MOS-LQO" || echo "(no score line)"
echo "=== SCIT-LCA L1 (expect lower) ==="
"$VB" --reference_file "$REF" --degraded_file "$EX12/scit_lca/L1/1089-134686-0000.wav" --use_speech_mode 2>&1 | grep -iE "MOS-LQO" || echo "(no score line)"
echo "=== batch-mode smoke (3 pairs) ==="
head -4 /mnt/h/H-CODE/speechtokenizer/output/experiments/_visqol_setup/batch_0.csv > /tmp/smoke_batch.csv
"$VB" --batch_input_csv /tmp/smoke_batch.csv --results_csv /tmp/smoke_results.csv --use_speech_mode 2>&1 | tail -3
echo "--- results ---"; cat /tmp/smoke_results.csv
echo "SMOKE_DONE"

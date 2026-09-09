#!/usr/bin/env bash
# Run ViSQOL (speech mode) over all shards in parallel inside WSL.
# Each shard: batch CSV in -> results CSV out (reference,degraded,moslqo).
set -uo pipefail
export PATH="$HOME/bin:$PATH"

VBIN="$HOME/visqol/bazel-bin/visqol"
VROOT="$HOME/visqol"          # CWD must be repo root for model file relative path
SETUP="/mnt/h/H-CODE/speechtokenizer/output/experiments/_visqol_setup"
N_SHARDS=7

if [ ! -x "$VBIN" ]; then echo "VISQOL BINARY MISSING: $VBIN"; exit 1; fi

cd "$VROOT"   # visqol needs to find model/ relative paths
pids=()
for s in $(seq 0 $((N_SHARDS-1))); do
  (
    "$VBIN" \
      --batch_input_csv "$SETUP/batch_${s}.csv" \
      --results_csv "$SETUP/results_${s}.csv" \
      --use_speech_mode \
      > "$SETUP/run_${s}.log" 2>&1
    echo "shard ${s} exit=$? -> results_${s}.csv"
  ) &
  pids+=($!)
done

fail=0
for p in "${pids[@]}"; do
  wait "$p" || fail=1
done
echo "ALL_SHARDS_DONE fail=$fail"

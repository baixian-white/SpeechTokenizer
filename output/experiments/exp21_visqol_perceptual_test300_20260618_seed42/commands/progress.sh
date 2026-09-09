#!/usr/bin/env bash
# Progress check for the parallel ViSQOL run.
S=/mnt/h/H-CODE/speechtokenizer/output/experiments/_visqol_setup
echo "visqol procs running: $(pgrep -c visqol 2>/dev/null || echo 0)"
total=0
for i in 0 1 2 3 4 5 6; do
  # each pair prints one "MOS-LQO:" line to the per-shard run log
  n=$(grep -c "MOS-LQO" "$S/run_${i}.log" 2>/dev/null || echo 0)
  done_csv=0
  [ -f "$S/results_${i}.csv" ] && done_csv=$(($(wc -l < "$S/results_${i}.csv") - 1))
  echo "shard ${i}: scored=${n} / 3257   (results_csv rows=${done_csv})"
  total=$((total + n))
done
echo "TOTAL scored so far: ${total} / 22800"

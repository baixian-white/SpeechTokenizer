# Failure Report

- experiment: exp4
- run_id: exp4_baseline_comparison_20260526_202329_seed42
- failed_step: Opus/AMR-WB traditional codec baseline
- reason: required traditional codec command not found (`ffmpeg` or `opusenc/opusdec`).
- produced_partial_artifacts: SCIT codes/reconstructions/payload, PCM baseline payload, sample manifest.
- next_step: install or provide a traditional codec tool, then rerun codec baseline commands using the same test list and packet schema.

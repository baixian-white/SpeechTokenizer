# Legacy Experiment Files Archive

This folder stores legacy experiment outputs, temporary files, old NAS artifacts, notebooks, and previous workspace copies that were moved out of the active project tree before starting the new experiment plan.

Purpose:

- Keep previous runs and generated data available for traceability.
- Prevent new experiments from accidentally reading old logs, checkpoints, NAS search databases, temporary files, or previous workspace copies.
- Preserve old files without deleting them.

Archive created for:

```text
pre_new_experiments_cleanup_20260526
```

Archived groups:

- `root/Log`: previous training logs.
- `root/runs`: previous run directories.
- `root/旧结果`: previous result folder.
- `root/tmp`: previous temporary files.
- `root/SpeechTokenizer-main*`: previous source/archive copy.
- `root/speechtokenizer_now*`: previous source/archive copy.
- `root/tmp_manuscript_extract.txt`: previous temporary manuscript extraction.
- `notebooks/`: previous exploratory notebooks.
- `nas/artifacts`: previous NAS database, records, and subset files.
- `nas/架构搜索结果`: previous NAS audio and spectrogram outputs.
- `nas/best_seanet_config.json`: previous exported NAS best config.

Clean directories recreated for new experiments:

```text
Log/
runs/
tmp/
output/experiments/
nas/artifacts/
nas/架构搜索结果/
```

Notes:

- `data/` and `model_hub/` were intentionally left in place because they may be required by training and inference.
- `output/doc/` was intentionally left in place because it contains the current paper and experiment planning documents.
- New experiments should write all run-specific outputs under `output/experiments/{run_id}/`.

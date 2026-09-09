# New Experiment Runs

Use this directory only for new experiment runs created after the legacy cleanup.

Each run should use a unique `run_id` directory:

```text
output/experiments/{run_id}/
```

Recommended run layout:

```text
configs/
commands/
logs/
checkpoints/
metrics/
samples/
reports/
artifacts/
```

Legacy experiment files were archived at:

```text
output/archive/legacy_experiment_files_20260526_pre_new_experiments/
```

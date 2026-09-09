# Exp3 Change Log

This is a monitored short debug/tracer LCA run:

- Uses Exp2 debug Base checkpoint, not a full 60 epoch Base model.
- Uses normalized subset manifests derived from the original split.
- `batch_size=2`, `num_workers=0`, `valid_num_workers=0`.
- `max_train_steps=3` to cover L=1/2/3 and clean/dropout/substitution once.
- Dropout implementation: previous-index replacement.
- Substitution: legal codebook index replacement.

Do not report this run as a full SCIT-Speech-LCA main result.

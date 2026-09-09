# Exp5 Change Log

This run uses a minimum debug/tracer ablation matrix. All variants use the same normalized train/valid subset and fixed sample list. Training variants keep max_train_steps from their source debug configs.

- `hand_encoder_base`: removes `nas_encoder_config`; otherwise follows Exp2 debug config.
- `no_semantic_distill`: sets `distill_loss_lambda=0`; otherwise follows Exp2 debug config.
- `random_l_off_lca`: uses LCA debug config with `random_l_sampling.values=[3,3,3]`.

Do not report this as full paper-grade ablation evidence.

# Exp2 Config Preparation

- run_id: exp2_scit_speech_distill30_retrain_20260528_seed42
- mode: formal
- config: output\experiments\exp2_scit_speech_distill30_retrain_20260528_seed42\configs\scit_speech_base_config.json
- results_folder: output\experiments\exp2_scit_speech_distill30_retrain_20260528_seed42\checkpoints
- nas_encoder_config: output\experiments\exp2_scit_speech_distill30_retrain_20260528_seed42\configs\best_seanet_config.json
- distill_loss_lambda: 30.0
- distill_loss_lambda_source: override
- experiment_tag: distill30_retrain
- experiment_note: Formal full retrain from scratch with distill_loss_lambda=30, motivated by S2-C positive finetune result.
- note: this script prepares training inputs only; it does not train or evaluate the model.

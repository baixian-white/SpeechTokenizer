# Environment for exp4

## Repository state
- Repo path: `h:/H-CODE/speechtokenizer`
- git HEAD: `cb9929cb60e05540b24d59bc68f117187219d810`
- git dirty (worktree changes):
```
 M config/spt_base_cfg.json
 M nas/SeaNet.py
 D nas/best_seanet_config.json
 M nas/model_components.py
 M nas/run_nas.sh
 M nas/search_autoencoder.py
 M nas/train_nas.py
 M scripts/train_example.py
 M scripts/train_example.sh
 M speechtokenizer/trainer/dataset.py
 M speechtokenizer/trainer/loss.py
 M speechtokenizer/trainer/trainer.py
 D "\345\257\271\346\257\224wav\350\257\255\351\237\263\345\267\256\350\267\235.ipynb"
 D "\350\257\273\345\217\226wav\346\226\207\344\273\266\344\277\241\346\201\257.ipynb"
?? nas/encoder_handoff.py
?? nas/encoder_only_model_variant.py
?? nas/evaluate_encoder_proxy.py
?? nas/run_staged_encoder_nas.py
?? nas/search_space.py
?? nas/teacher_guided_proxy.py
?? nas/validate_short_distill.py
?? output/
?? scripts/build_exp4_supplementary.py
?? scripts/channel_sim.py
?? scripts/codebook_usage_report.py
?? scripts/collect_environment.py
?? scripts/compare_reconstruction_audio.py
?? scripts/convert_single_flac_to_wav.py
?? scripts/create_experiment_run.py
?? scripts/evaluate_ablation_variants.py
?? scripts/evaluate_layer_reconstruction.py
?? scripts/evaluate_lca_vs_base.py
?? scripts/evaluate_sample_audio_quality.py
?? scripts/exp2_supplementary.py
?? scripts/experiment_utils.py
?? scripts/export_full_utterance_samples.py
?? scripts/export_loss_curves.py
?? scripts/extract_best_from_trainer.py
?? scripts/migrate_paths_e_to_h.py
?? scripts/normalize_filelist_paths.py
?? scripts/pack_indices.py
?? scripts/package_checkpoint.py
?? scripts/package_exp2_outputs.py
?? scripts/payload_accounting.py
?? scripts/prefetch_baseline_models.py
?? scripts/preflight_experiments.py
?? scripts/preflight_smoke_test.py
?? scripts/prepare_exp2_config.py
?? scripts/prepare_exp2_supplementary.py
?? scripts/review_experiment_outputs.py
?? scripts/run_asr_evaluation.py
?? scripts/run_exp2_training.bat
?? scripts/run_exp2_training.ps1
?? scripts/run_exp4_baselines.py
?? scripts/run_opus_baseline.py
?? scripts/summarize_exp2.py
?? scripts/train_decoder_only_finetune.py
?? scripts/train_distill_weight_ablation.py
?? scripts/train_lca.py
?? speechtokenizer/trainer/lca_trainer.py
?? tests/

```

## Hardware
- Platform: `Windows-10-10.0.22631-SP0`
- CUDA available: `True`
- CUDA device: `NVIDIA GeForce RTX 5070 Ti`
- VRAM (used at exp4 eval, single GPU): ~3 GB peak

## Conda environment
- Env name: `speechtokenizer`
- Env path: `C:\Users\Windows11\.conda\envs\speechtokenizer`
- Python: `3.10.19`

## Critical packages (full list in tool_versions.md)
- torch 2.9.1+cu128
- torchaudio 2.9.1+cu128
- whisper 20250625
- jiwer 4.0.0
- conda-forge ffmpeg 6.1.2 with libopus

## Reproducibility notes
- All decoded audio is 16 kHz mono WAV
- Whisper transcription uses `language='en'`, `condition_on_previous_text=False` (no inter-sample state)
- ChannelSim is not applied in exp4 (clean operating points only); exp3 covers perturbation evaluation
- All evaluation pairs share the same 8-sample fixed test set
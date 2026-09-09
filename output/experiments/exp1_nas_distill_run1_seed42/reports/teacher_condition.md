# Teacher Condition

- teacher_enabled: True
- teacher_config: model_hub\speechtokenizer_hubert_avg\config.json
- teacher_checkpoint: model_hub\speechtokenizer_hubert_avg\SpeechTokenizer.pt
- teacher_target: pre_rvq
- teacher_cache_mode: disk
- teacher_role: frozen SpeechTokenizer encoder pre-RVQ latent anchor.
- teacher_quantizer_role: frozen RVQ compatibility diagnostic only.
- short_distillation_enabled: True
- short_distillation_steps: stage2=100, stage3=500, final=1500
- short_distillation_optimized_module: NAS candidate encoder only.
- downstream_modules_for_proxy: frozen pretrained SpeechTokenizer transform/RVQ/decoder.
- teacher_is_method_contribution: false

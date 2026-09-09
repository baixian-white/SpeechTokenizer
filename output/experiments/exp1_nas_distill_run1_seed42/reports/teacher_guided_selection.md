# Staged NAS Selection

- pipeline: stage1 profile -> stage2 short distillation + proxy -> stage3 short distillation + refined proxy -> stage4 final short distillation + Pareto
- selection_mode: balanced
- short_distillation_steps: stage2=100, stage3=500, final=1500
- short_distillation_loss: 1.0*SmoothL1(Z_B,Z_A) + 1.0*cosine_distance + 0.5*temporal_delta_loss
- short_distillation_updates: NAS candidate encoder only
- frozen_proxy_components: pretrained SpeechTokenizer transform, RVQ, and decoder
- capacity_guard: min_n_filters=24, max_skip_blocks=1
- stage1_generated: 8192
- stage1_to_stage2: 512
- stage2_to_stage3: 64
- stage3_to_final: 8
- selected_candidate: nas_seed42_000896
- selected_config: artifacts/best_architecture/best_seanet_config.json
- stage1_score: 0.45*log(MAC_ratio)+0.45*log(param_ratio)+0.10*log(RTF_ratio)
- proxy_quality_penalty: weighted positive log excess over hand-designed encoder for semantic_proxy_loss, proxy_recon_l1, and proxy_mel_loss
- proxy_resource_score: weighted log ratios for encoder_macs, encoder_params, and encoder_rtf_mean
- quality_margins: semantic=0.1, recon=0.1, mel=0.1
- weights: semantic=1.0, recon=0.6, mel=0.6, macs=0.12, params=0.12, rtf=0.03
- pareto_objectives: teacher_latent_smooth_l1, teacher_latent_cosine_distance, teacher_temporal_delta_loss, rvq_quantized_feature_l1, semantic_proxy_loss, proxy_recon_l1, proxy_mel_loss, encoder_macs, encoder_params, encoder_rtf_mean
- selected_encoder_strides: [5, 4, 4, 4]
- selected_decoder_strides: [8, 5, 4, 2]
- decoder_condition: frozen_teacher_decoder

This is a staged proxy NAS result. It must not be presented as a fully trained SCIT-Speech result.

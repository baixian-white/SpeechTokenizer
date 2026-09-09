# Controller Preflight Report

- status: preflight_failed
- python: `C:\Users\Windows11\.conda\envs\speechtokenizer\python.exe`
- torch: `2.9.1+cu128`
- cuda: `{'available': True, 'torch_cuda': '12.8', 'device_count': 1, 'devices': ['NVIDIA GeForce RTX 5070 Ti']}`
- disk_free_gb: 58.61

## Fixed Condition Check
- passed

## Model/Data Smoke Test
- passed: `{'data_batch_audio_shape': [1, 1, 16000], 'data_batch_semantic_shape': [1, 768, 50], 'forward_output_shape': [1, 1, 16000], 'feature_shape': [1, 50, 768], 'commit_loss_is_finite': True, 'codes_shape': [3, 1, 50], 'decoded_shapes': {'L1': [1, 1, 16000], 'L2': [1, 1, 16000], 'L3': [1, 1, 16000]}, 'expected_latent_frames_for_segment': 50, 'observed_latent_frames': 50, 'codes_shape_valid': True, 'latent_frame_check': True}`

## Payload Toy Check
- result: `{'input_shape': [3, 5], 'sim_output_shape': [2, 5], 'sim_stats': {'shape': [2, 5], 'L': 2, 'codebook_size': 1024, 'p_drop': 0.2, 'p_sub': 0.1, 'seed': 42, 'total_indices': 10, 'replaced_by_previous': 2, 'substituted': 1, 'actual_p_drop': 0.2, 'actual_p_sub': 0.1}, 'packed_bytes': 13, 'ideal_bits_L2_for_5_frames': 100}`

## Per-Experiment Status
- exp1: preflight_passed (exp1_nas_semantic_encoder_20260526_202329_seed42)
- exp2: preflight_failed (exp2_scit_speech_training_20260526_202329_seed42)
- exp3: preflight_failed (exp3_low_load_channel_aware_adaptation_20260526_202329_seed42)
- exp4: preflight_failed (exp4_baseline_comparison_20260526_202329_seed42)
- exp5: preflight_failed (exp5_ablation_and_diagnosis_20260526_202329_seed42)

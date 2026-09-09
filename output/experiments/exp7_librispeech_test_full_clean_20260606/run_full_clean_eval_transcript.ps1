Set-Location 'H:\H-CODE\speechtokenizer'
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path 'output/experiments/exp7_librispeech_test_full_clean_20260606/logs' | Out-Null
Start-Transcript -Path 'output/experiments/exp7_librispeech_test_full_clean_20260606/logs/full_clean_eval_transcript.log' -Append
Write-Host '=== SCIT full LibriSpeech clean eval started ==='
Write-Host ('Started at: ' + (Get-Date))
Write-Host 'Step 1/2: test-clean all samples'
& 'C:\Users\Windows11\.conda\envs\speechtokenizer\python.exe' scripts/evaluate_clean_large_nosave.py --run-dir output/experiments/exp7_librispeech_test_full_clean_20260606/eval_clean_full_lca_nosave/test-clean --base-config output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/configs/scit_speech_base_config.json --base-checkpoint output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/SCIT-Speech-Base_best.pt --lca-config output/experiments/exp6_librispeech_test_subset_20260606/configs/full_lca_clean_eval_config.json --lca-checkpoint output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/checkpoints/SCIT-Speech-LCA_v2_step30000_robust_optimum.pt --sample-list output/experiments/exp7_librispeech_test_full_clean_20260606/artifacts/test-clean_all_files.txt --device cuda
Write-Host 'Step 2/2: test-other all samples'
& 'C:\Users\Windows11\.conda\envs\speechtokenizer\python.exe' scripts/evaluate_clean_large_nosave.py --run-dir output/experiments/exp7_librispeech_test_full_clean_20260606/eval_clean_full_lca_nosave/test-other --base-config output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/configs/scit_speech_base_config.json --base-checkpoint output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/SCIT-Speech-Base_best.pt --lca-config output/experiments/exp6_librispeech_test_subset_20260606/configs/full_lca_clean_eval_config.json --lca-checkpoint output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/checkpoints/SCIT-Speech-LCA_v2_step30000_robust_optimum.pt --sample-list output/experiments/exp7_librispeech_test_full_clean_20260606/artifacts/test-other_all_files.txt --device cuda
Write-Host ('Finished at: ' + (Get-Date))
Write-Host '=== SCIT full LibriSpeech clean eval finished ==='
Stop-Transcript

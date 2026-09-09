Set-Location 'H:\H-CODE\speechtokenizer'
$env:PYTHONUNBUFFERED='1'
New-Item -ItemType Directory -Force -Path 'H:\H-CODE\speechtokenizer\output\experiments\exp11_perturbed_asr_wer_20260609\logs' | Out-Null
'USER_VISIBLE_PERT_ASR_TEST_CLEAN_100_STARTED ' + (Get-Date) | Set-Content 'H:\H-CODE\speechtokenizer\output\experiments\exp11_perturbed_asr_wer_20260609\logs\user_visible_test_clean_100_marker.txt'
Start-Transcript -Path 'H:\H-CODE\speechtokenizer\output\experiments\exp11_perturbed_asr_wer_20260609\logs\test-clean_100_perturbed_asr_user_visible_transcript.log' -Append
Write-Host '=== SCIT perturbed ASR/WER test-clean 100 ==='
Write-Host ('Started at: ' + (Get-Date))
& 'C:\Users\Windows11\.conda\envs\speechtokenizer\python.exe' --version
& 'C:\Users\Windows11\.conda\envs\speechtokenizer\python.exe' scripts/evaluate_perturbed_asr_wer_onthefly.py `
  --run-dir output/experiments/exp11_perturbed_asr_wer_20260609/eval_perturbed_asr_wer/test-clean_100 `
  --base-config output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/configs/scit_speech_base_config.json `
  --base-checkpoint output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/SCIT-Speech-Base_best.pt `
  --lca-config output/experiments/exp6_librispeech_test_subset_20260606/configs/full_lca_perturb_eval_config.json `
  --lca-checkpoint output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/checkpoints/SCIT-Speech-LCA_v2_step30000_robust_optimum.pt `
  --sample-list output/experiments/exp7_librispeech_test_full_clean_20260606/artifacts/test-clean_all_files.txt `
  --max-samples 100 `
  --save-sample-count 5 `
  --whisper-model base.en `
  --device cuda
Write-Host ('EXITCODE=' + $LASTEXITCODE)
Write-Host ('Finished at: ' + (Get-Date))
Stop-Transcript


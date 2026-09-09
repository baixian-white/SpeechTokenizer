Set-Location 'H:\H-CODE\speechtokenizer'
$env:PYTHONUNBUFFERED='1'
New-Item -ItemType Directory -Force -Path 'H:\H-CODE\speechtokenizer\output\experiments\exp10_asr_wer_onthefly_20260609\logs' | Out-Null
'USER_VISIBLE_ASR_TEST_OTHER_300_STARTED ' + (Get-Date) | Set-Content 'H:\H-CODE\speechtokenizer\output\experiments\exp10_asr_wer_onthefly_20260609\logs\user_visible_test_other_300_marker.txt'
Start-Transcript -Path 'H:\H-CODE\speechtokenizer\output\experiments\exp10_asr_wer_onthefly_20260609\logs\test-other_300_asr_user_visible_transcript.log' -Append
Write-Host '=== SCIT ASR/WER test-other 300, user-visible PowerShell ==='
Write-Host ('Started at: ' + (Get-Date))
& 'C:\Users\Windows11\.conda\envs\speechtokenizer\python.exe' --version
& 'C:\Users\Windows11\.conda\envs\speechtokenizer\python.exe' scripts/evaluate_asr_wer_onthefly.py `
  --run-dir output/experiments/exp10_asr_wer_onthefly_20260609/eval_asr_wer_clean/test-other_300 `
  --base-config output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/configs/scit_speech_base_config.json `
  --base-checkpoint output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/SCIT-Speech-Base_best.pt `
  --lca-config output/experiments/exp6_librispeech_test_subset_20260606/configs/full_lca_clean_eval_config.json `
  --lca-checkpoint output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/checkpoints/SCIT-Speech-LCA_v2_step30000_robust_optimum.pt `
  --sample-list output/experiments/exp7_librispeech_test_full_clean_20260606/artifacts/test-other_all_files.txt `
  --max-samples 300 `
  --save-sample-count 5 `
  --whisper-model base.en `
  --device cuda
Write-Host ('EXITCODE=' + $LASTEXITCODE)
Write-Host ('Finished at: ' + (Get-Date))
Stop-Transcript


Set-Location 'H:\H-CODE\speechtokenizer'
$env:PYTHONUNBUFFERED='1'
New-Item -ItemType Directory -Force -Path 'H:\H-CODE\speechtokenizer\output\experiments\exp9_librispeech_test_full_packet_burst_20260608\logs' | Out-Null
'USER_VISIBLE_PACKET_BURST_TEST_OTHER_STARTED ' + (Get-Date) | Set-Content 'H:\H-CODE\speechtokenizer\output\experiments\exp9_librispeech_test_full_packet_burst_20260608\logs\user_visible_test_other_marker.txt'
Start-Transcript -Path 'H:\H-CODE\speechtokenizer\output\experiments\exp9_librispeech_test_full_packet_burst_20260608\logs\test-other_packet_burst_user_visible_transcript.log' -Append
Write-Host '=== SCIT packet/burst test-other, user-visible PowerShell ==='
Write-Host ('Started at: ' + (Get-Date))
Write-Host 'Python:'
& 'C:\Users\Windows11\.conda\envs\speechtokenizer\python.exe' --version
& 'C:\Users\Windows11\.conda\envs\speechtokenizer\python.exe' scripts/evaluate_packet_burst_large_nosave.py `
  --run-dir output/experiments/exp9_librispeech_test_full_packet_burst_20260608/eval_packet_burst_full_lca_nosave/test-other `
  --base-config output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/configs/scit_speech_base_config.json `
  --base-checkpoint output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/SCIT-Speech-Base_best.pt `
  --lca-config output/experiments/exp6_librispeech_test_subset_20260606/configs/full_lca_perturb_eval_config.json `
  --lca-checkpoint output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/checkpoints/SCIT-Speech-LCA_v2_step30000_robust_optimum.pt `
  --sample-list output/experiments/exp7_librispeech_test_full_clean_20260606/artifacts/test-other_all_files.txt `
  --save-sample-count 5 `
  --device cuda
Write-Host ('EXITCODE=' + $LASTEXITCODE)
Write-Host ('Finished at: ' + (Get-Date))
Stop-Transcript


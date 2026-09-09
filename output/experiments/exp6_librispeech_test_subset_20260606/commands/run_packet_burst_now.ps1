$ErrorActionPreference = 'Stop'
Set-Location 'H:\H-CODE\speechtokenizer'
$root = 'output\experiments\exp6_librispeech_test_subset_20260606'
$baseConfig = 'output\experiments\exp2_scit_speech_distill30_retrain_20260529_seed42\configs\scit_speech_base_config.json'
$baseCkpt = 'output\experiments\exp2_scit_speech_distill30_retrain_20260529_seed42\checkpoints\SCIT-Speech-Base_best.pt'
$lcaConfig = 'output\experiments\exp6_librispeech_test_subset_20260606\configs\full_lca_perturb_eval_config.json'
$lcaCkpt = 'output\experiments\exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42\checkpoints\SCIT-Speech-LCA_v2_step30000_robust_optimum.pt'
foreach ($split in @('test-clean','test-other')) {
  Write-Host "=== Packet/Burst $split ==="
  $sampleList = Join-Path $root "artifacts\${split}_100_files.txt"
  $runDir = Join-Path $root "eval_packet_burst_full_lca\$split"
  & 'C:\Users\Windows11\.conda\envs\speechtokenizer\python.exe' scripts/evaluate_packet_burst_loss.py --run-dir $runDir --base-config $baseConfig --base-checkpoint $baseCkpt --lca-config $lcaConfig --lca-checkpoint $lcaCkpt --sample-list $sampleList --max-samples 100 --device cuda --channel-seed 123
  if ($LASTEXITCODE -ne 0) { throw "packet/burst $split failed" }
}

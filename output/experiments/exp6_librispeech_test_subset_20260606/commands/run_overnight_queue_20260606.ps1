$ErrorActionPreference = 'Stop'
Set-Location 'H:\H-CODE\speechtokenizer'
$root = 'output\experiments\exp6_librispeech_test_subset_20260606'
$log = Join-Path $root 'reports\overnight_queue_20260606.log'
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
function Log($msg) {
  $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
  $line = "[$ts] $msg"
  Write-Host $line
  Add-Content -Path $log -Value $line -Encoding UTF8
}
function IsDone($path) { Test-Path $path }
Log 'Overnight queue started.'
$pertClean = Join-Path $root 'eval_perturb_full_lca\test-clean\metrics\base_vs_lca_results.json'
$pertOther = Join-Path $root 'eval_perturb_full_lca\test-other\metrics\base_vs_lca_results.json'
while (-not ((IsDone $pertClean) -and (IsDone $pertOther))) {
  $wClean = (Get-ChildItem (Join-Path $root 'eval_perturb_full_lca\test-clean') -Recurse -File -Filter '*.wav' -ErrorAction SilentlyContinue | Measure-Object).Count
  $wOther = (Get-ChildItem (Join-Path $root 'eval_perturb_full_lca\test-other') -Recurse -File -Filter '*.wav' -ErrorAction SilentlyContinue | Measure-Object).Count
  Log "Waiting for perturb eval: test-clean wavs=$wClean done=$(IsDone $pertClean); test-other wavs=$wOther done=$(IsDone $pertOther)"
  Start-Sleep -Seconds 300
}
Log 'Perturb eval completed. Starting packet/burst eval.'
$baseConfig = 'output\experiments\exp2_scit_speech_distill30_retrain_20260529_seed42\configs\scit_speech_base_config.json'
$baseCkpt = 'output\experiments\exp2_scit_speech_distill30_retrain_20260529_seed42\checkpoints\SCIT-Speech-Base_best.pt'
$lcaConfig = 'output\experiments\exp6_librispeech_test_subset_20260606\configs\full_lca_perturb_eval_config.json'
$lcaCkpt = 'output\experiments\exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42\checkpoints\SCIT-Speech-LCA_v2_step30000_robust_optimum.pt'
foreach ($split in @('test-clean','test-other')) {
  $sampleList = Join-Path $root "artifacts\${split}_100_files.txt"
  $runDir = Join-Path $root "eval_packet_burst_full_lca\$split"
  $doneJson = Join-Path $runDir 'metrics\packet_burst_results.json'
  if (Test-Path $doneJson) {
    Log "Packet/burst $split already completed; skipping."
    continue
  }
  Log "Starting packet/burst $split."
  & 'C:\Users\Windows11\.conda\envs\speechtokenizer\python.exe' scripts/evaluate_packet_burst_loss.py --run-dir $runDir --base-config $baseConfig --base-checkpoint $baseCkpt --lca-config $lcaConfig --lca-checkpoint $lcaCkpt --sample-list $sampleList --max-samples 100 --device cuda --channel-seed 123
  if ($LASTEXITCODE -ne 0) { throw "packet/burst $split failed with exit code $LASTEXITCODE" }
  Log "Finished packet/burst $split."
}
Log 'Overnight queue completed.'

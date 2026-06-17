[CmdletBinding()]
param(
    [int]$Seed = 42,
    [string]$RunId = "",
    [string]$BaseConfig = "config/spt_base_cfg.json",
    [string]$Exp1BestConfig = "output/experiments/exp1_nas_distill_run1_seed42/artifacts/best_architecture/best_seanet_config.json",
    [Nullable[Double]]$DistillLossLambda = $null,
    [string]$ExperimentTag = "",
    [string]$ExperimentNote = "",
    [switch]$DebugRun,
    [switch]$TracerRun
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RunId)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $RunId = "exp2_scit_speech_training_${stamp}_seed${Seed}"
}

$runDir = Join-Path "output/experiments" $RunId
$configPath = Join-Path $runDir "configs/scit_speech_base_config.json"

$prepareArgs = @(
    "scripts/prepare_exp2_config.py",
    "--run-id", $RunId,
    "--seed", "$Seed",
    "--base-config", $BaseConfig,
    "--exp1-best-config", $Exp1BestConfig
)

if ($DebugRun) {
    $prepareArgs += "--debug"
}
if ($TracerRun) {
    $prepareArgs += "--tracer"
}
if ($null -ne $DistillLossLambda) {
    $prepareArgs += @("--distill-loss-lambda", "$DistillLossLambda")
}
if (-not [string]::IsNullOrWhiteSpace($ExperimentTag)) {
    $prepareArgs += @("--experiment-tag", $ExperimentTag)
}
if (-not [string]::IsNullOrWhiteSpace($ExperimentNote)) {
    $prepareArgs += @("--experiment-note", $ExperimentNote)
}

Write-Host "Preparing Exp2 run: $RunId"
python @prepareArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Starting Exp2 training with config: $configPath"
accelerate launch scripts/train_example.py --config $configPath
exit $LASTEXITCODE

[CmdletBinding()]
param(
    [ValidateSet('Preflight','Checks','Prepare','Smoke','Formal','Switch','Stability','RunPlan','Summarize')]
    [string]$Mode = 'Preflight',
    [string]$Python = 'E:\Miniconda\envs\st_audio_py310\python.exe',
    [string]$QualityPython = 'E:\Miniconda\envs\st_audio_py310\python.exe',
    [ValidateSet('strict','balanced')][string]$MemoryPolicy = 'strict',
    [ValidateSet('local_cpu','gpu_worker')][string]$ComputeBackend = 'local_cpu',
    [string]$ClassifierCandidates = '',
    [string[]]$Conditions = @('A','B','C','D'),
    [string[]]$Scenarios = @('sequential','concurrent'),
    [ValidateSet('smoke','formal')][string]$Dataset = 'smoke',
    [int]$Repeats = 0,
    [ValidateSet('off','device')][string]$Playback = 'off',
    [double]$DurationSeconds = 1800,
    [string]$OutputDirectory = '',
    [string]$PlanPath = '',
    [string]$RunDirectory = ''
)

$ErrorActionPreference = 'Stop'
$env:MKL_THREADING_LAYER = 'sequential'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
$project = Join-Path $PSScriptRoot 'project'
if (-not (Test-Path -LiteralPath $project -PathType Container)) {
    $current = Get-Item -LiteralPath $PSScriptRoot
    while ($null -ne $current -and -not (Test-Path -LiteralPath (Join-Path $current.FullName 'scripts\speaker_identity_utils.py'))) {
        $current = $current.Parent
    }
    if ($null -eq $current) { throw 'Cannot locate the experiment project.' }
    $project = $current.FullName
}
# Enumerate the top-level demo directory without assuming its Unicode spelling.
$demoRoot = @(Get-ChildItem -LiteralPath $project -Directory | Where-Object { $_.Name -like '3*demo' })
$launcher = @($demoRoot | Get-ChildItem -Recurse -File -Filter 'run_metrics_experiment.py')
if ($launcher.Count -ne 1) { throw 'Expected one run_metrics_experiment.py entry point.' }
$demo = $launcher[0].Directory.FullName

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Experiment command failed with exit code $LASTEXITCODE. See the preceding diagnostics and run logs." }
}

function Invoke-CodeChecks {
        Invoke-CheckedPython @('-m','unittest','tests.test_metrics_experiment','tests.test_stage1_six_user','tests.test_demo_exp23_speaker_identity','tests.test_demo_speaker_identity','-v')
        $syntax = @('-m','py_compile') + @('metrics_protocol.py','metrics_client.py','quality_metrics.py','run_metrics_experiment.py','summarize_metrics_experiment.py','router_server.py' | ForEach-Object { Join-Path $demo $_ })
        Invoke-CheckedPython $syntax
        $manifest = Join-Path (Split-Path $project -Parent) 'FILE_SHA256.csv'
        if (Test-Path -LiteralPath $manifest) {
            $package = Split-Path $project -Parent
            foreach ($row in (Import-Csv -LiteralPath $manifest)) {
                $path = Join-Path $package $row.relative_path
                if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing packaged file: $path" }
                if ((Get-Item -LiteralPath $path).Length -ne [long]$row.bytes -or (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $row.sha256) {
                    throw "Packaged file mismatch: $path"
                }
            }
            Write-Host 'Package SHA-256: PASS'
        }
        Write-Host 'Code checks: PASS'
}

Push-Location -LiteralPath $project
try {
    if ($Mode -in @('Checks','Smoke','Formal','Switch','Stability','RunPlan')) {
        Invoke-CodeChecks
    }
    if ($Mode -eq 'Checks') {
        return
    }
    if ($Mode -eq 'Summarize') {
        if (-not $RunDirectory) { throw 'Supply -RunDirectory with the generated experiment directory.' }
        Invoke-CheckedPython @((Join-Path $demo 'summarize_metrics_experiment.py'), $RunDirectory)
        return
    }
    $command = @($launcher[0].FullName, '--quality-python', $QualityPython, '--ram-policy', $MemoryPolicy,
                 '--compute-backend', $ComputeBackend, '--classifier-candidates', $ClassifierCandidates)
    if ($Mode -eq 'RunPlan') {
        if (-not $PlanPath) { throw 'Supply -PlanPath with experiment_plan.json.' }
        Invoke-CheckedPython ($command + @('--run-plan', $PlanPath))
        return
    }
    if ($Mode -eq 'Formal') { $Dataset = 'formal' }
    if ($Mode -eq 'Smoke') { $Dataset = 'smoke' }
    if ($Mode -eq 'Switch') { $Scenarios = 'switch' }
    if ($Mode -eq 'Stability') {
        $Scenarios = 'stability'
        if ($Repeats -eq 0) { $Repeats = 1 }
    }
    $command += @('--mode', $Dataset, '--conditions', ($Conditions -join ','), '--scenarios', ($Scenarios -join ','),
                  '--playback', $Playback, '--duration-seconds', [string]$DurationSeconds)
    if ($Repeats -gt 0) { $command += @('--repeats', [string]$Repeats) }
    if ($OutputDirectory) { $command += @('--output-dir', $OutputDirectory) }
    if ($Mode -eq 'Prepare') { $command += '--prepare-only' }
    if ($Mode -eq 'Preflight') { $command += '--preflight-only' }
    Invoke-CheckedPython $command
}
finally {
    Pop-Location
}

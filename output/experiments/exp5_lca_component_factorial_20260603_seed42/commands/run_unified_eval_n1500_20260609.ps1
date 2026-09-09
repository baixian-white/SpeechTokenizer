# Crash-resilient V0..V4 driver:
# - calls evaluate_lca_vs_base.py with --no-save-wavs and a per-variant --skip-samples-file
# - on segfault (any non-zero exit), inspects the variant's csv to find the latest
#   completed sample, picks the next sample in valid_files.txt as the suspected
#   crash sample, appends it to the skip-list, then re-runs the variant; resume
#   logic inside the python script continues from the existing csv.
# - per-variant retry cap to avoid infinite loops.

$ErrorActionPreference = 'Continue'
Set-Location 'H:\H-CODE\speechtokenizer'

# Reduce GPU memory fragmentation under Windows WDDM where the device is shared
# with desktop GPU users (browsers, VS Code WebView, etc). PyTorch's docs
# recommend this when "reserved but unallocated" memory is large; it lets the
# allocator expand existing segments instead of carving fresh ones.
$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'

$ROOT = 'output\experiments\exp5_lca_component_factorial_20260603_seed42'
$EVAL_DIR = "$ROOT\eval_unified_n64_20260609"
$BASE_CFG = 'output\experiments\exp2_scit_speech_distill30_retrain_20260529_seed42\configs\scit_speech_base_config.json'
$BASE_CKPT = 'output\experiments\exp2_scit_speech_distill30_retrain_20260529_seed42\checkpoints\SCIT-Speech-Base_best.pt'
$SAMPLE_LIST = "$ROOT\artifacts\valid_files.txt"
$PY = 'C:\Users\Windows11\.conda\envs\speechtokenizer\python.exe'
$N = 256
$SEED = 42
$MAX_RETRIES_PER_VARIANT = 100  # OOM restarts and segfault skips share this counter

$VARIANTS = @(
    @{ Name = 'V0_clean_control_step30000';        CfgFile = "$ROOT\configs\eval\V0_clean_control_step30000.json";        Ckpt = "$ROOT\runs\V0_full_depth_clean_control\checkpoints\SpeechTokenizerTrainer_00030000" },
    @{ Name = 'V1_random_l_only_step25000';        CfgFile = "$ROOT\configs\eval\V1_random_l_only_step25000.json";        Ckpt = "$ROOT\runs\V1_random_l_only\checkpoints\SpeechTokenizerTrainer_00025000" },
    @{ Name = 'V2_channelsim_only_step17500';      CfgFile = "$ROOT\configs\eval\V2_channelsim_only_step17500.json";      Ckpt = "$ROOT\runs\V2_channelsim_only\checkpoints\SpeechTokenizerTrainer_00017500" },
    @{ Name = 'V3_random_l_channelsim_step32500';  CfgFile = "$ROOT\configs\eval\V3_random_l_channelsim_step32500.json";  Ckpt = "$ROOT\runs\V3_random_l_channelsim\checkpoints\SpeechTokenizerTrainer_00032500" },
    @{ Name = 'V4_full_lca_step30000';             CfgFile = "$ROOT\configs\eval\V4_full_lca_step30000.json";             Ckpt = 'output\experiments\exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42\checkpoints\SCIT-Speech-LCA_v2_step30000_robust_optimum.pt' }
)

New-Item -ItemType Directory -Force -Path $EVAL_DIR | Out-Null

# Read valid_files.txt once: index by line number for sample_id lookup
$validRows = Get-Content $SAMPLE_LIST
$validIds = @()
foreach ($line in $validRows) {
    $audioPath = $line.Split("`t")[0]
    $sid = [System.IO.Path]::GetFileNameWithoutExtension($audioPath)
    $validIds += $sid
}

foreach ($v in $VARIANTS) {
    $name = $v.Name
    $cfg = $v.CfgFile
    $ckpt = $v.Ckpt
    $runDir = "$EVAL_DIR\$name"
    $skipFile = "$runDir\skip_samples.txt"
    $csvPath = "$runDir\metrics\base_vs_lca_results.csv"
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null
    if (-not (Test-Path $skipFile)) { New-Item -ItemType File -Path $skipFile -Force | Out-Null }

    Write-Host ""
    Write-Host "=== Evaluating $name (n=$N, no-save-wavs, resume+skip) ===" -ForegroundColor Cyan

    $attempt = 0
    while ($attempt -le $MAX_RETRIES_PER_VARIANT) {
        $attempt++
        $started = Get-Date

        & $PY scripts\evaluate_lca_vs_base.py `
            --run-dir $runDir `
            --base-config $BASE_CFG `
            --base-checkpoint $BASE_CKPT `
            --lca-config $cfg `
            --lca-checkpoint $ckpt `
            --sample-list $SAMPLE_LIST `
            --max-samples $N `
            --device cuda `
            --channel-seed $SEED `
            --no-save-wavs `
            --skip-pesq `
            --skip-samples-file $skipFile

        $rc = $LASTEXITCODE
        $elapsed = (Get-Date) - $started
        Write-Host ("attempt {0} of {1}: exit {2}, elapsed {3:N1} min" -f $attempt, $name, $rc, $elapsed.TotalMinutes) -ForegroundColor Yellow

        if ($rc -eq 0) {
            Write-Host "Done $name in $($attempt) attempt(s)" -ForegroundColor Green
            break
        }

        # Distinguish OOM-driven restart (exit 2) from native segfault (-1073741819 = 0xC0000005).
        # OOM: cuda context corrupted, just restart fresh process; resume picks up.
        # Segfault: a specific sample triggered native crash; add it to skip list.
        # Always sleep before any restart so the OS / WDDM has time to release the
        # previous process's GPU resources; on Windows WDDM the cuda context release
        # is asynchronous and a fast restart will see "phantom" reserved memory.
        Start-Sleep -Seconds 8

        if ($rc -eq 2) {
            Write-Host "  OOM-driven restart: skipping skip-list update; resume from csv." -ForegroundColor Magenta
            continue
        }
        if ($rc -eq 1) {
            Write-Host "  exit 1 (likely model-load OOM): treating as transient, no skip-list update." -ForegroundColor Magenta
            continue
        }

        # Crash: identify next not-yet-completed sample and add to skip list.
        if (-not (Test-Path $csvPath)) {
            Write-Host "FAILED: $name - no csv produced on attempt $attempt; aborting variant." -ForegroundColor Red
            exit 1
        }
        $doneIds = @{}
        Import-Csv $csvPath | ForEach-Object { $doneIds[$_.sample_id] = $true }

        # Read current skip list
        $currentSkip = @{}
        if (Test-Path $skipFile) {
            Get-Content $skipFile | Where-Object { $_.Trim() -ne '' } | ForEach-Object { $currentSkip[$_.Trim()] = $true }
        }

        # Find first valid_id that is neither done nor skipped: that's the suspect crash sample.
        $suspect = $null
        foreach ($sid in $validIds[0..($N-1)]) {
            if (-not $doneIds.ContainsKey($sid) -and -not $currentSkip.ContainsKey($sid)) {
                $suspect = $sid
                break
            }
        }

        if ($null -eq $suspect) {
            Write-Host "FAILED: $name - all samples accounted for but exit $rc; aborting." -ForegroundColor Red
            exit 1
        }

        Write-Host "  segfault suspect: $suspect; adding to skip list and retrying..." -ForegroundColor Magenta
        Add-Content -Path $skipFile -Value $suspect
    }

    if ($attempt -gt $MAX_RETRIES_PER_VARIANT) {
        Write-Host "FAILED: $name exceeded retry cap ($MAX_RETRIES_PER_VARIANT)" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "ALL VARIANTS DONE" -ForegroundColor Green

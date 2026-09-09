$ErrorActionPreference = 'Stop'
$taskArchive = Join-Path $PSScriptRoot 'speechtokenizer_no_datasets_20260908_142824.tar.zst'
$taskPartial = $taskArchive + '.partial'
$taskVerification = $taskArchive + '.verification.json'
$taskProgress = $taskArchive + '.progress.json'

Write-Host '每 5 秒刷新；按 Ctrl+C 退出查看（不会停止压缩）。'
Write-Host '依次显示：恢复检查、继续压缩、完整校验。各阶段百分比分开计算。'

while ($true) {
    if (Test-Path -LiteralPath $taskVerification) {
        $taskResult = Get-Content -LiteralPath $taskVerification -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($taskResult.success) {
            Write-Host "压缩及校验完成，共 $($taskResult.verified_entries) 个条目。" -ForegroundColor Green
        } else {
            Write-Host "校验未通过：$($taskResult.error)" -ForegroundColor Red
        }
        break
    }

    if (Test-Path -LiteralPath $taskProgress) {
        try {
            $taskStatus = Get-Content -LiteralPath $taskProgress -Raw -Encoding UTF8 | ConvertFrom-Json
            $taskStageName = switch ($taskStatus.stage) {
                'recovering' { '恢复检查' }
                'compressing' { '压缩' }
                'verifying' { '完整校验' }
                'complete' { '全部完成' }
                'failed' { '处理失败' }
                default { $taskStatus.stage }
            }
            Write-Host ('{0}  {1} {2:N2}% | 更新于 {3}' -f (Get-Date -Format 'HH:mm:ss'), $taskStageName, $taskStatus.percent, $taskStatus.updated_at)
            if ($taskStatus.stage -eq 'failed') {
                Write-Host $taskStatus.error -ForegroundColor Red
                break
            }
            if ($taskStatus.stage -eq 'complete') { break }
            Start-Sleep -Seconds 5
            continue
        } catch {
            Write-Host '正在更新进度，请稍候。'
        }
    }

    $taskCurrent = if (Test-Path -LiteralPath $taskArchive) { $taskArchive } else { $taskPartial }
    $taskState = if ($taskCurrent -eq $taskArchive) { '压缩完成，等待校验结果' } else { '打包尚未完成' }
    try {
        $taskStream = [System.IO.File]::Open($taskCurrent, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
        try { $taskGiB = $taskStream.Length / 1GB } finally { $taskStream.Dispose() }
        Write-Host ('{0:HH:mm:ss}  {1} | 已生成 {2:N2} GiB' -f (Get-Date), $taskState, $taskGiB)
    } catch {
        Write-Host ('暂时无法读取文件大小：' + $_.Exception.Message)
    }
    Start-Sleep -Seconds 5
}

Write-Host $taskArchive

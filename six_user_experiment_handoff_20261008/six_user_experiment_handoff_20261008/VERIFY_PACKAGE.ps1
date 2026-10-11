[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$manifest = Join-Path $root 'HANDOFF_SHA256.csv'
if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) { throw "Missing $manifest" }
$bad = @()
foreach ($row in (Import-Csv -LiteralPath $manifest)) {
    $path = Join-Path $root ($row.relative_path -replace '/', '\')
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        $bad += "MISSING $($row.relative_path)"
        continue
    }
    $item = Get-Item -LiteralPath $path
    $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ([long]$item.Length -ne [long]$row.bytes -or $hash -ne $row.sha256.ToLowerInvariant()) {
        $bad += "MISMATCH $($row.relative_path)"
    }
}
if ($bad.Count) { $bad | ForEach-Object { Write-Error $_ }; exit 1 }
Write-Host "HANDOFF SHA-256: PASS ($((Import-Csv -LiteralPath $manifest).Count) files)"

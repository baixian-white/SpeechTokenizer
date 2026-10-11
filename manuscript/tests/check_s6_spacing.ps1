param(
    [string]$PdfPath = (Join-Path $PSScriptRoot '..\supplementary_methods_v3.pdf')
)

$ErrorActionPreference = 'Stop'
$raw = (& pdftotext -f 12 -l 12 -bbox-layout $PdfPath -) -join "`n"
if ($LASTEXITCODE -ne 0) {
    throw "pdftotext failed for $PdfPath"
}

# pdfTeX can emit control characters in math text; they are irrelevant to these headings.
$xml = [xml]([regex]::Replace($raw, '[\x00-\x08\x0B\x0C\x0E-\x1F]', ''))
$ns = [System.Xml.XmlNamespaceManager]::new($xml.NameTable)
$ns.AddNamespace('x', 'http://www.w3.org/1999/xhtml')
$lines = @($xml.SelectNodes('//x:line', $ns) | ForEach-Object {
    [pscustomobject]@{
        Y = [double]$_.GetAttribute('yMin')
        X = [double]$_.GetAttribute('xMin')
        Text = (($_.SelectNodes('x:word', $ns) | ForEach-Object { $_.InnerText }) -join ' ')
    }
})

function Get-LeftLine([string]$prefix) {
    $match = @($lines | Where-Object { $_.X -lt 300 -and $_.Text.StartsWith($prefix) })
    if ($match.Count -ne 1) {
        throw "Expected one left-column line beginning '$prefix'; found $($match.Count)."
    }
    return $match[0]
}

$checks = @(
    @{ Name = 'S6 heading to S6.1'; Start = 'S6. Frozen-codec'; End = 'S6.1. Frozen inputs'; Max = 45 },
    @{ Name = 'S6.1 to first paragraph'; Start = 'S6.1. Frozen inputs'; End = 'The speaker experiment used'; Max = 35 },
    @{ Name = 'S6.2 to first paragraph'; Start = 'S6.2. Token embeddings'; End = 'Each RVQ layer has'; Max = 35 }
)

$failed = $false
foreach ($check in $checks) {
    $gap = (Get-LeftLine $check.End).Y - (Get-LeftLine $check.Start).Y
    '{0}: {1:N1} pt (limit {2} pt)' -f $check.Name, $gap, $check.Max
    if ($gap -gt $check.Max -or $gap -le 0) {
        $failed = $true
    }
}
$leftBottom = ($lines | Where-Object { $_.X -lt 290 } | Measure-Object -Property Y -Maximum).Maximum
'S6 left-column body bottom: {0:N1} pt (minimum 680 pt)' -f $leftBottom
if ($leftBottom -lt 680) {
    $failed = $true
}
if ($failed) {
    throw 'S6 has stretched spacing or a prematurely empty left column.'
}

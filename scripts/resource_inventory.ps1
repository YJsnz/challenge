param(
    [string[]]$Manifests = @('configs/resource_downloads.json','datasets/raw/industrial_behaviours/metadata/downloads.json'),
    [string]$Output = 'outputs/resource-download-report.json',
    [string]$Workspace = (Split-Path -Parent $PSScriptRoot)
)
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath($Workspace)
$records = [Collections.Generic.List[object]]::new()
foreach ($manifest in $Manifests) {
    $items = Get-Content -LiteralPath (Join-Path $root $manifest) -Raw | ConvertFrom-Json
    foreach ($item in $items) {
        $path = [IO.Path]::GetFullPath((Join-Path $root $item.path))
        if (-not $path.StartsWith($root+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) { throw 'Manifest path escapes workspace' }
        $state, $receipt = 'missing', $null
        if ((Test-Path -LiteralPath $path) -and (Test-Path -LiteralPath ($path+'.receipt.json'))) {
            $receipt = Get-Content -LiteralPath ($path+'.receipt.json') -Raw | ConvertFrom-Json
            if ($receipt.source_url -eq $item.url -and (Get-Item -LiteralPath $path).Length -eq $receipt.bytes -and
                ($null -eq $item.bytes -or $receipt.bytes -eq $item.bytes) -and
                (-not $item.sha256 -or $receipt.sha256 -eq $item.sha256)) { $state='downloaded_and_checked' }
            else { $state='receipt_mismatch' }
        } elseif (Test-Path -LiteralPath ($path+'.partial')) { $state='partial' }
        $records.Add([pscustomobject]@{path=$item.path;group=$item.group;state=$state;
            expected_bytes=$item.bytes;downloaded_bytes=if($receipt){$receipt.bytes}else{0};
            sha256=if($receipt){$receipt.sha256}else{$null};publisher_hash_available=[bool]$item.sha256})
    }
}
$report = [pscustomobject]@{created_at=(Get-Date).ToUniversalTime().ToString('o');
    expected_files=$records.Count;verified_files=@($records|Where-Object state -eq 'downloaded_and_checked').Count;
    total_downloaded_bytes=($records|Measure-Object downloaded_bytes -Sum).Sum;files=$records.ToArray();
    note='Receipts were created after full file hashing during download. Inventory checks receipts and current file sizes; no model loading or media decoding.'}
$outPath = [IO.Path]::GetFullPath((Join-Path $root $Output))
if (-not $outPath.StartsWith($root+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) { throw 'Report path escapes workspace' }
New-Item -ItemType Directory -Path (Split-Path -Parent $outPath) -Force | Out-Null
$report | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $outPath -Encoding utf8
$report | Select-Object expected_files,verified_files,total_downloaded_bytes

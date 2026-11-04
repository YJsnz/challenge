#Requires -Version 7.0
param(
    [ValidateRange(1,32)][int]$Workers = 8,
    [ValidateRange(1,10)][int]$Attempts = 3,
    [string]$Proxy = '',
    [string]$Workspace = (Split-Path -Parent $PSScriptRoot)
)
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath($Workspace)
$logRoot = Join-Path $root 'outputs/download-logs'
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
$statusPath = Join-Path $root 'outputs/resource-download-status.json'
$status = [ordered]@{state='downloading';pid=$PID;started_at=(Get-Date).ToUniversalTime().ToString('o');
    workers=$Workers;attempts=$Attempts;finished_at=$null;verified_files=$null;expected_files=$null;error=$null}
$status | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding utf8
try {
    & (Join-Path $PSScriptRoot 'download_resources.ps1') -Manifest (Join-Path $root 'configs/resource_downloads.json') -Proxy $Proxy -Workspace $root
    $downloadScript = Join-Path $PSScriptRoot 'download_resources.ps1'
    $results = @(0..31 | ForEach-Object -Parallel {
        $shard = $_
        $taskRoot, $taskLogs = $using:root, $using:logRoot
        $scriptPath, $taskProxy, $maxAttempts = $using:downloadScript, $using:Proxy, $using:Attempts
        $log = Join-Path $taskLogs ('industrial-'+$shard+'.log')
        $taskStatus = Join-Path $taskLogs ('industrial-'+$shard+'.status.json')
        $result = [ordered]@{shard=$shard;state='downloading';attempt=0;error=$null}
        for ($attempt=1; $attempt -le $maxAttempts; $attempt++) {
            $result.attempt = $attempt
            $result | ConvertTo-Json | Set-Content -LiteralPath $taskStatus -Encoding utf8
            try {
                & $scriptPath -Manifest (Join-Path $taskRoot 'configs/industrial_video_downloads.json') -Workspace $taskRoot -Proxy $taskProxy -ShardCount 32 -ShardIndex $shard *>> $log
                $result.state, $result.error = 'complete', $null
                break
            } catch {
                $result.error = $_.Exception.Message
                ('Attempt '+$attempt+': '+$result.error) | Add-Content -LiteralPath $log -Encoding utf8
                if ($attempt -lt $maxAttempts) { Start-Sleep -Seconds 2 }
            }
        }
        if ($result.state -ne 'complete') { $result.state='failed' }
        $result | ConvertTo-Json | Set-Content -LiteralPath $taskStatus -Encoding utf8
        [pscustomobject]$result
    } -ThrottleLimit $Workers)
    & (Join-Path $PSScriptRoot 'resource_inventory.ps1') -Workspace $root
    $report = Get-Content -LiteralPath (Join-Path $root 'outputs/resource-download-report.json') -Raw | ConvertFrom-Json
    $status.verified_files, $status.expected_files = $report.verified_files, $report.expected_files
    if (@($results | Where-Object state -ne 'complete').Count -or $report.verified_files -ne $report.expected_files) {
        throw 'Some resources remain incomplete; inspect the inventory and per-shard logs, then rerun to resume.'
    }
    $status.state='complete'
} catch {
    $status.state, $status.error = 'failed', $_.Exception.Message
} finally {
    $status.finished_at=(Get-Date).ToUniversalTime().ToString('o')
    $status | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding utf8
}
if ($status.state -ne 'complete') { Write-Error $status.error; exit 1 }

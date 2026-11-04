param(
    [Parameter(Mandatory=$true)][string]$Manifest,
    [string]$Group = '',
    [ValidateRange(1,32)][int]$ShardCount = 1,
    [ValidateRange(0,31)][int]$ShardIndex = 0,
    [string]$Proxy = '',
    [string]$Workspace = (Split-Path -Parent $PSScriptRoot)
)
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath($Workspace)
$items = Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
$selected = @($items | Where-Object { -not $Group -or $_.group -eq $Group })
if ($ShardIndex -ge $ShardCount) { throw 'ShardIndex must be less than ShardCount' }
$selected = @(for ($index=0; $index -lt $selected.Count; $index++) {
    if ($index % $ShardCount -eq $ShardIndex) { $selected[$index] }
})
if (-not $selected.Count) { throw 'No resources selected' }
foreach ($item in $selected) {
    $target = [IO.Path]::GetFullPath((Join-Path $root $item.path))
    if (-not $target.StartsWith($root + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Resource path escapes workspace'
    }
    $uri = [uri]$item.url
    if ($uri.Scheme -ne 'https') { throw 'Only HTTPS downloads are supported' }
    $parent = Split-Path -Parent $target
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    $partial = $target + '.partial'
    if (-not (Test-Path -LiteralPath $target)) {
        $argsList = @('--location','--fail','--retry','4','--retry-delay','2',
                      '--connect-timeout','25','--speed-limit','1024','--speed-time','60',
                      '--continue-at','-','--silent','--show-error','--output',$partial)
        if ($Proxy) { $argsList += @('--proxy', $Proxy) }
        $argsList += $item.url
        Write-Output ('Downloading ' + $item.path)
        & curl.exe @argsList
        if ($LASTEXITCODE -eq 33) {
            # Some download endpoints answer a Range request with a full response.
            # Keep the previous bytes and retry without Range; never append that response.
            if (Test-Path -LiteralPath $partial) {
                $backup = $partial + '.resume-backup-' + [guid]::NewGuid().ToString('N')
                Move-Item -LiteralPath $partial -Destination $backup
                Write-Output ('Retained previous partial file: ' + $backup)
            }
            $argsList = @($argsList | Where-Object { $_ -notin @('--continue-at','-') })
            Write-Output ('Retrying without Range: ' + $item.path)
            & curl.exe @argsList
        }
        if ($LASTEXITCODE -ne 0) { throw ('Download failed: ' + $item.path) }
    }
    $verify = if (Test-Path -LiteralPath $target) { $target } else { $partial }
    $length = (Get-Item -LiteralPath $verify).Length
    if ($null -ne $item.bytes -and $length -ne [long]$item.bytes) { throw ('Size mismatch: ' + $item.path) }
    $hash = (Get-FileHash -LiteralPath $verify -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($item.sha256 -and $hash -ne $item.sha256.ToLowerInvariant()) { throw ('SHA256 mismatch: ' + $item.path) }
    if ($verify -eq $partial) { Move-Item -LiteralPath $partial -Destination $target }
    [pscustomobject]@{path=$item.path;bytes=$length;sha256=$hash;source_url=$item.url;
        publisher_sha256_verified=[bool]$item.sha256;checked_at=(Get-Date).ToUniversalTime().ToString('o')} |
        ConvertTo-Json | Set-Content -LiteralPath ($target+'.receipt.json') -Encoding utf8
    Write-Output ('Verified ' + $item.path + ' (' + $length + ' bytes)')
}

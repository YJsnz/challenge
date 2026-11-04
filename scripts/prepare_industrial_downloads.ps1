param([string]$Proxy = '', [string]$Workspace = (Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath($Workspace)
$metadata = Join-Path $root 'datasets/raw/industrial_behaviours/metadata'
New-Item -ItemType Directory -Path $metadata -Force | Out-Null
$request = @{TimeoutSec=40; Headers=@{Accept='application/vnd.mendeley-public-dataset.1+json'}}
if ($Proxy) { $request.Proxy = $Proxy }
$base = 'https://data.mendeley.com/public-api/datasets/xjmtb22pff'
$folders = Invoke-RestMethod -Uri ($base+'/folders/1') @request
$folders | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $metadata 'folders.json') -Encoding utf8
$folderMap = @{}
foreach ($folder in $folders) { $folderMap[$folder.id] = $folder }
$manifest = [Collections.Generic.List[object]]::new()
$inventory = [Collections.Generic.List[object]]::new()
$folderIds = @('root') + @($folders.id)
$counter = 0
foreach ($id in $folderIds) {
    $start = 0
    do {
        $uri = $base+'/files?folder_id='+$id+'&version=1&%24start='+$start+'&%24limit=1000'
        $response = Invoke-RestMethod -Uri $uri @request
        $files = @($response | ForEach-Object { $_ })
        foreach ($file in $files) {
            if ($file.status -ne 'COMPLETED') { throw ('Unready publisher file: '+$file.id) }
            $parts = [Collections.Generic.List[string]]::new()
            $parentId = $file.folder_id
            $visited = @{}
            while ($parentId) {
                if ($visited.ContainsKey($parentId)) { throw 'Publisher folder cycle' }
                $visited[$parentId] = $true
                $folder = $folderMap[$parentId]
                if (-not $folder) { throw 'Missing publisher parent folder' }
                $parts.Insert(0,$folder.name)
                $parentId = $folder.parent_id
            }
            $parts.Add($file.filename)
            foreach ($name in $parts) {
                if ($name -in @('.','..') -or $name.IndexOfAny([IO.Path]::GetInvalidFileNameChars()) -ge 0) { throw 'Unsafe publisher filename' }
            }
            $manifest.Add([pscustomobject]@{group=('industrial-'+($counter % 4));
                path=('datasets/raw/industrial_behaviours/'+($parts -join '/'));
                url=$file.content_details.download_url; bytes=[long]$file.content_details.size;
                sha256=$file.content_details.sha256_hash})
            $inventory.Add($file)
            $counter++
        }
        $start += $files.Count
    } while ($files.Count -eq 1000)
    Write-Output ('Listed folder '+$id+': '+$start+' files')
}
if ($manifest.Count -ne @($manifest.path | Sort-Object -Unique).Count) { throw 'Duplicate publisher paths' }
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $metadata 'downloads.json') -Encoding utf8
$inventory | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $metadata 'publisher_files.json') -Encoding utf8
$total = ($manifest | Measure-Object bytes -Sum).Sum
Write-Output ('Total: '+$manifest.Count+' files; '+[long]$total+' bytes')
if ($total -gt (Get-PSDrive -Name ([IO.Path]::GetPathRoot($root).Substring(0,1))).Free - 5GB) {
    throw 'Insufficient free space with 5 GB reserve'
}

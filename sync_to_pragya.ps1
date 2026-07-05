# sync_to_pragya.ps1
# Push changed .py files to PRAGYA HPC via scp.
# Usage: .\sync_to_pragya.ps1

$ErrorActionPreference = "Stop"

$RemoteUser = "mt6210961"
$RemoteHost = "pragya.iitd.ac.in"
$RemotePath = "/home/maths/mtech/mt6210961/Static_coarsening/static_coarsening_2/befgc_hetero"
$LocalPath  = $PSScriptRoot

Write-Host "Syncing .py files to PRAGYA ($RemoteHost)..." -ForegroundColor Cyan

ssh "$RemoteUser@$RemoteHost" "mkdir -p '$RemotePath'"

$pyFiles = Get-ChildItem -Path $LocalPath -Recurse -Filter *.py |
    Where-Object { $_.FullName -notmatch '\\__pycache__\\' }

if (-not $pyFiles) {
    Write-Host "No .py files found to sync." -ForegroundColor Yellow
    exit 0
}

foreach ($file in $pyFiles) {
    $relative = $file.FullName.Substring($LocalPath.Length).TrimStart('\')
    $remoteRelative = $relative -replace '\\', '/'
    $remoteTarget = "$RemotePath/$remoteRelative"
    $remoteDir = Split-Path $remoteTarget -Parent

    Write-Host "  -> $remoteRelative" -ForegroundColor Gray

    $remoteDirUnix = $remoteDir -replace '\\', '/'
    ssh "$RemoteUser@$RemoteHost" "mkdir -p '$remoteDirUnix'"
    scp -q $file.FullName "${RemoteUser}@${RemoteHost}:$remoteTarget"
}

Write-Host "Sync complete." -ForegroundColor Green

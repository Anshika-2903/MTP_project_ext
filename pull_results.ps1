# pull_results.ps1
# Pull logs back from PRAGYA.
# Usage: .\pull_results.ps1

$ErrorActionPreference = "Stop"

$RemoteUser = "mt6210961"
$RemoteHost = "pragya.iitd.ac.in"
$RemotePath = "/home/maths/mtech/mt6210961/Static_coarsening/static_coarsening_2/befgc_hetero"
$LocalPath  = Join-Path $PSScriptRoot "results"

New-Item -ItemType Directory -Force -Path $LocalPath | Out-Null

Write-Host "Pulling logs + results from PRAGYA..." -ForegroundColor Cyan
scp -q "${RemoteUser}@${RemoteHost}:${RemotePath}/*.log" $LocalPath 2>$null
scp -q "${RemoteUser}@${RemoteHost}:${RemotePath}/benchmark_results.jsonl" $LocalPath 2>$null
Write-Host "Done. See $LocalPath" -ForegroundColor Green

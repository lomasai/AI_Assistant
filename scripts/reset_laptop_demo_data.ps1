param(
    [switch]$RemoveDeviceOverlay
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$DemoDb = Resolve-Path -Path "memory\laptop-demo.db" -ErrorAction SilentlyContinue
if ($DemoDb) {
    Remove-Item -LiteralPath $DemoDb.Path -Force
    Write-Host "Removed memory\laptop-demo.db"
}

if ($RemoveDeviceOverlay -and (Test-Path "config\device.yaml")) {
    Remove-Item -LiteralPath "config\device.yaml" -Force
    Write-Host "Removed config\device.yaml"
}

Write-Host "Laptop demo data reset complete."

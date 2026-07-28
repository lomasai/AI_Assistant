param(
    [string]$AdminToken = "",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    throw "Virtual environment not found. Run: python -m venv .venv; .\.venv\Scripts\pip.exe install -r requirements.txt"
}

if ($AdminToken.Trim()) {
    $env:ADMIN_API_TOKEN = $AdminToken.Trim()
}

Copy-Item -Path "config\laptop-demo.yaml" -Destination "config\device.yaml" -Force

Write-Host "Starting laptop mock demo on http://127.0.0.1:$Port"
Write-Host "Physical hardware output remains disabled by config/laptop-demo.yaml."
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port $Port

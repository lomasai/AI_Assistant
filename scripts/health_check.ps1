param(
    [int]$Port = 8000,
    [string]$AdminToken = ""
)

$ErrorActionPreference = "Stop"
$Headers = @{}
if ($AdminToken.Trim()) {
    $Headers["X-Admin-Token"] = $AdminToken.Trim()
}

$BaseUrl = "http://127.0.0.1:$Port"
Invoke-RestMethod -Uri "$BaseUrl/api/v1/health" -Headers $Headers | ConvertTo-Json -Depth 8
Invoke-RestMethod -Uri "$BaseUrl/camera/status" | ConvertTo-Json -Depth 8
Invoke-RestMethod -Uri "$BaseUrl/api/v1/audio/health" | ConvertTo-Json -Depth 8
Invoke-RestMethod -Uri "$BaseUrl/api/v1/engagement/health" -Headers $Headers | ConvertTo-Json -Depth 8
Invoke-RestMethod -Uri "$BaseUrl/api/v1/hardware/health" -Headers $Headers | ConvertTo-Json -Depth 8

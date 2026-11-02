# run_studio.ps1 -- start Render Studio (builds the UI on first run) and open it in the browser.
#   powershell -ExecutionPolicy Bypass -File studio\run_studio.ps1 [-Port 8765] [-NoBrowser] [-Rebuild]
param(
    [int]$Port = 8765,
    [switch]$NoBrowser,
    [switch]$Rebuild          # rebuild the UI even if studio\ui\dist exists
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$py = "$root\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "missing .venv; see README" }

$ui = "$root\studio\ui"
if ($Rebuild -or -not (Test-Path "$ui\dist\index.html")) {
    if (-not (Test-Path "$ui\node_modules")) { Write-Host "installing UI packages ..."; Push-Location $ui; npm install; Pop-Location }
    Write-Host "building UI ..."; Push-Location $ui; npm run build; Pop-Location
}

& $py -c "import fastapi, uvicorn, psutil, httpx" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "installing backend packages ..."
    uv pip install --python $py fastapi "uvicorn[standard]" python-multipart httpx psutil
}

$env:VG_WORK = "$root\work_ex"
$env:PYTHONIOENCODING = "utf-8"
$url = "http://127.0.0.1:$Port"
if (-not $NoBrowser) { Start-Job -ScriptBlock { param($u) Start-Sleep 2; Start-Process $u } -ArgumentList $url | Out-Null }
Write-Host "Render Studio: $url  (Ctrl+C stops it)"
& $py -m uvicorn studio.server:app --host 127.0.0.1 --port $Port --log-level warning

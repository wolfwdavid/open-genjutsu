# run_render.ps1 -- run a Wan2.2-Animate render OUTSIDE the agent shell (its background jobs are
# stopped when the system runs low on memory; a render started here is not).
#
#   powershell -ExecutionPolicy Bypass -File run_render.ps1 -Side cat
#   powershell -ExecutionPolicy Bypass -File run_render.ps1 -Side user -Ref refs\user_ref.png
#
# Safe to re-run: finished windows are skipped (--resume). Close heavy apps first: the Arc iGPU
# shares system RAM and a render uses ~28 GB of the 32 GB.
param(
    [ValidateSet("cat", "user")] [string]$Side = "cat",
    [string]$Ref = "",
    [string[]]$Blocks = @("A", "B")
)
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$log = Join-Path $root "work\round4\render_$Side.log"

if ($Side -eq "cat") {
    if (-not $Ref) { $Ref = "refs\cat_body_base_hq.png" }
    $perf = "R"; $poseDir = @("--pose-dir", "work\pose_R_cat")
    $prompt = "A real orange and white tabby cat with fur, pointed cat ears, whiskers and a pink cat nose, sitting upright on a wooden stool at a hanging studio microphone in front of an orange backdrop. The cat wears black sunglasses with dark opaque lenses. The cat bobs its head and lifts its furry front paws to the music. Photorealistic cat, soft frontal studio light."
    $neg = "human, person, man, woman, human skin, human hands, fingers, human arms, jeans, t-shirt, headphones, blue glasses, clear lenses"
} else {
    if (-not $Ref) { $Ref = "refs\user_ref.png" }
    $perf = "L"; $poseDir = @()
    $prompt = "The man from the reference photo, full body in exactly his own outfit, wearing black sunglasses with dark opaque lenses, standing at a hanging studio microphone in front of an orange backdrop, nodding and gesturing to the music. Photorealistic, soft frontal studio light."
    $neg = "clear glasses, white sunglasses, no sunglasses"
}
if (-not (Test-Path $Ref)) { throw "reference image not found: $Ref" }

# --- ComfyUI server (reuse if already up)
function Test-Comfy { try { Invoke-RestMethod -Uri "http://127.0.0.1:8188/system_stats" -TimeoutSec 3 | Out-Null; $true } catch { $false } }
$server = $null
if (-not (Test-Comfy)) {
    Write-Host "starting ComfyUI ..."
    $server = Start-Process -FilePath "$root\.venv-comfy\Scripts\python.exe" -PassThru -WindowStyle Minimized `
        -ArgumentList @("vendor\ComfyUI\main.py", "--listen", "127.0.0.1", "--port", "8188",
                        "--use-pytorch-cross-attention", "--disable-auto-launch",
                        "--cache-ram", "8", "--disable-pinned-memory") `
        -RedirectStandardOutput "$root\work\round4\comfy_server.log" -RedirectStandardError "$root\work\round4\comfy_server.err.log"
    for ($i = 0; $i -lt 120 -and -not (Test-Comfy); $i++) { Start-Sleep 2 }
    if (-not (Test-Comfy)) { throw "ComfyUI did not start; see work\round4\comfy_server.err.log" }
}

try {
    foreach ($b in $Blocks) {
        $name = "${Side}_$b"
        Write-Host "=== $name ==="
        $bg = @()
        if ($Side -eq "user") {
            $catFrames = Join-Path $root "work\wan\cat_$b\frames"
            if (-not (Test-Path $catFrames)) { throw "render the cat first: $catFrames not found" }
            $bg = @("--bg-frames-dir", $catFrames)
        }
        # PowerShell 5.1 turns native stderr lines into errors; never abort on a Python warning
        $ErrorActionPreference = "Continue"
        & "$root\.venv\Scripts\python.exe" wan_animate.py --block $b --side $perf --ref $Ref --name $name `
            @poseDir @bg --length 33 --resume --reuse-stage --prompt $prompt --negative-extra $neg 2>&1 |
            Tee-Object -FilePath $log -Append
        $code = $LASTEXITCODE
        $ErrorActionPreference = "Stop"
        if ($code -ne 0) { throw "render $name failed (exit $code); re-run this script to resume" }
    }
    Write-Host "ALL DONE: work\wan\${Side}_A.mp4, work\wan\${Side}_B.mp4"
} finally {
    if ($server) {
        # the venv python is a launcher: stop every process running this ComfyUI
        Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*vendor\ComfyUI\main.py*" } |
            ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    }
}

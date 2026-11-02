# run_cat_fix.ps1 -- re-render the two weak stretches of the cat pass and splice them in.
#   1. block A close-up (strided frames 179..238): fresh start at the camera cut, no carried drift
#   2. block B from frame 28 on: foreleg reach capped (wide arm spreads), seeded from frames 0..27
# Run it OUTSIDE the agent shell, like run_render.ps1:
#   powershell -ExecutionPolicy Bypass -File run_cat_fix.ps1
# Safe to re-run: finished windows are skipped; the splice backs up the old frames once.
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$log = Join-Path $root "work\round4\render_cat_fix.log"
$py = "$root\.venv\Scripts\python.exe"

$prompt = "A real orange and white tabby cat with fur, pointed cat ears, whiskers and a pink cat nose, sitting upright on a wooden stool at a hanging studio microphone in front of an orange backdrop. The cat wears black sunglasses with dark opaque black lenses. The cat bobs its head and lifts its furry front paws to the music. Photorealistic cat, natural colours, soft frontal studio light."
$neg = "human, person, man, woman, human skin, human hands, fingers, human arms, long arms, jeans, t-shirt, headphones, blue glasses, clear lenses, blue lenses, purple lenses, coloured lenses, reflective lenses, oversaturated, neon colours"

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

$jobs = @(
    @{ Block = "A"; Name = "cat_A_close"; Range = "179:239"; Seed = $null; Dst = "work\wan\cat_A\frames"; Start = 179 },
    @{ Block = "B"; Name = "cat_B_fix";   Range = "28:92";   Seed = "work\wan\cat_B\frames"; Dst = "work\wan\cat_B\frames"; Start = 28 }
)
try {
    foreach ($j in $jobs) {
        Write-Host "=== $($j.Name) ==="
        $extra = @("--range", $j.Range)
        if ($j.Seed) {
            # seed from the ORIGINAL frames (the backup, if a previous splice already ran)
            $seed = $j.Seed
            $bk = Join-Path (Split-Path $j.Dst -Parent) "frames_backup"
            if (Test-Path $bk) { $seed = $bk }
            $extra += @("--seed-frames", $seed)
        }
        $ErrorActionPreference = "Continue"
        & $py wan_animate.py --block $j.Block --side R --ref refs\cat_body_base_hq.png --name $j.Name `
            --pose-dir work\pose_R_cat --length 33 --resume @extra --prompt $prompt --negative-extra $neg 2>&1 |
            Tee-Object -FilePath $log -Append
        $code = $LASTEXITCODE
        $ErrorActionPreference = "Stop"
        if ($code -ne 0) { throw "render $($j.Name) failed (exit $code); re-run this script to resume" }
        & $py splice_frames.py --src "work\wan\$($j.Name)\frames" --dst $j.Dst --start $j.Start 2>&1 | Tee-Object -FilePath $log -Append
    }
    Write-Host "ALL DONE" ; "ALL DONE" | Out-File -FilePath $log -Append
} finally {
    if ($server) {
        Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*vendor\ComfyUI\main.py*" } |
            ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    }
}

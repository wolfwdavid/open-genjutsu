# run_example.ps1 -- recast the Genjutsu example (work_ex/cut.mp4): the user's cat replaces the
# grey cat on the LEFT (fur + glasses only: cap, flannel and stool stay from the example) and the
# user replaces the dancer on the RIGHT. Run it OUTSIDE the agent shell:
#   powershell -ExecutionPolicy Bypass -File run_example.ps1 -Side cat            # ~1.5 h
#   powershell -ExecutionPolicy Bypass -File run_example.ps1 -Side user           # ~1.5 h, after the cat
#   ... -Probe                                                                     # first 33 frames only (~15 min)
#   ... -Side cat -FixShot 2                                                       # re-render one shot of the cat and splice it in
# Safe to re-run: finished windows are skipped (--resume).
param(
    [ValidateSet("cat", "user")] [string]$Side = "cat",
    [switch]$Probe,
    [int]$FixShot = 0,         # 1..5: re-render only that camera shot of the cat (tight mask, no finger points)
    [double]$ReserveVram = 4   # GB kept free: forces a partial load of the 11.2 GB Wan model so KSampler has room (2 still full-loads and OOMs)
)
# shot boundaries in strided frames (cut frames 104/184/276/346 of the 24 fps example, stride 2)
$shots = @{ 1 = "0:52"; 2 = "52:92"; 3 = "92:138"; 4 = "138:173"; 5 = "173:181" }
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$env:VG_WORK = Join-Path $root "work_ex"
$py = "$root\.venv\Scripts\python.exe"
$tag = $(if ($Probe) { "probe" } elseif ($FixShot) { "fix$FixShot" } else { "full" })
$log = Join-Path $root "work_ex\round4\render_${Side}_$tag.log"

$common = @("--block", "E", "--blocks", "work_ex\blocks.json", "--width", "768", "--height", "576", "--length", "33", "--stride", "2")
if ($Side -eq "cat") {
    $name = "cat_E"
    # whole cat (no cap, no flannel): the full character mask, not the fur-only one
    $args1 = @("--side", "L", "--ref", "refs\cat_body_base_hq.png",
               "--block-px", "20", "--mask-grow", "6")   # 20 src px = 16 model px: aligned with the latent grid
    $prompt = "A real orange and white tabby cat with orange tabby fur, white chest and white paws, pink cat nose and whiskers, wearing classic black sunglasses with thick black frames and solid black opaque lenses you cannot see through, sitting upright on a wooden stool at a hanging studio microphone in front of an orange backdrop, rapping and gesturing with its front paws. The cat wears nothing else: no hat, no clothes, just fur. Photorealistic cat fur, soft studio light."
    $neg = "hat, cap, baseball cap, shirt, flannel, plaid, clothes, clothing, jacket, grey fur, gray cat, blue glasses, blue frames, clear lenses, transparent lenses, tinted lenses, visible eyes behind glasses, blue lenses, purple lenses, black patches, black blocks, human, human hands, fingers, oversaturated"
} else {
    $name = "user_E"
    if (-not (Test-Path "work_ex\wan\cat_E\frames")) { throw "render the cat first (run_example.ps1 -Side cat)" }
    # paint over the deflickered cat frames when they exist (deflicker.py; kept in sync by -FixShot)
    $bg = $(if (Test-Path "work_ex\wan\cat_E\frames_df") { "work_ex\wan\cat_E\frames_df" } else { "work_ex\wan\cat_E\frames" })
    Write-Host "background: $bg"
    $args1 = @("--side", "R", "--ref", "refs\user_ref.png", "--bg-frames-dir", $bg)
    $prompt = "The man from the reference photo, in his black suit, black shirt and black tie, wearing black sunglasses with dark opaque lenses, dancing and gesturing to the music next to a hanging studio microphone in front of an orange backdrop. Photorealistic, soft studio light."
    $neg = "orange tracksuit, woman, long hair, clear glasses, no sunglasses, white sunglasses"
}
if ($FixShot) {
    if ($Side -ne "cat") { throw "-FixShot is for the cat pass" }
    $range = $shots[$FixShot]
    if (-not $range) { throw "-FixShot must be 1..5" }
    $name = "cat_E_shot$FixShot"
    $args1 = @("--side", "L", "--ref", "refs\cat_body_base_hq.png", "--mask-dir", "work_ex\mask_Ltight",
               "--pose-dir", "work_ex\pose_L_nohands", "--block-px", "20", "--mask-grow", "0",
               "--range", $range, "--seed", "$(20260926 + 101 * $FixShot)")
    $prompt = $prompt.Replace("sitting upright on a wooden stool", "close-up of its upper body, leaning forward, compact natural cat proportions, sitting on a wooden stool")
    $neg += ", long neck, stretched body, elongated torso, melted, blurry paws, smeared paws, missing lens, one lens"
    $common += @("--resume")
} elseif ($Probe) { $name = "${name}_probe"; $common += @("--probe", "33") } else { $common += @("--resume") }

function Test-Comfy { try { Invoke-RestMethod -Uri "http://127.0.0.1:8188/system_stats" -TimeoutSec 3 | Out-Null; $true } catch { $false } }
$server = $null
if (-not (Test-Comfy)) {
    Write-Host "starting ComfyUI ..."
    $server = Start-Process -FilePath "$root\.venv-comfy\Scripts\python.exe" -PassThru -WindowStyle Minimized `
        -ArgumentList @("vendor\ComfyUI\main.py", "--listen", "127.0.0.1", "--port", "8188",
                        "--use-pytorch-cross-attention", "--disable-auto-launch",
                        "--cache-ram", "8", "--disable-pinned-memory", "--reserve-vram", "$ReserveVram") `
        -RedirectStandardOutput "$root\work_ex\round4\comfy_server.log" -RedirectStandardError "$root\work_ex\round4\comfy_server.err.log"
    for ($i = 0; $i -lt 120 -and -not (Test-Comfy); $i++) { Start-Sleep 2 }
    if (-not (Test-Comfy)) { throw "ComfyUI did not start; see work_ex\round4\comfy_server.err.log" }
}
try {
    Write-Host "=== $name ==="
    $ErrorActionPreference = "Continue"
    & $py wan_animate.py @common @args1 --name $name --prompt $prompt --negative-extra $neg 2>&1 |
        ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
    $code = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    if ($code -ne 0) { throw "render $name failed (exit $code); re-run the same command to resume" }
    if ($FixShot) {
        $start = $range.Split(":")[0]
        & $py splice_frames.py --src "work_ex\wan\$name\frames" --dst "work_ex\wan\cat_E\frames" --start $start 2>&1 |
            ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
        if (Test-Path "work_ex\wan\cat_E\frames_df") {   # re-derive so the user pass never sees a stale background
            & $py deflicker.py --in "work_ex\wan\cat_E\frames" --src "work_ex\wan\cat_E\in\bg" --out "work_ex\wan\cat_E\frames_df" 2>&1 |
                ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
        }
    }
    Write-Host "ALL DONE: work_ex\wan\$name.mp4"; "ALL DONE" | Out-File -FilePath $log -Append -Encoding utf8
} finally {
    if ($server) {
        Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*vendor\ComfyUI\main.py*" } |
            ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    }
}

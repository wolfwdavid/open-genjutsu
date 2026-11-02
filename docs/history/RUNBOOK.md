# RUNBOOK - cat duet rebuild from work/cut.mp4 + faces/

Rebuilds work/final.mp4 (720x1280, 29.1 s, original audio) from scratch:
USER's face on the LEFT performer (inswapper under his white sunglasses),
LivePortrait-animated CAT HEAD with blue round glasses on the RIGHT performer.

All commands run from the repo root in Git Bash on Windows. Always prefix
python with PYTHONIOENCODING=utf-8 (cp1252 console). Runtimes below were
measured on 2026-09-24/25 on a Core Ultra 9 285H + Arc 140T iGPU while other jobs
shared the GPU; expect them to be 20-40 % faster on an idle box. "Fix-2" below
= the 2026-09-25 rebuild (swap.py occlusion handling + composite.py hand/lens/
wide-shot/collar fixes). "Fix-3" = the 2026-09-25 round-3 rebuild (swap.py track-driven
yaw gate for the extreme-profile close-up f573-580; composite.py watershed hand/face split,
per-shot canonical-face reference, forehead zone, chain-line/mic-zone bright paint-back,
wide-shot dread-fragment filter); its numbers are the current reference.

Two venvs, never mixed:
- .venv      python 3.11, onnxruntime-directml 1.24.4, insightface 2.0 (buffalo_l in
             ~/.insightface/models/buffalo_l), opencv 5.0, numpy, scipy, pillow(+heif),
             scikit-image, tqdm.       -> track.py, make_cat_ref.py, swap.py, composite.py, finish.py
- .venv-lp   python 3.11, torch 2.14.0+xpu, torchvision 0.29.0+xpu, numpy 2.4.6,
             opencv-python-headless 5.0, pyyaml, scipy, imageio(+ffmpeg), pillow, rich, tqdm.
             -> animate_cat.py only (LivePortrait animals v1.1 on the Intel XPU)

ffmpeg / ffprobe must be on PATH. Fonts: compose.py picks C:/Windows/Fonts/segoeuib.ttf.

## 0. Inputs and one-time setup

Inputs that must exist (all gitignored):
- work/cut.mp4        29.07 s, 960x532, 30 fps, 872 frames (the COLORS duet cut)
- work/audio.m4a      original audio of the cut (29.118 s), used only by finish.py:
                        ffmpeg -y -i work/cut.mp4 -vn -c:a copy work/audio.m4a
- faces/user/selected/*.jpg   6 frontal selfie frames of the user (identity cos 0.91-0.96)
- faces/cat/IMG_9325.PNG      hero cat photo (frontal, sharpest)
- models/inswapper_128.onnx, models/gfpgan_1.4.onnx   (facefusion-assets models-3.0.0)
- models/isnet-general-use.onnx                        (rembg ISNet, make_cat_ref downloads it once)
- models/xseg_1.onnx  (70 MB, facefusion-assets models-3.1.0) OPTIONAL: XSeg face-occluder used by
  swap.py's occlusion handling (`--occluder-model`); without it the skin/reference rules still run
- vendor/LivePortrait          unmodified clone of KlingAIResearch/LivePortrait @ 9b294b3 with
  vendor/LivePortrait/pretrained_weights/liveportrait_animals/base_models_v1.1/
  {appearance_feature_extractor,motion_extractor,warping_module,spade_generator}.pth
  (no XPose / insightface weights needed: we crop with our own tracks)

Venvs:
```
uv venv .venv --python 3.11
uv pip install --python .venv/Scripts/python.exe -r requirements.txt

uv venv .venv-lp --python 3.11
uv pip install --python .venv-lp/Scripts/python.exe torch==2.14.0+xpu torchvision==0.29.0+xpu \
    --index-url https://download.pytorch.org/whl/xpu
uv pip install --python .venv-lp/Scripts/python.exe numpy opencv-python-headless pyyaml scipy \
    imageio imageio-ffmpeg pillow rich tqdm
```
Do NOT install insightface, onnxruntime-* or LivePortrait's own requirements.txt into .venv-lp.

## 1. track.py  (.venv, ~2-3 min)

Per-frame 30 Hz face tracks for both performers -> contract (A).
```
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe track.py
```
Writes work/tracks30.json (872 entries for L and R: bbox, 5 kps, lmk106, yaw/pitch/roll,
eye_c, iod, present/held; One-Euro smoothed, reset at cuts 104/239/461/581/688),
work/track_overlay.mp4 and work/track_sheet.jpg. Runtime was not logged; the script was
saved at 22:43:27 and tracks30.json landed at 22:46:20, so under 3 min for 872 frames
(buffalo_l det@640 on DirectML, re-detect @1280 when a face is missing). `--no-video`
skips overlay + sheet. Check work/track_sheet.jpg: both boxes present at 1/5/12/17/21.8/27 s.

Knobs: `--min-cutoff 1.0 --beta 0.02` (One-Euro; lower cutoff = smoother, more lag),
`--min-det 0.45`, `--cuts` (comma list of frame indices).

## 2. make_cat_ref.py  (.venv, ~5 s)

Cat LivePortrait source + alpha + landmarks -> contract (B).
```
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe make_cat_ref.py
```
Writes refs/cat_src_512.png, refs/cat_src_glasses_512.png (blue round glasses baked in),
refs/cat_alpha_512.png, refs/cat_meta.json, assets/glasses_blue_512.png, work/cat_ref_sheet.jpg.
Check the sheet: head upright, ears inside, eyes at ~0.42 of the height, set-orange padding.

The clipped image-right ear tip is rebuilt by mirroring the left ear (`--no-mirror-ear` to skip);
the glasses have short symmetric temple stubs (<= 0.15 iod).
Knobs: `--points x1,y1,...` (hero-photo landmarks: eyes, nose, mouth in 1125x1484 px),
`--side-factor 4.1` (crop side / inter-ocular distance; bigger = smaller head in the 512),
`--eye-y 0.42`, `--glasses-r 0.44 --glasses-stroke 7` (lens radius x iod, frame stroke px),
`--warm 0.07`, `--no-matte` (skip ISNet), `--set-frame 300` (frame sampled for the set colour).
IMPORTANT: animate_cat.py loads the glasses PNG at start - rerun step 3 after any change here.

## 3. animate_cat.py  (.venv-lp on xpu, ~5 min: 0.24-0.35 s/frame fp16 batch 4)

Drives the cat head with the RIGHT performer's pose -> contract (C).
```
PYTHONIOENCODING=utf-8 .venv-lp/Scripts/python.exe animate_cat.py \
    --src refs/cat_src_glasses_512.png --video work/cut.mp4 --tracks work/tracks30.json \
    --sheet work/cat_anim_sheet.jpg
```
Writes work/cat_anim/00000.png .. 00871.png (512x512 BGR) + work/cat_anim/motion.json +
work/cat_anim_sheet.jpg. Observed (QA-fix run, one pass): 872 frames rendered in 247.4 s
(0.284 s/frame, batch 4) while swap.py shared the GPU; render all 872 in ONE pass so motion.json
describes every frame (a chained-range run only records the last range).
Pose-only transfer relative to the neutral driving frame (t=13.47 s): expression/lips/eyes
stay as in the photo (mouth CLOSED by decision). `--start N --end M` renders a range
(the PNGs are per-frame, so ranges can be chained; motion.json is rewritten each run for the
frames rendered). 3 s preview without PNGs:
```
... --start 465 --end 555 --test-video work/cat_anim_3s.mp4 --no-png
```
Knobs (clamps and gains, all in degrees / driver kp units):
`--yaw-scale 0.8 --yaw-clamp 35`, `--pitch-scale 0.9 --pitch-clamp 20`,
`--roll-scale 1.0 --roll-clamp 20`, `--t-scale 0.0` (translation OFF: all translation comes from
the track in composite.py; 1.0 made the head pop sideways) `--t-clamp 0.06`, `--scale-clamp 0.12`,
`--smooth-sigma 1.0` (Gaussian over frames, reset at cuts) and `--smooth-sigma-small 3.0` for shots
whose median face width < `--small-w 40` (wide shots: noisy driver pose), `--neutral-t 13.47`,
`--driving-multiplier 1.0`, `--pose-source lp|tracks`, `--batch 4`, `--fp32`, `--device xpu|cpu`.
Raising yaw/pitch scale makes the cat follow the mic-turns more but LivePortrait warps
badly past ~30 deg - keep the clamps. If the GPU is shared with swap.py (step 4) both slow down;
run them one after the other.

## 4. swap.py --only L  (.venv on DirectML, ~14 min: 1.0-1.1 fps at --scale 2 with occlusion handling)

User's face on the LEFT performer -> contract (D).
```
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe swap.py --input work/cut.mp4 --out work/user_L.mp4 \
    --a faces/user/selected --only L --scale 2 --out-scale-back \
    --tracks work/tracks30.json --occlusion mask --min-yaw-gate 65 --yaw-gate-min-w 100
```
(`--tracks work/tracks30.json` and `--occlusion mask` are the defaults and are spelled out only so
the command is self-describing; the two yaw-gate flags are NOT defaults - without them the
extreme-profile artifact at f573-580 comes back.)
Observed (fix-3 run): 872 frames in 847.4 s (1.0 f/s, 14.1 min wall, GPU shared); stats swapped_L 872,
yaw_gated 8, yaw_ramp 2, proxy_L 21, occ_masked_L 316, held_L 80, small_face 211. Every frame outside
f571-580 is pixel-identical to the fix-2 render (face-box mean abs diff 0.00).

Yaw gate (fix-3): `--min-yaw-gate 65` leaves the LEFT face UNSWAPPED (original performer) while the
head is in extreme profile. With --tracks the gate is precomputed per frame from the One-Euro-smoothed
track yaw (yaw_gate_weights): it engages when 2 of the frames f..f+2 exceed the gate (one-frame spikes
never engage; the 1-frame lead compensates the smoothed track's lag), releases with `--yaw-gate-hyst 3`
deg hysteresis, applies only to faces >= `--yaw-gate-min-w 100` input px, is wrapped in a
`--yaw-gate-ramp 2` frame linear crossfade on both sides and never crosses a cut. Without tracks the
detector yaw drives the 3-out/4-in hold ramp instead of a hard per-frame skip. The gated runs are
printed before the render ("yaw gate L: ... 8 frames left unswapped [(573, 580)], 2 crossfade frames
[(571, 572)]"). WHY 65 / 100 px: track yaw at f573-580 is 61.8-79.4 deg, but the medium shots f239-437
and f786-857 also sit at 65-79 deg (219 frames) and their swaps read fine - the artifact only shows on
the 111-142 px close-up faces where the 128-px swapper output is upsampled ~1.9x, hence the size floor.
A plain gate without the floor would remove 25 % of the identity and flicker (9 on/off toggles).
Yaw evidence for every frame (track yaw, kps yaw, detector yaw): work/fix3_swap/yaw_all.json,
thresholds 55-75 tabulated by work/fix3_swap/gate_stats.py. The hold crossfade state (hold_w /
prev_frac) also resets at every cut, so an occlusion hold never fades across a cut.
work/tracks30.json is read by default: L/R roles by IoU, and when the detector misses the LEFT
face (hand across the mouth, 21 px wide-shot face) the swap runs on the track's held kps
(`proxy_L` in the stats) so the identity never drops out. After swap + GFPGAN the performer's
opaque white sunglasses are pasted back from the ORIGINAL frame (white-frame / black-lens pixels
inside an eye-line ellipse, `--no-eyewear` to skip), and faces narrower than `--small-face-px 40`
input px get GFPGAN at `--small-face-strength 0.3` (no shimmer in the wide shots).
Observed (QA-fix run): 872 frames, swapped_L 872 (proxy_L for the detector misses), 1.4 f/s
while animate_cat.py shared the GPU. Observed (fix-2 run, occlusion handling ON): 872 frames in
814 s (1.1 f/s, 13.6 min wall) while other jobs shared the GPU; per-shot pre-pass (skin model +
arm-free reference frames + XSeg) ~5 s per shot, Occluder.mask() ~0.3 s/frame on top of the swap.
Output 960x532 30 fps with the cut's audio re-encoded (aac 192k). Every reference photo's cosine
to the set is printed; photos under 0.6 are dropped.
Use ONLY faces/user/selected (IMG_0176 clean-shaven and IMG_4311 eyes-closed would dilute the identity).

Occlusion handling (fix-2, ON by default, needs --tracks for lmk106 + cuts): when a hand, fist,
wrist chain or the mic covers the LEFT face, `class Occluder` builds a per-frame occluder mask
(per-shot skin model, arm/hand seeds = skin blobs outside the dilated face hull with area beside
the face, growth through pixels that differ (Lab) from pose/time-nearest arm-free frames of the
same shot, bridged across the wrist chain, XSeg non-face pixels when models/xseg_1.onnx exists)
and pastes the ORIGINAL pixels back over the swap inside the swap+GFPGAN paste region (dilated
0.03 w, feathered `--occ-feather 3` px). When more than `--mask-hold-lo 0.5` of the face hull is
covered the swap ramps out (3 frames out / 4 in, one-frame spikes debounced) and at
`--mask-hold-frac 0.75`+ the frame is left unswapped: identity is intentionally lost on those
~60 frames (f569-573, f699-731, f735-786) instead of painting the user's face on the hand.
Flags: `--occlusion mask|hold|off` (default mask; hold = skip the swap when > `--hold-frac 0.25`
of the hull is covered; off = v1 behaviour), `--occluder-model models/xseg_1.onnx`,
`--occ-diff-thr 18` (Lab distance to the reference frames that counts as changed),
`--occ-feather 3`, `--occ-debug-dir DIR` (per-frame panels roi | skin | motion | changed | mask |
swap | out, slow). The DIS optical-flow branch exists but is OFF (`Occluder(flow_gap=0)` in the
code, no CLI flag): DIS/Farneback were unreliable on this blurry, fast footage.
Outside the two occlusion windows only f441 (the RIGHT performer's hand crossing the LEFT face)
and f504 (peace sign beside the temple) fire; clean frames are pixel-identical to v1.

Knobs: `--enhance-strength 0.8` (GFPGAN blend; 0.6 = softer/more original skin, 1.0 = crisp
but can look pasted), `--no-enhance`, `--scale 2` (needed: wide-shot faces are 21-31 px),
`--tracks work/tracks30.json` (assign L/R by IoU with the tracks instead of identity),
`--min-yaw-gate DEG` (0 = off) with `--yaw-gate-hyst 3 --yaw-gate-min-w 0 --yaw-gate-ramp 2`
(see above; use 65 / 100 for this footage), `--start S --duration S` for a dry run.

## 5. composite.py  (.venv, CPU only, ~4 min: 0.27 s/frame)

Cat head over the RIGHT performer on top of user_L.mp4 -> contract (E).
```
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe composite.py --out work/cat_R.mp4 \
    --sheet work/comp_sheet.jpg --sheet-frames 30,150,360,510,654,810 --sheet-orig
```
Observed (QA-fix run): 872 frames in 160.3 s (0.184 s/frame). Observed (fix-2 run): 872 frames in
195.3 s (0.224 s/frame). Observed (fix-3 run, current): 872 frames in 232.8 s (0.267 s/frame,
3.7 f/s; 0.20 s/frame in the wide shots, 0.26-0.27 in the medium/close shots where the watershed +
reference distance run), 237.8 s total wall including the pre-passes (hair temporal max over the
211 wide-shot frames 2.9 s, face-reference medians over the 661 medium/close frames 1.5 s) and the
sheet; stats held 2, centroid_fallback 0, missing cat PNGs 0; ffmpeg rc 0; cat_R.mp4 7.31 MB.
960x532, 30 fps, audio copied from user_L.mp4. Defaults read work/user_L.mp4, work/tracks30.json,
work/cat_anim/, refs/cat_meta.json, refs/cat_alpha_512.png, refs/cat_src_glasses_512.png.
composite.py exits with rc 2 and a clear message when the input has 0 frames (e.g. user_L.mp4 is a
stub while swap.py is still writing it) - check that swap.py has finished first.
3 s test window: `--start 465 --end 555 --out work/comp_3s.mp4` (90 frames, ~38 s).
Debug panels: `--debug-dir work/comp_dbg --debug-every 15`.

Per shot (fix-3): besides the wide-shot hair temporal max, every medium/close shot gets a
head-aligned (translation + scale) MEDIAN of its frames = the performer's canonical face
(precompute_face_ref, `--ref-size 192` px canvas of `--ref-k 3.0` x face width). Per frame the Lab
distance of the eroded skin to that reference tells face from hand: distance < `--ref-thr-lo 30` is
face (forehead, cheeks), > `--ref-thr-hi 45` in a blob >= (`--ref-hand-w 0.20` w)^2 px inside the
body is a hand in front of the face. `--no-face-ref` falls back to the pale-palm rule only.

Per frame (fix-3): analyze() classifies the original pixels (backdrop / polo / skin / bright /
dark hair), labels hand-like skin blobs (those that extend outside the own-body region, which now
includes a FOREHEAD zone up to `--forehead-up 0.30` w above the bbox top inside the head ROI), then
splits hand from face with a marker WATERSHED (cv2.watershed on the blurred frame): hand seeds =
erosion-split blobs (`--hand-split 0.06` w) that reach outside the body and hold no face core,
strict-pale palm blobs (V > `--pale-v 140` & S < `--pale-s 100`, >= (`--palm-w 0.25` w)^2 px) and
blobs that differ from the canonical face; face seeds = core-holding blobs, dark skin inside the
eroded lmk106 hull, the upper-neck band under the jaw (`--neck-face-down 0.20` h) and forehead skin
that matches the canonical face. analyze() runs TWO frames ahead (the main loop holds 5 analyses)
so process() can AND the lit-palm specks over 3 frames (5 frames inside the lenses / face core) and
require watershed hand pixels inside the face core to be 3-frame stable, connected to the hand seed
through a skin path >= `--hand-path-w 0.04` w wide and brighter than `--core-hand-v 120`.
process() removes the head (ellipse + rounded jaw block, keyed dreads, jaw-shaped neck polygon down
to the collar, small dark collar dots), fills (quadratic backdrop surface + normalized-convolution
polo fill that samples ONLY orange pixels), lays the cat on and paints hands / mic / chains back per
pixel. Hand protection is PER PIXEL (proven hand skin dilated 1 px; `--hand-protect-w` is legacy
and unused): the merged hand+face blob is never shielded, which is what let the forehead / hairline
come back above the cat's glasses in round 2. Paint-back rule: keep_skin = kept & (~own | spec) |
hand_pure. Holes enclosed by hand skin up to (`--hand-hole-w 0.12` w)^2 px are accessories (ring,
watch) and come back, bigger holes (the OK sign at f775) show the cat. A PURE hand blob (holding <
`--hand-core-frac 0.10` of the face core, > (`--hand-pure-out-w 0.15` w)^2 px outside the own-body
region) comes back with ALL its pixels, dark fist included. Bright paint-back INSIDE the cat is
allowed only (1) in the mic zone for components that extend outside the head ROI, (2) from
`--chain-margin 0.06` w above the cat's chin line downward (chains), (3) in components touching
restored hand skin (rings, watch) - never inside the eroded hull / lens / face-core zone (the
sunglasses' hinge reflection used to slip through at f510/f575). The 2 px paint-back margin never
grows into pixels darker than `--dark-v 110` (shaded cheek next to a lit hand). Wide shots: only
non-hair skin blobs >= (`--fg-min-blob-small 0.45` w)^2 px count as hands / exclude pixels from the
dread key (measured on f585: 457/859 dark dread pixels are skin-class, so dread fragments used to
paint themselves back as "hands"), bright/chain paint-back is limited to the mic zone, the dread key
reaches `--hair-down-small 3.5` h below the face box, and the chroma blur is skipped (it haloed the
2-px glasses rim into a doubled rim at s ~ 0.08).

Placement: the cat's eye line starts `--eye-up -0.18` face widths BELOW the human eye line and
the head is then pulled down (`--chin-pull 0.85`, max `--chin-pull-max 0.45` w) until the cat's
chin reaches the collar line `--collar-frac 0.25` (human chin + 0.25 h), so the head sits on
the shoulders; there is NO painted neck any more (`--neck` re-enables the old fur band) and
nothing below the collar is removed except dread pixels, so the pearl chains stay intact.
Wide shots (face width < `--small-w 40`) use a bigger removal ROI (`--head-up-small 0.5
--head-side-small 0.5 --hair-half-w-small 2.2 --hair-down-small 3.0`; fix-2 widened these from
1.7 / 2.5 because the surviving strand at f30 sat just outside the 1.7 w rect), an absolute dread
threshold `--hair-v-small 120` (was 110) AND a chroma gate `Cr < --hair-cr-small 168 & Cb >
--hair-cb-small 106` (measured: dreads Cr 139-156 / Cb 114-121 vs shadowed polo Cr ~171 / Cb ~104,
backdrop Cr 192 / Cb 84, so dark polo folds are no longer keyed), plus a per-SHOT head-aligned
temporal max of the dread mask (`--no-hair-tmax` disables; a pixel must be keyed in >=
`--hair-tmax-min 2` frames of the shot, accumulator pad `--hair-tmax-pad 240` px) restricted to the
RIGHT performer's side and excluding per-frame skin and bright pixels. The tiny cat is kept as sharp
as the mic: softening sigma = max(min(sigma, `--small-sigma-cap 0.4`), `--small-feather 0.4`) px
(v1 used 1.5 px feather, visibly blurrier than the plate). Profiles shift the head ellipse toward
the back of the head (`--yaw-shift 0.3` x w x sin(yaw)) and widen the hair rect (`--hair-half-w-profile
1.6` above `--profile-yaw 35`). Fill = quadratic backdrop surface on a `--ring-w 0.40` w ring
(`--ring-w-small 0.50` in wide shots), polo part by NC fill (below the collar line) whose weak
fallback samples ONLY orange pixels with a `--fill-sigma-min 5` px sigma floor (`--fill-sigma-w
0.10` w), feathered `--fill-feather 2.5` px. The human neck is filled BEFORE the cat is laid on:
`neck_poly` = convex hull of the lmk106 jaw contour plus the same points pushed down
`--neck-fill-down 0.35` x h (minus dilated bright = chains, minus hand protect), and dark connected
components of 3 px .. (`--dot-max-w 0.08` w)^2 in the collar band that touch nothing bright are
removed (pearl gaps preserved) - no grey haze between cat chin and collar any more.
Hands over the face: skin blobs come back in front of the cat when > `--hand-out-frac 0.35` of the
blob (or > (`--hand-out-w 0.35` w)^2 px) lies outside the face hull + neck column; inside the face
only lit pixels (V > `--hull-v 170`, or V > `--lens-v 95` inside the lenses with lens-hand ON) that
belong to a blob with > (`--lit-out-w 0.06` w)^2 px outside the own-body region AND survive the
3-frame temporal AND do - this is what removed the pink specks inside the blue lenses.
Chains/mic: bright hysteresis (`--bright-grow 3 --bright-s2 110 --bright-v2 120`), min blob
`--bright-min-blob 0.04` w; bright paint-back outside the cat is limited to the mic zone so the
hair bead behind the head is not restored.

Knobs worth touching:
- `--cat-scale 1.38`  cat head width / effective face width (1.45 read mascot-large; 1.55+
  covers the mic).
- `--eye-up -0.18` / `--collar-frac 0.25` / `--chin-pull 0.85`  see above; the pull is precomputed
  per frame and Gaussian-smoothed `--pull-smooth 2.0` frames inside each shot (0 = per frame).
- `--roll-mode anim|track|trackfull|none`  anim (default) = LivePortrait's own roll only;
  track adds the residual (tracks30 roll - roll at the neutral frame) - use if the cat looks
  too upright when he tilts; `--roll-clamp 25`.
- `--centroid-smooth 0`  temporal smoothing of the glasses-centroid anchor (0 = raw; the LP
  render carries no translation any more, so the raw centroid is clean).
- cat look: `--cat-sat 1.0 --cat-contrast 1.0 --cat-gain --cat-gamma --cat-warm`, softening
  sigma = `--blur-a 0.30` + `--blur-b 0.90` x on-screen scale (clamped `--blur-min 0.4
  --blur-max 1.4`; grows with the cat so the close shot is softest), `--chroma-blur 1.0` px
  Cr/Cb blur, `--grain 2.0`.
- key/edges: `--key-lo 18 --key-hi 45 --gate-dilate 22 --edge-feather 1.2`.
- removal / fill: `--head-up 0.35 --head-side 0.25 --jaw-half-w 0.55`, dreads `--hair-v 70
  --hair-v-fringe 135 --hair-grow-w 0.08 --brown-v 160 --hair-half-w 1.15`, `--fill-mode nc|telea`,
  `--neck-half-w 0.35` (neck-skin removal column).
- foreground paint-back: `--excl-ax 0.62 --excl-ay 0.55` (own-body ellipse), `--fg-min-blob 0.12`,
  `--hull-erode 0.06`, `--face-core-erode 0.15`, `--no-lens-hand` (lens rule is ON by default now),
  the fix-2 hand flags (`--hand-split 0.06 --hand-core-frac 0.10 --hand-pure-out-w 0.15
  --lit-out-w 0.06`; `--hand-protect-w 0.05` is legacy/unused) and the fix-3 flags with their
  defaults: `--forehead-up 0.30`, `--hand-path-w 0.04`, `--core-hand-v 120`, `--hand-hole-w 0.12`,
  `--chain-margin 0.06`, `--pale-v 140 --pale-s 100 --palm-w 0.25`, `--neck-face-down 0.20`,
  `--dark-v 110`, `--fg-min-blob-small 0.45`, `--hair-down-small 3.5` (was 3.0), face reference
  `--no-face-ref --ref-size 192 --ref-k 3.0 --ref-thr-lo 30 --ref-thr-hi 45 --ref-hand-w 0.20`.
  Raising --ref-thr-hi makes fewer things count as a hand in front of the face (safer for the
  forehead, worse for same-tone fists); lowering --pale-v brings more palm back.

## 6. finish.py  (.venv, ~11.5 s)

Original audio back on, 9:16 compose with caption, contact sheet, sync checks -> contract (F).
```
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe finish.py
```
The caption "the cat killed it ngl" is the built-in default (as are `--font-size 44 --video-y 440`,
the centred band layout of commit e46dad8), so the bare command above equals
`finish.py --caption "the cat killed it ngl"`. Observed (fix-2 run): total wall 11.5 s (remux 0.1 s,
compose 2.9 s, sheet + ffprobe + sync checks the rest); final.mp4 5.60 MB. Observed (fix-3 run):
total wall 11.4 s (remux 0.1 s, compose 3.0 s), final.mp4 5.58 MB, 720x1280, 872 frames, video
29.067 s (start 0.033), audio aac 44.1 kHz stereo 29.094 s, container 29.100 s; audio sync lag
+0.00 ms vs audio.m4a and cut.mp4 (peak corr 1.000), video motion-series lag 0 frames 872/872
(peak corr 0.989) -> "checks audio PASS, video PASS", rc 0.
Steps and observed output:
1. re-mux work/audio.m4a over work/cat_R.mp4 -> work/cat_R_audio.mp4 (-c:v copy -c:a copy
   -shortest, 0.1 s). The video is delayed by the source cut's own video/audio start offset
   (cut.mp4 video pts 0.033008 s vs audio 0.0; `--av-offset auto`), so frame i sits at the same
   song instant as in the source (final.mp4 pts identical to cut.mp4 for all 872 frames).
2. compose.py -> work/final.mp4, 720x1280, libx264 crf 18 (2.5 s).
3. work/final_sheet.jpg = 6 frames at 1, 5, 12, 17, 21.8, 27 s (3x2 grid, 2160x2560) and
   ffprobe: final.mp4 = 29.100 s file / 29.067 s video, 720x1280, 30.000 fps, 872 frames,
   5.84 MB, audio aac 44.1 kHz stereo 29.094 s.
4. sync: audio of final.mp4 cross-correlated with work/audio.m4a and work/cut.mp4 ->
   lag +0.00 ms both (peak corr 1.000), threshold 40 ms; video motion-energy series of
   final.mp4 vs cut.mp4 -> lag 0 frames, 872/872 (peak corr 0.989). Exit code 2 on failure.
`--skip-build` reruns only the sheet + checks on an existing final.mp4.

Knobs: `--caption TEXT` (compose.py drawtext, ~90 px above the video band), `--font-size 44`,
`--video-y 440` (top of the 532-px band in the 1280-px frame; 330 = the old meme layout with the
band high in the frame), `--bg white --caption-color black --font PATH`, `--crf 18`,
`--av-offset auto|0|SEC`, `--max-offset-ms 40`, `--sheet-times 1,5,12,17,21.8,27`,
`--no-video-check`.

## Full rebuild, in order

```
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe track.py
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe make_cat_ref.py
PYTHONIOENCODING=utf-8 .venv-lp/Scripts/python.exe animate_cat.py --src refs/cat_src_glasses_512.png --video work/cut.mp4 --tracks work/tracks30.json --sheet work/cat_anim_sheet.jpg
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe swap.py --input work/cut.mp4 --out work/user_L.mp4 --a faces/user/selected --only L --scale 2 --out-scale-back --tracks work/tracks30.json --occlusion mask --min-yaw-gate 65 --yaw-gate-min-w 100
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe composite.py --out work/cat_R.mp4 --sheet work/comp_sheet.jpg --sheet-frames 30,150,360,510,654,810 --sheet-orig
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe finish.py --caption "the cat killed it ngl"
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe work/fix3_final/comp_vs_example.py
```
Total ~27 min wall (track 3 + cat ref 0.1 + animate 4-5 + swap 14.1 + composite 4 + finish 0.2),
GPU-bound in swap.py and animate_cat.py (run those two sequentially, not in parallel).
Fix-2 measured: swap 814 s, composite 198 s, finish 11.5 s.
Fix-3 measured (current): swap 847.4 s, composite 237.8 s wall (232.8 s render), finish 11.4 s.
Changing only composite knobs: rerun 5 then 6 (~4.2 min).

Dependencies: steps 3, 4, 5 need step 1; step 3 needs step 2; step 5 needs 3 and 4;
step 6 needs 5. Changing make_cat_ref output invalidates cat_anim/ (rerun 3 then 5 then 6).
Changing only the caption: rerun 6.

QA sheets after a rebuild: work/comp_sheet.jpg (composite, `--sheet-orig` puts the user_L frame
above each composited frame), work/final_sheet.jpg (finish.py, 6 frames at 1/5/12/17/21.8/27 s of
the 720x1280 final) and work/comp_vs_example.jpg (rows f150/f510/f810 of cat_R.mp4: full frame |
zoom on the cat | example still ex_2/ex_3/ex_6; regenerated by
`PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe work/fix3_final/comp_vs_example.py`, which crops
4.2 x face-height around the tracks30 R centre, clamped to the frame). Expected ffprobe of
work/final.mp4: 720x1280, 30/1, 872 frames, video 29.067 s (start 0.033), audio aac 44.1 kHz
stereo 29.094 s, container 29.100 s.

## Fixed in fix-3 (2026-09-25, round-3 QA: vs-example 6.5 / meme 7.5 -> verdict "fix")
Composite (all verified on 2x-6x before/after crops, work/fix3_comp/ + work/fix3_final/):
- D1 MAJOR: forehead / hairline / sunglasses-top visible above the cat's glasses when a hand was
  near the face (f766-785, f436-441, f806-813, f598) - gone (per-pixel hand protection, forehead
  zone, watershed hand/face split, canonical-face reference). OK ring at f775 whole, four fingers
  back at f810, dark hair arcs above the head at f436-441 / f810-813 gone.
- D2 MEDIUM: sunglasses hinge/lens highlights painted back inside the cat (f244-300, f417-446,
  f575-580, f688+) and the earring bead beside the ear (f143-152) - gone; chains, rings, watch intact.
- D3 MEDIUM: wide shots f581-603 "claw" cluster + peach blob on the chest, f30-34 dark chunks at
  ear level - gone (dread fragments no longer count as hands).
- D5 MINOR: ghost-doubled blue glasses rim in wide shots (f603, f666-687) - single crisp rim (chroma
  blur skipped in wide shots).
Swap:
- MEDIUM: extreme-profile swap artifact f573-580 (fake eye on the backdrop, protruding skin patch,
  blocky paste) - the LEFT performer is the original on these 8 frames (yaw gate), 2 crossfade
  frames f571-572 inside the existing occlusion hold; every other frame pixel-identical to fix-2.

## Known Issues (after fix-3, 2026-09-25)
Composite (RIGHT performer / cat):
- D4 MINOR, partial: a 2-3 px dark red-brown speck remains inside the cat's left lens at f808-810
  (was a 6-8 px blob; lit cheek highlight next to the lit fingers, V 110-130 passes --core-hand-v /
  --dark-v). The pink palm smear at f812-815 is now one coherent motion-blurred hand over the cat's
  face (physically correct occlusion) - judge in motion.
- Wide shot f581: a faint 2 px dark dread residue left of the collar (was a claw cluster + blob).
  The dark shapes on the chest in the wide shots (f30-34, f581-603) are the chain + cross, not dreads.
- A fist pressing on the chin (f174) is partly UNDER the cat chin where the blob still merges with
  the face - reads as the cat resting its chin on the fist (unchanged since fix-2).
- Dread removal: soft polo/backdrop-coloured feathered patches remain on the shoulders (e.g. f244
  right shoulder edge, f580 collar), and a faint feathered fill arc is visible around the tiny cat
  at f30 / f585-603 when zoomed 4x+. Below the collar nothing is removed on purpose, so dread tips
  behind the shoulders stay (f575 right of the collar, f720 left collar).
- Cat pose carries 0.8x yaw / 0.9x pitch with 35/20 deg clamps, so the biggest turns to the mic
  still read partly frontal (LivePortrait warps badly beyond the clamps).
- Wide shots (0-3.5 s, 19.4-22.9 s): 40-60 px cat head, identity not readable at that size (kept as
  sharp as the mic).
- Runtime: 0.267 s/frame (was 0.224) because of the watershed + reference distance; ~4 min per clip.
Swap (LEFT performer / user):
- Where a hand covers > 50 % of the face hull the swap ramps out and the original performer shows
  (f569-580 incl. the yaw-gated f573-580, f699-711, f717-728, f752-784; held_L 80 + yaw_gated 8) -
  by design, instead of the user's face painted on the hand.
- f560-561: beard/chin painted through the motion-blurred same-tone fist (track yaw 59-61 is below
  the gate; lowering the gate to 60 would also drop the good fist-on-cheek frames f560-569). Needs
  the Occluder to catch the blurred fist, not the yaw gate.
- f748: dark curl + lens glint under the fingers for 1-2 frames (face 77 px < the 100 px gate floor;
  lowering the floor to 75 would gate the medium shots f786-857 whose swap is fine). Needs the
  Occluder (hand at the temple f746-748 is unmasked), not the yaw gate.
- The blurred hand over the glasses at f698 and the same-tone fist on the cheek f558-567 are not
  masked (pale skin patch, no features).
- The swapped face is lighter than the performer's neck/ears (inswapper carries the source skin
  tone; no colour transfer applied, to keep identity cos > 0.6).
- The round-2 note "f580 drops the swap for a single frame" is superseded: f573-580 are gated as one
  run and the wide shot f581+ is unchanged.
Finish:
- Audio is the original m4a copied bit-exact; the 33 ms video delay is carried as an mp4 edit
  list exactly like the source cut.mp4 (pass `--av-offset 0` if a target player ignores edit lists).

Round-3 QA material: work/fix3_comp/ (composite before/after crops + test windows),
work/fix3_swap/ (yaw evidence, fix3_573_580_big.jpg, fix3_transition.jpg, fix3_identity.jpg,
round-2 render backup user_L_r2.mp4), work/fix3_final/ (composite_run.log, finish_run.log, round-2
sheet backups *_r2.jpg, qa_*.jpg crops of the final cat_R.mp4, crops.py, comp_vs_example.py).

## Wan-Animate recast of the example clip (work_ex/, rounds 4-5)
Whole characters are regenerated with Wan2.2-Animate in ComfyUI (vendor/ComfyUI, .venv-comfy) instead of a head
swap: the cat replaces the grey cat on the LEFT, the user replaces the dancer on the RIGHT. Everything runs with
`VG_WORK=work_ex`. Renders go through run_example.ps1 and must be started from the user's own PowerShell
(agent background jobs get killed under memory pressure).

| step | command | time |
|---|---|---|
| cat pass | `powershell -ExecutionPolicy Bypass -File run_example.ps1 -Side cat` -> work_ex/wan/cat_E/frames (181 f @ 12 fps, 768x576) | ~1.5 h |
| re-do one shot | `... -Side cat -FixShot N` (tight mask, hand-free skeleton, new seed; auto-splices, old frames -> cat_E/frames_backup) | ~23 min (shot 2) |
| user pass | `... -Side user` (paints over cat_E/frames_df if present, else cat_E/frames; ref refs/user_ref.png) -> work_ex/wan/user_E/frames | ~1 h 42 min |
| deflicker | `python deflicker.py --in <frames> --src work_ex/wan/cat_E/in/bg --out <frames>_df` (only for renders decoded with --decode-frames 16) | ~20 s |
| 12 -> 24 fps | `python to30.py --in <frames> --out <dir24> --method rife --gpu 0 --n-out 361` | ~10 s |
| lip sync | `.venv-lp/Scripts/python.exe lip_pass.py run --frames <dir24> --out <dir24>_lips --strength 1.2` (needs work_ex/mouth.json, lip_dir.npy) | ~10 min |
| encode | `ffmpeg -framerate 24 -i <dir24>_lips/%05d.png -vf scale=960:720:flags=lanczos -c:v libx264 -crf 16 -preset slow -pix_fmt yuv420p work_ex/wan/<name>.mp4` | ~30 s |
| finish | `python finish.py --blocks work_ex/blocks.json --input work_ex/wan/<name>.mp4 --out work_ex/final.mp4` | ~10 s |

Resume reads vendor/ComfyUI/output/wan/<name>/wNN: to re-render with a new reference or prompt, move that folder
(and work_ex/wan/<name>) aside first, or the old windows are reused.

run_example.ps1 flags: `-Side cat|user`, `-Probe` (first 33 frames only, ~15 min: always probe a new character or
reference before a full render), `-FixShot 1..5`, `-ReserveVram GB` (default 4). Every render resumes finished windows.

Ops notes
- **GPU memory (-ReserveVram).** The Arc 140T iGPU shares a 16.4 GB budget with the desktop, browser and editor.
  With ComfyUI's default 0.6 GB reserve (and still with 2 GB) the 11.2 GB Wan model was loaded fully and KSampler
  hit "XPU out of memory ... Tried to allocate 592 MiB" ~2 min into window 0, every time. 4 GB forces a partial load
  (9.4 GB on the GPU, 1.7 GB offloaded) at the same ~28 s/frame. Check the load line in
  work_ex/round4/comfy_server.err.log: it must say `loaded partially`.
- **System RAM.** During the user pass free RAM fell to 0.6 GB; close Chrome/VS Code while rendering.
- "device lost": re-run the same command (resume). Never reuse a ComfyUI server left by another job.
- **Temporal decode must cover the window.** The Wan VAE is causal. With `--decode-frames 16` (the old default),
  VAEDecodeTiled started a new temporal chunk every 12 frames, and each one decoded brighter and washed out, fading
  over ~8 frames: a whole-frame pulse about once a second (cat pass +15 RGB, user pass up to +28; RIFE mistook
  them for 17 shot cuts). Default is now 64, so a 33-frame window is one chunk; spatial tiling stays at 512 px.
  All frames rendered before 2026-09-27 have the pulse, so run deflicker.py on them.
- deflicker.py: per-channel 33-point quantile curve per frame, lower envelope (rolling min over 13 frames, smoothed)
  within each shot, frames remapped to it. The flash is a tone curve (shadows ~0, mids x1.2-1.3, highlights +30), not
  a gain: a mean-gain version dimmed whites by 15-25, and a source-relative envelope drifted with the recast
  histograms. Result on user_E: max in-shot step 21.7 -> 6.8, mean 2.71 -> 0.88 (source 4.9 / 0.96). Flash frames
  stay slightly hazy (the haze is not a pure tone change).

Known issues (2026-09-27)
- **User identity drifts per shot** (work_ex/round4/user_identity_check.jpg): only shot 1 resembles the user; later
  shots show other men (one bearded), a white shirt instead of black, ghost outlines at f40/f150. refs/user_ref.png is
  a 310x1024 full-body image (face ~40 px), and each shot cut starts a fresh window with only that reference for
  identity. Fix: close-up face photos -> a reference with a large face, probe, then re-render the user pass.
- Cat pass has the (milder) decode flicker; shot 2 re-rendered (compact body, full black sunglasses); a pinkish paw
  smear at ~6.5 s and streaks at ~4.5 s remain.

Render Studio (studio/run_studio.ps1, http://127.0.0.1:8765): watches these logs live (progress, ETA, model load mode,
OOM / device-lost alerts), builds user_ref.png from dropped close-ups, compares any two frame lanes per shot with a
wipe, records approve / re-roll per shot (work_ex/studio/review.json) and hands the chosen mp4 to video_editor_ai.
Details: studio/README.md.

## 2026-10-04: merged into video_editor_ai
This folder is now `video_editor_ai/recast/`. The PowerShell launchers and the Studio UI are gone:
`recast/runner.py` holds the same commands (`python -m recast.runner --help`), `recast_worker.py`
runs them as GPU-agent jobs, and the web UI is `/recast` in the editor. Data dirs (work_ex, venvs,
ComfyUI, models) moved here unchanged; ComfyUI's `extra_model_paths.yaml` points at the new path.

## Wan2.2-Animate-2 lane (2026-10-08, Kaggle T4)

`wan_animate2.py` drives Animate-2 from the clip itself (no skeleton, no mask, no background plate):
each side is a crop of `work_ex/cut.mp4` (cat = left 544 px, user = right 544 px, rendered at
480x640) and the output is a whole new frame of the reference character against the prompted
studio; compositing the two sides back into one frame is a separate step. Runs on Kaggle through
`kaggle_run.py --engine animate2` (backend repo); see `kaggle/README.md` for weights, the Rebels
loader (`model class=WAN_Animate2` must appear in the ComfyUI log) and the distilled sampler
settings. Prompts: `runner.CAT_A2_PROMPT` / `USER_A2_PROMPT` (two-part "Character Description /
Background description" convention) plus a pose-branch prompt describing the motion.

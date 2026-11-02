# COLORS duet face-swap — local pipeline

Goal: reproduce the viral "Musk & Altman at the COLORS mic" format with the user
and a friend swapped onto the two rappers, using a free local pipeline on the
Intel Arc GPU. Same 29 s cut as the repost.

Inputs
- Source: `~/Downloads/retarded_repost_central_TikTokDownloader.com_ac7c8.mp4` (68 s, 960x532, 30 fps)
- Reference (repost): `~/Downloads/The post praises ... .mp4` (29 s, 720x1280)
- Face refs: USER TO SUPPLY — 3-5 clear photos each of person A and person B

## Plan
- [x] 1. Find the 29 s segment: cross-correlate repost audio against source audio → start offset
- [x] 2. Cut segment, upscale/crop to 9:16 (720x1280) to match repost framing
- [x] 3. Environment: venv, onnxruntime-directml (Intel Arc), insightface, face enhancer
- [x] 4. Tool: `swap.py` — detect two faces per frame, assign L/R by x-position,
       swap A→left rapper, B→right rapper, enhance, re-mux original audio
- [x] 5a. 3 s self-test dry run OK (2 faces every frame, clean blend)
- [x] 5b. Full 29 s self-test PASSED: 872 frames, 849 two-face / 22 one-face / 1 none / 0 ambiguous; contact sheet work/selftest_sheet.jpg clean (1829 s render under GPU contention)
- [ ] 5c. Real run with user photos (BLOCKED: waiting on faces/A, faces/B)
- [x] 6. compose.py written; test pending
- [ ] 6b. Optional: caption overlay ("X killed it ngl" style) via ffmpeg drawtext
- [ ] 7. Verify: frame check at 5 points, audio sync, no face flicker
- [x] 8. git init + commit tooling (no source videos / face photos committed)

## Review
(fill after completion)

Notes
- segment offset 29.574 s (audio xcorr 0.853)
- ~1.0 s/frame at 2x scale on Arc 140T: detect 0.08, swap 0.21, enhance 0.26 per face
- wide shots have 18-27 px faces at native res -> always use --scale 2
- both performers wear sunglasses; swap keeps them (only lower face/skin visibly changes in wide shots)

## SCOPE CHANGE 2026-09-24 evening
Cast is now: USER (human face swap, existing pipeline) + CAT wearing glasses, cat is the one rapping.
inswapper/buffalo_l are human-only -> cat needs a different method (LivePortrait-animals, head compositing, or V2V).
User also has an Intel AI Boost NPU (Core Ultra 9 285H) -> evaluate OpenVINO NPU for detection/embedding.
- [x] 9. Ultracode workflow cat-duet-design (17 agents, DONE -> tasks/DESIGN.md) (run wf_26b2c825-42b): research 4 -> design 3 -> refute 6 -> synthesize
- [x] 10. Photos answered + collected: faces/user (selfie MOV -> 6 frames in faces/user/selected, cos 0.91-0.96), faces/cat (11 files; hero candidate IMG_9325, sharpness 602)
- [ ] 11. Implement cat pipeline per final design (separate workflow)

TARGET LOOK (user example, 21:03): work/example/hotel_lobby_genjutsu.mp4 -- FULL-BODY Genjutsu recast:
grey cat on a stool in red backwards cap + blue glasses + plaid flannel replaces LEFT rapper;
creator in her own orange tracksuit replaces RIGHT rapper. Not a face swap.
- [ ] 9b. Workflow script patched (EXAMPLE injected into design/refute/judge, +closest-to-example lens).
       Plan: let wf_26b2c825-42b finish -> resume with edited script (research cached, design onward reruns).

## Research findings (workflow wf_26b2c825-42b, 3/4 researchers, 21:15)
- FOOTAGE: RIGHT/orange performer raps 19.4 s (66.8 %) in 4 bursts; LEFT/striped only 1.0 s of ad-libs.
  => CAT = RIGHT (orange shirt), USER = LEFT (striped shirt). Cat head yaw -50..+10, pitch up toward mic,
  face 60-135 px when rapping, 21-31 px in the two wide shots (0-3.5 s, 19.4-22.9 s) where nobody raps.
  Per-frame tracks (cx,cy,w,h,yaw,pitch,roll,mouth_inner @15 Hz) in work/analysis.json; plot work/analysis.png.
- CAT METHOD: LivePortrait animals v1.1 in PyTorch on torch XPU is FEASIBLE (5D grid_sample OK on Arc 140T,
  ~0.1-0.25 s/frame est.); human driving video works by design; needs custom paste-back + glasses baked
  into the source photo; XPose (CUDA-only) bypassed by cropping the single cat photo with our own detector.
  Fallback: 2.5D cutout + jaw flap (zero new models). Full-body generative V2V (Wan VACE etc.) = NO on Intel.
  Audio-driven fallback for mouth: JoyVASA (animal mode).
- NPU: ~13 TOPS vs GPU 77; OpenVINO NPU FP16 static-shape only, no 5D GridSample; onnxruntime-openvino
  cannot share a venv with onnxruntime-directml. Best case 10-15 % by offloading detector/landmarks.
- PHOTOS: user 5 (3 min, 8 max), no glasses/hat, frontal + slight turns. Cat: ONE photo used (send 3-5
  candidates), frontal, both ears, eyes open, mouth CLOSED, sharp, plain bg, >=600 px face; glasses BAKED into
  the photo (real or edited once), not tracked per frame.
- NPU BENCHMARK (4/4): NPU real + healthy (driver 32.0.100.4023). det_10g 12-17 ms, w600k_r50 6-7 ms on NPU,
  decision-identical to CPU (vs ~80 ms detect+embed on DirectML). inswapper on NPU fast (124 ms) but FP16 wrecks
  output (PSNR 22 dB) -> keep on GPU. gfpgan on NPU correct but slower (308 ms vs 260). Pin openvino==2025.4.1
  (2026.4 driver-compiler fails on det_10g AvgPool) or update NPU driver to 32.0.100.5540. Do NOT install
  onnxruntime-openvino in the project venv; add `openvino` and shim insightface det/rec sessions.
  Est. frame time two-face: 1.02 s -> 0.96 s (det/embed on NPU); one-face 0.55 -> 0.49; pipelined 0.33.

## Prep done (both lanes) 22:30
- work/gj_A.mp4 3.47-19.40 s (15.93 s), work/gj_B.mp4 22.93-29.07 s (6.17 s), work/audio.m4a 29.1 s
- ffmpeg scdet cuts: 3.50 s / 19.40 s / 22.97 s (hidden reframe at 7.97 s and 15.40 s not caught at thr 0.25 -> use analysis cuts 104/239/461/581/688)
- faces/A -> faces/cat, faces/B -> faces/user (user had put the cat in A). swap.py/README defaults updated.
- DECISION 22:35 (user): FREE LOCAL lane. Cat mouth stays CLOSED (match example, no lip pass). User keeps glasses on
  screen (striped rapper's white sunglasses stay; inswapper lock runs under them). Cat = RIGHT/orange, user = LEFT/striped.
- [x] 11. Free lane BUILT via ultracode workflow wf_a7a58872-044 (11 agents, 2h04m, 1.84M tokens): track.py | make_cat_ref.py | swap.py flags + user_L | animate_cat.py (.venv-lp XPU) -> composite.py -> finish.py -> 3 QA + fix + recheck

## Review 2026-09-25 00:25 -- free lane v1 shipped to work/final.mp4
Pipeline (tasks/RUNBOOK.md): track.py -> make_cat_ref.py -> animate_cat.py (.venv-lp, XPU 0.28 s/f) -> swap.py --only L -> composite.py -> finish.py
Outputs: work/final.mp4 720x1280 29.07 s (audio bit-exact, A/V aligned to cut.mp4 pts), work/cat_R.mp4 960x532.
QA round 1: 3 reviewers -> fix; 15 defects (9 major). Fixer round 1 applied: ear mirror, temple stubs, yaw 0.8/pitch 0.9 scaling,
  t_scale 0, per-shot smoothing, swap.py --tracks + dropout hold + glasses re-paint, hair keying, chain protection, fill hysteresis.
QA recheck: fix | vs example 5.5 | meme 6.5. REMAINING MAJORS:
  (1) REGRESSION: hands/forearms below the cat carved into rectangles (f172-189, wide f594-603) -- composite.py hand mask bbox fill.
  (2) swap.py paints user face + glasses onto a hand crossing the LEFT face (f560-575, f712-729, f748-758) -- needs occlusion-aware swap mask.
  Minors: wide-shot dread streaks, pink specks under hands, grey neck haze, cat softer than plate in wide shots, top-heavy 9:16 layout.
NEXT: round-2 fix (composite hand mask -> convex-hull/skin-blob instead of rect; swap.py skin-occlusion mask via inswapper fake_diff + hand skin blob), re-QA.

## Round 2 (wf_d2886d3a-a82, 4 agents, 2h22m) -- 2026-09-25 02:41
Fixed: hand rectangles, lens specks, wide-shot collar strands, neck haze/dots, wide-shot cat sharpness,
  user face painted on hands (swap.py Occluder: skin model + XSeg + ramp-out), 9:16 band centred.
QA recheck2: fix | vs example 6.5 | meme 7.5.
REMAINING: [MAJOR] forehead/hairline/sunglasses-top visible above cat when hand at temple (f766-785, f436-439, f806)
  -- new hand-protect exclusion lets non-hand head pixels survive. [MED] bright sunglasses-hinge dot painted back on cat
  forehead (f244-300, f575-580). [MED] wide-shot dread cluster on chest f581-603, chunks f30-34. [MED] extreme-profile
  swap artifact f573-580 (yaw 69-79) -> --min-yaw-gate 65. Minors: lens smudges f808-814, wide ghost rim, fill outline.
- [ ] 12. Round 3: composite forehead-cover + bright paint-back gate + wide chest cluster; swap --min-yaw-gate 65; rebuild; recheck

## Round 4 (2026-09-25): FULL-BODY local recast (plan: local planning notes)
Decisions (user): no paid API, local build only; user shoots 3 new full-body photos in outfit + SUNGLASSES (-> refs/user_*.jpg);
wide shots dropped (final = block A f104-580 + block B f688-871 = 661 frames, 22.0 s); user MUST wear sunglasses in the result.
Route: Wan2.2-Animate-14B Replacement (GGUF Q4_K_M, lightx2v 4-step, relight LoRA) on torch XPU via headless ComfyUI,
two passes (LEFT -> user, RIGHT -> costumed cat on a stool). Gated by a probe; fallback = 2D RTMW-driven puppet (puppet.py).
- [x] 13. Phase 0a: ComfyUI-XPU in vendor/ComfyUI (.venv-comfy) + ComfyUI-GGUF + WanAnimatePreprocess + VideoHelperSuite, API up
- [x] 14. Phase 0b: weights -> models/comfy/ (Wan2.2-Animate GGUF, umt5, clip_vision_h, VAE, LoRAs, ViTPose+yolov10m ONNX)
- [x] 15. Phase 1a: prep_blocks.py (gj_B2 from f688, blocks_ref.mp4, audio_blocks.m4a, blocks.json) + finish.py block mode + judge_take.py -- done 2026-09-25, logs/reports in work/round4/ (null render: 661 f, audio lag 0 ms, video lag 0; judge on blocks_ref: L sunglasses 0.979, R performer face 0.997; on cat_R: L sunglasses 0.979, R 0.000, identity 0.51)
- [x] 16. Phase 1b: pose_body.py (RTMW-x whole-body on DirectML, L/R, skeleton PNGs, character masks) + QA sheet
- [x] 17. Phase 1c (stand-in user ref, manual cat costume; generative costume pending): make_refs.py (user ref w/ sunglasses check; cat cutout base; costume candidates)
- [x] 18. Phase 0c (PASS 2026-09-26: 23.6 s/frame at 832x464, 33-frame window 12.9 min, block A ~1.6 h/side; full-body replacement with black sunglasses confirmed on work/round4/probe_sheet.jpg): PROBE 33/49/77-frame windows, block A LEFT, 832x464 -> s/frame, fits? GATE <= ~6 h/side
- [ ] 19. Phase 2: pass 1 LEFT=user (A, B2) -> judge; pass 2 RIGHT=cat -> judge
- [ ] 20. Phase 3: finish.py -> work/final_v2.mp4, QA workflow (3 reviewers -> fix -> recheck), runbook/lessons, commit

## Round 5 (2026-09-26): recast built on the Genjutsu EXAMPLE itself (user direction)
User: "based off this one" -- cat on the LEFT rapping, user on the RIGHT dancing, both in BLACK sunglasses.
Source = work/example/hotel_lobby_genjutsu.mp4 -> work_ex/ (VG_WORK=work_ex), 361 f @ 24 fps, cuts 104/184/276/346, its own audio.
- [x] pose_body on the example (cat L held 6 %, dancer R 0.3 %); fur-only mask for the cat (make_fur_mask.py: grey fur + glasses, mic excluded)
      so the example's cap, flannel and stool stay pixel-exact; pose-based face crops; driver: --mask-dir, --block-px, --range, --seed-frames, fresh windows at cuts
- [ ] probe cat (run_example.ps1 -Side cat -Probe) -> full cat -> probe/full user -> RIFE 12->24 fps -> finish -> QA
- COLORS-based cat pass (work/wan/cat_A, cat_B) superseded; run_cat_fix.ps1 not needed.

### Round 5 status (2026-09-26 19:45) -- RESUME HERE
Done:
- [x] Cat pass on the example: work_ex/wan/cat_E (181 f @ 12 fps, 768x576), bare orange/white tabby in black sunglasses on the stool
      (no cap/flannel per user), full-character mask, run via run_example.ps1 -Side cat (2 GPU "device lost" resets, resumed).
- [x] RIFE 12->24 fps: work_ex/round4/cat_E24 (361 f). Audio: work_ex/audio.m4a = example audio padded (apad) to 15.12 s so the mux keeps all 361 frames.
- [x] Lip sync: vocals isolated (.venv-audio, audio-separator UVR-MDX-NET-Voc_FT) -> mouth_curve.py -> work_ex/mouth.json;
      lip_pass.py calibrate (mouth direction from faces/cat open-mouth photos -> work_ex/lip_dir.npy) + run --strength 1.2
      -> work_ex/round4/cat_E24_lips (361 f) -> preview work_ex/final_cat_lips.mp4 (audio sync PASS; video check's -1 lag is a
      false positive, dancer-side diff is minimal at lag 0).
Next:
- [ ] USER runs: powershell -ExecutionPolicy Bypass -File run_example.ps1 -Side user   (~1.4 h, resumable; paints the user over
      the dancer on work_ex/wan/cat_E/frames; black suit + black sunglasses; ref refs/user_ref.png)
- [ ] Then: RIFE user_E frames 12->24 (to30.py --n-out 361), re-apply the lip pass on those 24 fps frames
      (lip_pass.py run --frames <user24> --out <user24_lips> --strength 1.2; only the cat's mouth region changes),
      encode 960x720 @24, finish.py --blocks work_ex/blocks.json (VG_WORK=work_ex) -> work_ex/final.mp4, judge sunglasses on the user.
- Optional re-dos the user may ask for: coat colour drifts per shot (~12 s browner), mirrored sunglasses near the end, pink blob ~6.5 s.
Ops notes: renders must run from the user's own PowerShell (agent background jobs get killed under memory pressure);
if a render shows "device lost", just re-run (resume). TdrDelay registry fix offered, not applied (user's call).
Never reuse a ComfyUI server left by another job (stale VRAM -> OOM); kill and let the script start a fresh one.
- 2026-09-26 19:50 IN PROGRESS: shot-2 re-render (4.3-7.7 s, cut frames 104-183, strided 52-91) started by the user:
  run_example.ps1 -Side cat -FixShot 2 (tight mask work_ex/mask_Ltight + fur, hand-free skeleton work_ex/pose_L_nohands,
  seed 20261128, anti-stretch prompt). Log work_ex/round4/render_cat_fix2.log; it auto-splices into work_ex/wan/cat_E/frames
  at 52 (old frames backed up to work_ex/wan/cat_E/frames_backup).
  WHEN DONE: to30.py --in work_ex/wan/cat_E/frames --out work_ex/round4/cat_E24 --method rife --gpu 0 --n-out 361;
  VG_WORK=work_ex .venv-lp/Scripts/python.exe lip_pass.py run --frames work_ex/round4/cat_E24 --out work_ex/round4/cat_E24_lips --strength 1.2;
  encode 24 fps 960x720 -> work_ex/wan/cat_E24_lips.mp4; VG_WORK=work_ex finish.py --blocks work_ex/blocks.json
  --input work_ex/wan/cat_E24_lips.mp4 --out work_ex/final_cat_lips.mp4; show the user frames 104-183 vs the example
  (they asked: "what's going on with the cat's body at 5 seconds": stretched/melted body, smeared paws, half sunglasses).
  If still bad: re-run with another seed (edit the --seed formula or delete work_ex/wan/cat_E_shot2 + ComfyUI output/wan/cat_E_shot2).
- 2026-09-26 21:00 DONE: shot-2 re-render. First 3 attempts OOM'd in KSampler (ComfyUI full-loaded the 11.2 GB Wan model,
  no room for activations); fixed with run_example.ps1 -ReserveVram (default 4 GB -> partial load, ~28 s/frame, no speed loss).
  Spliced at 52..91 (old frames in work_ex/wan/cat_E/frames_backup). Rebuilt: RIFE -> cat_E24, lip pass -> cat_E24_lips,
  work_ex/final_cat_lips.mp4 (previous build kept as *_prev). Before/after: work_ex/round4/shot2_final_compare.jpg.
  Result: no stretched body, full black sunglasses; remaining: pinkish paw smear at ~6.5 s, streaks at lower-left ~4.5 s.
  NEXT: user approves shot 2 (or another seed), then run_example.ps1 -Side user.
- 2026-09-26 22:48 user_E render DONE (181 f, 101.9 min, no OOM with -ReserveVram 4). Review found two defects:
  1. FLICKER (both passes): wan_animate.py decoded with VAEDecodeTiled temporal_size=16/overlap=4 -> a new causal-VAE chunk
     every 12 frames -> whole frame flashes brighter/washed-out at window-relative frames 12 and 24, decays over ~8 f
     (cat pass +15 RGB, user pass up to +28; RIFE flagged 17 fake "cuts"). Root cause fixed: --decode-frames default 64
     (whole window in one temporal chunk). Existing frames: deflicker.py (per-channel quantile envelope per shot) cuts the
     max in-shot step 21.7 -> 6.8 (source 4.9), mean 2.71 -> 0.88 (source 0.96); flash frames stay slightly hazy.
     Output: work_ex/wan/user_E/frames_df; check sheet work_ex/round4/deflicker_check.jpg.
  2. USER IDENTITY DRIFTS per shot (work_ex/round4/user_identity_check.jpg): f5/f40 resemble the user, f70/f100/f178 are
     other men, f150/f165 bearded; white shirt instead of black; ghost outlines at f40/f150.
     Cause: refs/user_ref.png is full-body 310x1024 -> face ~40 px; each shot starts a fresh window with only the ref for identity.
  DECISION PENDING (user): better face reference (close-up photo) + re-render user side (probe first), and whether to
  re-render the cat with the flicker-free decode or keep the deflickered cat.
- 2026-09-27 cat frames deflickered: work_ex/wan/cat_E/frames_df (max in-shot step 13.8 -> 4.0, flash pairs ~17 -> 5-7.6;
  source 4.5-6.9). Sheet work_ex/round4/cat_deflicker_check.jpg. run_example.ps1 -Side user now paints over frames_df when it
  exists, and -FixShot re-runs deflicker after each splice so frames_df never goes stale.
  Old drifting user render moved aside (resume would have reused it): work_ex/wan/user_E_oldref(+.mp4),
  vendor/ComfyUI/output/wan/user_E_oldref, work_ex/round4/user_E24_oldref.
- 2026-10-04 PAUSED the render work (user: UI/UX first). Repo split into cat-duet-recast (this) + cat-duet-faceswap-archive.
  Face close-ups NOT found anywhere (refs/ unchanged since 09-26; only ID-card photos in Downloads, not usable).
- 2026-10-04 Render Studio built: studio/ (FastAPI + Svelte 5/Tailwind 4, dark editing-suite). Views: Monitor (log parser,
  frame-weighted s/frame, ETA, partial/complete model load, OOM alerts), References (drop photos -> make_refs.py user
  --no-sunglasses-check -> candidate vs current -> promote), Shot review (5 shots, 8 lanes incl. ffmpeg-exact source
  frames, wipe, approve/re-roll + notes), Settings + "Send to editor" (video_editor_ai API with its session cookie, else
  stage + open). Verified: every endpoint by curl incl. a real reference build (face 563 px from IMG_0176), all views in
  Chrome; the <video> preview only stalls in a hidden automation tab (Chrome defers media there; ffprobe over HTTP OK).
  NEXT (render lane): user drops close-ups in Studio > References -> build -> promote -> run_example.ps1 -Side user -Probe.

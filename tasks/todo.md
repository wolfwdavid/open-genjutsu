# Open Source Higgsfield submission (deadline 2026-11-03 08:30 PT)

Plan written 2026-11-02 evening. Repo assembled from `synaps-optic-context/recast` (newest
scripts) and `hugginface_profile/video_generator` (Render Studio, run scripts).

## Repo
- [x] Merge both copies, strip personal paths, keep refs/face photos and models out
- [x] MIT license
- [x] Export the ComfyUI graph (`workflows/`, API + editor format, validated against ComfyUI 0.37.0)
- [x] Judge-facing README, `docs/SETUP.md` (CUDA + XPU), `docs/OWN_CLIP.md`
- [x] Demo render and QA sheets tracked (`demo/`, `docs/images/`)
- [x] Public GitHub repo: https://github.com/wolfwdavid/open-genjutsu
- [x] comfy-build.yaml scanned, patched (GGUF/ONNX added, XPU pins emptied), validate + push --dry-run green (9 models, 22 GB upper bound)
- [x] Platform build cut and green: Build 2ba4aa66-00cf-4905-9dcc-347f5acbb29c, release b0400e09-05c1-43fd-80e5-61c72f361e97 v1 (linux/nvidia cu130, ready 2026-11-02 23:27 UTC). 2 uploads (ViTPose, 2.5 GB), 7 public fetches. Not deployed (would bill hourly).
- [x] Deployment dep-9bb2eea3-380e-4ebf-b212-de5cca0b7024 (rtx-pro-6000-mig-2g-48gb, us, min 0 max 1) ready; run_cloud.py probe job 8b4b7584 succeeded: 33 frames in 81 s wall
- [ ] Clean-clone test on the second laptop: follow `docs/SETUP.md` steps 0 to 2, note anything that breaks

## POV Switch (second app, user asked 2026-11-02 evening, skipped questions)
- [x] game/analyze.py: cuts by HSV-histogram correlation, motion peaks, YOLOv10 characters, 1:1 crops, "someone else" card; tested on both sample clips
- [x] game/graphs.py: Qwen-Image-Edit-2509 POV graph + Wan2.2 I2V two-expert outcome graph, validated by ComfyUI 0.37.0 (queued + cancelled locally; node-level validation passed on the deployment too)
- [x] game/server.py + game/ui/index.html: upload, moments, characters, POV, outcome, continue, tree
- [x] comfy-build.yaml: 8 open weights resolved by sha256 to public HF sources; release v2 green
- [x] Deployment on v2; cat POV (41 s), girl POV (19 s), cat outcome (62 s) verified by eye; POV prompt rewritten once (first try showed the character instead of their view)
- [x] Demo frames into docs/images, README numbers
- [x] Outcome prompt rewritten (viewer invisible + third-person negative) after the girl's first outcome drifted to third person; re-run is first person with hair at the edge
- [x] Halo Spartan POV via the "someone else here" card: works (HUD text removed, marine ahead, rifle at the edge)

- [x] Switch eyes (from_branch): verified cat outcome -> girl's eyes; image-2 conditioning rejected, text continuity kept

## Real-time lane: Matrix-Game 2.0 (started 2026-11-03 00:30 UTC, user said "keep going")
Goal: an action-driven first-person chunk (WASD + camera) from the POV frame, so the game feels live instead
of one 20-60 s clip per move. Production deployment (release v2) was to stay untouched until v3 was proven.
- [x] Research: no ComfyUI node and no diffusers pipeline exist for 2.0; the official code needs flash-attn only
      in the action module and the cross-attention; everything else already falls back to SDPA
- [x] `../ComfyUI-MatrixGame2` (https://github.com/wolfwdavid/ComfyUI-MatrixGame2, MIT): vendored `mg2core/` with an SDPA `attention.py`,
      `runner.py` (streaming pipeline as a resident `Session`, one 12-frame block per action, KV caches kept
      between prompts), nodes `MatrixGame2Loader` + `MatrixGame2Walk` (writes a state JSON next to the mp4)
- [x] CPU smoke test (random weights, 1 layer, 352x640): 3 blocks incl. KV-cache rollover, session restart, all OK
- [x] Node registers in ComfyUI 0.37 (`/object_info`), graph accepted by `/prompt` with placeholder weights
- [x] comfy-build.yaml: 3 Skywork/Matrix-Game-2.0 weights by sha256 + the node as a `source: local` zip
      (no public repo needed); release v3 `7655ecaa-acb7-4c2b-8cdc-1853ee584b0c` green 2026-11-03 01:02 UTC
      (diffusers 0.42.1, einops, ftfy installed from the pack's requirements)
- [x] game/server.py `/api/walk`, game/graphs.py `walk_clip`, UI pad + WASD/QE/RF keys, WALK tree nodes
- [ ] BLOCKED: staging deployment on v3 refused with `deploy_payment_required`; `comfy cloud status` shows the
      "NYC Build Night" workspace at -$19.41 (-4095 credits) and production was stopped with stopReason
      `credits` at 16:25 UTC. No Matrix-Game job has run on a GPU yet. Needs credits, then:
      `comfy deploy up comfy-build.yaml --create --name staging --release 7655ecaa-acb7-4c2b-8cdc-1853ee584b0c --gpu rtx-pro-6000-mig-2g-48gb --region us --min 0 --max 1 --no-editor`
      and `comfy deploy run comfy-build.yaml --deployment staging --workflow <probe> --asset-root <dir with pov.png>`
      (probe graph: %TEMP%\wf\mg2_walk.api.json)
- [ ] After a green probe: time per block, move production to v3, README numbers, demo shot

## Demo video (user records)
- [ ] Screen capture per `demo/SHOT_LIST.md`, under 2 min
- [ ] Upload unlisted, paste link into README

## Submit
- [ ] Sign-up form: https://form.typeform.com/to/Ayn9Cwfj
- [ ] Submission form: https://form.typeform.com/to/V1dSQrkS (repo, video link, categories: Best Overall + Open Weights Only)
- [ ] Social post with #OpenSourceHiggsfield

## Review (2026-11-02)
- 81 files tracked, 11 MB. No refs photos, faces, settings.json or model weights in the index (checked with git ls-files).
- Workflow export validated against ComfyUI 0.37.0 /prompt: every node type and widget accepted; only the placeholder input folders were rejected, as expected.
- Docs claims cross-checked against the scripts: LoRA defaults, pose_body --cuts, face_crops --from-pose, make_refs _hq output.
- Open risk: demo/cat_rap_example.mp4 is derived from Higgsfield's own example clip (their performer is in frame). Higgsfield sent Comfy a cease-and-desist about the challenge. User decides whether to keep it or swap for a clip they own.
- Not done: CUDA setup path is untested on NVIDIA hardware (stated in README).

## Review (2026-11-03 12:20 ET)
- The laptop slept from 20:12 ET (01:12 UTC) to 12:12 ET; the 08:30 PT deadline passed while it was asleep.
  Nothing ran in between. The Matrix-Game lane was built and release v3 cut before the sleep.
- The Comfy workspace is a shared team workspace (86 seats); its balance went negative overnight and the
  production deployment was stopped for credits, so neither app can be demoed on the platform until it is topped up.
- Local servers restarted: game 8766 (new /api/walk), Studio 8765 and ComfyUI 8188 untouched.

## Review (2026-11-05)
- Pushed open-genjutsu main and pushed the node pack to https://github.com/wolfwdavid/ComfyUI-MatrixGame2 (private for now, MIT). The README links resolve once it is made public: `gh repo edit wolfwdavid/ComfyUI-MatrixGame2 --visibility public --accept-visibility-change-consequences`.
- Node pack README now has a Status section: CPU-verified and built into release v3, not yet run on a GPU with real weights.
- The build spec still packages the node as `source: local`; switching it to the git repo is optional and needs a new release.
- Still open: GPU probe on release v3 (needs workspace credits), demo video, sign-up and submission forms, social post.

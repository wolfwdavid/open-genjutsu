# Open Source Higgsfield submission (deadline 2026-11-03 09:00 PT)

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
- [ ] Deployment moved to v2 (staging 17 models), then one POV job + one outcome job verified by eye
- [ ] Demo frames into docs/images, README numbers

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

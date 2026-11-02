# Open Source Higgsfield submission (deadline 2026-11-03 09:00 PT)

Plan written 2026-11-02 evening. Repo assembled from `synaps-optic-context/recast` (newest
scripts) and `hugginface_profile/video_generator` (Render Studio, run scripts).

## Repo
- [x] Merge both copies, strip personal paths, keep refs/face photos and models out
- [x] MIT license
- [x] Export the ComfyUI graph (`workflows/`, API + editor format, validated against ComfyUI 0.37.0)
- [x] Judge-facing README, `docs/SETUP.md` (CUDA + XPU), `docs/OWN_CLIP.md`
- [x] Demo render and QA sheets tracked (`demo/`, `docs/images/`)
- [ ] Public GitHub repo under wolfwdavid, pushed (local commit 8c9f51a ready; `gh repo create` needs the user to run it)
- [ ] Clean-clone test on the second laptop: follow `docs/SETUP.md` steps 0 to 2, note anything that breaks

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

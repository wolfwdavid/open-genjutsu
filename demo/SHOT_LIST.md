# Demo video: shot list (target 1:45, hard max 2:00)

Judges score the work, not the editing. Screen-record on the dual-screen laptop: Render Studio on
the top screen, ComfyUI canvas or terminal on the bottom. Talk over it or add captions. No music
under the voice-over except where the render plays.

| time | on screen | say |
|---|---|---|
| 0:00-0:12 | `demo/cat_rap_example.mp4` full screen, sound on | "This is Higgsfield's Genjutsu motion transfer, rebuilt in ComfyUI with open weights. One photo of my cat replaced the cat in the clip. Every move, cut and the audio are the original's. The mouth is synced to the vocals." |
| 0:12-0:25 | Split: Higgsfield example frame left, our frame right (`docs/images/contact_sheet.jpg`) | "The closed version runs in their cloud. This ran on a laptop iGPU with 16 GB shared memory in about ninety minutes." |
| 0:25-0:45 | ComfyUI canvas, drag in `workflows/wan_animate_replace.json`, pan across it slowly | "The core is Wan2.2-Animate-14B in replacement mode. Pose skeleton, character mask, face crops and the reference plate go in. The graph renders 33-frame windows chained with continue motion, so clips can be any length." |
| 0:45-1:00 | `docs/images/pose_mask_sheet.jpg` then `docs/images/fur_mask_sheet.jpg` | "Inputs are built per performer: YOLOX and RTMW whole-body pose, ISNet mattes, and a skeleton retargeted to cat proportions so the model draws a cat, not a furry human. Fur-only masks keep the costume from the source." |
| 1:00-1:15 | `docs/images/lip_pass_strip.jpg`, then 3 s of the render on a close-up | "The lip sync is LivePortrait animals. We isolate the vocals, turn them into a mouth-open curve, and add only the mouth delta back through a feathered mask, so the Wan render's fur and lighting stay." |
| 1:15-1:30 | Render Studio: Monitor view with a log loaded, then Shot review with the wipe | "Render Studio wraps it: live ETA and out-of-memory alerts, drag-in reference photos, and per-shot review with a wipe so you re-roll one shot instead of the whole clip." |
| 1:30-1:45 | GitHub repo page, scroll README; end on the model table | "Everything is MIT, every weight is open, setup for CUDA and Intel Arc is in the README. Repo link in the description." |

Record at 1080p. Export H.264. Upload unlisted to YouTube (the form asks for a link) and put the
same link in the README under the demo render.

## Social post (Most Viral is a category)

Post the render clip with the repo link and `#OpenSourceHiggsfield`, tagging @ComfyUI.
Suggested text: "Rebuilt Higgsfield's Genjutsu motion transfer with open weights in ComfyUI.
One photo of my cat, one clip, a laptop iGPU. Code + workflow: <repo>"

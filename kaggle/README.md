# Kaggle render lane

Runs the Wan2.2-Animate step of the recast on a free Kaggle GPU (2x T4, 30 h/week) instead of
the local Arc 140T. `wan_animate.py` needs ComfyUI on the same disk, so the whole render step
runs inside a Kaggle script kernel; the script itself is unchanged.

Run the driver with the system Python (it has Kaggle CLI 2.2; the project `.venv` has an old
`kaggle` package without the CLI).

```
python -m kaggle auth login                                   # once, OAuth in the browser
python recast/kaggle/kaggle_run.py run --side cat --mode probe # bundle, push, wait, pull
python recast/kaggle/kaggle_run.py run --side cat --mode full --name cat_E_k
python recast/kaggle/kaggle_run.py run --side user --mode probe --ref refs/user_ref.png \
       --bg-dir work_ex/wan/cat_E_k/frames
python recast/kaggle/kaggle_run.py run --shot 2 --name cat_E_k            # re-render shot 2
```

- `kaggle_run.py` copies the inputs (`work_ex` pose/mask/face dirs, `cut.mp4`, refs,
  `wan_animate.py`, about 260 MB) into `_dataset/` and uploads them as the private dataset
  `recast-work-ex`; unchanged inputs are not re-uploaded. It then writes `_kernel/` (the
  kernel script with `PARAMS` injected plus `kernel-metadata.json`), pushes the kernel
  `recast-wan-render`, polls its status, and pulls the output into `work_ex/wan/<name>/`
  with the same layout as a local run, so RIFE, the lip pass and `finish.py` follow unchanged.
- `render_kaggle.py` (the kernel): pins ComfyUI, ComfyUI-GGUF and VideoHelperSuite to the
  local commits, keeps the image's CUDA torch, downloads the six weight files from public
  Hugging Face repos (19.3 GB), starts ComfyUI with `--fp32-vae` (the T4 has no bf16),
  runs the render, and copies `frames/`, `run.log`, `render.json` and the raw ComfyUI window
  outputs to `/kaggle/working`.
- Prompts and arguments come from `recast/runner.py`, so Kaggle takes match local takes.
- Limits: one kernel run is one probe, one take or one shot fix (12 h session cap); resuming
  a take across runs is not wired. Weights are re-downloaded every run (a few minutes).

## Wan2.2-Animate-2 lane (`--engine animate2`, 2026-10-08)

Animate-2 (Aug 2026) is driven by the video itself: no skeleton, no character mask, no background
plate. That removes the "cat turns human" failure of the skeleton lane, but there is no in-footage
replacement either: each side is rendered on its own from a crop of the clip (`wan_animate2.py`,
left 544 px for the cat, right 544 px for the user, at 480x640) and composited afterwards.

```
python recast/kaggle/kaggle_run.py run --engine animate2 --side cat  --mode probe --name cat_A2_probe
python recast/kaggle/kaggle_run.py run --engine animate2 --side user --mode full  --name user_A2 --ref refs/user_ref.png
```

- Weights: `realrebelai/Wan-Animate-2_GGUFs` `Distilled/Wan-Animate-2-14B-TURBO-Q4_K_M.gguf` (11.3 GB,
  downloaded from Hugging Face inside the kernel unless the weights dataset holds it); text encoder,
  CLIP vision and VAE are the Animate lane's files. Distilled settings: 10 steps, lcm, simple, cfg 1,
  shift 5, no LoRA.
- Loader: a GGUF carries no `model_type`, so the stock GGUF loader silently loads Animate-2 as plain
  Wan 2.1 I2V and ignores the driving video. The kernel installs `RealRebelAI/Rebels_w3a8_Loader`
  (pinned by `LOADER_COMMIT`) and the graph uses `RebelsGGUFUnetLoaderMeta` with `model_type=animate2`;
  `logs/comfy.log` must contain `model class=WAN_Animate2`.
- The pose branch is cached on the CPU (`--pose-cache cpu`) so it runs once per window, not once per step.
- ComfyUI pin `88ab4a0` already contains the native `WanAnimate2ToVideo` / `WanAnimate2Cache` nodes.
- The reference image should match the driving clip's opening pose; the background comes from the
  prompt (`runner.CAT_A2_PROMPT`: "Character Description: ... Background description: ...").

## SeedVR2 restore/upscale (`--engine seedvr2`, 2026-10-08)

One-step video restoration with SeedVR2 3B (native ComfyUI nodes, in the pinned ComfyUI since
8e2e54e). Runs on a take's deflickered frames, before RIFE, split at the shot cuts:

```
python recast/kaggle/kaggle_run.py run --engine seedvr2 --mode full --src work_ex/wan/cat_E/frames_df \
       --name cat_E_up --kernel recast-seedvr2 --weights ""
```

- Weights from `Comfy-Org/SeedVR2`: `seedvr2_3b_fp16` (6.8 GB, under 2 min to download) and the
  0.5 GB VAE. fp16 because the T4 has no fast int8/fp8 path.
- First T4 run: sampling the 52-frame first shot at 2x (1536x1152) took about 10 min including the
  model load; the tiled decode then ran out of memory at tile 512 / 64 frames, so the decode now
  uses tile 256 / 32 frames (`--decode-tile`, `--decode-frames`).
- `runner.post_chain(..., upscaled=<frames dir>)` makes RIFE read the restored frames and encodes
  at 1440x1080 instead of 960x720.
- The driver waits until the new dataset version's files are listed before pushing a kernel: a
  kernel pushed earlier mounts the previous version (it now fails in seconds when its script is missing).

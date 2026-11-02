# Setup

Three Python environments, one ComfyUI checkout, and about 22 GB of weights. Everything lives
inside the repo folder and is gitignored. Windows paths are shown; on Linux swap
`.venv/Scripts/python.exe` for `.venv/bin/python`.

Tested configuration: Windows 11, Python 3.11, `uv` 0.12, ComfyUI v0.37.0, Intel Arc 140T (16 GB
shared). The NVIDIA path below uses the stock ComfyUI CUDA wheels and has not been run by the
author; if something differs, `docs/history/comfy_setup.md` has the full XPU log to compare against.

## 0. Tools

- Python 3.11, [`uv`](https://docs.astral.sh/uv/), `git`, `ffmpeg` on PATH, Node 20+ (Studio only).
- `rife-ncnn-vulkan`: download the release zip from
  https://github.com/nihui/rife-ncnn-vulkan/releases and unzip into `vendor/rife-ncnn-vulkan/`.

## 1. Pipeline venv (`.venv`)

```
uv venv .venv --python 3.11
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

`requirements.txt` pins `onnxruntime-directml` (Windows, any GPU). On Linux with NVIDIA replace it
with `onnxruntime-gpu` and edit the provider list at the top of `track.py` and in `pose_body.py`
and `make_refs.py` to prefer `CUDAExecutionProvider`.

Extra models for this venv, placed in `models/`:

| file | source |
|---|---|
| `isnet-general-use.onnx` | https://github.com/danielgatis/rembg/releases (ISNet general use) |
| YOLOX-m HumanArt + RTMW-x wholebody ONNX | fetched by `rtmlib` on first run of `pose_body.py` |
| `buffalo_l` (insightface) | fetched by insightface on first run of `track.py` |

## 2. ComfyUI venv (`.venv-comfy`) and checkout

```
git clone https://github.com/comfyanonymous/ComfyUI vendor/ComfyUI
cd vendor/ComfyUI/custom_nodes
git clone https://github.com/city96/ComfyUI-GGUF
git clone https://github.com/kijai/ComfyUI-WanAnimatePreprocess
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
cd ../../..
uv venv .venv-comfy --python 3.11
```

**NVIDIA (CUDA):**

```
uv pip install --python .venv-comfy/Scripts/python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-comfy/Scripts/python.exe -r vendor/ComfyUI/requirements.txt
uv pip install --python .venv-comfy/Scripts/python.exe gguf sentencepiece protobuf onnx onnxruntime-gpu opencv-python imageio-ffmpeg matplotlib
```

**Intel Arc (XPU):** torch must stay `+xpu`; use a constraints file and skip the two optional
ComfyUI packages that re-resolve torch. The exact commands and versions are in
`docs/history/comfy_setup.md` sections 1 to 3.

Launch (both):

```
.venv-comfy/Scripts/python.exe vendor/ComfyUI/main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch --reserve-vram 4
```

Add `--use-pytorch-cross-attention` on XPU. `run_example.ps1` starts and stops the server for you.

## 3. Weights for ComfyUI

Create `vendor/ComfyUI/extra_model_paths.yaml`:

```yaml
comfy_local:
  base_path: <absolute path to this repo>/models/comfy/
  diffusion_models: unet/
  unet: unet/
  text_encoders: text_encoders/
  clip_vision: clip_vision/
  vae: vae/
  loras: loras/
  detection: detection/
```

| folder under `models/comfy/` | file | source |
|---|---|---|
| `unet/` | `Wan2.2-Animate-14B-Q4_K_M.gguf` | https://huggingface.co/QuantStack/Wan2.2-Animate-14B-GGUF |
| `text_encoders/` | `umt5-xxl-encoder-Q5_K_M.gguf` | https://huggingface.co/city96/umt5-xxl-encoder-gguf |
| `clip_vision/` | `clip_vision_h.safetensors` | https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged |
| `vae/` | `wan_2.1_vae.safetensors` | https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged |
| `loras/` | `lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors` | https://huggingface.co/Kijai/WanVideo_comfy (Lightx2v) |
| `loras/` | `WanAnimate_relight_lora_fp16.safetensors` | https://huggingface.co/Kijai/WanVideo_comfy (LoRAs/Wan22_Animate) |
| `detection/` | `vitpose_h_wholebody_model.onnx`, `yolov10m.onnx` | linked from the ComfyUI-WanAnimatePreprocess README |

Loader dropdowns are rebuilt on every `/object_info` call, so weights added while the server
runs are picked up without a restart.

## 4. Lip-pass venv (`.venv-lp`)

```
uv venv .venv-lp --python 3.11
uv pip install --python .venv-lp/Scripts/python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-lp/Scripts/python.exe -r vendor/lp_harness/requirements_base.txt onnxruntime-gpu
git clone https://github.com/KwaiVGI/LivePortrait vendor/LivePortrait
```

Download the **animals v1.1** weights and `landmark.onnx` from
https://huggingface.co/KwaiVGI/LivePortrait into `vendor/LivePortrait/pretrained_weights/` as the
LivePortrait README describes, and copy `~/.insightface/models/buffalo_l` into
`vendor/LivePortrait/pretrained_weights/insightface/models/buffalo_l`. `vendor/README.md` has the
XPU-specific notes. Smoke test: `.venv-lp/Scripts/python.exe vendor/lp_harness/smoke_lp.py`.

Vocal isolation for the mouth curve uses
[`audio-separator`](https://github.com/nomadkaraoke/python-audio-separator) with the
`UVR-MDX-NET-Voc_FT` model in its own venv (`.venv-audio`); any isolated-vocals WAV works.

## 5. Render Studio (optional)

```
powershell -ExecutionPolicy Bypass -File studio\run_studio.ps1
```

Builds the Svelte UI on first run and serves http://127.0.0.1:8765 from the `.venv` Python
(FastAPI, httpx). See `studio/README.md`.

## Sanity checks

- `GET http://127.0.0.1:8188/system_stats` shows your device and `comfyui_version`.
- `python workflows/export_workflow.py --comfy http://127.0.0.1:8188` must print both files; it
  fails on the first node type the server does not know, which pinpoints a missing custom node.
- `run_example.ps1 -Side cat -Probe` renders 33 frames and reports seconds per frame
  (Arc 140T: ~28 s/frame; an RTX 4090 should be well under 5).

## 6. Cloud render lane (optional, skips sections 2 and 3)

If you do not want to run the 14B model locally, use the Comfy developer platform build in
`comfy-build.yaml`. You still need section 1 for the prep scripts and section 4 for the lip pass.

```
pip install -U comfy-cli
comfy cloud login
comfy build release ls comfy-build.yaml            # a green release exists (v1)
comfy deploy up comfy-build.yaml --gpu rtx-pro-6000-mig-2g-48gb --region us --min 0 --max 1
comfy deploy status comfy-build.yaml --watch       # until ready (~5 min: 22 GB of models are staged)
python run_cloud.py --block E --side L --ref refs/cat_body_base_hq.png --name cat_cloud --prompt "..." \
    --width 768 --height 576 --length 33 --stride 2 --probe 33
```

`run_cloud.py` takes the same flags as `wan_animate.py`. A deployment with `--min 0` bills only
while a job runs. `comfy deploy stop comfy-build.yaml` when you are done.

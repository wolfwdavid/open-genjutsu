# Headless ComfyUI on torch XPU (step 13)

Verified 2026-09-25 on Intel Arc 140T (16 GB), Windows 11, Python 3.11, uv 0.12.
ComfyUI `88ab4a0` (v0.37.0, master of 2026-09-24). Everything lives under the repo and is gitignored:
`.venv-comfy/`, `vendor/ComfyUI/`, `models/comfy/`.

## 1. Create the venv

```
uv venv .venv-comfy --python 3.11
uv pip install --python .venv-comfy/Scripts/python.exe torch==2.14.0+xpu torchvision==0.29.0+xpu \
    --index-url https://download.pytorch.org/whl/xpu
.venv-comfy/Scripts/python.exe -c "import torch;print(torch.__version__, torch.xpu.is_available())"
#   -> 2.14.0+xpu True
```

## 2. Clone ComfyUI + custom nodes

```
git clone https://github.com/comfyanonymous/ComfyUI vendor/ComfyUI
cd vendor/ComfyUI/custom_nodes
git clone https://github.com/city96/ComfyUI-GGUF
git clone https://github.com/kijai/ComfyUI-WanAnimatePreprocess
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
```

Native Wan-Animate node: `vendor/ComfyUI/comfy_extras/nodes_wan.py` (`class WanAnimateToVideo`, line ~1113;
the same file also registers `WanAnimate2ToVideo` and `WanAnimate2Cache`).

## 3. Install requirements (torch must stay +xpu)

Do NOT run the stock `pip install -r requirements.txt`: it re-resolves `torch` and pulls
`comfyui-workflow-templates`, which is ~300 MB of demo media that the headless API never needs
(on this link that download timed out after 30+ min). Use a filtered requirements file plus a
constraints file that pins torch:

```
# constraints.txt
torch==2.14.0+xpu
torchvision==0.29.0+xpu

# comfy-reqs.txt = vendor/ComfyUI/requirements.txt minus the two lines
#   comfyui-workflow-templates==...   (demo templates, optional; startup logs an ERROR banner but continues)
#   comfyui-embedded-docs==...        (node docs, optional; startup logs INFO)
# plus the custom-node deps (onnxruntime-gpu replaced by the DirectML build):
gguf>=0.13.0
sentencepiece
protobuf
onnx
onnxruntime-directml
opencv-python
imageio-ffmpeg
matplotlib            # imported by WanAnimatePreprocess/pose_utils but missing from its requirements.txt

UV_HTTP_TIMEOUT=900 uv pip install --python .venv-comfy/Scripts/python.exe \
    -r comfy-reqs.txt -c constraints.txt \
    --extra-index-url https://download.pytorch.org/whl/xpu --index-strategy unsafe-best-match
```

`--index-strategy unsafe-best-match` is needed because the XPU index also hosts old `tqdm`/`numpy`
wheels and the uv default first-index rule would otherwise refuse the PyPI versions.
Result: 112 packages, torch/torchvision unchanged (`2.14.0+xpu` / `0.29.0+xpu`), onnxruntime 1.24.4 with
providers `['DmlExecutionProvider', 'CPUExecutionProvider']`. Notable versions: transformers 5.16.1,
opencv-python 4.11.0.86, gguf 0.19.0, av 18.0.0, kornia 0.8.3, spandrel 0.4.2, comfy-kitchen 0.2.35,
comfy-aimdo 0.5.5, comfyui-frontend-package 1.53.6.

Custom-node requirement outcomes:
- ComfyUI-GGUF: gguf, sentencepiece, protobuf - OK
- ComfyUI-WanAnimatePreprocess: onnx 1.23.0, opencv-python - OK; `onnxruntime-gpu` skipped (CUDA-only),
  `onnxruntime-directml` installed instead; `matplotlib` added (undeclared import)
- ComfyUI-VideoHelperSuite: opencv-python, imageio-ffmpeg 0.6.0 - OK
- Nothing failed.

## 4. Model paths

`vendor/ComfyUI/extra_model_paths.yaml` (already written) maps a `comfy_local` entry at
`base_path: <repo>/models/comfy/`:

| ComfyUI folder key  | subfolder(s) under models/comfy/ | used by                                        |
|---------------------|----------------------------------|------------------------------------------------|
| diffusion_models    | unet/, diffusion_models/         | UnetLoaderGGUF / UNETLoader (Wan2.2-Animate)   |
| unet                | unet/                            | legacy alias                                   |
| text_encoders       | text_encoders/                   | CLIPLoaderGGUF / CLIPLoader (umt5-xxl)         |
| clip_vision         | clip_vision/                     | CLIPVisionLoader (clip_vision_h)               |
| vae                 | vae/                             | VAELoader (wan_2.1_vae)                        |
| loras               | loras/                           | LoraLoaderModelOnly (relight lora etc.)        |
| detection           | detection/                       | OnnxDetectionModelLoader (vitpose + yolo .onnx)|

All seven folders exist (empty until the weights job lands). The startup log confirms each
`Adding extra search path ...` line. Loader dropdown lists are rebuilt on every `/object_info` call,
so weights added while the server runs are picked up without a restart.

## 5. Launch (headless)

```
PYTHONIOENCODING=utf-8 .venv-comfy/Scripts/python.exe vendor/ComfyUI/main.py \
    --listen 127.0.0.1 --port 8188 --use-pytorch-cross-attention --disable-auto-launch
```

Attention flag that worked: `--use-pytorch-cross-attention` (log: `Using pytorch attention`, `Device: xpu:0
Intel(R) Arc(TM) 140T GPU (16GB)`, `pytorch version: 2.14.0+xpu`). `--use-split-cross-attention` was not
needed and was not tested. Startup to `Starting server` takes about 20 s (VHS import alone is ~5 s).
Set `PYTHONIOENCODING=utf-8`: VHS node categories contain emoji and cp1252 consoles choke on them.

`GET http://127.0.0.1:8188/system_stats` reported `devices[0] = {"name": "xpu:0 Intel(R) Arc(TM) 140T GPU (16GB)",
"type": "xpu", "vram_total": 17576935424}`, `pytorch_version: 2.14.0+xpu`, `comfyui_version: 0.37.0`.
`GET /object_info` returned 1015 node types.

## 6. Node class names (all confirmed present in /object_info)

- Core Wan-Animate: `WanAnimateToVideo` (category `model/conditioning/wan/animate`), plus `WanAnimate2ToVideo`,
  `WanAnimate2Cache`. Inputs: required `positive, negative, vae, width, height, length, batch_size,
  continue_motion_max_frames, video_frame_offset`; optional `clip_vision_output, reference_image, face_video,
  pose_video, background_video, character_mask, continue_motion`.
- ComfyUI-GGUF (category `bootleg`): `UnetLoaderGGUF`, `UnetLoaderGGUFAdvanced`, `CLIPLoaderGGUF`, `DualCLIPLoaderGGUF`.
- ComfyUI-VideoHelperSuite: `VHS_LoadVideo` (upload), `VHS_LoadVideoPath` (absolute path - the one to drive from
  the API), `VHS_VideoCombine`.
- ComfyUI-WanAnimatePreprocess (category `WanAnimatePreprocess`): `OnnxDetectionModelLoader`,
  `PoseAndFaceDetection`, `DrawViTPose`, `PoseRetargetPromptHelper`, `PoseDetectionOneToAllAnimation`.
- Also present and needed for the graph: `CLIPVisionLoader`, `CLIPVisionEncode`, `VAELoader`, `CLIPLoader`,
  `LoraLoaderModelOnly`, `ModelSamplingSD3`, `KSampler`, `LoadImage`.

## 7. Caveats

- `OnnxDetectionModelLoader.onnx_device` only offers `CUDAExecutionProvider` / `CPUExecutionProvider`
  (`custom_nodes/ComfyUI-WanAnimatePreprocess/nodes.py:36`). Use `CPUExecutionProvider` from the API; it works
  with onnxruntime-directml. Passing `CUDAExecutionProvider` makes onnxruntime raise (provider not available).
  To run ViTPose/YOLO on the GPU, add `"DmlExecutionProvider"` to that list (one-line local patch;
  `models/onnx_models.py` passes the string straight to `InferenceSession`).
- Startup prints an `********** ERROR *********** comfyui-workflow-templates is not installed` banner. Harmless:
  it only disables the template gallery in the web UI; the API and all nodes work. `/system_stats` shows
  `installed_templates_version: null` for the same reason.
- `Could not autodetect AIMDO implementation, assuming Nvidia`: comfy-aimdo (pinned-memory loader) only knows
  cuda/rocm (`comfy_aimdo/control.py:72-81`); on XPU ComfyUI falls back to its normal "RAM pressure cache"
  path. Harmless.
- `On windows we are currently forcing single GPU mode` is the standard Windows notice; irrelevant here.
- comfy_kitchen reports its triton backend as available-but-disabled and cuda unavailable; the eager backend is used.
- Nothing GPU-heavy was executed; only server start, `/system_stats` and `/object_info` were exercised.
- On Windows the venv `python.exe` is a launcher that spawns the base interpreter: to stop the server, kill every
  process whose command line contains `vendor/ComfyUI/main.py` (launcher + real interpreter), e.g.
  `Get-CimInstance Win32_Process | ? CommandLine -like "*vendor/ComfyUI/main.py*" | % { Stop-Process -Id $_.ProcessId -Force }`.

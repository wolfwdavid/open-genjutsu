"""Kaggle script kernel: one Wan2.2-Animate or Wan2.2-Animate-2 render (probe, take or shot fix) of
the cat-duet recast.

kaggle_run.py replaces the PARAMS line below and pushes this file as a private GPU kernel.
Everything heavy lives under /kaggle/tmp (not persisted); only work_ex/wan/<name> minus the
staged inputs is copied to /kaggle/working, which Kaggle keeps as the kernel output.
"""
PARAMS = {}  # injected by kaggle_run.py: dataset, name, args, engine, script, reserve_vram, comfy/gguf/vhs/loader commits

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

TMP = Path("/kaggle/tmp")
OUT = Path("/kaggle/working")


def find_dataset() -> Path:
    """The mount point moved between Kaggle versions (/kaggle/input/<slug>, .../datasets/<owner>/<slug>);
    find the folder that holds our code/wan_animate.py instead of guessing."""
    root = Path("/kaggle/input")
    for p in sorted(root.rglob("wan_animate.py")):
        if p.parent.name == "code":
            return p.parent.parent
    listing = [str(q) for q in root.rglob("*") if q.is_dir()][:40]
    sys.exit(f"dataset {PARAMS['dataset']} not found under {root}; dirs: {listing}")


DATA = Path("/kaggle/input")  # resolved in main()
COMFY = TMP / "ComfyUI"
RECAST = TMP / "recast"
WORK = RECAST / "work_ex"
PORT = 8188
URL = f"http://127.0.0.1:{PORT}"

# (ComfyUI models subfolder, file name expected by the render script, HF repo, path in repo)
WEIGHTS_COMMON = [  # text encoder, image encoder and VAE are shared by both Wan-Animate generations
    ("text_encoders", "umt5-xxl-encoder-Q5_K_M.gguf", "city96/umt5-xxl-encoder-gguf", "umt5-xxl-encoder-Q5_K_M.gguf"),
    ("clip_vision", "clip_vision_h.safetensors", "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
     "split_files/clip_vision/clip_vision_h.safetensors"),
    ("vae", "wan_2.1_vae.safetensors", "Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/vae/wan_2.1_vae.safetensors"),
]
WEIGHTS_BY_ENGINE = {
    "animate": [
        ("unet", "Wan2.2-Animate-14B-Q4_K_M.gguf", "QuantStack/Wan2.2-Animate-14B-GGUF", "Wan2.2-Animate-14B-Q4_K_M.gguf"),
        ("loras", "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", "Kijai/WanVideo_comfy",
         "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"),
        ("loras", "WanAnimate_relight_lora_fp16.safetensors", "Kijai/WanVideo_comfy",
         "LoRAs/Wan22_relight/WanAnimate_relight_lora_fp16.safetensors"),
    ],
    "animate2": [  # distilled (TURBO) Animate-2, 11.3 GB; loaded through the Rebels loader with model_type=animate2
        ("unet", "Wan-Animate-2-14B-TURBO-Q4_K_M.gguf", "realrebelai/Wan-Animate-2_GGUFs",
         "Distilled/Wan-Animate-2-14B-TURBO-Q4_K_M.gguf"),
    ],
    "seedvr2": [  # one-step restoration/upscale; fp16 because the T4 has no int8/fp8 fast path for convrot
        ("unet", "seedvr2_3b_fp16.safetensors", "Comfy-Org/SeedVR2", "diffusion_models/seedvr2_3b_fp16.safetensors"),
        ("vae", "seedvr2_ema_vae_fp16.safetensors", "Comfy-Org/SeedVR2", "vae/seedvr2_ema_vae_fp16.safetensors"),
    ],
}
NEEDS_COMMON = {"animate", "animate2"}  # the Wan text encoder / CLIP vision / VAE


def weights() -> list[tuple[str, str, str, str]]:
    engine = PARAMS.get("engine", "animate")
    return (WEIGHTS_COMMON if engine in NEEDS_COMMON else []) + WEIGHTS_BY_ENGINE[engine]

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def sh(cmd: str, cwd: Path | None = None) -> None:
    log("+ " + cmd)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def setup_comfy() -> None:
    sh(f"git clone -q https://github.com/comfyanonymous/ComfyUI {COMFY}")
    sh(f"git checkout -q {PARAMS['comfy_commit']}", cwd=COMFY)
    nodes = COMFY / "custom_nodes"
    sh(f"git clone -q https://github.com/city96/ComfyUI-GGUF {nodes / 'ComfyUI-GGUF'}")
    sh(f"git checkout -q {PARAMS['gguf_commit']}", cwd=nodes / "ComfyUI-GGUF")
    sh(f"git clone -q https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite {nodes / 'ComfyUI-VideoHelperSuite'}")
    sh(f"git checkout -q {PARAMS['vhs_commit']}", cwd=nodes / "ComfyUI-VideoHelperSuite")
    if PARAMS.get("engine") == "animate2":
        # the node pack sits in a subfolder of its repo (the root has no __init__.py), so clone aside and copy it in
        loader = TMP / "Rebels_w3a8_Loader"
        sh(f"git clone -q https://github.com/RealRebelAI/Rebels_w3a8_Loader {loader}")
        sh(f"git checkout -q {PARAMS['loader_commit']}", cwd=loader)
        shutil.copytree(loader / "ComfyUI-Rebels-W4A8", nodes / "ComfyUI-Rebels-W4A8")
    # keep the image's CUDA torch: ComfyUI's requirements would otherwise pull a new torch/torchaudio
    req = TMP / "requirements.txt"
    lines = []
    for f in (COMFY / "requirements.txt", nodes / "ComfyUI-GGUF" / "requirements.txt",
              nodes / "ComfyUI-VideoHelperSuite" / "requirements.txt"):
        for ln in f.read_text().splitlines():
            s = ln.strip()
            if s and not s.startswith("#") and not s.split("=")[0].split(">")[0].strip() in ("torch", "torchvision", "torchaudio"):
                lines.append(s)
    req.write_text("\n".join(lines) + "\n")
    sh(f"{sys.executable} -m pip install -q -r {req}")
    import torch  # noqa: E402
    log(f"torch {torch.__version__} cuda {torch.cuda.is_available()} "
        f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")


def fetch_weights() -> None:
    """Link the weights from the attached Kaggle weights dataset; download from HF only what is missing.
    HF rate-limits repeated anonymous 11 GB pulls (429 on the xet CAS path), so the fallback disables xet
    and retries with backoff."""
    local = {}
    for p in Path("/kaggle/input").rglob("*"):
        if p.is_file() and p.suffix in (".gguf", ".safetensors"):
            local[p.name] = p
    for folder, name, repo, path in weights():
        dst = COMFY / "models" / folder / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            continue
        if name in local:
            # copy, don't symlink: /kaggle/input is a slow network mount and ComfyUI re-reads the
            # memory-mapped GGUF every window when the model is partially loaded
            t = time.time()
            shutil.copyfile(local[name], dst)
            log(f"{name}: copied {dst.stat().st_size / 1e9:.2f} GB from the dataset in {time.time() - t:.0f}s")
            continue
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        from huggingface_hub import hf_hub_download
        cache = TMP / "hf"
        for attempt in range(5):
            t = time.time()
            try:
                src = Path(hf_hub_download(repo, path, local_dir=cache / repo.replace("/", "__")))
                os.link(src, dst)
                log(f"{name}: {dst.stat().st_size / 1e9:.2f} GB in {time.time() - t:.0f}s")
                break
            except Exception as e:  # noqa: BLE001
                wait = 60 * (attempt + 1)
                log(f"{name}: download failed ({str(e)[:120]}); retry in {wait}s")
                time.sleep(wait)
        else:
            sys.exit(f"could not fetch {name}")


def stage_inputs() -> None:
    RECAST.mkdir(parents=True, exist_ok=True)
    for py in (DATA / "code").glob("*.py"):
        shutil.copy(py, RECAST / py.name)
    for d in ("work_ex", "refs"):
        if (DATA / d).is_dir():
            shutil.copytree(DATA / d, RECAST / d, dirs_exist_ok=True)
        elif (DATA / f"{d}.zip").exists():  # --dir-mode zip uploads that Kaggle left packed
            shutil.unpack_archive(DATA / f"{d}.zip", RECAST / d)
        else:
            sys.exit(f"dataset has no {d}/ or {d}.zip under {DATA}: {sorted(p.name for p in DATA.iterdir())}")
    for extra in PARAMS.get("extra_dirs", []):  # e.g. a cat take's frames for the user pass
        src = DATA / "extra" / extra.replace("/", "__")
        if src.exists():
            shutil.copytree(src, RECAST / extra, dirs_exist_ok=True)
    n = sum(1 for _ in WORK.rglob("*.png"))
    log(f"inputs staged under {RECAST}: {n} png")


def start_comfy() -> subprocess.Popen:
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    logf = open(OUT / "logs" / "comfy.log", "w")
    cmd = [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(PORT), "--disable-auto-launch",
           "--use-pytorch-cross-attention", "--disable-pinned-memory", "--fp32-vae",
           "--reserve-vram", str(PARAMS.get("reserve_vram", 1.5))]
    log("+ " + " ".join(cmd))
    p = subprocess.Popen(cmd, cwd=COMFY, stdout=logf, stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            with urllib.request.urlopen(URL + "/system_stats", timeout=5) as r:
                stats = json.load(r)
            dev = stats["devices"][0]
            log(f"ComfyUI up: {dev['name']} vram {dev['vram_total'] / 1e9:.1f} GB")
            return p
        except Exception:
            if p.poll() is not None:
                sys.exit("ComfyUI exited while starting; see logs/comfy.log")
            time.sleep(5)
    sys.exit("ComfyUI did not answer within 10 min")


def render() -> int:
    env = dict(os.environ, VG_WORK=str(WORK), PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, PARAMS.get("script", "wan_animate.py"), *PARAMS["args"], "--comfy", URL, "--comfy-dir", str(COMFY)]
    log("+ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=RECAST, env=env).returncode


def collect() -> None:
    src = WORK / "wan" / PARAMS["name"]
    dst = OUT / "wan" / PARAMS["name"]
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("in"))
    frames = sorted((dst / "frames").glob("*.png")) if (dst / "frames").exists() else []
    log(f"collected {len(frames)} frames -> {dst}")
    # the raw ComfyUI window outputs let a later run resume a take
    cout = COMFY / "output" / "wan" / PARAMS["name"]
    if cout.exists():
        shutil.copytree(cout, OUT / "comfy_output" / "wan" / PARAMS["name"], dirs_exist_ok=True)


def main() -> None:
    global DATA
    DATA = find_dataset()
    log(f"dataset mounted at {DATA}")
    log(f"params: {json.dumps({k: v for k, v in PARAMS.items() if k != 'args'})}")
    log("args: " + " ".join(PARAMS["args"]))
    stage_inputs()      # cheap and most likely to fail: do it before the 19 GB download
    script = RECAST / PARAMS.get("script", "wan_animate.py")
    if not script.exists():  # a stale dataset version: fail in seconds, not after the weight download
        sys.exit(f"{script.name} is not in the mounted dataset; was the kernel pushed before the dataset version was ready?")
    setup_comfy()
    fetch_weights()
    p = start_comfy()
    rc = 1
    try:
        rc = render()
    finally:
        try:
            collect()
        finally:
            p.terminate()
            try:
                p.wait(30)
            except subprocess.TimeoutExpired:
                p.kill()
    (OUT / "result.json").write_text(json.dumps({"name": PARAMS["name"], "rc": rc, "seconds": time.time() - T0}))
    log(f"done rc={rc}")
    if rc:
        sys.exit(rc)


if __name__ == "__main__":
    main()

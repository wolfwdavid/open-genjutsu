#!/usr/bin/env python
"""
wan_animate.py -- drive a headless ComfyUI (torch XPU) to run Wan2.2-Animate-14B in
character-REPLACEMENT mode over one block of the cut, one side at a time, in chained windows.

Inputs per side (all indexed by CUT frame number, produced by pose_body.py / face_crops.py):
  work/pose_<side>/%05d.png      OpenPose/DWPose-style skeleton of that performer only (960x532)
  work/mask_<side>/%05d.png      8-bit character mask of that performer (960x532)
  work/face_<side>/%05d.png      512x512 face crop of that performer (optional, --face none)
  <block src video>              the frames to keep as background (gj_A.mp4 / gj_B2.mp4)
  refs/<ref>.png                 reference image of the replacement character

The script renders the block in windows of --length frames (must be 4k+1), each window a
separate ComfyUI prompt; window i>0 receives the last --continue frames of window i-1 as
`continue_motion`. Output frames are written as PNG sequences under work/wan/<name>/frames/
and stitched to work/wan/<name>.mp4 at --out-fps (frame stride is --stride: 2 renders every
second cut frame = 15 fps, the closest to the model's native 16 fps).

Usage
  .venv/Scripts/python.exe wan_animate.py --block A --side L --ref refs/user_ref.png \
      --prompt "..." --name user_A [--length 77 --stride 2 --width 832 --height 464]
  ... --probe 33            # only the first window, report s/frame, exit
  ... --dry-run             # print the prompt JSON and exit (no server needed)
ComfyUI must be running: see tasks/comfy_setup.md (default http://127.0.0.1:8188).
"""
from __future__ import annotations

import os
import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = Path(os.environ["VG_WORK"]).resolve() if os.environ.get("VG_WORK") else ROOT / "work"

NEG_PROMPT = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
              "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
              "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")


# ----------------------------------------------------------------------------- comfy client
class Comfy:
    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.client_id = str(uuid.uuid4())

    def _get(self, path: str):
        with urllib.request.urlopen(self.url + path, timeout=60) as r:
            return json.loads(r.read().decode("utf-8"))

    def stats(self):
        return self._get("/system_stats")

    def object_info(self, cls: str):
        return self._get(f"/object_info/{cls}")

    def queue(self, prompt: dict) -> str:
        body = json.dumps({"prompt": prompt, "client_id": self.client_id}).encode("utf-8")
        req = urllib.request.Request(self.url + "/prompt", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            res = json.loads(r.read().decode("utf-8"))
        if "error" in res:
            raise RuntimeError(json.dumps(res, indent=1)[:4000])
        return res["prompt_id"]

    def wait(self, prompt_id: str, poll: float = 5.0, log=None) -> dict:
        t0 = time.time()
        last = -1
        while True:
            hist = self._get(f"/history/{prompt_id}")
            if prompt_id in hist:
                h = hist[prompt_id]
                st = h.get("status", {})
                if st.get("status_str") == "error" or not st.get("completed", False):
                    msgs = st.get("messages", [])
                    err = [m for m in msgs if m[0] == "execution_error"]
                    if err:
                        raise RuntimeError("ComfyUI execution error: " + json.dumps(err[0][1])[:4000])
                    if st.get("completed", False) is False and st.get("status_str") == "error":
                        raise RuntimeError("ComfyUI reported error: " + json.dumps(st)[:2000])
                return h
            el = int(time.time() - t0)
            if log and el // 60 != last:
                last = el // 60
                q = self._get("/queue")
                log(f"  ... waiting {el} s (running {len(q.get('queue_running', []))}, "
                    f"pending {len(q.get('queue_pending', []))})")
            time.sleep(poll)

    def interrupt(self):
        req = urllib.request.Request(self.url + "/interrupt", data=b"", method="POST")
        urllib.request.urlopen(req, timeout=30).read()


# ----------------------------------------------------------------------------- graph
class Graph:
    """Builds a ComfyUI API-format prompt (dict of node_id -> {class_type, inputs})."""

    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.n = 0

    def add(self, class_type: str, **inputs) -> str:
        self.n += 1
        nid = str(self.n)
        self.nodes[nid] = {"class_type": class_type, "inputs": inputs}
        return nid

    @staticmethod
    def out(nid: str, idx: int = 0):
        return [nid, idx]


def load_image_dir(g: Graph, directory: Path, start: int, count: int) -> str:
    """VHS_LoadImagesPath: loads `count` PNGs from `directory` starting at sorted index `start`."""
    return g.add("VHS_LoadImagesPath", directory=str(directory), image_load_cap=count,
                 skip_first_images=start, select_every_nth=1)


def build_window(a, win: dict, models: dict, prev_frames_dir: Path | None) -> tuple[dict, str]:
    """One window = one prompt. Returns (prompt, save_prefix)."""
    g = Graph()
    # models
    if a.unet.lower().endswith(".gguf"):
        model = g.out(g.add("UnetLoaderGGUF", unet_name=a.unet))
    else:
        model = g.out(g.add("UNETLoader", unet_name=a.unet, weight_dtype="default"))
    for lora, strength in models["loras"]:
        model = g.out(g.add("LoraLoaderModelOnly", model=model, lora_name=lora, strength_model=strength))
    model = g.out(g.add("ModelSamplingSD3", model=model, shift=a.shift))
    if a.text_encoder.lower().endswith(".gguf"):
        clip = g.out(g.add("CLIPLoaderGGUF", clip_name=a.text_encoder, type="wan"))
    else:
        clip = g.out(g.add("CLIPLoader", clip_name=a.text_encoder, type="wan", device="cpu"))
    vae = g.out(g.add("VAELoader", vae_name=a.vae))
    clipv = g.out(g.add("CLIPVisionLoader", clip_name=a.clip_vision))
    pos = g.out(g.add("CLIPTextEncode", clip=clip, text=a.prompt))
    neg = g.out(g.add("CLIPTextEncode", clip=clip, text=a.negative))
    # reference image (already copied into ComfyUI/input by the caller)
    ref = g.out(g.add("LoadImage", image=a.ref_name))
    cv_out = g.out(g.add("CLIPVisionEncode", clip_vision=clipv, image=ref, crop="none"))
    # per-window sequences: pose, background, mask, face  (dirs hold the block's STRIDED frames)
    n = win["length"]
    s = win["seq_start"]
    pose = g.out(load_image_dir(g, win["pose_dir"], s, n))
    pose = g.out(g.add("ImageScale", image=pose, upscale_method="lanczos", width=a.width,
                       height=a.height, crop="disabled"))
    bg = g.out(load_image_dir(g, win["bg_dir"], s, n))
    bg = g.out(g.add("ImageScale", image=bg, upscale_method="lanczos", width=a.width,
                     height=a.height, crop="disabled"))
    mask_img = g.out(load_image_dir(g, win["mask_dir"], s, n))
    mask_img = g.out(g.add("ImageScale", image=mask_img, upscale_method="nearest-exact",
                           width=a.width, height=a.height, crop="disabled"))
    mask = g.out(g.add("ImageToMask", image=mask_img, channel="red"))
    # (the staged bg frames are already blacked out inside the grown+blockified mask)
    kw = dict(positive=pos, negative=neg, vae=vae, width=a.width, height=a.height, length=n,
              batch_size=1, clip_vision_output=cv_out, reference_image=ref, pose_video=pose,
              background_video=bg, character_mask=mask,
              continue_motion_max_frames=a.continue_frames, video_frame_offset=0)
    if win["face_dir"] is not None:
        face = g.out(load_image_dir(g, win["face_dir"], s, n))
        kw["face_video"] = face
    if prev_frames_dir is not None:
        cont = g.out(load_image_dir(g, prev_frames_dir, win["prev_count"] - a.continue_frames,
                                    a.continue_frames))
        kw["continue_motion"] = cont
    w2v = g.add("WanAnimateToVideo", **kw)
    latent = g.out(g.add("KSampler", model=model, positive=g.out(w2v, 0), negative=g.out(w2v, 1),
                         latent_image=g.out(w2v, 2), seed=a.seed + win["index"], steps=a.steps,
                         cfg=a.cfg, sampler_name=a.sampler, scheduler=a.scheduler, denoise=1.0))
    latent = g.out(g.add("TrimVideoLatent", samples=latent, trim_amount=g.out(w2v, 3)))
    if a.decode_tile > 0:
        # tiled decode: the full 832x464x33 decode ran out of device memory on the 16 GB iGPU
        img = g.out(g.add("VAEDecodeTiled", samples=latent, vae=vae, tile_size=a.decode_tile,
                          overlap=64, temporal_size=a.decode_frames, temporal_overlap=4))
    else:
        img = g.out(g.add("VAEDecode", samples=latent, vae=vae))
    img = g.out(g.add("ImageFromBatch", image=img, batch_index=g.out(w2v, 4), length=n))
    prefix = f"wan/{a.name}/w{win['index']:02d}/f"
    g.add("SaveImage", images=img, filename_prefix=prefix)
    return g.nodes, prefix


# ----------------------------------------------------------------------------- data prep
def stage_sequences(a, blocks: dict, log) -> dict:
    """Copy the strided per-frame inputs of the block into flat dirs under work/wan/<name>/in/."""
    blk = next(b for b in blocks["blocks"] if b["name"] == a.block)
    frames = list(range(blk["cut_start"], blk["cut_end"], a.stride))
    base = WORK / "wan" / a.name / "in"
    dirs = {"pose": base / "pose", "bg": base / "bg", "mask": base / "mask"}
    if a.face != "none":
        dirs["face"] = base / "face"
    if a.reuse_stage and all(d.exists() and len(list(d.glob("*.png"))) == len(frames) for d in dirs.values()):
        log(f"reusing staged inputs under {base}")
        return {"frames": frames, "dirs": dirs, "block": blk}
    if base.exists():
        shutil.rmtree(base)
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    # background frames come from the block source video (frame k of the src = cut_start + k)
    import cv2  # local import: the comfy venv may not have it, the project venv does
    cap = cv2.VideoCapture(str(ROOT / blk["src"]))
    src_frames = {}
    k = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        f = blk["cut_start"] + k
        if f in frames:
            src_frames[f] = fr
        k += 1
    cap.release()
    bg_size_hw = next(iter(src_frames.values())).shape[:2] if src_frames else (532, 960)
    if a.bg_frames_dir:
        # background = an earlier pass (e.g. the cat render): frame i of that dir = frames[i]
        bdir = Path(a.bg_frames_dir)
        for i, f in enumerate(frames):
            bp = bdir / f"{i:05d}.png"
            if not bp.exists():
                sys.exit(f"--bg-frames-dir: missing {bp} (expected {len(frames)} frames)")
            im = cv2.imread(str(bp))
            ref_hw = bg_size_hw
            if im.shape[:2] != ref_hw:
                im = cv2.resize(im, (ref_hw[1], ref_hw[0]), interpolation=cv2.INTER_LANCZOS4)
            src_frames[f] = im
        log(f"background frames from {bdir}")
    missing = [f for f in frames if f not in src_frames]
    if missing:
        sys.exit(f"block {a.block}: {len(missing)} source frames missing (e.g. {missing[:5]})")
    import numpy as np
    src_w = next(iter(src_frames.values())).shape[1]
    block_px = a.block_px if a.block_px else max(8, int(round(32 * src_w / a.width)))  # BlockifyMask(32) at model res
    grow_px = max(0, int(round(a.mask_grow * src_w / a.width)))
    for i, f in enumerate(frames):
        p = (Path(a.pose_dir) if a.pose_dir else WORK / f"pose_{a.side}") / f"{f:05d}.png"
        m = (Path(a.mask_dir) if a.mask_dir else WORK / f"mask_{a.side}") / f"{f:05d}.png"
        if not p.exists() or not m.exists():
            sys.exit(f"missing {p} or {m}; run pose_body.py first")
        shutil.copy(p, dirs["pose"] / f"{i:05d}.png")
        mask = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)
        if grow_px:
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow_px + 1,) * 2))
        # blockify: any block containing mask pixels becomes fully masked (gives the model room)
        h, w = mask.shape
        H, W = -(-h // block_px), -(-w // block_px)
        padded = np.zeros((H * block_px, W * block_px), np.uint8)
        padded[:h, :w] = mask
        blocks = padded.reshape(H, block_px, W, block_px).max(axis=(1, 3))
        mask = np.kron(blocks, np.ones((block_px, block_px), np.uint8))[:h, :w]
        # carve the hanging mic back out: foreground (isnet) pixels that pose_body.py kept out of
        # BOTH character masks inside the top-centre band are the mic + cable
        both = WORK / "mask_both" / f"{f:05d}.png"
        matte = WORK / "matte_isnet" / f"{f:05d}.png"
        if both.exists() and matte.exists():
            mb = cv2.imread(str(both), cv2.IMREAD_GRAYSCALE) > 127
            mt = cv2.imread(str(matte), cv2.IMREAD_GRAYSCALE) > 127
            band = np.zeros_like(mask, bool)
            band[:int(0.5 * h), int(0.42 * w):int(0.62 * w)] = True
            mic = mt & ~cv2.dilate(mb.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & band
            mic = cv2.dilate(mic.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            mask[mic] = 0
        cv2.imwrite(str(dirs["mask"] / f"{i:05d}.png"), mask * 255)
        bg = src_frames[f].copy()
        bg[mask > 0] = 0
        cv2.imwrite(str(dirs["bg"] / f"{i:05d}.png"), bg)
        if "face" in dirs:
            p = WORK / f"face_{a.side}" / f"{f:05d}.png"
            if not p.exists():
                sys.exit(f"missing {p}; run face_crops.py first (or --face none)")
            shutil.copy(p, dirs["face"] / f"{i:05d}.png")
    log(f"staged {len(frames)} frames (stride {a.stride}) of block {a.block} side {a.side} -> {base}")
    return {"frames": frames, "dirs": dirs, "block": blk}


def round_4k1(x: int) -> int:
    """Smallest 4k+1 >= x."""
    return max(1, ((x - 2) // 4 + 1) * 4 + 1) if x > 1 else 1


def plan_windows(n_frames: int, length: int, cont: int, probe: int | None) -> list[dict]:
    """Windows of `length` frames (4k+1). Window 0 renders frames 0..L-1. Every later window
    starts `cont` frames before the previous window's end: those frames are fed back as
    continue_motion and dropped from its output (trim_image), so it contributes L-cont new
    frames. The last window may extend past the block (the node holds the last pose frame);
    the surplus frames are discarded on copy."""
    wins = []
    start = 0
    idx = 0
    while start < n_frames:
        want = (probe if probe is not None else length)
        if idx == 0:
            seq_start = 0
            L = min(want, round_4k1(n_frames))
            L = ((L - 1) // 4) * 4 + 1
            new = min(L, n_frames)
        else:
            seq_start = start - cont
            L = min(want, round_4k1(cont + (n_frames - start)))
            L = ((L - 1) // 4) * 4 + 1
            new = min(L - cont, n_frames - start)
            if new <= 0:
                raise ValueError(f"window length {L} <= continue frames {cont}")
        wins.append({"index": idx, "seq_start": seq_start, "length": L,
                     "new_start": start, "new_count": new})
        start += new
        idx += 1
        if probe is not None:
            break
    return wins


def plan_range(s0: int, s1: int, length: int, cont: int, seeded: bool) -> list[dict]:
    """Windows covering strided frames [s0, s1). The first window starts fresh unless `seeded`
    (then it re-uses the `cont` frames before s0 as continue_motion)."""
    wins, start, first = [], s0, True
    while start < s1:
        rem = s1 - start
        if first and not seeded:
            L = ((min(length, round_4k1(rem)) - 1) // 4) * 4 + 1
            wins.append({"seq_start": start, "length": L, "new_start": start, "new_count": min(L, rem), "fresh": True})
        else:
            L = ((min(length, round_4k1(cont + rem)) - 1) // 4) * 4 + 1
            new = min(L - cont, rem)
            if new <= 0:
                raise ValueError(f"window length {L} <= continue frames {cont}")
            wins.append({"seq_start": start - cont, "length": L, "new_start": start, "new_count": new, "fresh": False})
        start += wins[-1]["new_count"]
        first = False
    return wins


def cut_indices(blk: dict, stride: int) -> list[int]:
    """Shot cuts inside the block as strided indices (first strided frame of each new shot)."""
    try:
        cuts = json.load(open(WORK / "tracks30.json"))["cuts"]
    except Exception:
        cuts = [104, 239, 461, 581, 688]
    out = []
    for c in cuts:
        if blk["cut_start"] < c < blk["cut_end"]:
            out.append(-(-(c - blk["cut_start"]) // stride))
    return out


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--block", required=True, help="block name in --blocks (A, B for the COLORS cut; E for the example)")
    ap.add_argument("--side", required=True, choices=["L", "R"])
    ap.add_argument("--ref", required=True, help="reference image (png/jpg)")
    ap.add_argument("--name", required=True, help="output name under work/wan/")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default=NEG_PROMPT)
    ap.add_argument("--negative-extra", default="", help="appended to the negative prompt")
    ap.add_argument("--blocks", default=str(WORK / "blocks.json"))
    ap.add_argument("--face", default="crops", choices=["crops", "none"])
    ap.add_argument("--stride", type=int, default=2, help="render every Nth cut frame (2 = 15 fps)")
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=464)
    ap.add_argument("--length", type=int, default=77, help="window length in frames (4k+1)")
    ap.add_argument("--continue-frames", type=int, default=5, help="overlap frames re-used as continue_motion (4k+1)")
    ap.add_argument("--probe", type=int, default=None, help="render only the first window of N frames and report")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--shift", type=float, default=8.0)
    ap.add_argument("--sampler", default="euler")
    ap.add_argument("--scheduler", default="simple")
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--mask-grow", type=int, default=8)
    ap.add_argument("--unet", default="Wan2.2-Animate-14B-Q4_K_M.gguf")
    ap.add_argument("--text-encoder", default="umt5-xxl-encoder-Q5_K_M.gguf")
    ap.add_argument("--clip-vision", default="clip_vision_h.safetensors")
    ap.add_argument("--vae", default="wan_2.1_vae.safetensors")
    ap.add_argument("--lora", action="append", default=None,
                    help="lora_name:strength (repeatable); default lightx2v distill 1.0 + relight 1.0")
    ap.add_argument("--comfy", default="http://127.0.0.1:8188")
    ap.add_argument("--comfy-dir", default=str(ROOT / "vendor" / "ComfyUI"))
    ap.add_argument("--out-fps", type=float, default=None, help="fps of the stitched mp4 (default 30/stride)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pose-dir", default=None, help="skeleton PNG dir (default work/pose_<side>; e.g. work/pose_R_cat)")
    ap.add_argument("--bg-frames-dir", default=None, help="use these PNGs (0-based, one per strided frame) as the background, e.g. work/wan/cat_A/frames")
    ap.add_argument("--decode-tile", type=int, default=512, help="VAEDecodeTiled tile px (0 = plain VAEDecode)")
    ap.add_argument("--decode-frames", type=int, default=64,
                    help="frames decoded at a time (tiled decode); must cover the whole window: the Wan VAE is causal, "
                         "so each temporal chunk restarts its cache and flashes brighter (16 gave a pulse every 12 frames)")
    ap.add_argument("--mask-dir", default=None, help="character-mask PNG dir (default <WORK>/mask_<side>; e.g. mask_Lfur)")
    ap.add_argument("--block-px", type=int, default=0, help="mask blockify size in source px (0 = 32 at model res)")
    ap.add_argument("--range", default=None, help="render only strided frames START:END of the block (outputs 0-based)")
    ap.add_argument("--seed-frames", default=None, help="dir of already-rendered strided frames: continue from the ones before --range START")
    ap.add_argument("--no-fresh-at-cuts", action="store_true", help="carry continue_motion across shot cuts (old behaviour)")
    ap.add_argument("--resume", action="store_true", help="skip windows already rendered by an earlier run")
    ap.add_argument("--reuse-stage", action="store_true", help="keep existing staged inputs if complete")
    a = ap.parse_args()
    if a.negative_extra:
        a.negative = a.negative + ", " + a.negative_extra

    out_dir = WORK / "wan" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = open(out_dir / "run.log", "a", encoding="utf-8")

    def log(msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    loras = [("lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", 1.0),
             ("WanAnimate_relight_lora_fp16.safetensors", 1.0)]
    if a.lora:
        loras = []
        for s in a.lora:
            name, _, st = s.rpartition(":")
            loras.append((name, float(st)))
    models = {"loras": loras}

    blocks = json.load(open(a.blocks))
    staged = stage_sequences(a, blocks, log)
    n = len(staged["frames"])
    if a.probe is not None:
        wins = plan_windows(n, a.length, a.continue_frames, a.probe)
        for w in wins:
            w["fresh"] = True
        r0 = 0
    else:
        r0, r1 = 0, n
        if a.range:
            r0, r1 = (int(v) for v in a.range.split(":"))
            r1 = min(r1, n)
        cuts = [] if a.no_fresh_at_cuts else [c for c in cut_indices(staged["block"], a.stride) if r0 < c < r1]
        bounds = [r0] + cuts + [r1]
        wins = []
        for si in range(len(bounds) - 1):
            wins += plan_range(bounds[si], bounds[si + 1], a.length, a.continue_frames,
                               seeded=(si == 0 and a.seed_frames is not None))
        for i, w in enumerate(wins):
            w["index"] = i
        if cuts:
            log(f"fresh windows at shot cuts (strided idx): {cuts}")
        n = r1 - r0
    log(f"{n} frames -> {len(wins)} window(s): " + ", ".join(
        f"w{w['index']}[seq {w['seq_start']}+{w['length']}, new {w['new_count']}]" for w in wins))

    comfy_dir = Path(a.comfy_dir)
    ref_src = Path(a.ref)
    a.ref_name = f"{a.name}_ref{ref_src.suffix.lower()}"
    if not a.dry_run:
        (comfy_dir / "input").mkdir(parents=True, exist_ok=True)
        shutil.copy(ref_src, comfy_dir / "input" / a.ref_name)

    frames_dir = out_dir / "frames"
    if not a.dry_run:
        if frames_dir.exists() and not a.resume:
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)

    client = None if a.dry_run else Comfy(a.comfy)
    if client:
        st = client.stats()
        devs = [d.get("name") for d in st.get("devices", [])]
        log(f"ComfyUI {st.get('system', {}).get('comfyui_version', '?')} devices {devs}")

    prev_dir = Path(a.seed_frames).resolve() if a.seed_frames else None
    prev_count = r0 if a.seed_frames else 0
    total_new = 0
    t_all = time.time()
    for w in wins:
        if w.get("fresh"):
            prev_dir, prev_count = None, 0
        w.update(pose_dir=staged["dirs"]["pose"], bg_dir=staged["dirs"]["bg"],
                 mask_dir=staged["dirs"]["mask"], face_dir=staged["dirs"].get("face"),
                 prev_count=prev_count)
        prompt, prefix = build_window(a, w, models, prev_dir)
        if a.dry_run:
            print(json.dumps(prompt, indent=1, ensure_ascii=False))
            if w["index"] == 0:
                return 0
            continue
        win_out = comfy_dir / "output" / f"wan/{a.name}/w{w['index']:02d}"
        done = sorted(win_out.glob("*.png")) if win_out.exists() else []
        if a.resume and len(done) >= w["new_count"]:
            for i, p in enumerate(done[:w["new_count"]]):
                shutil.copy(p, frames_dir / f"{total_new + i:05d}.png")
            total_new += w["new_count"]
            prev_dir, prev_count = win_out, len(done)
            log(f"window {w['index']}: already rendered ({len(done)} frames), skipped")
            continue
        if win_out.exists():
            shutil.rmtree(win_out)
        log(f"window {w['index']}: seq {w['seq_start']}..{w['seq_start'] + w['length'] - 1} "
            f"({w['length']} frames, {a.width}x{a.height}, steps {a.steps})")
        t0 = time.time()
        pid = client.queue(prompt)
        client.wait(pid, log=log)
        dt = time.time() - t0
        pngs = sorted(win_out.glob("*.png")) if win_out.exists() else []
        if not pngs:
            log(f"window {w['index']}: no output frames under {win_out}")
            return 2
        log(f"window {w['index']}: {len(pngs)} frames in {dt:.0f} s = {dt / max(1, len(pngs)):.1f} s/frame "
            f"({dt / w['length']:.1f} s per rendered frame incl. overlap)")
        # ImageFromBatch dropped the continued frames: the saved frames are the NEW ones
        keep = pngs[:w["new_count"]]
        if len(pngs) < w["new_count"]:
            log(f"WARNING: window {w['index']} returned {len(pngs)} frames, expected >= {w['new_count']}")
        for i, p in enumerate(keep):
            shutil.copy(p, frames_dir / f"{total_new + i:05d}.png")
        total_new += len(keep)
        prev_dir, prev_count = win_out, len(pngs)
        if a.probe is not None:
            proj = dt / len(pngs) * n
            log(f"PROBE: {dt / len(pngs):.1f} s/frame -> projected {proj / 3600:.1f} h for the {n}-frame block")
            break

    if a.dry_run:
        return 0
    log(f"total {total_new} frames in {(time.time() - t_all) / 60:.1f} min")
    if total_new < n and a.probe is None:
        log(f"WARNING: expected {n} frames, got {total_new}")
    fps = a.out_fps or 30.0 / a.stride
    mp4 = out_dir.with_suffix(".mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", f"{fps:g}",
                    "-i", str(frames_dir / "%05d.png"), "-c:v", "libx264", "-crf", "14",
                    "-pix_fmt", "yuv420p", "-r", f"{fps:g}", str(mp4)], check=True)
    log(f"wrote {mp4} @ {fps:g} fps")
    json.dump({"block": a.block, "side": a.side, "stride": a.stride, "range_start": r0, "frames": staged["frames"][:total_new],
               "n": total_new, "width": a.width, "height": a.height, "length": a.length,
               "windows": [{k: v for k, v in w.items() if not isinstance(v, Path)} for w in wins],
               "ref": a.ref, "prompt": a.prompt, "steps": a.steps, "seed": a.seed},
              open(out_dir / "render.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""
wan_animate2.py -- drive a headless ComfyUI to run Wan2.2-Animate-2 (14B, distilled GGUF) over one
block of the cut, one side at a time, in chained windows.

Animate-2 differs from the Wan2.2-Animate lane in wan_animate.py: it takes the DRIVING VIDEO itself
(no skeleton, no character mask, no background plate) and renders a whole new frame of the reference
character performing that motion against a prompted background. There is no in-footage replacement
mode, so each side is rendered on its own and composited later.

Inputs per side (indexed by CUT frame number like the Animate lane):
  <block src video>     the frames to drive from, cropped to the side's region (--crop) and strided
  refs/<ref>.png        reference image; it must roughly match the driving clip's opening pose

The block is rendered in windows of --length frames (4k+1), one ComfyUI prompt each; window i>0
receives the last frame of window i-1 as `continue_motion` (Animate-2 keeps exactly one frame) and
its driving frames start one frame earlier. Output frames are written as PNG sequences under
work/wan/<name>/frames/ and stitched to work/wan/<name>.mp4 at --out-fps.

Model loading: a GGUF of Animate-2 carries no architecture metadata, so the stock GGUF loader
silently loads it as plain Wan 2.1 I2V (the driving video is then ignored). The Rebels loader node
(RebelsGGUFUnetLoaderMeta, github RealRebelAI/Rebels_w3a8_Loader) supplies model_type=animate2;
the ComfyUI console must print "model class=WAN_Animate2" on load.

Usage
  python wan_animate2.py --block E --side L --ref refs/cat_body_base_hq.png --prompt "..." --name cat_A2
  ... --probe 33            # only the first window, report s/frame, exit
  ... --dry-run             # print the prompt JSON and exit (no server needed)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from wan_animate import (NEG_PROMPT, Comfy, Graph, cut_indices, load_image_dir, plan_range,
                         plan_windows)

ROOT = Path(__file__).resolve().parent
WORK = Path(os.environ["VG_WORK"]).resolve() if os.environ.get("VG_WORK") else ROOT / "work"

# Driving-video crop per side as fractions of the source frame (x0, y0, x1, y1). Measured on the
# example clip (960x720): the left performer covers x 0..527, the right one x 375..959.
SIDE_CROP = {"L": (0.0, 0.0, 0.567, 1.0), "R": (0.433, 0.0, 1.0, 1.0)}
CONTINUE_FRAMES = 1  # WanAnimate2ToVideo.CONTINUE_MOTION_FRAMES
POSE_PROMPT = "A reference video of a performer moving and gesturing to music in a studio."


# ----------------------------------------------------------------------------- graph
def build_window(a, win: dict, models: dict, prev_frames_dir: Path | None) -> tuple[dict, str]:
    """One window = one prompt. Returns (prompt, save_prefix)."""
    g = Graph()
    if a.unet.lower().endswith(".gguf"):
        if a.model_type:
            model = g.out(g.add("RebelsGGUFUnetLoaderMeta", unet_name=a.unet, model_type=a.model_type))
        else:
            model = g.out(g.add("UnetLoaderGGUF", unet_name=a.unet))
    else:
        model = g.out(g.add("UNETLoader", unet_name=a.unet, weight_dtype="default"))
    for lora, strength in models["loras"]:
        model = g.out(g.add("LoraLoaderModelOnly", model=model, lora_name=lora, strength_model=strength))
    if a.pose_cache != "none":
        # caches the pose branch per block so it runs once instead of on every step (~2x faster)
        model = g.out(g.add("WanAnimate2Cache", model=model, device=a.pose_cache, dtype=a.pose_cache_dtype))
    model = g.out(g.add("ModelSamplingSD3", model=model, shift=a.shift))
    if a.text_encoder.lower().endswith(".gguf"):
        clip = g.out(g.add("CLIPLoaderGGUF", clip_name=a.text_encoder, type="wan"))
    else:
        clip = g.out(g.add("CLIPLoader", clip_name=a.text_encoder, type="wan", device="default"))
    vae = g.out(g.add("VAELoader", vae_name=a.vae))
    clipv = g.out(g.add("CLIPVisionLoader", clip_name=a.clip_vision))
    pos = g.out(g.add("CLIPTextEncode", clip=clip, text=a.prompt))
    neg = g.out(g.add("CLIPTextEncode", clip=clip, text=a.negative))
    pos_pose = g.out(g.add("CLIPTextEncode", clip=clip, text=a.pose_prompt))
    # reference image (already copied into ComfyUI/input by the caller)
    ref = g.out(g.add("LoadImage", image=a.ref_name))
    cv_ref = g.out(g.add("CLIPVisionEncode", clip_vision=clipv, image=ref, crop="none"))
    # driving frames of this window; the node itself fits them to width x height (area, centre crop)
    n = win["length"]
    drive = g.out(load_image_dir(g, win["drive_dir"], win["seq_start"], n))
    first = g.out(g.add("ImageFromBatch", image=drive, batch_index=0, length=1))
    cv_pose = g.out(g.add("CLIPVisionEncode", clip_vision=clipv, image=first, crop="none"))
    kw = dict(positive=pos, negative=neg, vae=vae, width=a.width, height=a.height, length=n, batch_size=1,
              reference_image=ref, pose_video=drive, clip_vision_output=cv_ref, positive_pose=pos_pose,
              clip_vision_output_pose=cv_pose, video_frame_offset=0, pose_strength=a.pose_strength,
              pose_start_percent=0.0, pose_end_percent=a.pose_end, reference_image_strength=a.ref_strength)
    if prev_frames_dir is not None:
        kw["continue_motion"] = g.out(load_image_dir(g, prev_frames_dir, win["prev_count"] - CONTINUE_FRAMES,
                                                     CONTINUE_FRAMES))
    a2v = g.add("WanAnimate2ToVideo", **kw)
    latent = g.out(g.add("KSampler", model=model, positive=g.out(a2v, 0), negative=g.out(a2v, 1),
                         latent_image=g.out(a2v, 2), seed=a.seed + win["index"], steps=a.steps,
                         cfg=a.cfg, sampler_name=a.sampler, scheduler=a.scheduler, denoise=1.0))
    latent = g.out(g.add("TrimVideoLatent", samples=latent, trim_amount=g.out(a2v, 3)))
    if a.decode_tile > 0:
        img = g.out(g.add("VAEDecodeTiled", samples=latent, vae=vae, tile_size=a.decode_tile,
                          overlap=64, temporal_size=a.decode_frames, temporal_overlap=4))
    else:
        img = g.out(g.add("VAEDecode", samples=latent, vae=vae))
    # trim_image (output 4) drops the continued frame; the saved frames are the NEW ones
    img = g.out(g.add("ImageFromBatch", image=img, batch_index=g.out(a2v, 4), length=n))
    prefix = f"wan/{a.name}/w{win['index']:02d}/f"
    g.add("SaveImage", images=img, filename_prefix=prefix)
    return g.nodes, prefix


# ----------------------------------------------------------------------------- data prep
def parse_crop(spec: str | None, side: str, w: int, h: int) -> tuple[int, int, int, int]:
    """--crop x0:y0:x1:y1 in source pixels, else the side's default fraction box snapped to 16 px."""
    if spec:
        x0, y0, x1, y1 = (int(v) for v in spec.split(":"))
    else:
        fx0, fy0, fx1, fy1 = SIDE_CROP[side]
        x0, x1 = int(round(fx0 * w / 16)) * 16, int(round(fx1 * w / 16)) * 16
        y0, y1 = int(round(fy0 * h / 16)) * 16, int(round(fy1 * h / 16)) * 16
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 16 or y1 - y0 < 16:
        sys.exit(f"bad crop {x0}:{y0}:{x1}:{y1} for a {w}x{h} source")
    return x0, y0, x1, y1


def stage_sequences(a, blocks: dict, log) -> dict:
    """Crop + stride the block's source frames into work/wan/<name>/in/drive/."""
    blk = next(b for b in blocks["blocks"] if b["name"] == a.block)
    frames = list(range(blk["cut_start"], blk["cut_end"], a.stride))
    base = WORK / "wan" / a.name / "in"
    drive = base / "drive"
    if a.reuse_stage and drive.exists() and len(list(drive.glob("*.png"))) == len(frames):
        log(f"reusing staged driving frames under {drive}")
        return {"frames": frames, "drive_dir": drive, "block": blk}
    if base.exists():
        shutil.rmtree(base)
    drive.mkdir(parents=True)
    import cv2  # local import: the comfy venv may not have it, the project venv does
    cap = cv2.VideoCapture(str(ROOT / blk["src"]))
    if not cap.isOpened():
        sys.exit(f"cannot open {ROOT / blk['src']}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x0, y0, x1, y1 = parse_crop(a.crop, a.side, w, h)
    want = {f: i for i, f in enumerate(frames)}
    k, got = 0, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        f = blk["cut_start"] + k
        if f in want:
            cv2.imwrite(str(drive / f"{want[f]:05d}.png"), fr[y0:y1, x0:x1])
            got += 1
        k += 1
    cap.release()
    if got < len(frames):
        sys.exit(f"block {a.block}: only {got} of {len(frames)} source frames read")
    log(f"staged {got} driving frames (stride {a.stride}, crop {x0}:{y0}:{x1}:{y1} of {w}x{h}) "
        f"of block {a.block} side {a.side} -> {drive}")
    return {"frames": frames, "drive_dir": drive, "block": blk, "crop": [x0, y0, x1, y1]}


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--block", required=True, help="block name in --blocks (E for the example)")
    ap.add_argument("--side", required=True, choices=["L", "R"])
    ap.add_argument("--ref", required=True, help="reference image (png/jpg)")
    ap.add_argument("--name", required=True, help="output name under work/wan/")
    ap.add_argument("--prompt", required=True, help="Animate-2 style: 'Character Description: ... Background description: ...'")
    ap.add_argument("--pose-prompt", default=POSE_PROMPT, help="prompt for the pose branch (describes the motion)")
    ap.add_argument("--negative", default=NEG_PROMPT)
    ap.add_argument("--negative-extra", default="", help="appended to the negative prompt")
    ap.add_argument("--blocks", default=str(WORK / "blocks.json"))
    ap.add_argument("--crop", default=None, help="driving crop x0:y0:x1:y1 in source px (default: the side's region)")
    ap.add_argument("--stride", type=int, default=2, help="render every Nth cut frame (2 = 12 fps of a 24 fps cut)")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--length", type=int, default=33, help="window length in frames (4k+1)")
    ap.add_argument("--probe", type=int, default=None, help="render only the first window of N frames and report")
    ap.add_argument("--steps", type=int, default=10, help="distilled weights: 10 steps, cfg 1, lcm")
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--sampler", default="lcm")
    ap.add_argument("--scheduler", default="simple")
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--unet", default="Wan-Animate-2-14B-TURBO-Q4_K_M.gguf")
    ap.add_argument("--model-type", default="animate2", help="metadata the Rebels GGUF loader injects ('' = stock loader)")
    ap.add_argument("--text-encoder", default="umt5-xxl-encoder-Q5_K_M.gguf")
    ap.add_argument("--clip-vision", default="clip_vision_h.safetensors")
    ap.add_argument("--vae", default="wan_2.1_vae.safetensors")
    ap.add_argument("--lora", action="append", default=None, help="lora_name:strength (repeatable); default none")
    ap.add_argument("--pose-cache", default="cpu", choices=["cpu", "gpu", "none"],
                    help="WanAnimate2Cache device (cpu = system RAM, safe; gpu only if it fits beside the model)")
    ap.add_argument("--pose-cache-dtype", default="default", choices=["default", "int8", "int4"])
    ap.add_argument("--pose-strength", type=float, default=1.0)
    ap.add_argument("--pose-end", type=float, default=1.0, help="pose_end_percent (e.g. 0.7 loosens fine detail)")
    ap.add_argument("--ref-strength", type=float, default=1.0, help="reference_image_strength (>1 tightens identity)")
    ap.add_argument("--comfy", default="http://127.0.0.1:8188")
    ap.add_argument("--comfy-dir", default=str(ROOT / "vendor" / "ComfyUI"))
    ap.add_argument("--out-fps", type=float, default=None, help="fps of the stitched mp4 (default 24/stride)")
    ap.add_argument("--decode-tile", type=int, default=512, help="VAEDecodeTiled tile px (0 = plain VAEDecode)")
    ap.add_argument("--decode-frames", type=int, default=64, help="frames per temporal decode chunk (cover the window)")
    ap.add_argument("--range", default=None, help="render only strided frames START:END of the block (outputs 0-based)")
    ap.add_argument("--seed-frames", default=None, help="dir of already-rendered strided frames: continue from the one before --range START")
    ap.add_argument("--no-fresh-at-cuts", action="store_true", help="carry continue_motion across shot cuts")
    ap.add_argument("--resume", action="store_true", help="skip windows already rendered by an earlier run")
    ap.add_argument("--reuse-stage", action="store_true", help="keep existing staged inputs if complete")
    ap.add_argument("--dry-run", action="store_true")
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

    loras = []
    for s in a.lora or []:
        name, _, st = s.rpartition(":")
        loras.append((name, float(st)))
    models = {"loras": loras}

    blocks = json.load(open(a.blocks))
    staged = stage_sequences(a, blocks, log)
    n = len(staged["frames"])
    if a.probe is not None:
        wins = plan_windows(n, a.length, CONTINUE_FRAMES, a.probe)
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
            wins += plan_range(bounds[si], bounds[si + 1], a.length, CONTINUE_FRAMES,
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
        for cls in ("WanAnimate2ToVideo", "RebelsGGUFUnetLoaderMeta" if a.model_type else "UnetLoaderGGUF"):
            try:
                client.object_info(cls)
            except Exception as e:  # noqa: BLE001
                log(f"ComfyUI has no node {cls}: {e}")
                return 3

    prev_dir = Path(a.seed_frames).resolve() if a.seed_frames else None
    prev_count = r0 if a.seed_frames else 0
    total_new = 0
    t_all = time.time()
    for w in wins:
        if w.get("fresh"):
            prev_dir, prev_count = None, 0
        w.update(drive_dir=staged["drive_dir"], prev_count=prev_count)
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
    fps = a.out_fps or float(blocks.get("fps", 24)) / a.stride
    mp4 = out_dir.with_suffix(".mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", f"{fps:g}",
                    "-i", str(frames_dir / "%05d.png"), "-c:v", "libx264", "-crf", "14",
                    "-pix_fmt", "yuv420p", "-r", f"{fps:g}", str(mp4)], check=True)
    log(f"wrote {mp4} @ {fps:g} fps")
    json.dump({"engine": "animate2", "block": a.block, "side": a.side, "stride": a.stride, "range_start": r0,
               "frames": staged["frames"][:total_new], "n": total_new, "width": a.width, "height": a.height,
               "length": a.length, "crop": staged.get("crop"), "unet": a.unet, "model_type": a.model_type,
               "windows": [{k: v for k, v in w.items() if not isinstance(v, Path)} for w in wins],
               "ref": a.ref, "prompt": a.prompt, "pose_prompt": a.pose_prompt, "steps": a.steps,
               "sampler": a.sampler, "shift": a.shift, "seed": a.seed},
              open(out_dir / "render.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""
run_cloud.py -- the Wan2.2-Animate replacement render of wan_animate.py, but on a Comfy developer
platform deployment instead of a local ComfyUI: the laptop does the cheap per-frame prep (pose,
masks, face crops: pose_body.py / face_crops.py), the deployment does the 14B render.

Same inputs and output layout as wan_animate.py (same --block/--side/--ref/--prompt/--length/...
flags; frames land in <WORK>/wan/<name>/frames/ and <WORK>/wan/<name>.mp4), so finish.py, to30.py
and lip_pass.py run unchanged afterwards.

How a window travels
  * the staged PNG sequences of the block (pose / bg / mask / face) are encoded once into one
    near-lossless H.264 clip each under <WORK>/wan/<name>/cloud/; the per-window graph loads them
    with VHS_LoadVideo (skip_first_frames / frame_load_cap select the window)
  * the reference plate is a LoadImage of a file in the same folder
  * window i>0 loads the last --continue-frames frames of window i-1's returned clip as
    continue_motion
  * the output is one VHS_VideoCombine mp4 per window; it is decoded back to PNG frames locally
  * each window is one `comfy deploy run` (the CLI uploads the files the graph names, deduped by
    hash, waits, and downloads the outputs)

Usage
  python run_cloud.py --block E --side L --ref refs/cat_body_base_hq.png --name cat_cloud \
      --prompt "..." --width 768 --height 576 --length 33 --stride 2 --probe 33
  ... --dry-run          # write the first window's API-format graph and exit (no upload, no job)
Needs: comfy-cli signed in (`comfy cloud login`), comfy-build.yaml next to this file with a
deployment in `ready` state (`comfy deploy status comfy-build.yaml`).
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

import wan_animate as wa

ROOT = Path(__file__).resolve().parent
WORK = wa.WORK
SPEC = ROOT / "comfy-build.yaml"


# ----------------------------------------------------------------------------- graph
def load_video(g: wa.Graph, path: Path, start: int, count: int) -> str:
    return g.add("VHS_LoadVideo", video=str(path), force_rate=0, custom_width=0, custom_height=0,
                 frame_load_cap=count, skip_first_frames=start, select_every_nth=1)


def build_window_cloud(a, win: dict, models: dict, clips: dict, prev_clip: Path | None) -> dict:
    """One window = one API-format prompt whose inputs are files (uploaded by `comfy deploy run`)."""
    g = wa.Graph()
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
        clip = g.out(g.add("CLIPLoader", clip_name=a.text_encoder, type="wan", device="default"))
    vae = g.out(g.add("VAELoader", vae_name=a.vae))
    clipv = g.out(g.add("CLIPVisionLoader", clip_name=a.clip_vision))
    pos = g.out(g.add("CLIPTextEncode", clip=clip, text=a.prompt))
    neg = g.out(g.add("CLIPTextEncode", clip=clip, text=a.negative))
    ref = g.out(g.add("LoadImage", image=str(clips["ref"])))
    cv_out = g.out(g.add("CLIPVisionEncode", clip_vision=clipv, image=ref, crop="none"))
    n, s = win["length"], win["seq_start"]
    pose = g.out(load_video(g, clips["pose"], s, n))
    pose = g.out(g.add("ImageScale", image=pose, upscale_method="lanczos", width=a.width,
                       height=a.height, crop="disabled"))
    bg = g.out(load_video(g, clips["bg"], s, n))
    bg = g.out(g.add("ImageScale", image=bg, upscale_method="lanczos", width=a.width,
                     height=a.height, crop="disabled"))
    mask_img = g.out(load_video(g, clips["mask"], s, n))
    mask_img = g.out(g.add("ImageScale", image=mask_img, upscale_method="nearest-exact",
                           width=a.width, height=a.height, crop="disabled"))
    mask = g.out(g.add("ImageToMask", image=mask_img, channel="red"))
    kw = dict(positive=pos, negative=neg, vae=vae, width=a.width, height=a.height, length=n,
              batch_size=1, clip_vision_output=cv_out, reference_image=ref, pose_video=pose,
              background_video=bg, character_mask=mask,
              continue_motion_max_frames=a.continue_frames, video_frame_offset=0)
    if "face" in clips:
        kw["face_video"] = g.out(load_video(g, clips["face"], s, n))
    if prev_clip is not None:
        kw["continue_motion"] = g.out(load_video(g, prev_clip, win["prev_count"] - a.continue_frames,
                                                 a.continue_frames))
    w2v = g.add("WanAnimateToVideo", **kw)
    latent = g.out(g.add("KSampler", model=model, positive=g.out(w2v, 0), negative=g.out(w2v, 1),
                         latent_image=g.out(w2v, 2), seed=a.seed + win["index"], steps=a.steps,
                         cfg=a.cfg, sampler_name=a.sampler, scheduler=a.scheduler, denoise=1.0))
    latent = g.out(g.add("TrimVideoLatent", samples=latent, trim_amount=g.out(w2v, 3)))
    if a.decode_tile > 0:
        img = g.out(g.add("VAEDecodeTiled", samples=latent, vae=vae, tile_size=a.decode_tile,
                          overlap=64, temporal_size=a.decode_frames, temporal_overlap=4))
    else:
        img = g.out(g.add("VAEDecode", samples=latent, vae=vae))
    img = g.out(g.add("ImageFromBatch", image=img, batch_index=g.out(w2v, 4), length=n))
    g.add("VHS_VideoCombine", images=img, frame_rate=a.clip_fps, loop_count=0,
          filename_prefix=f"{a.name}_w{win['index']:02d}", format="video/h264-mp4",
          pix_fmt="yuv420p", crf=a.clip_crf, save_metadata=False, trim_to_audio=False,
          pingpong=False, save_output=True)
    return g.nodes


# ----------------------------------------------------------------------------- clips
def encode_dir(src: Path, dst: Path, fps: float, crf: int, log) -> int:
    n = len(list(src.glob("*.png")))
    if dst.exists():
        probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                                "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(dst)],
                               capture_output=True, text=True)
        if probe.stdout.strip() == str(n):
            log(f"clip {dst.name}: {n} frames, reused")
            return n
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", f"{fps:g}", "-i", str(src / "%05d.png"),
                    "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
                    "-r", f"{fps:g}", str(dst)], check=True)
    log(f"clip {dst.name}: {n} frames, {dst.stat().st_size / 1e6:.1f} MB")
    return n


def decode_clip(src: Path, dst: Path) -> list[Path]:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-start_number", "0",
                    str(dst / "%05d.png")], check=True)
    return sorted(dst.glob("*.png"))


# ----------------------------------------------------------------------------- deploy run
def deploy_run(workflow: Path, asset_root: Path, out_dir: Path, timeout: int, log) -> list[Path]:
    cmd = ["comfy", "--json", "deploy", "run", str(SPEC), "--workflow", str(workflow),
           "--asset-root", str(asset_root), "--output-dir", str(out_dir), "--timeout", str(timeout)]
    log("$ " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (out_dir / "run.stderr.txt").write_text(r.stderr or "", encoding="utf-8")
    try:
        env = json.loads(r.stdout)
    except json.JSONDecodeError:
        log(f"comfy deploy run returned no envelope (exit {r.returncode}); stdout:\n{r.stdout[:2000]}")
        raise SystemExit(2)
    (out_dir / "run.envelope.json").write_text(json.dumps(env, indent=1), encoding="utf-8")
    if not env.get("ok"):
        err = env.get("error") or {}
        log(f"job failed: {err.get('code')}: {err.get('message')}")
        if err.get("hint"):
            log(f"hint: {err['hint']}")
        raise SystemExit(2)
    d = env["data"]
    assets = d.get("assets") or {}
    for f in assets.get("files", []) or []:
        log(f"asset {f.get('name') or f.get('path')}: {'uploaded' if f.get('uploaded') else 'held'}")
    job = d.get("job") or {}
    log(f"job {job.get('id')} {job.get('status')} release v{job.get('releaseVersion', '?')}")
    outs = []
    for o in d.get("outputs", []) or []:
        p = Path(o["path"]) if o.get("path") else None
        log(f"output node {o.get('node_id')}: {o.get('name')} ({o.get('type')}) -> {p}")
        if p and p.suffix.lower() in (".mp4", ".mkv", ".webm"):
            outs.append(p)
    if not outs:
        outs = sorted(out_dir.rglob("*.mp4"))
    return outs


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--block", required=True)
    ap.add_argument("--side", required=True, choices=["L", "R"])
    ap.add_argument("--ref", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default=wa.NEG_PROMPT)
    ap.add_argument("--negative-extra", default="")
    ap.add_argument("--blocks", default=str(WORK / "blocks.json"))
    ap.add_argument("--face", default="crops", choices=["crops", "none"])
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=464)
    ap.add_argument("--length", type=int, default=77)
    ap.add_argument("--continue-frames", type=int, default=5)
    ap.add_argument("--probe", type=int, default=None)
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
    ap.add_argument("--lora", action="append", default=None)
    ap.add_argument("--out-fps", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pose-dir", default=None)
    ap.add_argument("--bg-frames-dir", default=None)
    ap.add_argument("--decode-tile", type=int, default=512)
    ap.add_argument("--decode-frames", type=int, default=64)
    ap.add_argument("--mask-dir", default=None)
    ap.add_argument("--block-px", type=int, default=0)
    ap.add_argument("--range", default=None)
    ap.add_argument("--seed-frames", default=None)
    ap.add_argument("--no-fresh-at-cuts", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--reuse-stage", action="store_true")
    ap.add_argument("--clip-crf", type=int, default=4, help="H.264 crf of the input / output clips")
    ap.add_argument("--job-timeout", type=int, default=3600, help="seconds to wait per window")
    a = ap.parse_args()
    if a.negative_extra:
        a.negative = a.negative + ", " + a.negative_extra
    a.clip_fps = a.out_fps or 30.0 / a.stride

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
        loras = [(s.rpartition(":")[0], float(s.rpartition(":")[2])) for s in a.lora]
    models = {"loras": loras}

    blocks = json.load(open(a.blocks))
    staged = wa.stage_sequences(a, blocks, log)
    n = len(staged["frames"])
    if a.probe is not None:
        wins = wa.plan_windows(n, a.length, a.continue_frames, a.probe)
        for w in wins:
            w["fresh"] = True
        r0 = 0
    else:
        r0, r1 = 0, n
        if a.range:
            r0, r1 = (int(v) for v in a.range.split(":"))
            r1 = min(r1, n)
        cuts = [] if a.no_fresh_at_cuts else [c for c in wa.cut_indices(staged["block"], a.stride) if r0 < c < r1]
        bounds = [r0] + cuts + [r1]
        wins = []
        for si in range(len(bounds) - 1):
            wins += wa.plan_range(bounds[si], bounds[si + 1], a.length, a.continue_frames,
                                  seeded=(si == 0 and a.seed_frames is not None))
        for i, w in enumerate(wins):
            w["index"] = i
        n = r1 - r0
    log(f"{n} frames -> {len(wins)} window(s): " + ", ".join(
        f"w{w['index']}[seq {w['seq_start']}+{w['length']}, new {w['new_count']}]" for w in wins))

    # the clips the graph names: one per staged sequence, plus the reference plate
    cloud = out_dir / "cloud"
    cloud.mkdir(exist_ok=True)
    clips = {}
    for key in ("pose", "bg", "mask", "face"):
        d = staged["dirs"].get(key)
        if d is None:
            continue
        clips[key] = cloud / f"{key}.mp4"
        encode_dir(Path(d), clips[key], a.clip_fps, a.clip_crf, log)
    ref_src = Path(a.ref)
    clips["ref"] = cloud / f"ref{ref_src.suffix.lower()}"
    shutil.copy(ref_src, clips["ref"])

    frames_dir = out_dir / "frames"
    if not a.dry_run:
        if frames_dir.exists() and not a.resume:
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)

    prev_clip: Path | None = None
    prev_count = 0
    if a.seed_frames:
        prev_clip = cloud / "seed.mp4"
        prev_count = encode_dir(Path(a.seed_frames).resolve(), prev_clip, a.clip_fps, a.clip_crf, log)
    total_new = 0
    t_all = time.time()
    for w in wins:
        if w.get("fresh"):
            prev_clip, prev_count = None, 0
        w["prev_count"] = prev_count
        prompt = build_window_cloud(a, w, models, clips, prev_clip)
        win_dir = cloud / f"w{w['index']:02d}"
        win_dir.mkdir(exist_ok=True)
        wf = win_dir / "workflow.api.json"
        wf.write_text(json.dumps(prompt, indent=1, ensure_ascii=False), encoding="utf-8")
        if a.dry_run:
            print(json.dumps(prompt, indent=1, ensure_ascii=False))
            log(f"dry run: wrote {wf}")
            return 0
        done_clip = win_dir / "out.mp4"
        if a.resume and done_clip.exists():
            pngs = decode_clip(done_clip, win_dir / "frames")
            if len(pngs) >= w["new_count"]:
                for i, p in enumerate(pngs[:w["new_count"]]):
                    shutil.copy(p, frames_dir / f"{total_new + i:05d}.png")
                total_new += w["new_count"]
                prev_clip, prev_count = done_clip, len(pngs)
                log(f"window {w['index']}: already rendered ({len(pngs)} frames), skipped")
                continue
        log(f"window {w['index']}: seq {w['seq_start']}..{w['seq_start'] + w['length'] - 1} "
            f"({w['length']} frames, {a.width}x{a.height}, steps {a.steps}) -> deployment")
        t0 = time.time()
        outs = deploy_run(wf, cloud, win_dir / "outputs", a.job_timeout, log)
        dt = time.time() - t0
        if not outs:
            log(f"window {w['index']}: the job returned no video output")
            return 2
        shutil.copy(outs[0], done_clip)
        pngs = decode_clip(done_clip, win_dir / "frames")
        log(f"window {w['index']}: {len(pngs)} frames in {dt:.0f} s wall = {dt / max(1, len(pngs)):.1f} s/frame "
            f"(upload + queue + render + download)")
        keep = pngs[:w["new_count"]]
        if len(pngs) < w["new_count"]:
            log(f"WARNING: window {w['index']} returned {len(pngs)} frames, expected >= {w['new_count']}")
        for i, p in enumerate(keep):
            shutil.copy(p, frames_dir / f"{total_new + i:05d}.png")
        total_new += len(keep)
        prev_clip, prev_count = done_clip, len(pngs)
        if a.probe is not None:
            log(f"PROBE: {dt / max(1, len(pngs)):.1f} s/frame wall -> projected {dt / max(1, len(pngs)) * n / 60:.0f} min for the {n}-frame block")
            break

    log(f"total {total_new} frames in {(time.time() - t_all) / 60:.1f} min")
    if total_new < n and a.probe is None:
        log(f"WARNING: expected {n} frames, got {total_new}")
    mp4 = out_dir.with_suffix(".mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", f"{a.clip_fps:g}",
                    "-i", str(frames_dir / "%05d.png"), "-c:v", "libx264", "-crf", "14",
                    "-pix_fmt", "yuv420p", "-r", f"{a.clip_fps:g}", str(mp4)], check=True)
    log(f"wrote {mp4} @ {a.clip_fps:g} fps")
    json.dump({"block": a.block, "side": a.side, "stride": a.stride, "range_start": r0,
               "frames": staged["frames"][:total_new], "n": total_new, "width": a.width, "height": a.height,
               "length": a.length, "windows": [{k: v for k, v in w.items() if not isinstance(v, Path)} for w in wins],
               "ref": a.ref, "prompt": a.prompt, "steps": a.steps, "seed": a.seed, "where": "comfy-deploy"},
              open(out_dir / "render.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

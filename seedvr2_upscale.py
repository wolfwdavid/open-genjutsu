#!/usr/bin/env python
"""
seedvr2_upscale.py -- restore and upscale a PNG frame sequence with SeedVR2 (one-step video
restoration, ComfyUI native nodes since ComfyUI 8e2e54e) through a headless ComfyUI.

The sequence is cut into segments at shot cuts (a temporal model must not blend across a cut);
each segment is one ComfyUI prompt. Inside a prompt the latent is split into temporal chunks that
are sampled one at a time and crossfaded back together, so VRAM stays bounded on a 16 GB card.

Graph per segment (the official "Video Upscale: SeedVR2" template, without its video I/O):
  frames -> ResizeImageMaskNode (scale by multiplier, lanczos) -> SeedVR2Preprocess (pad to 16 px, 4n+1)
  -> VAEEncodeTiled -> SeedVR2TemporalChunk -> SeedVR2Conditioning -> KSampler (1 step, cfg 1, euler)
  -> SeedVR2TemporalMerge -> VAEDecodeTiled -> SeedVR2PostProcessing (trim + colour match) -> PNG

Usage
  python seedvr2_upscale.py --in work_ex/wan/cat_E/frames_df --name cat_E_up --scale 2 --cuts auto
  ... --probe 21     # only the first 21 frames, report s/frame, exit
  ... --dry-run      # print the first prompt JSON and exit
Outputs: <WORK>/wan/<name>/frames/%05d.png (+ <name>.mp4, render.json, run.log).
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

from wan_animate import Comfy, Graph, load_image_dir

ROOT = Path(__file__).resolve().parent
WORK = Path(os.environ["VG_WORK"]).resolve() if os.environ.get("VG_WORK") else ROOT / "work"


def build_segment(a, seg: dict) -> tuple[dict, str]:
    g = Graph()
    model = g.out(g.add("UNETLoader", unet_name=a.unet, weight_dtype="default"))
    vae = g.out(g.add("VAELoader", vae_name=a.vae))
    frames = g.out(load_image_dir(g, seg["dir"], seg["start"], seg["count"]))
    resized = g.out(g.add("ResizeImageMaskNode", input=frames, resize_type="scale by multiplier",
                          **{"resize_type.multiplier": a.scale}, scale_method="lanczos"))
    pre = g.out(g.add("SeedVR2Preprocess", resized_images=resized))
    latent = g.out(g.add("VAEEncodeTiled", pixels=pre, vae=vae, tile_size=a.tile, overlap=128,
                         temporal_size=64, temporal_overlap=8))
    if a.chunk > 0:
        chunk = g.add("SeedVR2TemporalChunk", latent=latent, temporal_overlap=a.chunk_overlap,
                      chunking_mode="manual", **{"chunking_mode.frames_per_chunk": a.chunk})
        latents, overlap = g.out(chunk, 0), g.out(chunk, 1)
    else:
        latents = latent
    cond = g.add("SeedVR2Conditioning", model=model, vae_conditioning=latents)
    sampled = g.out(g.add("KSampler", model=model, positive=g.out(cond, 0), negative=g.out(cond, 1),
                          latent_image=latents, seed=a.seed + seg["index"], steps=1, cfg=1.0,
                          sampler_name="euler", scheduler="simple", denoise=a.denoise))
    if a.chunk > 0:
        sampled = g.out(g.add("SeedVR2TemporalMerge", latents=sampled, temporal_overlap=overlap))
    # decode is the memory peak (the 3B model is still resident): smaller tiles than the encode.
    # First T4 run OOMed here at tile 512 / 64 frames on 1536x1152 output (2026-10-08).
    img = g.out(g.add("VAEDecodeTiled", samples=sampled, vae=vae, tile_size=a.decode_tile,
                      overlap=a.decode_tile // 4, temporal_size=a.decode_frames, temporal_overlap=8))
    img = g.out(g.add("SeedVR2PostProcessing", images=img, original_resized_images=resized,
                      color_correction_method=a.color))
    prefix = f"wan/{a.name}/s{seg['index']:02d}/f"
    g.add("SaveImage", images=img, filename_prefix=prefix)
    return g.nodes, prefix


def parse_cuts(spec: str, n: int, stride: int) -> list[int]:
    """Segment boundaries as frame indices of the input sequence. 'auto' reads the shot cuts of the
    example clip (tracks30.json, cut-frame numbers) and maps them through --stride."""
    if not spec or spec == "none":
        return []
    if spec == "auto":
        try:
            cuts = json.load(open(WORK / "tracks30.json"))["cuts"]
        except (OSError, KeyError, ValueError):
            return []
        return sorted({-(-c // stride) for c in cuts if 0 < -(-c // stride) < n})
    return sorted({int(c) for c in spec.split(",") if 0 < int(c) < n})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True, help="PNG frame directory (sorted names)")
    ap.add_argument("--name", required=True, help="output name under work/wan/")
    ap.add_argument("--scale", type=float, default=2.0, help="upscale factor before restoration")
    ap.add_argument("--cuts", default="auto", help="'auto' (shot cuts of the clip), 'none', or comma frame indices")
    ap.add_argument("--stride", type=int, default=2, help="cut-frame stride of the input (for --cuts auto)")
    ap.add_argument("--chunk", type=int, default=21, help="pixel frames per temporal chunk (4n+1; 0 = whole segment)")
    ap.add_argument("--chunk-overlap", type=int, default=1, help="latent frames crossfaded between chunks")
    ap.add_argument("--tile", type=int, default=512, help="VAE tile size in px")
    ap.add_argument("--decode-tile", type=int, default=256, help="VAE decode tile size in px")
    ap.add_argument("--decode-frames", type=int, default=32, help="frames decoded per temporal tile")
    ap.add_argument("--color", default="lab", choices=["lab", "wavelet", "adain", "none"])
    ap.add_argument("--denoise", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--unet", default="seedvr2_3b_fp16.safetensors")
    ap.add_argument("--vae", default="seedvr2_ema_vae_fp16.safetensors")
    ap.add_argument("--probe", type=int, default=None, help="only the first N frames")
    ap.add_argument("--fps", type=float, default=12.0, help="fps of the preview mp4")
    ap.add_argument("--comfy", default="http://127.0.0.1:8188")
    ap.add_argument("--comfy-dir", default=str(ROOT / "vendor" / "ComfyUI"))
    ap.add_argument("--resume", action="store_true", help="skip segments already rendered")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.chunk and (a.chunk - 1) % 4:
        sys.exit("--chunk must be 4n+1 (5, 9, 13, 17, 21, ...)")

    src = Path(a.src)
    if not src.is_absolute():
        src = (ROOT / src).resolve()
    pngs = sorted(src.glob("*.png"))
    if not pngs:
        sys.exit(f"no PNG frames in {src}")
    n = len(pngs) if a.probe is None else min(a.probe, len(pngs))

    out_dir = WORK / "wan" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = open(out_dir / "run.log", "a", encoding="utf-8")

    def log(msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    bounds = [0] + [c for c in parse_cuts(a.cuts, n, a.stride) if c < n] + [n]
    segs = [{"index": i, "dir": src, "start": s, "count": e - s} for i, (s, e) in enumerate(zip(bounds, bounds[1:])) if e > s]
    log(f"{n} frames of {src} -> {len(segs)} segment(s): " + ", ".join(f"s{s['index']}[{s['start']}+{s['count']}]" for s in segs))

    if a.dry_run:
        print(json.dumps(build_segment(a, segs[0])[0], indent=1))
        return 0

    comfy_dir = Path(a.comfy_dir)
    frames_dir = out_dir / "frames"
    if frames_dir.exists() and not a.resume:
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    client = Comfy(a.comfy)
    st = client.stats()
    log(f"ComfyUI {st.get('system', {}).get('comfyui_version', '?')} devices {[d.get('name') for d in st.get('devices', [])]}")
    for cls in ("SeedVR2Preprocess", "SeedVR2TemporalChunk", "SeedVR2PostProcessing"):
        try:
            client.object_info(cls)
        except Exception as e:  # noqa: BLE001
            log(f"ComfyUI has no node {cls}: {e}")
            return 3

    t_all = time.time()
    written = 0
    for seg in segs:
        seg_out = comfy_dir / "output" / f"wan/{a.name}/s{seg['index']:02d}"
        done = sorted(seg_out.glob("*.png")) if seg_out.exists() else []
        if not (a.resume and len(done) >= seg["count"]):
            if seg_out.exists():
                shutil.rmtree(seg_out)
            prompt, _ = build_segment(a, seg)
            t0 = time.time()
            client.wait(client.queue(prompt), log=log)
            dt = time.time() - t0
            done = sorted(seg_out.glob("*.png")) if seg_out.exists() else []
            if not done:
                log(f"segment {seg['index']}: no output under {seg_out}")
                return 2
            log(f"segment {seg['index']}: {len(done)} frames in {dt:.0f} s = {dt / len(done):.1f} s/frame")
        else:
            log(f"segment {seg['index']}: already rendered, skipped")
        if len(done) != seg["count"]:
            log(f"WARNING: segment {seg['index']} returned {len(done)} frames, expected {seg['count']}")
        for i, p in enumerate(done[:seg["count"]]):
            shutil.copy(p, frames_dir / f"{seg['start'] + i:05d}.png")
        written += min(len(done), seg["count"])

    total = time.time() - t_all
    log(f"total {written} frames in {total / 60:.1f} min ({total / max(1, written):.1f} s/frame)")
    from PIL import Image  # local import: only needed once frames exist
    first = sorted(frames_dir.glob("*.png"))[0]
    size = Image.open(first).size
    mp4 = out_dir.with_suffix(".mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", f"{a.fps:g}", "-i", str(frames_dir / "%05d.png"),
                    "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(mp4)], check=True)
    log(f"wrote {mp4} ({size[0]}x{size[1]})")
    json.dump({"engine": "seedvr2", "src": str(src), "n": written, "size": size, "scale": a.scale, "unet": a.unet,
               "segments": [{k: v for k, v in s.items() if k != "dir"} for s in segs], "chunk": a.chunk,
               "color": a.color, "seconds": total}, open(out_dir / "render.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""
animate_cat.py -- drive a LivePortrait (animals v1.1) cat head with the RIGHT
performer's head motion from work/cut.mp4 and write contract (C):

    work/cat_anim/%05d.png   512x512 BGR animated cat head, one per video frame
    work/cat_anim/motion.json

Motion policy (final decisions): pose-only transfer.  The driver's
pitch/yaw/roll, translation and scale RELATIVE TO A NEUTRAL DRIVING FRAME
(t=13.47 s, mouth closed, near-frontal) are held across dropouts, smoothed
per shot (reset at the cuts), scaled, clamped, and applied on top of the cat
source's own pose.  The expression deltas are the SOURCE's own (zero
expression transfer) so the mouth stays closed and the eyes stay as in the
photo.

Runs from .venv-lp (torch xpu).  Imports vendor/LivePortrait via sys.path, the
same way vendor/lp_harness/smoke_lp.py does; vendor sources are not modified.

Usage (defaults reproduce the deliverable):
  PYTHONIOENCODING=utf-8 .venv-lp/Scripts/python.exe animate_cat.py \
      --src refs/cat_src_glasses_512.png --video work/cut.mp4 --tracks work/tracks30.json \
      --sheet work/cat_anim_sheet.jpg
  # 3 s test clip 15.5-18.5 s with a driver|cat side-by-side mp4 (no PNGs):
  ... --start 465 --end 555 --test-video work/cat_anim_3s.mp4 --no-png
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor" / "LivePortrait"))

import torch  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

from src.config.inference_config import InferenceConfig  # noqa: E402
from src.live_portrait_wrapper import LivePortraitWrapperAnimal  # noqa: E402
from src.utils.camera import get_rotation_matrix  # noqa: E402
from src.utils.helper import calc_motion_multiplier  # noqa: E402

DEFAULT_CUTS = [104, 239, 461, 581, 688]


# ----------------------------------------------------------------------------- tracks
def load_tracks30(path: Path) -> dict:
    """Contract (A): per-frame smoothed RIGHT-performer track at 30 Hz."""
    d = json.load(open(path))
    R = d["R"]
    n = len(R)
    tr = {
        "n": n,
        "fps": float(d.get("fps", 30.0)),
        "cuts": list(d.get("cuts", DEFAULT_CUTS)),
        "cx": np.array([r["cx"] for r in R], np.float64),
        "cy": np.array([r["cy"] for r in R], np.float64),
        "w": np.array([r["w"] for r in R], np.float64),
        "h": np.array([r["h"] for r in R], np.float64),
        "yaw": np.array([r.get("yaw", 0.0) for r in R], np.float64),
        "pitch": np.array([r.get("pitch", 0.0) for r in R], np.float64),
        "roll": np.array([r.get("roll", 0.0) for r in R], np.float64),
        "held": np.array([bool(r.get("held", False)) or not bool(r.get("present", True)) for r in R]),
    }
    return tr


def load_tracks15(path: Path, n_frames: int, fps: float, det_thr: float = 0.3) -> dict:
    """Fallback: build a 30 Hz track by linearly interpolating work/analysis.json
    tracks.R (15 Hz).  Frames whose 15 Hz slot has no detection are HELD
    (previous good values)."""
    a = json.load(open(path))
    T = a["tracks"]
    R = T["R"]
    ts = np.array(T["t"], np.float64)
    keys = ["cx", "cy", "w", "h", "yaw", "pitch", "roll"]
    valid = np.array([bool(r) and float(r.get("det", 1.0)) >= det_thr for r in R])
    vals = {k: np.array([float(r[k]) if v else np.nan for r, v in zip(R, valid)], np.float64) for k in keys}
    out = {k: np.zeros(n_frames) for k in keys}
    held = np.zeros(n_frames, bool)
    step = float(np.median(np.diff(ts))) if len(ts) > 1 else 1.0 / 15
    last = None
    for f in range(n_frames):
        t = f / fps
        k = int(round(t / step))
        k = min(max(k, 0), len(ts) - 1)
        if not valid[k]:
            held[f] = True
            if last is None:
                j = int(np.argmax(valid))  # before the first detection: first valid sample
                last = {kk: vals[kk][j] for kk in keys}
            for kk in keys:
                out[kk][f] = last[kk]
            continue
        # neighbours for linear interpolation
        if t >= ts[k]:
            k1, k2 = k, min(k + 1, len(ts) - 1)
        else:
            k1, k2 = max(k - 1, 0), k
        if not valid[k1]:
            k1 = k
        if not valid[k2]:
            k2 = k
        if k1 == k2 or ts[k2] == ts[k1]:
            cur = {kk: vals[kk][k] for kk in keys}
        else:
            a1 = min(max((t - ts[k1]) / (ts[k2] - ts[k1]), 0.0), 1.0)
            cur = {kk: (1 - a1) * vals[kk][k1] + a1 * vals[kk][k2] for kk in keys}
        for kk in keys:
            out[kk][f] = cur[kk]
        last = cur
    out.update({"n": n_frames, "fps": fps, "cuts": list(DEFAULT_CUTS), "held": held})
    return out


# ----------------------------------------------------------------------------- driver crops
def crop_driver(frame: np.ndarray, cx: float, cy: float, w: float, size: int = 256,
                side_mult: float = 2.2, cy_shift: float = -0.1) -> tuple[np.ndarray, tuple]:
    """1:1 crop of side 2.2*w centred on (cx, cy - 0.1*w), replicate-padded, -> size x size RGB."""
    side = max(int(round(side_mult * w)), 8)
    ccx, ccy = cx, cy + cy_shift * w
    x0 = int(round(ccx - side / 2))
    y0 = int(round(ccy - side / 2))
    pad = side
    fr = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    crop = fr[y0 + pad:y0 + pad + side, x0 + pad:x0 + pad + side]
    interp = cv2.INTER_AREA if side > size else cv2.INTER_CUBIC
    crop = cv2.resize(crop, (size, size), interpolation=interp)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (x0, y0, side)


def read_driver_crops(video: Path, tr: dict, n_frames: int) -> tuple[list, list]:
    cap = cv2.VideoCapture(str(video))
    crops, boxes = [], []
    f = 0
    while f < n_frames:
        ok, fr = cap.read()
        if not ok:
            break
        c, b = crop_driver(fr, tr["cx"][f], tr["cy"][f], tr["w"][f])
        crops.append(c)
        boxes.append(b)
        f += 1
    cap.release()
    while len(crops) < n_frames:  # container reported fewer frames than expected
        crops.append(crops[-1].copy())
        boxes.append(boxes[-1])
    return crops, boxes


# ----------------------------------------------------------------------------- smoothing
def segments(n: int, cuts: list[int]) -> list[tuple[int, int]]:
    b = [0] + [c for c in sorted(cuts) if 0 < c < n] + [n]
    return [(b[i], b[i + 1]) for i in range(len(b) - 1)]


def hold_last(x: np.ndarray, held: np.ndarray) -> np.ndarray:
    """Hold the last non-held value across held frames ((N,) or (N,k) arrays)."""
    y = x.copy()
    last = None
    for i in range(len(y)):
        if held[i] and last is not None:
            y[i] = last
        elif not held[i]:
            last = y[i].copy() if y.ndim > 1 else y[i]
    return y


def smooth_segments(x: np.ndarray, cuts: list[int], sigma: float, sigma_per_seg: list[float] | None = None) -> np.ndarray:
    """Centered Gaussian smoothing, reset at cuts (offline, whole track known).
    sigma_per_seg overrides sigma for each segment (e.g. heavier smoothing in wide shots)."""
    y = x.copy()
    for i, (s, e) in enumerate(segments(len(x), cuts)):
        sg = sigma if sigma_per_seg is None else sigma_per_seg[i]
        if sg > 0 and e - s >= 2:
            y[s:e] = gaussian_filter1d(x[s:e], sigma=sg, axis=0, mode="nearest")
    return y


# ----------------------------------------------------------------------------- LivePortrait
def build_wrapper(device: str, half: bool) -> LivePortraitWrapperAnimal:
    cfg = InferenceConfig(flag_force_cpu=True, flag_use_half_precision=half,
                          flag_stitching=False, flag_pasteback=False, flag_do_crop=False)
    w = LivePortraitWrapperAnimal(cfg)  # loads on CPU
    if device != "cpu":
        for m in [w.appearance_feature_extractor, w.motion_extractor, w.warping_module, w.spade_generator]:
            m.to(device)
        w.device = device  # device[:4] == 'xpu' -> autocast device_type
    return w


@torch.no_grad()
def extract_motion(w: LivePortraitWrapperAnimal, crops: list, batch: int) -> dict:
    """Run the motion extractor on all driver crops. Returns numpy arrays."""
    n = len(crops)
    pitch = np.zeros(n); yaw = np.zeros(n); roll = np.zeros(n)
    scale = np.zeros(n); t = np.zeros((n, 3)); kp = None; exp = None
    for s in range(0, n, batch):
        e = min(n, s + batch)
        x = np.stack(crops[s:e]).astype(np.float32) / 255.0
        x = torch.from_numpy(x).permute(0, 3, 1, 2).to(w.device)
        info = w.get_kp_info(x)
        pitch[s:e] = info["pitch"][:, 0].cpu().numpy()
        yaw[s:e] = info["yaw"][:, 0].cpu().numpy()
        roll[s:e] = info["roll"][:, 0].cpu().numpy()
        scale[s:e] = info["scale"].reshape(-1).cpu().numpy()
        t[s:e] = info["t"].reshape(-1, 3).cpu().numpy()
        if kp is None:
            kp = np.zeros((n,) + tuple(info["kp"].shape[1:]), np.float32)
            exp = np.zeros((n,) + tuple(info["exp"].shape[1:]), np.float32)
        kp[s:e] = info["kp"].cpu().numpy()
        exp[s:e] = info["exp"].cpu().numpy()
    return {"pitch": pitch, "yaw": yaw, "roll": roll, "scale": scale, "t": t, "kp": kp, "exp": exp}


def put_label(img: np.ndarray, text: str, y: int = 26, fs: float = 0.6) -> None:
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="refs/cat_src_glasses_512.png")
    ap.add_argument("--video", default="work/cut.mp4")
    ap.add_argument("--tracks", default="work/tracks30.json", help="contract (A) 30 Hz tracks")
    ap.add_argument("--tracks-15hz", default="work/analysis.json",
                    help="fallback 15 Hz analysis.json used when --tracks does not exist")
    ap.add_argument("--out-dir", default="work/cat_anim")
    ap.add_argument("--motion-json", default=None, help="default <out-dir>/motion.json")
    ap.add_argument("--n-frames", type=int, default=872)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--neutral-t", type=float, default=13.47, help="neutral driving frame time (s)")
    ap.add_argument("--start", type=int, default=0, help="first frame to render (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="last frame to render (exclusive)")
    ap.add_argument("--driving-multiplier", type=float, default=1.0)
    ap.add_argument("--yaw-scale", type=float, default=0.8)
    ap.add_argument("--yaw-clamp", type=float, default=35.0)
    ap.add_argument("--pitch-scale", type=float, default=0.9)
    ap.add_argument("--pitch-clamp", type=float, default=20.0)
    ap.add_argument("--roll-scale", type=float, default=1.0)
    ap.add_argument("--roll-clamp", type=float, default=20.0)
    ap.add_argument("--t-scale", type=float, default=0.0, help="translation transfer gain (0: all translation comes from the track in composite.py; >0 made the head pop)")
    ap.add_argument("--t-clamp", type=float, default=0.06, help="max |tx|,|ty| delta in driver kp space")
    ap.add_argument("--scale-clamp", type=float, default=0.12, help="max |scale ratio - 1|")
    ap.add_argument("--smooth-sigma", type=float, default=1.0, help="Gaussian sigma in frames (~5-frame window)")
    ap.add_argument("--smooth-sigma-small", type=float, default=3.0, help="sigma for shots whose median face width < --small-w (wide shots: noisy pose)")
    ap.add_argument("--small-w", type=float, default=40.0)
    ap.add_argument("--pose-source", choices=["lp", "tracks"], default="lp",
                    help="lp = LivePortrait motion-extractor angles (default); tracks = solvePnP angles from the tracks")
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--batch", type=int, default=4, help="warp_decode batch size")
    ap.add_argument("--motion-batch", type=int, default=16)
    ap.add_argument("--no-png", action="store_true", help="skip writing PNGs and motion.json (tests)")
    ap.add_argument("--test-video", default=None, help="write driver|cat side-by-side mp4 of the rendered range")
    ap.add_argument("--sheet", default=None, help="write a driver|cat contact sheet jpg")
    ap.add_argument("--sheet-frames", default="30,150,300,480,650,800")
    args = ap.parse_args()

    root = ROOT
    t_all = time.time()
    n = args.n_frames
    fps = args.fps
    end = n if args.end is None else min(args.end, n)
    start = max(0, args.start)

    # -- tracks
    tp = root / args.tracks
    if tp.exists():
        tr = load_tracks30(tp)
        src_tracks = str(args.tracks)
    else:
        tr = load_tracks15(root / args.tracks_15hz, n, fps)
        src_tracks = str(args.tracks_15hz) + " (15 Hz, interpolated)"
    cuts = [c for c in tr["cuts"] if 0 < c < n]
    print("tracks:", src_tracks, "frames", tr["n"], "cuts", cuts, "held", int(tr["held"].sum()))

    # -- source
    src_path = root / args.src
    src_bgr = cv2.imread(str(src_path))
    if src_bgr is None:
        print("ERROR: cannot read source", src_path)
        return 2
    if src_bgr.shape[:2] != (512, 512):
        print("WARN: source is", src_bgr.shape[:2], "-> resizing to 512x512")
        src_bgr = cv2.resize(src_bgr, (512, 512), interpolation=cv2.INTER_AREA)
    src_rgb = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)
    print("source:", src_path)

    # -- driver crops
    t0 = time.time()
    crops, boxes = read_driver_crops(root / args.video, tr, n)
    print("driver crops", len(crops), "read in", round(time.time() - t0, 1), "s")

    # -- LivePortrait
    dev = args.device
    if dev == "xpu" and not torch.xpu.is_available():
        print("WARN: xpu not available, falling back to cpu")
        dev = "cpu"
    half = not args.fp32
    t0 = time.time()
    w = build_wrapper(dev, half)
    print("LivePortrait animal loaded on", dev, "half", half, "in", round(time.time() - t0, 1), "s; torch", torch.__version__)

    with torch.no_grad():
        I_s = w.prepare_source(src_rgb)
        x_s_info = w.get_kp_info(I_s)
        x_c_s = x_s_info["kp"]                 # 1xNx3 canonical keypoints
        f_s = w.extract_feature_3d(I_s)
        x_s = w.transform_keypoint(x_s_info)   # source's own pose + expression
        s_pitch = float(x_s_info["pitch"]); s_yaw = float(x_s_info["yaw"]); s_roll = float(x_s_info["roll"])
        print("source pose  pitch %.1f yaw %.1f roll %.1f scale %.3f" % (s_pitch, s_yaw, s_roll, float(x_s_info["scale"])))

        t0 = time.time()
        M = extract_motion(w, crops, args.motion_batch)
        if dev == "xpu":
            torch.xpu.synchronize()
        t_me = time.time() - t0
        print("motion extractor: %d frames in %.1f s (%.4f s/frame)" % (n, t_me, t_me / n))

    # -- neutral frame + raw deltas
    nidx = min(max(int(round(args.neutral_t * fps)), 0), n - 1)
    if args.pose_source == "lp":
        P, Y, Rl = M["pitch"], M["yaw"], M["roll"]
    else:
        P, Y, Rl = tr["pitch"], tr["yaw"], tr["roll"]
    print("neutral frame %d (t=%.2f s): driver pitch %.1f yaw %.1f roll %.1f scale %.3f" % (
        nidx, nidx / fps, P[nidx], Y[nidx], Rl[nidx], M["scale"][nidx]))

    held = tr["held"].copy()
    d_pitch = P - P[nidx]
    d_yaw = Y - Y[nidx]
    d_roll = Rl - Rl[nidx]
    d_t = M["t"] - M["t"][nidx]
    d_t[:, 2] = 0.0
    r_scale = M["scale"] / max(M["scale"][nidx], 1e-6)

    # hold last good pose across held frames, then smooth per shot segment
    d_pitch = hold_last(d_pitch, held); d_yaw = hold_last(d_yaw, held); d_roll = hold_last(d_roll, held)
    d_t = hold_last(d_t, held); r_scale = hold_last(r_scale, held)
    sg = args.smooth_sigma
    seg_sig = []
    for s0, e0 in segments(n, cuts):
        wmed = float(np.median(tr["w"][s0:e0]))
        seg_sig.append(args.smooth_sigma_small if wmed < args.small_w else sg)
    print("per-shot smoothing sigma:", [(s0, e0, round(sig, 1)) for (s0, e0), sig in zip(segments(n, cuts), seg_sig)])
    d_pitch = smooth_segments(d_pitch, cuts, sg, seg_sig); d_yaw = smooth_segments(d_yaw, cuts, sg, seg_sig)
    d_roll = smooth_segments(d_roll, cuts, sg, seg_sig); d_t = smooth_segments(d_t, cuts, sg, seg_sig)
    r_scale = smooth_segments(r_scale, cuts, sg, seg_sig)

    # scale + clamp (clamp last so the applied pose never exceeds the limits)
    a_pitch = np.clip(d_pitch * args.pitch_scale, -args.pitch_clamp, args.pitch_clamp)
    a_yaw = np.clip(d_yaw * args.yaw_scale, -args.yaw_clamp, args.yaw_clamp)
    a_roll = np.clip(d_roll * args.roll_scale, -args.roll_clamp, args.roll_clamp)
    a_t = np.clip(d_t * args.t_scale, -args.t_clamp, args.t_clamp)
    a_t[:, 2] = 0.0
    a_scale = np.clip(r_scale, 1.0 - args.scale_clamp, 1.0 + args.scale_clamp)
    print("applied deltas: |yaw| max %.1f  |pitch| max %.1f  |roll| max %.1f  |t| max %.3f  scale %.3f..%.3f" % (
        np.abs(a_yaw).max(), np.abs(a_pitch).max(), np.abs(a_roll).max(), np.abs(a_t).max(), a_scale.min(), a_scale.max()))
    n_clamped = int((np.abs(d_yaw * args.yaw_scale) > args.yaw_clamp).sum()
                    + (np.abs(d_pitch * args.pitch_scale) > args.pitch_clamp).sum()
                    + (np.abs(d_roll * args.roll_scale) > args.roll_clamp).sum())
    print("frame-angles hitting a rotation clamp:", n_clamped)

    # translation gain: driver kp space -> source kp space (convex-hull area ratio, as LivePortrait does)
    with torch.no_grad():
        info_n = {k: torch.from_numpy(np.asarray(v[nidx:nidx + 1])).to(w.device) for k, v in M.items()}
        R_d_n = get_rotation_matrix(info_n["pitch"].float(), info_n["yaw"].float(), info_n["roll"].float())
        t_n = info_n["t"].float().clone(); t_n[..., 2] = 0
        x_d_n_abs = x_s_info["scale"] * (x_c_s @ R_d_n + info_n["exp"].float()) + t_n
        mm = float(calc_motion_multiplier(x_s, x_d_n_abs))
    print("motion multiplier (translation gain) %.3f" % mm)

    # -- render
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    idxs = list(range(start, end))
    B = max(1, args.batch)
    dm = args.driving_multiplier
    rendered = {}
    times = []
    if dev == "xpu":
        torch.xpu.reset_peak_memory_stats()
    t_render = time.time()
    with torch.no_grad():
        exp_s = x_s_info["exp"]
        scale_s = x_s_info["scale"]
        t_s = x_s_info["t"].clone(); t_s[..., 2] = 0
        for bs in range(0, len(idxs), B):
            ids = idxs[bs:bs + B]
            k = len(ids)
            t1 = time.time()
            pitch_t = torch.tensor([s_pitch + a_pitch[i] for i in ids], dtype=torch.float32, device=w.device)
            yaw_t = torch.tensor([s_yaw + a_yaw[i] for i in ids], dtype=torch.float32, device=w.device)
            roll_t = torch.tensor([s_roll + a_roll[i] for i in ids], dtype=torch.float32, device=w.device)
            R_new = get_rotation_matrix(pitch_t, yaw_t, roll_t)  # kx3x3
            sc = scale_s.expand(k, 1) * torch.tensor([a_scale[i] for i in ids], dtype=torch.float32, device=w.device)[:, None]
            tt = t_s.expand(k, 3) + mm * torch.tensor(np.stack([a_t[i] for i in ids]), dtype=torch.float32, device=w.device)
            x_d = sc[..., None] * (x_c_s.expand(k, -1, -1) @ R_new + exp_s.expand(k, -1, -1)) + tt[:, None, :]
            x_s_k = x_s.expand(k, -1, -1)
            x_d = x_s_k + (x_d - x_s_k) * dm
            out = w.warp_decode(f_s.expand(k, -1, -1, -1, -1), x_s_k, x_d)
            imgs = w.parse_output(out["out"])  # kx512x512x3 RGB
            if dev == "xpu":
                torch.xpu.synchronize()
            times.append((time.time() - t1) / k)
            for j, i in enumerate(ids):
                bgr = cv2.cvtColor(imgs[j], cv2.COLOR_RGB2BGR)
                if args.test_video or args.sheet:
                    rendered[i] = bgr
                if not args.no_png:
                    cv2.imwrite(str(out_dir / ("%05d.png" % i)), bgr)
            if (bs // B) % 25 == 0:
                print("  rendered %d/%d  %.3f s/frame" % (bs + k, len(idxs), float(np.mean(times[-25:]))), flush=True)
    t_render = time.time() - t_render
    print("render: %d frames in %.1f s (%.4f s/frame, batch %d)" % (len(idxs), t_render, t_render / max(1, len(idxs)), B))
    if dev == "xpu":
        print("xpu max memory allocated MB", round(torch.xpu.max_memory_allocated() / 2 ** 20))

    # -- motion.json (poses = applied deltas relative to the neutral frame, after clamps)
    if not args.no_png:
        mj = root / (args.motion_json or (args.out_dir + "/motion.json"))
        rec = {
            "n": n, "src": str(args.src), "fps": fps, "cuts": cuts,
            "neutral_frame": nidx, "neutral_t": nidx / fps,
            "source_pose": {"pitch": s_pitch, "yaw": s_yaw, "roll": s_roll},
            "params": {"yaw_scale": args.yaw_scale, "yaw_clamp": args.yaw_clamp, "pitch_scale": args.pitch_scale,
                       "pitch_clamp": args.pitch_clamp, "roll_scale": args.roll_scale, "roll_clamp": args.roll_clamp,
                       "t_scale": args.t_scale, "t_clamp": args.t_clamp, "scale_clamp": args.scale_clamp,
                       "smooth_sigma": args.smooth_sigma, "smooth_sigma_small": args.smooth_sigma_small, "small_w": args.small_w, "driving_multiplier": dm, "pose_source": args.pose_source,
                       "motion_multiplier": mm, "tracks": src_tracks},
            "rendered_range": [start, end],
            "frames": [
                {"f": i, "yaw": round(float(a_yaw[i]), 2), "pitch": round(float(a_pitch[i]), 2),
                 "roll": round(float(a_roll[i]), 2), "held": bool(held[i]),
                 "tx": round(float(a_t[i, 0]), 4), "ty": round(float(a_t[i, 1]), 4), "scale": round(float(a_scale[i]), 4),
                 "drv_yaw": round(float(Y[i]), 1), "drv_pitch": round(float(P[i]), 1), "drv_roll": round(float(Rl[i]), 1),
                 "crop": [int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2])]}
                for i in range(n)
            ],
        }
        mj.parent.mkdir(parents=True, exist_ok=True)
        with open(mj, "w", newline="\n") as fh:
            json.dump(rec, fh)
        print("wrote", mj)

    # -- test video: driver crop | cat, 1024x512
    if args.test_video and rendered:
        tv = root / args.test_video
        tv.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1024x512",
               "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(tv)]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for i in idxs:
            drv = cv2.cvtColor(cv2.resize(crops[i], (512, 512), interpolation=cv2.INTER_CUBIC), cv2.COLOR_RGB2BGR)
            fr = np.concatenate([drv, rendered[i]], 1)
            put_label(fr, "f%d t=%.2fs yaw %+.0f pitch %+.0f roll %+.0f" % (i, i / fps, a_yaw[i], a_pitch[i], a_roll[i]))
            p.stdin.write(fr.tobytes())
        p.stdin.close()
        p.wait()
        print("wrote", tv, "rc", p.returncode)

    # -- contact sheet: driver | cat pairs, 2 pairs per row
    if args.sheet:
        want = [int(x) for x in args.sheet_frames.split(",") if x.strip()]
        tiles = []
        for i in want:
            if i in rendered:
                cat = rendered[i]
            else:
                pth = out_dir / ("%05d.png" % i)
                cat = cv2.imread(str(pth)) if pth.exists() else np.zeros((512, 512, 3), np.uint8)
            drv = cv2.cvtColor(cv2.resize(crops[i], (512, 512), interpolation=cv2.INTER_CUBIC), cv2.COLOR_RGB2BGR)
            pair = np.concatenate([drv, cat], 1)
            put_label(pair, "f%d t=%.2fs  yaw %+.0f pitch %+.0f roll %+.0f%s" % (
                i, i / fps, a_yaw[i], a_pitch[i], a_roll[i], "  HELD" if held[i] else ""), y=30, fs=0.75)
            tiles.append(pair)
        rows = []
        for r in range(0, len(tiles), 2):
            right = tiles[r + 1] if r + 1 < len(tiles) else np.zeros_like(tiles[r])
            rows.append(np.concatenate([tiles[r], right], 1))
        sheet = np.concatenate(rows, 0)
        sheet = cv2.resize(sheet, (sheet.shape[1] // 2, sheet.shape[0] // 2), interpolation=cv2.INTER_AREA)
        sp = root / args.sheet
        cv2.imwrite(str(sp), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print("wrote", sp)

    print("total wall %.1f s" % (time.time() - t_all))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""to30.py - 2x temporal upsampling of a PNG frame sequence.

Given a directory of sequential 0-based frames NNNNN.png rendered at half rate
(e.g. every second cut frame at 15 fps), produce a directory with every frame
(2*n_in-1 frames, or --n-out with last-frame padding) so the block lines up
frame-for-frame with the 30 fps song.

Methods (--method):
  rife          rife-ncnn-vulkan (Vulkan, works on Intel Arc).  Default.
  minterpolate  ffmpeg minterpolate (mci/aobmc/vsbmc) through a lossless
                intermediate video.  CPU fallback.
  dis           OpenCV DIS optical flow, symmetric midpoint warp.  CPU fallback
                that needs nothing but cv2.

Output frame k (0-based):
  even k = 2*j          -> input frame j (copied bit-exact)
  odd  k = 2*j + 1      -> interpolated midpoint between input j and j+1

--gt <dir> compares the result against a ground-truth directory of the same
naming (PSNR / SSIM of the interpolated odd frames) and writes a contact sheet.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RIFE_DIR = os.path.join(HERE, "vendor", "rife-ncnn-vulkan")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


def list_frames(d: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(d, "*.png")))
    if not files:
        raise SystemExit(f"no PNG frames in {d}")
    return files


def find_rife_exe(rife_dir: str) -> str | None:
    for cand in glob.glob(os.path.join(rife_dir, "**", "rife-ncnn-vulkan.exe"), recursive=True):
        return cand
    return None


def find_rife_model(exe: str, model: str) -> str:
    base = os.path.dirname(exe)
    cand = os.path.join(base, model)
    if os.path.isdir(cand):
        return cand
    have = sorted(os.path.basename(p) for p in glob.glob(os.path.join(base, "rife*")) if os.path.isdir(p))
    raise SystemExit(f"model {model!r} not found under {base}; have: {have}")


def copy_frame(src: str, dst: str) -> None:
    shutil.copyfile(src, dst)


def fix_count(out_dir: str, n_out: int) -> None:
    """Trim or pad the sequential 0-based NNNNN.png set in out_dir to n_out."""
    files = list_frames(out_dir)
    if len(files) > n_out:
        for f in files[n_out:]:
            os.remove(f)
    elif len(files) < n_out:
        last = files[-1]
        for k in range(len(files), n_out):
            copy_frame(last, os.path.join(out_dir, f"{k:05d}.png"))


def detect_cuts(src: list[str], thresh: float) -> list[int]:
    """Indices j where input j -> j+1 is a shot change (mean abs gray diff on a
    96x54 thumbnail above thresh; genuine fast motion on this footage tops out
    around 14, cuts sit at 26+)."""
    if thresh <= 0:
        return []
    def thumb(p):
        g = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, (96, 54), interpolation=cv2.INTER_AREA).astype(np.float32)
    cuts, prev = [], thumb(src[0])
    for j in range(1, len(src)):
        cur = thumb(src[j])
        if float(np.abs(cur - prev).mean()) > thresh:
            cuts.append(j - 1)
        prev = cur
    return cuts


def hold_at_cuts(in_dir: str, out_dir: str, cuts: list[int]) -> None:
    """At a shot change no interpolator can synthesise the missing frame; a
    blended ghost looks far worse than a duplicate.  The odd frame is given
    to the new shot (on this footage the real cut lands on the odd frame)."""
    src = list_frames(in_dir)
    for j in cuts:
        copy_frame(src[j + 1], os.path.join(out_dir, f"{2*j+1:05d}.png"))


# ----------------------------------------------------------------------------
# method: rife-ncnn-vulkan
# ----------------------------------------------------------------------------
def run_rife(in_dir: str, out_dir: str, n_in: int, args) -> float:
    exe = args.rife_bin or find_rife_exe(DEFAULT_RIFE_DIR)
    if not exe or not os.path.isfile(exe):
        raise SystemExit("rife-ncnn-vulkan.exe not found; download the windows release into "
                         f"{DEFAULT_RIFE_DIR} or pass --rife-bin")
    model = find_rife_model(exe, args.model)
    n_target = 2 * n_in - 1
    # rife -n places output i at time i*n_in/n_rife, so n_rife MUST be exactly
    # 2*n_in for odd outputs to land on timestep 0.5 (with -n 2*n_in-1 the
    # fraction drifts from 0.5 to ~1.0 over the run; measured -7 dB).  The
    # surplus 2*n_in-th frame is discarded.
    n_rife = 2 * n_in
    tmp = tempfile.mkdtemp(prefix="to30_rife_", dir=os.path.dirname(os.path.abspath(out_dir)))
    try:
        cmd = [exe, "-i", in_dir, "-o", tmp, "-n", str(n_rife), "-m", model,
               "-g", str(args.gpu), "-j", args.rife_threads, "-f", "%08d.png"]
        if args.tta:
            cmd.append("-x")
        log("rife: " + " ".join(cmd))
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.time() - t0
        if r.returncode != 0:
            sys.stderr.write(r.stdout + r.stderr)
            raise SystemExit(f"rife-ncnn-vulkan failed rc={r.returncode}")
        produced = sorted(glob.glob(os.path.join(tmp, "*.png")))
        log(f"rife: {len(produced)} frames in {dt:.1f}s  ({len(produced)/dt:.2f} out-frames/s)")
        if len(produced) != n_rife:
            raise SystemExit(f"rife produced {len(produced)} frames, expected {n_rife}")
        produced = produced[:n_target]
        black = [k for k in range(1, n_target, 2)
                 if cv2.imread(produced[k], cv2.IMREAD_GRAYSCALE).max() == 0]
        if black:
            raise SystemExit(f"rife emitted {len(black)} black frames (first {black[:3]}); "
                             "on Intel Arc keep proc=1 in --rife-threads (default 2:1:2)")
        # output k sits at time k/2.  Even k are the source frames re-encoded
        # through rife; copy the originals for those so they stay bit-exact,
        # keep rife's frames for odd k.
        src = list_frames(in_dir)
        for k, p in enumerate(produced):
            dst = os.path.join(out_dir, f"{k:05d}.png")
            if k % 2 == 0:
                copy_frame(src[k // 2], dst)
            else:
                shutil.move(p, dst)
        return dt
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------------
# method: ffmpeg minterpolate
# ----------------------------------------------------------------------------
def run_minterpolate(in_dir: str, out_dir: str, n_in: int, args) -> float:
    src = list_frames(in_dir)
    tmp = tempfile.mkdtemp(prefix="to30_mi_", dir=os.path.dirname(os.path.abspath(out_dir)))
    try:
        vid = os.path.join(tmp, "in.mkv")
        t0 = time.time()
        # lossless intermediate at 15 fps (rgb ffv1 keeps PNG bit-exact)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "15",
                        "-i", os.path.join(in_dir, "%05d.png"),
                        "-c:v", "ffv1", "-pix_fmt", "bgr0", vid], check=True)
        # minterpolate to 30 fps: out frame 2j = src j, 2j+1 = midpoint.
        # The filter starves at the tail (needs lookahead), so clone the last
        # frame twice first; the surplus is trimmed below.
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", vid,
                        "-vf", "tpad=stop=2:stop_mode=clone,"
                               "minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
                        "-start_number", "0", os.path.join(tmp, "%05d.png")], check=True)
        dt = time.time() - t0
        produced = sorted(glob.glob(os.path.join(tmp, "*.png")))
        log(f"minterpolate: {len(produced)} frames in {dt:.1f}s ({len(produced)/dt:.2f} out-frames/s)")
        n_target = 2 * n_in - 1
        if len(produced) < n_target:
            raise SystemExit(f"minterpolate produced {len(produced)} frames, expected >= {n_target}")
        for k in range(n_target):
            dst = os.path.join(out_dir, f"{k:05d}.png")
            if k % 2 == 0:
                copy_frame(src[k // 2], dst)
            else:
                shutil.move(produced[k], dst)
        return dt
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------------
# method: OpenCV DIS optical flow midpoint
# ----------------------------------------------------------------------------
def _warp(img: np.ndarray, flow: np.ndarray, t: float) -> np.ndarray:
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx = gx + flow[..., 0] * t
    my = gy + flow[..., 1] * t
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def dis_midpoint(a: np.ndarray, b: np.ndarray, dis) -> np.ndarray:
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    fab = dis.calc(ga, gb, None)   # a -> b
    fba = dis.calc(gb, ga, None)   # b -> a
    # backward warping: midpoint pixel p samples a at p + 0.5*f_ba(p) (approx)
    # and b at p + 0.5*f_ab(p); flows are approximated at p (not at midpoint).
    wa = _warp(a, fba, 0.5)
    wb = _warp(b, fab, 0.5)
    return cv2.addWeighted(wa, 0.5, wb, 0.5, 0)


def run_dis(in_dir: str, out_dir: str, n_in: int, args) -> float:
    src = list_frames(in_dir)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setFinestScale(0)
    t0 = time.time()
    prev = cv2.imread(src[0], cv2.IMREAD_COLOR)
    copy_frame(src[0], os.path.join(out_dir, "00000.png"))
    for j in range(1, n_in):
        cur = cv2.imread(src[j], cv2.IMREAD_COLOR)
        mid = dis_midpoint(prev, cur, dis)
        cv2.imwrite(os.path.join(out_dir, f"{2*j-1:05d}.png"), mid)
        copy_frame(src[j], os.path.join(out_dir, f"{2*j:05d}.png"))
        prev = cur
    dt = time.time() - t0
    log(f"dis: {2*n_in-1} frames in {dt:.1f}s ({(2*n_in-1)/dt:.2f} out-frames/s)")
    return dt


# ----------------------------------------------------------------------------
# quality check against ground truth
# ----------------------------------------------------------------------------
def evaluate(out_dir: str, gt_dir: str, n_in: int, sheet: str | None) -> dict:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    n_target = 2 * n_in - 1
    rows, repeats = [], []
    for k in range(1, n_target, 2):
        gt = cv2.imread(os.path.join(gt_dir, f"{k:05d}.png"), cv2.IMREAD_COLOR)
        pr = cv2.imread(os.path.join(out_dir, f"{k:05d}.png"), cv2.IMREAD_COLOR)
        if gt is None or pr is None:
            raise SystemExit(f"missing frame {k} in gt/out")
        psnr = peak_signal_noise_ratio(gt, pr, data_range=255)
        ssim = structural_similarity(cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY),
                                     cv2.cvtColor(pr, cv2.COLOR_BGR2GRAY), data_range=255)
        rows.append((k, psnr, ssim))
        # A source with a 25->30 fps pull-up repeats every 6th frame; such a
        # "real" odd frame is a copy of a neighbour, which no midpoint
        # interpolator should reproduce.  Flag it so motion-only stats exist.
        nb = [cv2.imread(os.path.join(gt_dir, f"{m:05d}.png"), cv2.IMREAD_COLOR) for m in (k - 1, k + 1)]
        if max(peak_signal_noise_ratio(gt, x, data_range=255) for x in nb if x is not None) > 38:
            repeats.append(k)

    def stats(rs):
        ps = np.array([r[1] for r in rs]); ss = np.array([r[2] for r in rs])
        return dict(n=len(rs), psnr_mean=float(ps.mean()), psnr_min=float(ps.min()),
                    ssim_mean=float(ss.mean()), ssim_min=float(ss.min()))
    motion = [r for r in rows if r[0] not in repeats]
    res = dict(stats(rows), repeats=len(repeats), motion=stats(motion) if motion else None,
               worst=sorted(rows, key=lambda r: r[1])[:4])
    if sheet:
        tiles = []
        for k, psnr, ssim in res["worst"]:
            gt = cv2.imread(os.path.join(gt_dir, f"{k:05d}.png"))
            pr = cv2.imread(os.path.join(out_dir, f"{k:05d}.png"))
            diff = np.clip(cv2.absdiff(gt, pr).astype(np.int32) * 4, 0, 255).astype(np.uint8)
            row = np.concatenate([gt, pr, diff], axis=1)
            strip = np.full((32, row.shape[1], 3), 30, np.uint8)
            cv2.putText(strip, f"frame {k}   PSNR {psnr:.2f} dB   SSIM {ssim:.3f}      real | interpolated | absdiff x4",
                        (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(np.concatenate([strip, row], axis=0))
        cv2.imwrite(sheet, np.concatenate(tiles, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return res


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_dir", required=True, help="dir of NNNNN.png (0-based, sequential)")
    ap.add_argument("--out", dest="out_dir", required=True)
    ap.add_argument("--n-out", type=int, default=None, help="exact output count (default 2*n_in-1; pads by duplicating last frame)")
    ap.add_argument("--method", choices=["rife", "minterpolate", "dis"], default="rife")
    ap.add_argument("--rife-bin", default=None, help="path to rife-ncnn-vulkan.exe")
    ap.add_argument("--model", default="rife-v4.6", help="rife model dir name (rife-v4.6 default)")
    ap.add_argument("--gpu", type=int, default=0, help="vulkan gpu id (-1 = cpu)")
    ap.add_argument("--rife-threads", default="2:1:2",
                    help="rife -j load:proc:save. proc MUST stay 1 on Intel Arc (Vulkan driver 101.8132): "
                         "proc>=2 silently emits black frames")
    ap.add_argument("--tta", action="store_true", help="rife TTA (-x), slower, slightly better")
    ap.add_argument("--cut-thresh", type=float, default=20.0,
                    help="shot-cut detector threshold (mean abs gray diff, 0..255) on the input pairs; "
                         "0 disables. At a cut the previous frame is held instead of a blended ghost")
    ap.add_argument("--gt", default=None, help="ground-truth dir (all frames) for PSNR/SSIM check")
    ap.add_argument("--sheet", default=None, help="contact sheet jpg of the 4 worst frames (with --gt)")
    args = ap.parse_args()

    src = list_frames(args.in_dir)
    n_in = len(src)
    n_target = 2 * n_in - 1
    n_out = args.n_out or n_target
    if n_out < n_target:
        log(f"warning: --n-out {n_out} < 2*n_in-1 = {n_target}; trimming")

    if os.path.isdir(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)

    runner = dict(rife=run_rife, minterpolate=run_minterpolate, dis=run_dis)[args.method]
    dt = runner(args.in_dir, args.out_dir, n_in, args)
    cuts = detect_cuts(src, args.cut_thresh)
    if cuts:
        hold_at_cuts(args.in_dir, args.out_dir, cuts)
        log("shot cuts, odd frame duplicated from next input (out<-in): " + ", ".join(f"{2*j+1}<-{j+1}" for j in cuts))
    fix_count(args.out_dir, n_out)
    log(f"{args.method}: {n_in} in -> {n_out} out ({n_target} native, {n_out - n_target:+d} pad) in {dt:.1f}s")

    if args.gt:
        res = evaluate(args.out_dir, args.gt, n_in, args.sheet)
        log(f"{args.method} vs gt: n={res['n']}  PSNR mean {res['psnr_mean']:.2f} min {res['psnr_min']:.2f} dB  "
            f"SSIM mean {res['ssim_mean']:.4f} min {res['ssim_min']:.4f}")
        if res["motion"]:
            m = res["motion"]
            log(f"{args.method} vs gt, motion frames only (excl. {res['repeats']} pull-up repeats): n={m['n']}  "
                f"PSNR mean {m['psnr_mean']:.2f} min {m['psnr_min']:.2f} dB  "
                f"SSIM mean {m['ssim_mean']:.4f} min {m['ssim_min']:.4f}")
        log("worst frames: " + ", ".join(f"{k} ({p:.1f} dB / {s:.3f})" for k, p, s in res["worst"]))


if __name__ == "__main__":
    main()

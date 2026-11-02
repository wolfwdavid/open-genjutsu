#!/usr/bin/env python
"""deflicker.py - remove the brightness pulses a temporally tiled Wan VAE decode leaves in a render.

The Wan VAE is causal: VAEDecodeTiled with temporal_size < window length restarts its cache at
every temporal chunk, and each restart decodes brighter, then fades over ~8 frames (with
--decode-frames 16: a flash every 12 frames).  The pulse is a global per-channel tone curve
(shadows barely move, mids x1.2-1.3, highlights +30), so it is corrected per quantile:

    Q_t(p)   = p-th percentile of render frame t, per channel (p = 0..100, 33 points)
    target_t = lower envelope of Q(p) within the shot (rolling min over --win frames, then smoothed):
               the flash only ever brightens, so the envelope is the flicker-free tone curve
    render_t = piecewise-linear map  Q_t(p) -> target_t(p)   (kept monotone)

The envelope is taken on the render itself, not relative to the source: the recast characters
change the frame's histogram (black suit vs orange tracksuit), which made source-relative offsets
drift (mid residual 8.3 vs 7.4, mean step 1.31 vs 0.88; source clip 0.96).
The source clip is only used to find shot cuts (mean abs gray diff > --cut-thresh); the envelope
never crosses a cut.

Shot cuts are found in the source (mean abs gray diff > --cut-thresh) and never smoothed across.

  python deflicker.py --in work_ex/wan/user_E/frames --src work_ex/wan/cat_E/in/bg --out work_ex/wan/user_E/frames_df
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from PIL import Image


def thumb(path: str, size=(192, 144)) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize(size, Image.BILINEAR), np.float32)


def shot_ranges(src: list[np.ndarray], thresh: float) -> list[tuple[int, int]]:
    gray = [s.mean(2) for s in src]
    cuts = [j + 1 for j in range(len(gray) - 1) if np.abs(gray[j + 1] - gray[j]).mean() > thresh]
    bounds = [0] + cuts + [len(src)]
    return [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]


def envelope(r: np.ndarray, win: int) -> np.ndarray:
    """Rolling minimum (centered, clamped at the shot edges) then a box smooth of the same width."""
    n, h = len(r), win // 2
    lo = np.stack([r[max(0, t - h):min(n, t + h + 1)].min(0) for t in range(n)])
    return np.stack([lo[max(0, t - h):min(n, t + h + 1)].mean(0) for t in range(n)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_dir", required=True, help="rendered frames NNNNN.png")
    ap.add_argument("--src", required=True, help="source frames, same count and numbering (shot-cut reference)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--win", type=int, default=13, help="envelope window in frames (> flash period of 12)")
    ap.add_argument("--cut-thresh", type=float, default=25.0)
    a = ap.parse_args()

    fin = sorted(glob.glob(os.path.join(a.in_dir, "*.png")))
    fsrc = sorted(glob.glob(os.path.join(a.src, "*.png")))
    if len(fin) != len(fsrc):
        raise SystemExit(f"frame count mismatch: {len(fin)} rendered vs {len(fsrc)} source")
    os.makedirs(a.out, exist_ok=True)

    ren = [thumb(f) for f in fin]
    src = [thumb(f) for f in fsrc]
    ps = np.linspace(0, 100, 33)
    qr = np.stack([np.percentile(x.reshape(-1, 3), ps, axis=0) for x in ren])   # (T, P, 3)
    target = qr.copy()
    shots = shot_ranges(src, a.cut_thresh)
    for lo, hi in shots:
        target[lo:hi] = envelope(qr[lo:hi], a.win)
    target = np.clip(np.maximum.accumulate(target, axis=1), 0, 255)

    m = np.stack([x.reshape(-1, 3).mean(0) for x in ren])
    s = np.stack([x.reshape(-1, 3).mean(0) for x in src])
    after = np.zeros_like(m)
    for t, f in enumerate(fin):
        img = np.asarray(Image.open(f).convert("RGB"), np.float32)
        for c in range(3):
            xp = qr[t, :, c] + np.arange(len(ps)) * 1e-3          # strictly increasing for np.interp
            img[..., c] = np.interp(img[..., c], xp, target[t, :, c])
        Image.fromarray(np.clip(img + 0.5, 0, 255).astype(np.uint8)).save(os.path.join(a.out, os.path.basename(f)))
        after[t] = thumb(os.path.join(a.out, os.path.basename(f))).reshape(-1, 3).mean(0)
    step = lambda x: np.abs(np.diff(x.mean(1)))
    inner = np.ones(len(m) - 1, bool)
    for lo, _ in shots[1:]:
        inner[lo - 1] = False
    print(f"shots {[lo for lo, _ in shots]}; largest quantile shift {np.abs(target - qr).max():.1f}")
    print(f"max frame-to-frame brightness step inside shots: before {step(m)[inner].max():.1f}, "
          f"after {step(after)[inner].max():.1f} (source {step(s)[inner].max():.1f})")
    print(f"wrote {len(fin)} frames -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

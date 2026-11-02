#!/usr/bin/env python
"""
face_crops.py -- 512x512 face crops of one performer for Wan2.2-Animate's `face_video` input.

Reads work/tracks30.json (face tracks from track.py, indexed by cut frame) and work/cut.mp4,
crops a square of side --scale x face width centred slightly above the face centre (same
geometry as animate_cat.crop_driver), replicate-pads at the borders, and writes
work/face_<side>/%05d.png for every frame in --start..--end (cut frame numbers).

Usage
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe face_crops.py --side L
  ... --side R --start 104 --end 872
"""
from __future__ import annotations

import os
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
WORK = Path(os.environ["VG_WORK"]).resolve() if os.environ.get("VG_WORK") else ROOT / "work"


def crop_square(fr: np.ndarray, cx: float, cy: float, side: float, out: int) -> np.ndarray:
    h, w = fr.shape[:2]
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    s = int(round(side))
    pad = max(0, -x0, -y0, x0 + s - w, y0 + s - h)
    if pad:
        fr = cv2.copyMakeBorder(fr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x0 += pad
        y0 += pad
    c = fr[y0:y0 + s, x0:x0 + s]
    return cv2.resize(c, (out, out), interpolation=cv2.INTER_AREA if s > out else cv2.INTER_CUBIC)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", required=True, choices=["L", "R"])
    ap.add_argument("--video", default=str(WORK / "cut.mp4"))
    ap.add_argument("--tracks", default=str(WORK / "tracks30.json"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None, help="exclusive cut frame")
    ap.add_argument("--scale", type=float, default=1.6, help="crop side = scale x face width")
    ap.add_argument("--up", type=float, default=0.05, help="centre shift up, x face width")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--from-pose", action="store_true", help="face box from <WORK>/pose_body.json instead of tracks30 face tracks")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--smooth", type=float, default=2.0, help="gaussian sigma (frames) on cx,cy,w")
    a = ap.parse_args()

    if a.from_pose:
        # face box from the whole-body pose (works where a face detector cannot: a cat)
        pj = json.load(open(WORK / "pose_body.json", encoding="utf-8"))
        side = pj[a.side]
        n = len(side)
        cuts = [0] + [c for c in pj.get("cuts", []) if 0 < c < n] + [n]
        cx, cy, w = np.zeros(n), np.zeros(n), np.zeros(n)
        last = None
        for i, e in enumerate(side):
            kp = np.array(e["kps"], np.float32) if e.get("kps") else None
            val = None
            if kp is not None:
                head = kp[[0, 1, 2, 3, 4]]
                good = head[head[:, 2] >= 0.5]
                if len(good) >= 2:
                    span = float(np.ptp(good[:, 0]))
                    eyes = kp[[1, 2]]
                    iod = float(np.linalg.norm(eyes[0, :2] - eyes[1, :2])) if (eyes[:, 2] >= 0.5).all() else 0.0
                    fw = max(span * 1.25, iod * 2.6, 24.0)
                    val = (float(good[:, 0].mean()), float(good[:, 1].mean()), fw)
            if val is None:
                val = last if last is not None else (0.0, 0.0, 24.0)
            cx[i], cy[i], w[i] = val
            last = val
    else:
        tr = json.load(open(a.tracks))
        side = tr[a.side]
        n = len(side)
        cuts = [0] + [c for c in tr.get("cuts", []) if 0 < c < n] + [n]
        cx = np.array([e["cx"] for e in side], float)
        cy = np.array([e["cy"] for e in side], float)
        w = np.array([max(8.0, e["w"]) for e in side], float)
    end = n if a.end is None else min(n, a.end)
    if a.smooth > 0:
        from scipy.ndimage import gaussian_filter1d
        for i in range(len(cuts) - 1):
            s, e = cuts[i], cuts[i + 1]
            for arr in (cx, cy, w):
                arr[s:e] = gaussian_filter1d(arr[s:e], a.smooth, mode="nearest")
    out_dir = Path(a.out_dir) if a.out_dir else WORK / f"face_{a.side}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(a.video)
    f = 0
    written = 0
    while True:
        ok, fr = cap.read()
        if not ok or f >= end:
            break
        if f >= a.start:
            side_px = a.scale * w[f]
            c = crop_square(fr, cx[f], cy[f] - a.up * w[f], side_px, a.size)
            cv2.imwrite(str(out_dir / f"{f:05d}.png"), c)
            written += 1
        f += 1
    cap.release()
    print(f"wrote {written} crops -> {out_dir} (frames {a.start}..{end - 1}, side {a.side})")
    # small QA sheet
    picks = [p for p in ((0, 104, 184, 276, 346, 360) if a.from_pose else (104, 239, 300, 461, 580, 688, 800, 870)) if a.start <= p < end]
    tiles = [cv2.imread(str(out_dir / f"{p:05d}.png")) for p in picks]
    if tiles:
        sheet = np.hstack([cv2.resize(t, (256, 256)) for t in tiles])
        (WORK / "round4").mkdir(exist_ok=True)
        cv2.imwrite(str(WORK / "round4" / f"face_{a.side}_sheet.jpg"), sheet)
        print(f"sheet -> work/round4/face_{a.side}_sheet.jpg ({picks})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

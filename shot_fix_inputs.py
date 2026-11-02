#!/usr/bin/env python
"""
shot_fix_inputs.py -- tighter inputs for re-rendering one shot of a character.

  mask_<side>tight/  = (isnet matte AND the generous character mask), holes filled, dilated --grow px.
                       Follows the real silhouette, so the model cannot grow extra body into the
                       padded area the generous mask allowed.
  pose_<side>_nohands/ = the Wan-style skeleton with the 42 hand points dropped (finger points make the
                       model paint smeared human-like hands on a cat's paws).

Usage
  VG_WORK=work_ex PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe shot_fix_inputs.py --side L --start 104 --end 184
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
WORK = Path(os.environ["VG_WORK"]).resolve() if os.environ.get("VG_WORK") else ROOT / "work"
sys.path.insert(0, str(ROOT))
from make_fur_mask import fill_holes  # noqa: E402
from pose_body import draw_wan_pose  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", default="L")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--grow", type=int, default=6)
    ap.add_argument("--thr", type=float, default=0.5)
    a = ap.parse_args()
    pj = json.load(open(WORK / "pose_body.json", encoding="utf-8"))
    seq = pj[a.side]
    W, H = pj["width"], pj["height"]
    end = len(seq) if a.end is None else min(len(seq), a.end)
    md, pd = WORK / f"mask_{a.side}tight", WORK / f"pose_{a.side}_nohands"
    md.mkdir(exist_ok=True)
    pd.mkdir(exist_ok=True)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.grow + 1,) * 2)
    areas = []
    for f in range(len(seq)):
        char = cv2.imread(str(WORK / f"mask_{a.side}" / f"{f:05d}.png"), cv2.IMREAD_GRAYSCALE)
        if a.start <= f < end:
            matte = cv2.imread(str(WORK / "matte_isnet" / f"{f:05d}.png"), cv2.IMREAD_GRAYSCALE)
            m = ((matte > 127) & (char > 127)).astype(np.uint8)
            n, lab, st, _ = cv2.connectedComponentsWithStats(m)
            if n > 1:
                big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
                keep = np.zeros_like(m)
                for i in range(1, n):
                    if st[i, cv2.CC_STAT_AREA] >= 0.05 * st[big, cv2.CC_STAT_AREA]:
                        keep[lab == i] = 1
                m = keep
            fur = cv2.imread(str(WORK / f"mask_{a.side}fur" / f"{f:05d}.png"), cv2.IMREAD_GRAYSCALE)
            if fur is not None:
                m |= (fur > 127).astype(np.uint8)          # never leave grey fur outside the mask
            m = cv2.dilate(fill_holes(m), k)
            areas.append(int(m.sum()))
            cv2.imwrite(str(md / f"{f:05d}.png"), m * 255)
        else:
            cv2.imwrite(str(md / f"{f:05d}.png"), char)
        e = seq[f]
        canvas = np.zeros((H, W, 3), np.uint8)
        if e.get("kps"):
            kp = np.array(e["kps"], np.float32)
            kp[91:133, 2] = 0.0
            draw_wan_pose(canvas, kp, a.thr, "v2", False)
        cv2.imwrite(str(pd / f"{f:05d}.png"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"tight masks {a.start}..{end - 1}: area px min {min(areas)} median {int(np.median(areas))} max {max(areas)} -> {md}")
    print(f"hand-free skeletons -> {pd}")
    # QA: generous vs tight on the shot
    cap = cv2.VideoCapture(str(WORK / "cut.mp4"))
    tiles = []
    for f in np.linspace(a.start, end - 1, 5).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, fr = cap.read()
        g = cv2.imread(str(WORK / f"mask_{a.side}" / f"{f:05d}.png"), cv2.IMREAD_GRAYSCALE) > 127
        t = cv2.imread(str(md / f"{f:05d}.png"), cv2.IMREAD_GRAYSCALE) > 127
        vis = fr.copy()
        vis[g & ~t] = (0.5 * vis[g & ~t] + np.array([140, 0, 0])).astype(np.uint8)
        vis[t] = (0.5 * vis[t] + np.array([0, 0, 140])).astype(np.uint8)
        tiles.append(cv2.resize(vis, (480, 360)))
    cv2.imwrite(str(WORK / "round4" / "tight_mask_sheet.jpg"), np.hstack(tiles))
    print(f"sheet (red = tight mask, blue = dropped padding) -> {WORK / 'round4' / 'tight_mask_sheet.jpg'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

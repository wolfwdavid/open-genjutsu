#!/usr/bin/env python
"""
make_fur_mask.py -- mask ONLY the cat's fur (and glasses) inside the cat's character mask, so a
Wan2.2-Animate replacement repaints the fur as the user's cat and the glasses as black
sunglasses while the costume (cap, plaid flannel) and the stool stay pixel-exact from the source.

  fur      = character mask (work*/mask_<side>) AND low saturation (grey fur, light paws)
  glasses  = ellipse over the eyes from the pose keypoints (eyes/nose/ears of pose_body.json),
             so the model is free to paint new sunglasses there
  cleanup  = open (drop specks) -> keep components > --min-area -> close (--close px) -> fill holes

Writes <WORK>/mask_<side>fur/%05d.png and <WORK>/round4/fur_sheet.jpg.

Usage
  VG_WORK=work_ex PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe make_fur_mask.py --side L
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
NOSE, LEYE, REYE, LEAR, REAR = 0, 1, 2, 3, 4


def fill_holes(m: np.ndarray) -> np.ndarray:
    h, w = m.shape
    ff = m.copy()
    msk = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, msk, (0, 0), 1)
    return m | (1 - ff)


def glasses_ellipse(kp: np.ndarray, shape, thr: float, scale: float) -> np.ndarray:
    out = np.zeros(shape, np.uint8)
    ok = kp[:, 2] >= thr
    if not (ok[LEYE] and ok[REYE]):
        return out
    le, re = kp[LEYE, :2], kp[REYE, :2]
    c = (le + re) / 2
    iod = float(np.linalg.norm(le - re))
    if iod < 4:
        return out
    ang = float(np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0])))
    cv2.ellipse(out, (int(c[0]), int(c[1])), (int(iod * scale), int(iod * scale * 0.5)), ang, 0, 360, 1, -1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", default="L", choices=["L", "R"])
    ap.add_argument("--video", default=str(WORK / "cut.mp4"))
    ap.add_argument("--sat", type=int, default=85, help="fur = saturation below this (HSV 0-255)")
    ap.add_argument("--val-min", type=int, default=40)
    ap.add_argument("--tail-sat", type=int, default=140, help="looser saturation limit below the hips (blurred tail)")
    ap.add_argument("--min-area", type=int, default=600)
    ap.add_argument("--close", type=int, default=13)
    ap.add_argument("--glasses-scale", type=float, default=1.05, help="ellipse half-width, x inter-ocular distance")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--mic-gap", type=float, default=0.9, help="drop grey above the shoulders beyond the far eye + gap x iod")
    ap.add_argument("--sheet-frames", default="0,104,184,276,346,360")
    a = ap.parse_args()

    pose = json.load(open(WORK / "pose_body.json", encoding="utf-8"))
    seq = pose[a.side]
    mdir = WORK / f"mask_{a.side}"
    out = WORK / f"mask_{a.side}fur"
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(a.video)
    sheet_f = [int(v) for v in a.sheet_frames.split(",")]
    tiles, f, areas = [], 0, []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        char = cv2.imread(str(mdir / f"{f:05d}.png"), cv2.IMREAD_GRAYSCALE)
        if char is None:
            sys.exit(f"missing {mdir / f'{f:05d}.png'}; run pose_body.py first")
        char = (char > 127).astype(np.uint8)
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        fur = ((hsv[..., 1] < a.sat) & (hsv[..., 2] > a.val_min)).astype(np.uint8) & char
        # the tail is motion-blurred into the orange backdrop: below the hips accept a looser
        # saturation, but only where the hue is not the wood/backdrop orange-yellow band
        kp0 = np.array(seq[f]["kps"], np.float32) if seq[f].get("kps") else None
        if kp0 is not None and (kp0[[11, 12], 2] >= a.thr).any():
            hip_y = int(np.mean([kp0[i, 1] for i in (11, 12) if kp0[i, 2] >= a.thr]))
            H_, S_, V_ = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
            tail = (S_ < a.tail_sat) & (V_ > a.val_min) & ~((H_ >= 5) & (H_ <= 30) & (S_ > a.sat))
            tail[:hip_y] = False
            fur |= tail.astype(np.uint8) & char
        fur = cv2.morphologyEx(fur, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(fur)
        keep = np.zeros_like(fur)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] >= a.min_area:
                keep[lab == i] = 1
        kp = np.array(seq[f]["kps"], np.float32) if seq[f].get("kps") else None
        if kp is not None:
            # the hanging mic is grey metal next to the head: drop grey pixels above the shoulders
            # that lie beyond the far side of the face
            ok = kp[:, 2] >= a.thr
            if ok[LEYE] and ok[REYE] and (ok[5] or ok[6]):
                iod = float(np.linalg.norm(kp[LEYE, :2] - kp[REYE, :2]))
                sh_y = float(np.mean([kp[i, 1] for i in (5, 6) if ok[i]]))
                x_far = float(max(kp[LEYE, 0], kp[REYE, 0])) + a.mic_gap * iod
                keep[: int(sh_y), int(x_far):] = 0
            keep |= glasses_ellipse(kp, keep.shape, a.thr, a.glasses_scale)
        keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (a.close, a.close)))
        keep = fill_holes(keep) & cv2.dilate(char, np.ones((9, 9), np.uint8))
        cv2.imwrite(str(out / f"{f:05d}.png"), keep * 255)
        areas.append(int(keep.sum()))
        if f in sheet_f:
            vis = fr.copy()
            vis[keep > 0] = (0.45 * vis[keep > 0] + np.array([0, 0, 140])).astype(np.uint8)
            tiles.append(cv2.resize(np.hstack([fr, vis]), None, fx=0.5, fy=0.5))
        f += 1
    print(f"wrote {f} fur masks -> {out}; area px min {min(areas)} median {int(np.median(areas))} max {max(areas)}")
    if tiles:
        (WORK / "round4").mkdir(exist_ok=True)
        cv2.imwrite(str(WORK / "round4" / "fur_sheet.jpg"), np.vstack(tiles))
        print(f"sheet -> {WORK / 'round4' / 'fur_sheet.jpg'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

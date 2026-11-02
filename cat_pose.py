#!/usr/bin/env python
"""
cat_pose.py -- retarget the RIGHT performer's whole-body pose to seated-cat proportions and
render it in the Wan-Animate skeleton style, so Wan2.2-Animate draws a cat instead of a human.

Per frame (cut frame index), from work/pose_body.json:
  * head: nose/eyes/ears kept, but the head is scaled about the neck by --head-scale (cats have
    big heads relative to the shoulders) and the ears are pushed up (pointed cat ears).
  * shoulders pulled in toward the neck by --shoulder-scale (narrow cat chest).
  * forelegs: shoulder->elbow->wrist chain shortened by --arm-scale, keeping each joint's
    direction, so the paws still follow the rapper's hands but stay close to the chest.
  * hands (42 finger points) dropped: fingers are what makes the model paint human hands.
  * hips moved up toward the shoulders (--torso-scale, short cat torso) and legs replaced by a
    short folded haunch stub (seated), knees/ankles dropped.
The retargeted 133-kp array is drawn with pose_body.draw_wan_pose (hands have conf 0).

Writes work/pose_<side>_cat/%05d.png (same file names as work/pose_<side>/) and a QA sheet
work/round4/cat_pose_sheet.jpg. wan_animate.py uses it via --pose-dir.

Usage
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe cat_pose.py [--side R --start 104 --end 872]
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
sys.path.insert(0, str(ROOT))
from pose_body import draw_wan_pose  # noqa: E402

# COCO-WholeBody body indices
NOSE, LEYE, REYE, LEAR, REAR = 0, 1, 2, 3, 4
LSH, RSH, LEL, REL, LWR, RWR = 5, 6, 7, 8, 9, 10
LHIP, RHIP, LKNE, RKNE, LANK, RANK = 11, 12, 13, 14, 15, 16
FEET = list(range(17, 23))
HANDS = list(range(91, 133))


def retarget(kp: np.ndarray, a) -> np.ndarray:
    k = kp.copy()
    ok = k[:, 2] >= a.thr
    if not (ok[LSH] and ok[RSH]):
        return None
    neck = (k[LSH, :2] + k[RSH, :2]) / 2
    sh_w = np.linalg.norm(k[LSH, :2] - k[RSH, :2])
    # head: scale about the neck, ears up
    for i in [NOSE, LEYE, REYE, LEAR, REAR] + list(range(23, 91)):
        if ok[i]:
            k[i, :2] = neck + (k[i, :2] - neck) * a.head_scale
    for i in (LEAR, REAR):
        if ok[i]:
            k[i, 1] -= a.ear_up * sh_w
    # shoulders in
    for i in (LSH, RSH):
        k[i, :2] = neck + (k[i, :2] - neck) * a.shoulder_scale
    # forelegs: shorten each segment, keep directions
    for sh, el, wr in ((LSH, LEL, LWR), (RSH, REL, RWR)):
        if ok[el]:
            d1 = kp[el, :2] - kp[sh, :2]
            k[el, :2] = k[sh, :2] + d1 * a.arm_scale
            if ok[wr]:
                d2 = kp[wr, :2] - kp[el, :2]
                k[wr, :2] = k[el, :2] + d2 * a.arm_scale
        elif ok[wr]:
            k[wr, :2] = k[sh, :2] + (kp[wr, :2] - kp[sh, :2]) * a.arm_scale
    # cap each foreleg's reach (shoulder->paw) so wide arm spreads stay cat-sized
    for sh, el, wr in ((LSH, LEL, LWR), (RSH, REL, RWR)):
        for j in (el, wr):
            if k[j, 2] >= a.thr:
                d = k[j, :2] - k[sh, :2]
                r = float(np.linalg.norm(d))
                lim = a.max_reach * sh_w * (0.55 if j == el else 1.0)
                if r > lim:
                    k[j, :2] = k[sh, :2] + d * (lim / r)
    # torso: hips up toward the shoulders
    for h, s in ((LHIP, LSH), (RHIP, RSH)):
        if ok[h]:
            k[h, :2] = kp[s, :2] + (kp[h, :2] - kp[s, :2]) * a.torso_scale
            k[h, 0] = neck[0] + (k[h, 0] - neck[0]) * a.hip_width
        else:
            # hips out of frame (close-ups): synthesise below the shoulders
            k[h, :2] = k[s, :2] + np.array([0.0, a.torso_scale * 1.6 * sh_w])
            k[h, 2] = 1.0
    # seated haunches: short stubs out and down from the hips, no ankles/feet
    for h, kn, sgn in ((LHIP, LKNE, 1.0), (RHIP, RKNE, -1.0)):
        side = 1.0 if k[h, 0] >= neck[0] else -1.0
        k[kn, :2] = k[h, :2] + np.array([side * a.haunch * sh_w, 0.35 * a.haunch * sh_w])
        k[kn, 2] = 1.0
    for i in [LANK, RANK] + FEET + HANDS:
        k[i, 2] = 0.0
    return k


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", default="R", choices=["L", "R"])
    ap.add_argument("--pose-json", default=str(WORK / "pose_body.json"))
    ap.add_argument("--start", type=int, default=104)
    ap.add_argument("--end", type=int, default=872)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--head-scale", type=float, default=1.25)
    ap.add_argument("--ear-up", type=float, default=0.18, help="x shoulder width")
    ap.add_argument("--shoulder-scale", type=float, default=0.72)
    ap.add_argument("--arm-scale", type=float, default=0.62)
    ap.add_argument("--torso-scale", type=float, default=0.70)
    ap.add_argument("--hip-width", type=float, default=1.15)
    ap.add_argument("--haunch", type=float, default=0.45, help="x shoulder width")
    ap.add_argument("--max-reach", type=float, default=0.95, help="paw reach cap, x shoulder width")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    data = json.load(open(a.pose_json))
    W, H = data["width"], data["height"]
    seq = data[a.side]
    out = Path(a.out_dir) if a.out_dir else WORK / f"pose_{a.side}_cat"
    out.mkdir(parents=True, exist_ok=True)
    last = None
    miss = 0
    for f in range(a.start, min(a.end, len(seq))):
        e = seq[f]
        kp = np.array(e["kps"], np.float32) if e.get("kps") else None
        k = retarget(kp, a) if kp is not None else None
        if k is None:
            k = last
            miss += 1
        canvas = np.zeros((H, W, 3), np.uint8)
        if k is not None:
            draw_wan_pose(canvas, k, a.thr, "v2", False)
            last = k
        cv2.imwrite(str(out / f"{f:05d}.png"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"wrote {a.end - a.start} cat skeletons -> {out} (held {miss})")
    # QA sheet: original skeleton | cat skeleton over the frame
    cap = cv2.VideoCapture(str(WORK / "cut.mp4"))
    rows = []
    for f in (150, 300, 400, 500, 750, 870):
        if not a.start <= f < a.end:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        orig = cv2.imread(str(WORK / f"pose_{a.side}" / f"{f:05d}.png"))
        cat = cv2.imread(str(out / f"{f:05d}.png"))
        over = cv2.addWeighted(fr, 0.5, cat, 1.0, 0)
        rows.append(np.hstack([cv2.addWeighted(fr, 0.5, orig, 1.0, 0), over]))
    if rows:
        sheet = np.vstack(rows)
        sheet = cv2.resize(sheet, (sheet.shape[1] // 2, sheet.shape[0] // 2))
        cv2.imwrite(str(WORK / "round4" / "cat_pose_sheet.jpg"), sheet)
        print("sheet -> work/round4/cat_pose_sheet.jpg")
    return 0


if __name__ == "__main__":
    sys.exit(main())

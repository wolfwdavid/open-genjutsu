#!/usr/bin/env python
"""
lip_pass.py -- make the rendered cat's mouth move with the vocals (LivePortrait animals, torch XPU).

calibrate:  the mouth-open direction in LivePortrait's expression space is measured on the user's
            own cat: exp(open-mouth photos) - exp(closed-mouth hero crop), kept only on the implicit
            keypoints that sit in the lower-central face (the mouth), zero elsewhere. Saved to
            <WORK>/lip_dir.npy. A test sheet renders one video frame's head at several strengths.
run:        for every frame of --frames (0-based PNGs, e.g. the 24 fps cat render):
              head crop (square around the eyes/nose from <WORK>/pose_body.json, side --crop x face)
              -> LivePortrait source = that crop; x_d = x_s + scale_s * m(t) * strength * lip_dir
              -> warp/decode -> paste back ONLY a feathered ellipse around the mouth
            m(t) comes from <WORK>/mouth.json (mouth_curve.py on the isolated vocals).
            Writes --out dir with the same file names, plus a side-by-side test mp4 if --test.

Usage (run with .venv-lp, the torch-XPU LivePortrait venv):
  VG_WORK=work_ex .venv-lp/Scripts/python.exe lip_pass.py calibrate --frames work_ex/round4/cat_E24
  VG_WORK=work_ex .venv-lp/Scripts/python.exe lip_pass.py run --frames work_ex/round4/cat_E24 \
      --out work_ex/round4/cat_E24_lips --strength 1.0 [--start 0 --end 72 --test]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
WORK = Path(os.environ["VG_WORK"]).resolve() if os.environ.get("VG_WORK") else ROOT / "work"
sys.path.insert(0, str(ROOT / "vendor" / "LivePortrait"))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from animate_cat import build_wrapper  # noqa: E402

# (photo, crop centre x, y, side) in ORIGINAL pixels; measured on work_ex/round4/cat_photos_grid.jpg
OPEN_PHOTOS = [("faces/cat/IMG_9321.PNG", 402, 540, 760), ("faces/cat/IMG_9322.PNG", 638, 827, 830)]
CLOSED_CROP = "refs/cat_src_512.png"   # the hero head, already a LivePortrait-style crop, mouth closed
NOSE, LEYE, REYE, LEAR, REAR = 0, 1, 2, 3, 4


def crop(img: np.ndarray, cx: float, cy: float, side: float, size: int) -> tuple[np.ndarray, tuple]:
    s = int(round(side))
    x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2))
    pad = max(0, -x0, -y0, x0 + s - img.shape[1], y0 + s - img.shape[0])
    src = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE) if pad else img
    c = src[y0 + pad:y0 + pad + s, x0 + pad:x0 + pad + s]
    return cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA if s > size else cv2.INTER_CUBIC), (x0, y0, s)


def to_t(img_bgr: np.ndarray, dev) -> torch.Tensor:
    x = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(x).permute(2, 0, 1)[None].to(dev)


@torch.no_grad()
def kp_info(w, img256: np.ndarray):
    return w.get_kp_info(to_t(img256, w.device))


def head_boxes(n: int, fw_scale: float, frame_hw) -> list:
    """Per frame (cx, cy, face_w) in the frame's pixel space, from pose_body.json (source px)."""
    pj = json.load(open(WORK / "pose_body.json", encoding="utf-8"))
    sx = frame_hw[1] / pj["width"]
    sy = frame_hw[0] / pj["height"]
    seq, out, last = pj["L"], [], None
    stride = max(1, round(len(seq) / n)) if n < len(seq) * 0.75 else 1
    for i in range(n):
        e = seq[min(len(seq) - 1, i * stride)]
        kp = np.array(e["kps"], np.float32) if e.get("kps") else None
        val = None
        if kp is not None:
            ok = kp[:5, 2] >= 0.5
            if ok[LEYE] and ok[REYE]:
                le, re = kp[LEYE, :2], kp[REYE, :2]
                iod = float(np.linalg.norm(le - re))
                c = (le + re) / 2
                if ok[NOSE]:
                    c = 0.5 * c + 0.5 * kp[NOSE, :2]
                val = (c[0] * sx, c[1] * sy, iod * 2.6 * sx)
        if val is None:
            val = last
        out.append(val)
        last = val
    # fill leading gaps, smooth lightly
    first = next(v for v in out if v is not None)
    out = [v if v is not None else first for v in out]
    arr = np.array(out, np.float32)
    from scipy.ndimage import gaussian_filter1d
    for k in range(3):
        arr[:, k] = gaussian_filter1d(arr[:, k], 1.0, mode="nearest")
    return arr


def mouth_mask(size: int, cx: float, cy: float, rx: float, ry: float, feather: float) -> np.ndarray:
    m = np.zeros((size, size), np.float32)
    cv2.ellipse(m, (int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360, 1.0, -1)
    return cv2.GaussianBlur(m, (0, 0), feather)


def calibrate(a, w) -> int:
    dev = w.device
    closed = cv2.resize(cv2.imread(str(ROOT / CLOSED_CROP)), (256, 256), interpolation=cv2.INTER_AREA)
    ic = kp_info(w, closed)
    deltas = []
    crops = [closed]
    for path, cx, cy, side in OPEN_PHOTOS:
        img = cv2.imread(str(ROOT / path))
        c, _ = crop(img, cx, cy, side, 256)
        crops.append(c)
        io = kp_info(w, c)
        deltas.append((io["exp"] - ic["exp"]).cpu().numpy()[0])      # 21 x 3
    d = np.mean(deltas, axis=0)
    # keypoint positions in the closed source (canonical, x,y in [-1,1]-ish): keep lower-central ones
    kc = ic["kp"].cpu().numpy()[0]                                  # 21 x 3
    mag = np.linalg.norm(d, axis=1)
    y_nose = float(np.median(kc[:, 1]))
    keep = (kc[:, 1] > y_nose) & (np.abs(kc[:, 0]) < 0.35)          # below mid-face, central
    order = np.argsort(-mag)
    print("keypoint | canon x,y | delta |mag| | kept")
    for i in order:
        print(f"  {i:2d}  {kc[i,0]:+.2f},{kc[i,1]:+.2f}   {mag[i]:.4f}   {'Y' if keep[i] else ''}")
    lip = np.zeros_like(d)
    lip[keep] = d[keep]
    if not keep.any():
        print("no keypoint passed the lower-central test; keeping the top 6 by magnitude")
        lip[order[:6]] = d[order[:6]]
    np.save(WORK / "lip_dir.npy", lip)
    print(f"saved {WORK / 'lip_dir.npy'} ({int((np.abs(lip).sum(1) > 0).sum())} keypoints)")
    # test sheet on one video frame
    frames = sorted(Path(a.frames).glob("*.png"))
    fr = cv2.imread(str(frames[a.test_frame]))
    boxes = head_boxes(len(frames), 2.6, fr.shape[:2])
    cx, cy, fw = boxes[a.test_frame]
    head, box = crop(fr, cx, cy + a.down * fw, fw * a.crop, 256)
    x_s_info = kp_info(w, head)
    f_s = w.extract_feature_3d(to_t(head, dev))
    x_s = w.transform_keypoint(x_s_info)
    lip_t = torch.from_numpy(lip)[None].to(dev).float()
    tiles = [cv2.resize(c, (256, 256)) for c in crops]
    for sgain in a.gains:
        x_d = x_s + x_s_info["scale"][:, :, None] * sgain * lip_t if x_s_info["scale"].dim() == 2 else x_s + x_s_info["scale"] * sgain * lip_t
        out = w.parse_output(w.warp_decode(f_s, x_s, x_d)["out"])[0]
        t = cv2.resize(cv2.cvtColor(out, cv2.COLOR_RGB2BGR), (256, 256))
        cv2.putText(t, f"x{sgain}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        tiles.append(t)
    sheet = np.hstack(tiles)
    cv2.imwrite(str(WORK / "round4" / "lip_calib.jpg"), sheet)
    print(f"sheet -> {WORK / 'round4' / 'lip_calib.jpg'} (closed ref | open photos | frame {a.test_frame} at gains {a.gains})")
    return 0


def run(a, w) -> int:
    dev = w.device
    lip = torch.from_numpy(np.load(WORK / "lip_dir.npy"))[None].to(dev).float()
    mouth = np.array(json.load(open(WORK / "mouth.json"))["mouth"], np.float32)
    frames = sorted(Path(a.frames).glob("*.png"))
    n = len(frames)
    if len(mouth) != n:
        sys.exit(f"mouth.json has {len(mouth)} values but {n} frames")
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fr0 = cv2.imread(str(frames[0]))
    boxes = head_boxes(n, 2.6, fr0.shape[:2])
    end = n if a.end is None else min(n, a.end)
    size = 512
    mm = mouth_mask(size, size * 0.5, size * a.mouth_y, size * a.mouth_rx, size * a.mouth_ry, size * 0.035)
    t0 = time.time()
    test = []
    for i in range(a.start, end):
        fr = cv2.imread(str(frames[i]))
        cx, cy, fw = boxes[i]
        side = fw * a.crop
        head256, (x0, y0, s) = crop(fr, cx, cy + a.down * fw, side, 256)
        g = float(mouth[i]) * a.strength
        res = fr.copy()
        if g > 0.02:
            with torch.no_grad():
                info = kp_info(w, head256)
                f_s = w.extract_feature_3d(to_t(head256, dev))
                x_s = w.transform_keypoint(info)
                sc = info["scale"]
                sc = sc[:, :, None] if sc.dim() == 2 else sc
                x_d = x_s + sc * g * lip
                out = w.parse_output(w.warp_decode(f_s, x_s, x_d)["out"])[0]
                ref = w.parse_output(w.warp_decode(f_s, x_s, x_s)["out"])[0]
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR).astype(np.float32)
            ref = cv2.cvtColor(ref, cv2.COLOR_RGB2BGR).astype(np.float32)
            # paste only the CHANGE (open - closed reconstruction) onto the original head:
            # keeps the original sharpness/colour, adds just the mouth motion
            head_hi, _ = crop(fr, cx, cy + a.down * fw, side, size)
            delta = (out - ref) * mm[..., None]
            new_head = np.clip(head_hi.astype(np.float32) + delta, 0, 255).astype(np.uint8)
            # write back into the frame at the crop box (handle borders)
            back = cv2.resize(new_head, (s, s), interpolation=cv2.INTER_AREA)
            mback = cv2.resize(mm, (s, s))[..., None]
            X0, Y0 = max(0, x0), max(0, y0)
            X1, Y1 = min(fr.shape[1], x0 + s), min(fr.shape[0], y0 + s)
            sub = back[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0].astype(np.float32)
            msub = mback[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
            res[Y0:Y1, X0:X1] = (res[Y0:Y1, X0:X1] * (1 - msub) + sub * msub).astype(np.uint8)
        cv2.imwrite(str(out_dir / frames[i].name), res)
        if a.test:
            test.append(np.hstack([fr, res]))
        if (i - a.start) % 24 == 0:
            print(f"  {i}/{end}  mouth {mouth[i]:.2f}  {(time.time() - t0) / max(1, i - a.start + 1):.2f} s/frame", flush=True)
    print(f"done {end - a.start} frames -> {out_dir} in {time.time() - t0:.0f} s")
    if test:
        h, w_ = test[0].shape[:2]
        p = WORK / "round4" / "lip_test.mp4"
        vw = cv2.VideoWriter(str(p) + ".tmp.mp4", cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w_, h))
        for t in test:
            vw.write(t)
        vw.release()
        print(f"side-by-side -> {p}.tmp.mp4")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["calibrate", "run"])
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--crop", type=float, default=1.9, help="head crop side = crop x face width")
    ap.add_argument("--down", type=float, default=0.18, help="crop centre shift down, x face width (towards the mouth)")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--gains", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--test-frame", type=int, default=24)
    ap.add_argument("--mouth-y", type=float, default=0.66, help="mouth ellipse centre y (fraction of crop)")
    ap.add_argument("--mouth-rx", type=float, default=0.17)
    ap.add_argument("--mouth-ry", type=float, default=0.13)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    dev = a.device if (a.device != "xpu" or torch.xpu.is_available()) else "cpu"
    w = build_wrapper(dev, half=False)
    if a.mode == "calibrate":
        return calibrate(a, w)
    if not a.out:
        sys.exit("--out required for run")
    return run(a, w)


if __name__ == "__main__":
    sys.exit(main())

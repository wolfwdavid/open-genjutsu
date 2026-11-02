#!/usr/bin/env python
"""
Per-frame 30 fps face tracks for the two performers in work/cut.mp4  (contract A).

Pipeline
  1. buffalo_l (DirectML) on every frame: detection @640, re-detect @1280 when
     fewer than two faces are found, 106 landmarks + identity embedding per face.
  2. Learn the LEFT / RIGHT identities from exactly-two-face frames (labelled by
     x-order, then re-labelled once by identity), assign every detection to
     L or R by embedding similarity with an x-position fallback (swap.py logic).
  3. Head pose (yaw / pitch / roll, degrees) = mean of two estimators that were
     cross-checked on this footage: solvePnP on the 5 keypoints against a generic
     3D template, and buffalo_l's learned 3D-68 landmark pose (both raw values
     are kept per frame under "pose_raw").  yaw < 0 = face turned toward
     image-left, pitch > 0 = chin up, roll > 0 = head tilted clockwise on screen.
  4. One-Euro smoothing of cx, cy, w, h, yaw, pitch, roll and the 5 kps
     (eye_c / iod / bbox derive from the smoothed values); state reset at every
     shot cut and after every detection gap.  lmk106 is lightly smoothed.
     Missing frames carry the last good values with present=false, held=true.

Outputs
  work/tracks30.json        contract (A)
  work/track_overlay.mp4    boxes / eye line / pose text for both performers
  work/track_sheet.jpg      6 overlay frames at 1, 5, 12, 17, 21.8, 27 s

Usage
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe track.py
  ... --min-cutoff 1.0 --beta 0.02        # smoothing tune
  ... --no-video                          # json only (skip overlay + sheet)
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.app.common import Face

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
PROVIDERS = ["DmlExecutionProvider", "CPUExecutionProvider"]
DEFAULT_CUTS = [104, 239, 461, 581, 688]
SHEET_TIMES = [1.0, 5.0, 12.0, 17.0, 21.8, 27.0]

# Generic 3D template for the 5 insightface keypoints in OpenCV camera convention
# (x right, y down, z away from the camera): the nose is closest to the camera.
KPS_3D = np.array(
    [
        [-30.0, -30.0, -15.0],  # left eye  (image-left)
        [30.0, -30.0, -15.0],   # right eye (image-right)
        [0.0, 0.0, -35.0],      # nose tip
        [-22.0, 30.0, -10.0],   # mouth left corner
        [22.0, 30.0, -10.0],    # mouth right corner
    ],
    dtype=np.float64,
)


# ----------------------------------------------------------------------------- detection
def build_app(det_size: int) -> FaceAnalysis:
    app = FaceAnalysis(
        name="buffalo_l",
        providers=PROVIDERS,
        allowed_modules=["detection", "landmark_2d_106", "landmark_3d_68", "recognition"],
    )
    app.prepare(ctx_id=0, det_size=(det_size, det_size))
    return app


def detect(app: FaceAnalysis, img: np.ndarray, input_size=None, min_det: float = 0.45) -> list:
    """Like app.get() but lets us re-run the detector at a larger input size."""
    bboxes, kpss = app.det_model.detect(img, input_size=input_size, max_num=0, metric="default")
    faces = []
    for i in range(bboxes.shape[0]):
        if float(bboxes[i, 4]) < min_det:
            continue
        face = Face(bbox=bboxes[i, 0:4], kps=kpss[i], det_score=float(bboxes[i, 4]))
        for name, model in app.models.items():
            if name == "detection":
                continue
            model.get(img, face)
        faces.append(face)
    return faces


# ----------------------------------------------------------------------------- pose
def rot_to_ypr(R: np.ndarray):
    """R = Rz(roll) Ry(theta) Rx(phi) decomposition -> (yaw, pitch, roll) in degrees with our signs."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy < 1e-6:
        return (float("nan"),) * 3
    phi = math.atan2(R[2, 1], R[2, 2])       # rotation about x (nose down when > 0)
    theta = math.atan2(-R[2, 0], sy)         # rotation about y (nose toward image-left when > 0)
    psi = math.atan2(R[1, 0], R[0, 0])       # rotation about z (clockwise on screen when > 0)
    return -math.degrees(theta), -math.degrees(phi), math.degrees(psi)


def fuse_pose(pnp, p3d):
    """Average the two estimators; fall back to whichever is finite."""
    a = np.array(pnp, float)
    b = np.array(p3d, float)
    if not np.all(np.isfinite(a)):
        return tuple(float(v) for v in b)
    if not np.all(np.isfinite(b)):
        return tuple(float(v) for v in a)
    return tuple(float(v) for v in (a + b) / 2)


def head_pose(kps: np.ndarray, W: int, H: int):
    """Yaw/pitch/roll in degrees from 5 keypoints via solvePnP.
    yaw < 0 : face turned toward image-left; pitch > 0 : chin up; roll > 0 : tilted clockwise on screen."""
    f = float(W)
    K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])
    ok, rvec, tvec = cv2.solvePnP(KPS_3D, kps.astype(np.float64), K, np.zeros(4), flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return (float("nan"),) * 3
    R, _ = cv2.Rodrigues(rvec)
    return rot_to_ypr(R)


def pose_selfcheck(W: int = 960, H: int = 532) -> None:
    """Project the template under known rotations and assert the sign convention."""
    f = float(W)
    K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])
    tvec = np.array([[0.0], [0.0], [900.0]])

    def project(rvec):
        pts, _ = cv2.projectPoints(KPS_3D, rvec, tvec, K, np.zeros(4))
        return pts.reshape(-1, 2)

    # turn toward image-left: rotate about +y so nose x decreases
    yaw, _, _ = head_pose(project(np.array([0.0, math.radians(25), 0.0])), W, H)
    kps = project(np.array([0.0, math.radians(25), 0.0]))
    assert kps[2, 0] < W / 2, "template check: nose should move image-left"
    assert yaw < -15, f"yaw sign wrong: {yaw}"
    # chin up: rotate about x so the nose y decreases (moves up)
    kps = project(np.array([math.radians(-25), 0.0, 0.0]))
    assert kps[2, 1] < H / 2, "template check: nose should move up"
    _, pitch, _ = head_pose(kps, W, H)
    assert pitch > 15, f"pitch sign wrong: {pitch}"
    # clockwise tilt on screen: right eye goes down
    kps = project(np.array([0.0, 0.0, math.radians(20)]))
    assert kps[1, 1] > kps[0, 1], "template check: right eye should be lower"
    _, _, roll = head_pose(kps, W, H)
    assert roll > 10, f"roll sign wrong: {roll}"
    print(f"pose self-check ok: yaw {yaw:+.1f} (turn image-left), pitch {pitch:+.1f} (chin up), roll {roll:+.1f} (clockwise)")


# ----------------------------------------------------------------------------- smoothing
class OneEuro:
    """Vectorised One-Euro filter (Casiez et al. 2012); works on scalars or ndarrays."""

    def __init__(self, freq: float, min_cutoff: float = 1.0, beta: float = 0.02, d_cutoff: float = 1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self.x_prev = None
        self.dx_prev = None

    def _alpha(self, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x.copy()
        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff)
        edx = self.dx_prev + a_d * (dx - self.dx_prev)
        cutoff = self.min_cutoff + self.beta * np.abs(edx)
        a = self._alpha(cutoff)
        xhat = self.x_prev + a * (x - self.x_prev)
        self.x_prev = xhat
        self.dx_prev = edx
        return xhat


# ----------------------------------------------------------------------------- identity
def learn_identities(frames: list[dict]):
    """Exactly-two-face frames labelled by x-order, then one identity re-labelling pass."""
    embL, embR = [], []
    for fr in frames:
        good = [f for f in fr["faces"] if f["det"] >= 0.6]
        if len(good) == 2:
            a, b = sorted(good, key=lambda f: f["cx"])
            embL.append(a["emb"])
            embR.append(b["emb"])
    if len(embL) < 3:
        sys.exit("could not find two performers in enough frames")
    Lid = np.mean(embL, 0); Lid /= np.linalg.norm(Lid)
    Rid = np.mean(embR, 0); Rid /= np.linalg.norm(Rid)
    keepL = [e for e in embL + embR if float(e @ Lid) > float(e @ Rid)]
    keepR = [e for e in embL + embR if float(e @ Lid) <= float(e @ Rid)]
    Lid = np.mean(keepL, 0); Lid /= np.linalg.norm(Lid)
    Rid = np.mean(keepR, 0); Rid /= np.linalg.norm(Rid)
    sep = 1 - float(np.dot(Lid, Rid))
    print(f"identities learned from {len(embL)} two-face frames; L/R cosine separation = {sep:.3f}")
    return Lid, Rid, len(embL), sep


def assign(faces: list[dict], Lid, Rid, W: int, last: dict, margin: float = 0.08):
    """{face_index: 'L'|'R'} by identity similarity; fallback x-position (swap.py logic).
    For a single ambiguous face the nearest recent track position is preferred over W/2."""
    if not faces:
        return {}, False
    sims = np.array([[float(f["emb"] @ Lid), float(f["emb"] @ Rid)] for f in faces])
    if len(faces) == 1:
        d = sims[0, 0] - sims[0, 1]
        if abs(d) >= margin / 2:
            return {0: "L" if d > 0 else "R"}, False
        cx = faces[0]["cx"]
        if last.get("L") is not None and last.get("R") is not None:
            role = "L" if abs(cx - last["L"]) < abs(cx - last["R"]) else "R"
        else:
            role = "L" if cx < W / 2 else "R"
        return {0: role}, True
    # best (i->L, j->R) pair over all detections by identity sum
    best, best_pair = -1e9, None
    for i in range(len(faces)):
        for j in range(len(faces)):
            if i == j:
                continue
            s = sims[i, 0] + sims[j, 1]
            if s > best:
                best, best_pair = s, (i, j)
    i, j = best_pair
    s_ij = sims[i, 0] + sims[j, 1]
    s_ji = sims[i, 1] + sims[j, 0]
    if abs(s_ij - s_ji) < margin:
        ri, rj = ("L", "R") if faces[i]["cx"] < faces[j]["cx"] else ("R", "L")
        return {i: ri, j: rj}, True
    return {i: "L", j: "R"}, False


# ----------------------------------------------------------------------------- video io
def open_writer(out_path: str, w: int, h: int, fps: float, crf: int = 20) -> subprocess.Popen:
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", str(crf), "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", out_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def draw_overlay(img: np.ndarray, f: int, t: float, rec: dict, cuts: list[int]) -> np.ndarray:
    out = img.copy()
    colours = {"L": (255, 160, 0), "R": (0, 120, 255)}  # BGR: L blue, R orange
    for who in ("L", "R"):
        r = rec[who][f]
        col = colours[who] if r["present"] else (160, 160, 160)
        x1, y1, x2, y2 = [int(round(v)) for v in r["bbox"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2 if r["present"] else 1)
        e1, e2 = r["kps"][0], r["kps"][1]
        cv2.line(out, (int(e1[0]), int(e1[1])), (int(e2[0]), int(e2[1])), (0, 255, 0), 1, cv2.LINE_AA)
        for k in range(5):
            cv2.circle(out, (int(r["kps"][k][0]), int(r["kps"][k][1])), 2, (0, 255, 255), -1, cv2.LINE_AA)
        if r["lmk106"] is not None:
            for p in r["lmk106"]:
                cv2.circle(out, (int(p[0]), int(p[1])), 1, (255, 255, 255), -1)
        label = f"{who} y{r['yaw']:+.0f} p{r['pitch']:+.0f} r{r['roll']:+.0f}" + ("" if r["present"] else " HELD")
        ty = y1 - 8 if y1 > 24 else y2 + 18
        cv2.putText(out, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    hdr = f"f {f:03d}  t {t:5.2f}s" + ("  CUT" if f in cuts else "")
    cv2.putText(out, hdr, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, hdr, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(WORK / "cut.mp4"))
    ap.add_argument("--json", default=str(WORK / "tracks30.json"))
    ap.add_argument("--overlay", default=str(WORK / "track_overlay.mp4"))
    ap.add_argument("--sheet", default=str(WORK / "track_sheet.jpg"))
    ap.add_argument("--cuts", default=",".join(map(str, DEFAULT_CUTS)))
    ap.add_argument("--det-size", type=int, default=640)
    ap.add_argument("--fallback-det-size", type=int, default=1280)
    ap.add_argument("--min-det", type=float, default=0.45)
    ap.add_argument("--min-cutoff", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--lmk-min-cutoff", type=float, default=2.5, help="light smoothing for lmk106")
    ap.add_argument("--no-video", action="store_true", help="skip overlay + sheet")
    args = ap.parse_args()
    cuts = [int(c) for c in args.cuts.split(",") if c.strip()]

    t0 = time.time()
    pose_selfcheck()
    app = build_app(args.det_size)
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{args.input}: {W}x{H} @ {fps:.2f} fps, {total} frames; cuts {cuts}")

    # ---- pass 1: detect every frame ---------------------------------------------------------
    frames: list[dict] = []
    fallback_used = fallback_helped = 0
    idx = 0
    next_report = time.time() + 15
    while True:
        ok, img = cap.read()
        if not ok:
            break
        faces = detect(app, img, min_det=args.min_det)
        if len(faces) < 2 and args.fallback_det_size > args.det_size:
            fallback_used += 1
            alt = detect(app, img, (args.fallback_det_size, args.fallback_det_size), min_det=args.min_det)
            if len(alt) > len(faces):
                fallback_helped += 1
                faces = alt
        rec = []
        for f in faces:
            x1, y1, x2, y2 = [float(v) for v in f.bbox]
            pnp = head_pose(f.kps, W, H)
            # insightface landmark_3d_68 pose is [pitch, yaw, roll]; signs match ours (verified: corr with PnP, rotation test)
            p3 = (float(f.pose[1]), float(f.pose[0]), float(f.pose[2])) if getattr(f, "pose", None) is not None else (np.nan,) * 3
            yaw, pitch, roll = fuse_pose(pnp, p3)
            rec.append(dict(pose_pnp=pnp, pose_3d68=p3,
                bbox=np.array([x1, y1, x2, y2]), cx=(x1 + x2) / 2, cy=(y1 + y2) / 2, w=x2 - x1, h=y2 - y1,
                det=float(f.det_score), kps=np.asarray(f.kps, np.float64), lmk=np.asarray(f.landmark_2d_106, np.float64),
                yaw=yaw, pitch=pitch, roll=roll, emb=f.normed_embedding.astype(np.float32),
            ))
        frames.append(dict(f=idx, t=idx / fps, faces=rec))
        if time.time() > next_report:
            done = (idx + 1) / total
            el = time.time() - t0
            print(f"  {idx+1}/{total} ({done*100:.0f}%)  {el:.0f}s elapsed, ~{el/done*(1-done):.0f}s left", flush=True)
            next_report = time.time() + 15
        idx += 1
    cap.release()
    n = len(frames)
    t_detect = time.time() - t0
    print(f"pass 1 done: {n} frames in {t_detect:.1f}s ({t_detect/n*1000:.0f} ms/frame); "
          f"fallback @{args.fallback_det_size} used on {fallback_used} frames, helped on {fallback_helped}")
    if n != total:
        print(f"WARNING: frame count {n} != container count {total}")

    # ---- identity + assignment ---------------------------------------------------------------
    Lid, Rid, n_seed, sep = learn_identities(frames)
    raw = {"L": [None] * n, "R": [None] * n}
    n_two = n_one = n_zero = n_ambig = n_side_swap = 0
    last = {"L": None, "R": None}
    for k, fr in enumerate(frames):
        if fr["f"] in cuts:
            last = {"L": None, "R": None}
        roles, ambiguous = assign(fr["faces"], Lid, Rid, W, last)
        n_ambig += ambiguous
        if len(roles) == 0:
            n_zero += 1
        elif len(roles) == 1:
            n_one += 1
        else:
            n_two += 1
        for i, role in roles.items():
            raw[role][k] = fr["faces"][i]
            last[role] = fr["faces"][i]["cx"]
        if raw["L"][k] is not None and raw["R"][k] is not None and raw["L"][k]["cx"] > raw["R"][k]["cx"]:
            n_side_swap += 1
    print(f"assignment: two-face {n_two}, one-face {n_one}, none {n_zero}, ambiguous->geometry {n_ambig}, "
          f"L right of R (identity vs geometry conflict) in {n_side_swap} frames")

    # ---- smoothing --------------------------------------------------------------------------
    rec_out = {"L": [None] * n, "R": [None] * n}
    stats = {}
    for who in ("L", "R"):
        f_pos = OneEuro(fps, args.min_cutoff, args.beta)      # cx, cy, w, h
        f_ang = OneEuro(fps, args.min_cutoff, args.beta)      # yaw, pitch, roll
        f_kps = OneEuro(fps, args.min_cutoff, args.beta)      # 5x2
        f_lmk = OneEuro(fps, args.lmk_min_cutoff, args.beta)  # 106x2
        filters = (f_pos, f_ang, f_kps, f_lmk)
        last_good = None
        prev_missing = True
        n_present = 0
        gaps, gap_len = [], 0
        raw_cx, sm_cx, raw_yaw, sm_yaw = [], [], [], []
        for k in range(n):
            r = raw[who][k]
            f = frames[k]["f"]
            if f in cuts:
                for fl in filters:
                    fl.reset()
            if r is None:
                gap_len += 1
                prev_missing = True
                if last_good is not None:
                    h = dict(last_good)
                    h.update(f=f, t=round(frames[k]["t"], 4), present=False, held=True)
                    rec_out[who][k] = h
                continue
            if gap_len:
                gaps.append(gap_len)
                gap_len = 0
            if prev_missing:
                for fl in filters:
                    fl.reset()
            prev_missing = False
            pos = f_pos([r["cx"], r["cy"], r["w"], r["h"]])
            ang = f_ang([r["yaw"], r["pitch"], r["roll"]])
            kps = f_kps(r["kps"])
            lmk = f_lmk(r["lmk"])
            cx, cy, w, h = [float(v) for v in pos]
            eye_c = (kps[0] + kps[1]) / 2
            iod = float(np.linalg.norm(kps[1] - kps[0]))
            row = dict(
                f=f, t=round(frames[k]["t"], 4), present=True, held=False, det=round(r["det"], 3),
                bbox=[round(cx - w / 2, 2), round(cy - h / 2, 2), round(cx + w / 2, 2), round(cy + h / 2, 2)],
                cx=round(cx, 2), cy=round(cy, 2), w=round(w, 2), h=round(h, 2),
                kps=[[round(float(p[0]), 2), round(float(p[1]), 2)] for p in kps],
                lmk106=[[round(float(p[0]), 1), round(float(p[1]), 1)] for p in lmk],
                yaw=round(float(ang[0]), 2), pitch=round(float(ang[1]), 2), roll=round(float(ang[2]), 2),
                eye_c=[round(float(eye_c[0]), 2), round(float(eye_c[1]), 2)], iod=round(iod, 2),
                pose_raw={"pnp": [None if not np.isfinite(v) else round(float(v), 1) for v in r["pose_pnp"]],
                          "lmk3d68": [None if not np.isfinite(v) else round(float(v), 1) for v in r["pose_3d68"]]},
            )
            rec_out[who][k] = row
            last_good = row
            n_present += 1
            raw_cx.append(r["cx"]); sm_cx.append(cx); raw_yaw.append(r["yaw"]); sm_yaw.append(float(ang[0]))
        if gap_len:
            gaps.append(gap_len)
        # leading gap (no last_good yet): back-fill from the first good frame
        first = next((k for k in range(n) if rec_out[who][k] is not None and rec_out[who][k]["present"]), None)
        if first is None:
            sys.exit(f"performer {who} never detected")
        for k in range(first):
            if rec_out[who][k] is None:
                h = dict(rec_out[who][first])
                h.update(f=frames[k]["f"], t=round(frames[k]["t"], 4), present=False, held=True)
                rec_out[who][k] = h
        # smoothness / lag metrics on consecutive present frames (within shots)
        pres = np.array([rw["present"] for rw in rec_out[who]])
        cxs_raw = np.array([raw[who][k]["cx"] if raw[who][k] is not None else np.nan for k in range(n)])
        cxs_sm = np.array([rw["cx"] for rw in rec_out[who]], float)
        yaw_raw = np.array([raw[who][k]["yaw"] if raw[who][k] is not None else np.nan for k in range(n)])
        yaw_sm = np.array([rw["yaw"] for rw in rec_out[who]], float)
        ok_pair = pres[1:] & pres[:-1] & ~np.isin(np.arange(1, n), cuts)
        d_raw = np.abs(np.diff(cxs_raw))[ok_pair]
        d_sm = np.abs(np.diff(cxs_sm))[ok_pair]
        dy_raw = np.abs(np.diff(yaw_raw))[ok_pair]
        dy_sm = np.abs(np.diff(yaw_sm))[ok_pair]
        trip = pres[2:] & pres[1:-1] & pres[:-2] & ~np.isin(np.arange(1, n - 1), cuts) & ~np.isin(np.arange(2, n), cuts)
        j_raw = np.abs(np.diff(cxs_raw, 2))[trip]
        j_sm = np.abs(np.diff(cxs_sm, 2))[trip]
        jy_raw = np.abs(np.diff(yaw_raw, 2))[trip]
        jy_sm = np.abs(np.diff(yaw_sm, 2))[trip]
        # lag estimate: on fast frames (|v| > 3 px/frame) lag ~ (raw - smoothed) / velocity
        v = np.diff(cxs_raw)
        fast = ok_pair & (np.abs(v) > 3.0)
        lag = ((cxs_raw[1:] - cxs_sm[1:]) / v)[fast]
        lag_med = float(np.median(lag)) if lag.size else float("nan")
        lag_p90 = float(np.percentile(lag, 90)) if lag.size else float("nan")
        stats[who] = dict(
            present=int(n_present), held=int(n - n_present), held_gaps=len(gaps),
            max_held_gap=int(max(gaps)) if gaps else 0,
            median_abs_dcx_raw_px=round(float(np.median(d_raw)), 3), median_abs_dcx_smooth_px=round(float(np.median(d_sm)), 3),
            median_abs_dyaw_raw_deg=round(float(np.median(dy_raw)), 3), median_abs_dyaw_smooth_deg=round(float(np.median(dy_sm)), 3),
            jitter_cx_2nd_diff_raw_px=round(float(np.median(j_raw)), 3), jitter_cx_2nd_diff_smooth_px=round(float(np.median(j_sm)), 3),
            jitter_yaw_2nd_diff_raw_deg=round(float(np.median(jy_raw)), 3), jitter_yaw_2nd_diff_smooth_deg=round(float(np.median(jy_sm)), 3),
            lag_frames_median=round(lag_med, 2), lag_frames_p90=round(lag_p90, 2), fast_frames=int(fast.sum()),
            w_median=round(float(np.median([rw["w"] for rw in rec_out[who] if rw["present"]])), 1),
            yaw_p10_med_p90=[round(float(np.percentile(yaw_sm[pres], q)), 1) for q in (10, 50, 90)],
            pitch_p10_med_p90=[round(float(np.percentile(np.array([rw['pitch'] for rw in rec_out[who]])[pres], q)), 1) for q in (10, 50, 90)],
        )
        print(f"{who}: present {n_present}/{n}, held {n - n_present} in {len(gaps)} gaps (max {stats[who]['max_held_gap']} frames); "
              f"median |dcx| raw {stats[who]['median_abs_dcx_raw_px']} -> smooth {stats[who]['median_abs_dcx_smooth_px']} px; "
              f"|dyaw| raw {stats[who]['median_abs_dyaw_raw_deg']} -> {stats[who]['median_abs_dyaw_smooth_deg']} deg; "
              f"jitter (2nd diff) cx {stats[who]['jitter_cx_2nd_diff_raw_px']} -> {stats[who]['jitter_cx_2nd_diff_smooth_px']} px, "
              f"yaw {stats[who]['jitter_yaw_2nd_diff_raw_deg']} -> {stats[who]['jitter_yaw_2nd_diff_smooth_deg']} deg; "
              f"lag on {int(fast.sum())} fast frames median {lag_med:.2f} p90 {lag_p90:.2f} frames; w median {stats[who]['w_median']} px")

    # side check on smoothed output
    both = [k for k in range(n) if rec_out["L"][k]["present"] and rec_out["R"][k]["present"]]
    side_bad = [frames[k]["f"] for k in both if rec_out["L"][k]["cx"] >= rec_out["R"][k]["cx"]]
    print(f"side check: L.cx < R.cx in {len(both) - len(side_bad)}/{len(both)} both-present frames" + (f"; VIOLATIONS at {side_bad[:20]}" if side_bad else ""))

    # held frame list (for the report)
    held_frames = {who: [rw["f"] for rw in rec_out[who] if not rw["present"]] for who in ("L", "R")}

    out = {
        "fps": float(fps), "n_frames": n, "width": W, "height": H, "cuts": cuts,
        "meta": {
            "input": str(Path(args.input).as_posix()),
            "detector": f"buffalo_l det@{args.det_size} (fallback @{args.fallback_det_size} when <2 faces), min det {args.min_det}, {PROVIDERS[0]}",
            "fallback_used_frames": fallback_used, "fallback_helped_frames": fallback_helped,
            "identity": {"seed_two_face_frames": n_seed, "cosine_separation": round(sep, 3), "two_face": n_two, "one_face": n_one,
                         "no_face": n_zero, "ambiguous_geometry_fallback": n_ambig, "identity_geometry_conflicts": n_side_swap},
            "pose": "mean of solvePnP(SQPNP) on 5 kps (generic 3D template, f=W) and buffalo_l landmark_3d_68 pose; both raw in pose_raw; "
                    "yaw<0 = turned toward image-left, pitch>0 = chin up, roll>0 = clockwise on screen",
            "smoothing": f"One-Euro min_cutoff {args.min_cutoff} Hz, beta {args.beta}, d_cutoff 1.0 at {fps:.0f} fps on cx,cy,w,h,yaw,pitch,roll,kps; "
                         f"lmk106 min_cutoff {args.lmk_min_cutoff}; bbox/eye_c/iod derived from smoothed values; reset at cuts and after every gap",
            "held": "present=false frames repeat the last good smoothed values (leading gaps back-fill from the first good frame)",
            "stats": stats, "held_frames": held_frames, "side_violations": side_bad,
            "elapsed_detect_s": round(t_detect, 1),
        },
        "L": rec_out["L"], "R": rec_out["R"],
    }
    with open(args.json, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"wrote {args.json} ({Path(args.json).stat().st_size/1e6:.1f} MB) after {time.time()-t0:.1f}s")

    if args.no_video:
        return

    # ---- pass 2: overlay video + sheet -------------------------------------------------------
    t1 = time.time()
    cap = cv2.VideoCapture(args.input)
    writer = open_writer(args.overlay, W, H, fps)
    sheet_frames = {int(round(s * fps)): None for s in SHEET_TIMES}
    k = 0
    while True:
        ok, img = cap.read()
        if not ok or k >= n:
            break
        ov = draw_overlay(img, frames[k]["f"], frames[k]["t"], rec_out, cuts)
        if k in sheet_frames:
            sheet_frames[k] = ov
        writer.stdin.write(ov.tobytes())
        k += 1
    writer.stdin.close()
    writer.wait()
    cap.release()
    tiles = [v for v in sheet_frames.values() if v is not None]
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
    sheet = np.vstack(rows)
    cv2.imwrite(args.sheet, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"wrote {args.overlay} and {args.sheet} {sheet.shape} in {time.time()-t1:.1f}s; total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

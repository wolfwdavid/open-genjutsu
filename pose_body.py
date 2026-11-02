#!/usr/bin/env python
"""
Whole-body 2D pose for BOTH performers of work/cut.mp4 (round 4, step 16) plus the
per-side conditioning images a Wan2.2-Animate *replacement* pass consumes.

Pipeline (every frame, 960x532 @ 30 fps)
  1. YOLOX-m (HumanArt) person boxes -> the two performers, LEFT / RIGHT by box
     centre x (the performers never swap sides).  When only one box is found the
     other side keeps its previous box (`held`) and the pose is re-estimated on it.
  2. RTMW-x whole-body (133 kps: 17 body + 6 feet + 68 face + 42 hands), one
     top-down pass per side.  Both ONNX models run on DirectML (CPU fallback).
  3. One-Euro smoothing (track.py, min_cutoff 1.5 / beta 0.03) of every keypoint,
     reset at shot cuts and after gaps.  Confidences are not smoothed; RTMW emits
     raw SimCC logits (visible joints 3-10, hallucinated out-of-frame joints < 2),
     so they are stored as conf = min(1, raw / CONF_SCALE) with CONF_SCALE = 4 and
     keypoints outside the frame are zeroed (the frame crops the performers at the
     hips, and RTMW invents legs below the picture).
  4. Skeleton images in the Wan-Animate / DWPose convention (draw_aapose_by_meta_new:
     20-point body = 18 OpenPose + 2 toes, hands, no face), one per side.
  5. Character masks: ISNet foreground matte of the frame (one DML pass per frame,
     cached in work/matte_isnet) x a generous per-side region (convex hull of the
     side's keypoints + hair/dread head extension, dilated by 0.35 x box width),
     unioned with a skeleton-driven body core (torso quad, head ellipse, limb
     capsules, hand hulls) because ISNet drops the orange shirt on the orange
     backdrop; where both sides overlap the pixel goes to the side whose arm/hand
     core covers it, else to the nearer hull; hanging mic + burned-in logo removed.

Outputs
  work/pose_body.json               poses (see --help / README block below)
  work/pose_L, work/pose_R          skeleton PNGs (black background)
  work/mask_L, work/mask_R, work/mask_both   8-bit character masks
  work/matte_isnet                  cached ISNet matte per frame
  work/round4/pose_sheet.jpg        QA sheet, work/round4/pose_run.log

Usage
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe pose_body.py
  ... --stage masks        # re-run masks + sheet from pose_body.json and the matte cache
  ... --stage sheet        # only the QA sheet
  ... --face               # also draw the 68 face points (Wan-Animate does not)
"""
from __future__ import annotations

import os
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
WORK = Path(os.environ["VG_WORK"]).resolve() if os.environ.get("VG_WORK") else ROOT / "work"
sys.path.insert(0, str(ROOT))
from track import DEFAULT_CUTS, OneEuro  # noqa: E402

YOLOX_ZIP = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip"
RTMW_ZIP = "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip"
ISNET = ROOT / "models" / "isnet-general-use.onnx"
SHEET_FRAMES = [104, 150, 239, 240, 300, 400, 461, 500, 580, 688, 750, 870]
CONF_SCALE = 4.0          # raw SimCC logit -> [0,1]; 0.5 (Wan-Animate's draw threshold) = raw 2.0
SIDES = ("L", "R")

# ----------------------------------------------------------------------------- logging
class Tee:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "a", encoding="utf-8")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.fh.write(s + "\n")
        self.fh.flush()


# ----------------------------------------------------------------------------- models
def build_models(device: str, log):
    """YOLOX + RTMW through rtmlib with our own provider choice ('dml' or 'cpu')."""
    import onnxruntime as ort
    from rtmlib.tools.base import RTMLIB_SETTINGS
    from rtmlib.tools.object_detection import YOLOX

    ort.set_default_logger_severity(3)
    from rtmlib.tools.pose_estimation import RTMPose

    RTMLIB_SETTINGS["onnxruntime"]["dml"] = "DmlExecutionProvider"
    det = YOLOX(YOLOX_ZIP, model_input_size=(640, 640), backend="onnxruntime", device=device, score_thr=0.3, nms_thr=0.45)
    pose = RTMPose(RTMW_ZIP, model_input_size=(288, 384), backend="onnxruntime", device=device)
    log(f"models: det providers={det.session.get_providers()} pose providers={pose.session.get_providers()}")
    return det, pose


def validate_dml(frame: np.ndarray, det, pose, log) -> bool:
    """Compare DML vs CPU on one frame; True if DML is trustworthy."""
    try:
        det_cpu, pose_cpu = build_models("cpu", log)
        d1, d2 = detect_persons(det, frame), detect_persons(det_cpu, frame)
        if d1.shape != d2.shape or (len(d1) and np.abs(d1[:, :4] - d2[:, :4]).max() > 2.0):
            log(f"DML check: detector mismatch {d1.tolist()} vs {d2.tolist()}")
            return False
        if len(d1) == 0:
            return True
        k1, s1 = pose(frame, [d1[0, :4]])
        k2, s2 = pose_cpu(frame, [d2[0, :4]])
        dpx = float(np.abs(k1 - k2).max())
        dsc = float(np.abs(s1 - s2).max())
        log(f"DML check: max keypoint diff {dpx:.2f}px, max score diff {dsc:.3f}")
        return dpx < 3.0 and dsc < 0.1 and np.all(np.isfinite(k1))
    except Exception as e:  # noqa: BLE001
        log(f"DML check failed: {e!r}")
        return False


def detect_persons(det, img: np.ndarray) -> np.ndarray:
    """rtmlib's YOLOX post-processing but keeping the scores: (N,5) x1,y1,x2,y2,score."""
    from rtmlib.tools.object_detection.post_processings import multiclass_nms

    pimg, ratio = det.preprocess(img)
    out = det.inference(pimg)[0]
    if out.shape[-1] == 5:                      # NMS baked into the ONNX: (1, N, [x1 y1 x2 y2 score])
        d = out[0].astype(np.float32)
        d[:, :4] /= ratio
        return d[d[:, 4] >= det.score_thr]
    grids, strides_all = [], []
    for stride in (8, 16, 32):
        hs, ws = det.model_input_size[0] // stride, det.model_input_size[1] // stride
        xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        strides_all.append(np.full((*grid.shape[:2], 1), stride))
    grids = np.concatenate(grids, 1)
    strides_all = np.concatenate(strides_all, 1)
    out[..., :2] = (out[..., :2] + grids) * strides_all
    out[..., 2:4] = np.exp(out[..., 2:4]) * strides_all
    pred = out[0]
    boxes, scores = pred[:, :4], pred[:, 4:5] * pred[:, 5:]
    xyxy = np.stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
                     boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2], 1) / ratio
    dets, _ = multiclass_nms(xyxy, scores, nms_thr=det.nms_thr, score_thr=det.score_thr)
    if dets is None:
        return np.zeros((0, 5), np.float32)
    return dets[:, :5].astype(np.float32)


# ----------------------------------------------------------------------------- assignment
def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def cx(b) -> float:
    return 0.5 * (b[0] + b[2])


def select_two(dets: np.ndarray, prev: dict, W: int, H: int) -> dict:
    """{'L': box|None, 'R': box|None} from the detections; LEFT/RIGHT by centre x."""
    cands = [d for d in dets if (d[2] - d[0]) * (d[3] - d[1]) >= 0.01 * W * H]
    cands.sort(key=lambda d: -d[4])
    picked = []
    for d in cands:
        if all(iou(d, p) < 0.6 for p in picked):
            picked.append(d)
    if not picked:
        return {"L": None, "R": None}
    if len(picked) == 1:
        d = picked[0]
        have = [s for s in SIDES if prev[s] is not None]
        side = min(have, key=lambda s: abs(cx(d) - cx(prev[s]))) if have else None
        if side is None or abs(cx(d) - cx(prev[side])) > 0.3 * W:
            side = "L" if cx(d) < W / 2 else "R"
        return {side: d[:4], ("R" if side == "L" else "L"): None}
    a, b = picked[0], picked[1]
    if len(picked) > 2 and prev["L"] is not None and prev["R"] is not None:
        best, pair = 0.2, None
        for i, pi in enumerate(picked):
            for j, pj in enumerate(picked):
                if i != j and iou(pi, prev["L"]) + iou(pj, prev["R"]) > best:
                    best, pair = iou(pi, prev["L"]) + iou(pj, prev["R"]), (pi, pj)
        if pair is not None:
            a, b = pair
    if cx(a) > cx(b):
        a, b = b, a
    return {"L": a[:4], "R": b[:4]}


def kps_bbox(kp: np.ndarray, sc: np.ndarray, W: int, H: int, thr: float = 0.5):
    """Box around the confident body + feet keypoints (used to keep a held box tracking)."""
    pts = kp[:23][sc[:23] > thr]
    if len(pts) < 6:
        return None
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    w, h = x1 - x0, y1 - y0
    return np.array([max(0, x0 - 0.15 * w), max(0, y0 - 0.25 * h), min(W, x1 + 0.15 * w), min(H, y1 + 0.1 * h)], np.float32)


# ----------------------------------------------------------------------------- drawing (Wan-Animate convention)
BODY_A = [0, 6, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 17, 20]
BODY_B = [0, 5, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 18, 21]
LIMB_SEQ = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], [2, 12], [12, 13], [13, 14],
            [2, 1], [1, 15], [15, 17], [1, 16], [16, 18], [14, 19], [11, 20]]
LIMB_COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
               [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
               [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [200, 200, 0], [100, 100, 0]]
HAND_EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12],
              [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]


def hsv_rgb(h: float):
    r, g, b = cv2.cvtColor(np.array([[[h * 179, 255, 255]]], np.uint8), cv2.COLOR_HSV2RGB)[0, 0]
    return int(r), int(g), int(b)


HAND_COLORS = [hsv_rgb(ie / len(HAND_EDGES)) for ie in range(len(HAND_EDGES))]


def stick_widths(H: int, W: int, mode: str):
    base = int(min(H, W) / 200)
    if mode == "v1":
        return max(base, 1), max(base, 1)
    return max(base - 1, 1), max(max(base - 1, 1) // 2, 1)      # v2 (Wan-Animate default)


def draw_wan_pose(canvas: np.ndarray, kp: np.ndarray, thr: float, stick: str, face: bool) -> np.ndarray:
    """draw_aapose_by_meta_new re-implementation. canvas is RGB; kp is (133,3) x,y,conf."""
    H, W = canvas.shape[:2]
    sw_body, sw_hand = stick_widths(H, W, stick)
    body = (kp[BODY_A] + kp[BODY_B]) / 2.0
    for (i, j), col in zip(LIMB_SEQ, LIMB_COLORS):
        p, q = body[i - 1], body[j - 1]
        if p[2] < thr or q[2] < thr:
            continue
        mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
        length = math.hypot(p[0] - q[0], p[1] - q[1])
        angle = math.degrees(math.atan2(p[1] - q[1], p[0] - q[0]))
        poly = cv2.ellipse2Poly((int(mx), int(my)), (int(length / 2), sw_body), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, poly, [int(c * 0.6) for c in col])
    for p, col in zip(body, LIMB_COLORS):
        if p[2] >= thr:
            cv2.circle(canvas, (int(p[0]), int(p[1])), sw_body, col, -1)
    for hand in (kp[91:112], kp[112:133]):
        for (a, b), col in zip(HAND_EDGES, HAND_COLORS):
            p, q = hand[a], hand[b]
            if p[2] < thr or q[2] < thr:
                continue
            if min(p[0], p[1], q[0], q[1]) > 0.01:
                cv2.line(canvas, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), col, sw_hand)
        for p in hand:
            if p[2] >= thr and p[0] > 0.01 and p[1] > 0.01:
                cv2.circle(canvas, (int(p[0]), int(p[1])), sw_hand, (0, 0, 255), -1)
    if face:
        for p in kp[23:91]:
            if p[2] >= thr:
                cv2.circle(canvas, (int(p[0]), int(p[1])), 1, (255, 255, 255), -1)
    return canvas


# ----------------------------------------------------------------------------- matting
class ISNet:
    def __init__(self, model: Path, log):
        import onnxruntime as ort

        provs = [p for p in ("DmlExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.sess = ort.InferenceSession(str(model), providers=provs)
        self.inp = self.sess.get_inputs()[0].name
        log(f"isnet providers={self.sess.get_providers()}")

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        """Letterboxed whole-frame matte, float32 [0,1] at frame size."""
        H, W = bgr.shape[:2]
        S = max(H, W)
        fill = tuple(int(v) for v in np.median(bgr[::8, ::8].reshape(-1, 3), 0))
        sq = np.full((S, S, 3), fill, np.uint8)
        oy, ox = (S - H) // 2, (S - W) // 2
        sq[oy:oy + H, ox:ox + W] = bgr
        rgb = cv2.cvtColor(cv2.resize(sq, (1024, 1024), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - 0.5)[None].transpose(0, 3, 1, 2)
        out = self.sess.run(None, {self.inp: np.ascontiguousarray(x)})[0][0, 0].astype(np.float32)
        out = (out - out.min()) / max(1e-6, float(out.max() - out.min()))
        m = cv2.resize(out, (S, S), interpolation=cv2.INTER_LINEAR)[oy:oy + H, ox:ox + W]
        return np.clip(m, 0, 1)


def static_grey_blob(frames: list[np.ndarray], x0: int, x1: int, y0: int, y1: int, sat: int, val: int,
                     frac: float, touch_top: bool, wmin: int, wmax: int, hmin: int, amax: int, grow: int):
    """Mask (frame size, uint8 0/255) of a stationary bright-grey blob inside the band."""
    H, W = frames[0].shape[:2]
    acc = np.zeros((y1 - y0, x1 - x0), np.float32)
    for fr in frames:
        hsv = cv2.cvtColor(fr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        acc += ((hsv[..., 1] < sat) & (hsv[..., 2] > val)).astype(np.float32)
    grey = (acc / len(frames) >= frac).astype(np.uint8)
    grey = cv2.morphologyEx(grey, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(grey, 8)
    best, best_area = None, 0
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if touch_top and y > 4:
            continue
        if not (wmin <= w <= wmax and h >= hmin and a <= amax):
            continue
        if a > best_area:
            best, best_area = i, a
    out = np.zeros((H, W), np.uint8)
    if best is None:
        return None
    blob = (lab == best).astype(np.uint8) * 255
    if grow:
        blob = cv2.dilate(blob, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    out[y0:y1, x0:x1] = blob
    return out


def person_region(kp: np.ndarray, box, side: str, W: int, H: int, thr: float, dil_frac: float):
    """(hull, dilated hull) uint8 masks for one side, or (None, None)."""
    pts = kp[kp[:, 2] > thr][:, :2]
    if len(pts) < 5 or box is None:
        return None, None
    head = kp[[0, 1, 2, 3, 4] + list(range(23, 91))]
    head = head[head[:, 2] > thr][:, :2]
    if len(head) >= 3:
        x0, y0 = head.min(0)
        x1, y1 = head.max(0)
        hw, hh = max(x1 - x0, 10.0), max(y1 - y0, 10.0)
        top, wide = (0.25, 0.15) if side == "L" else (0.15, 0.10)
        rect = np.array([[x0 - wide * hw, y0 - top * hh], [x1 + wide * hw, y0 - top * hh],
                         [x1 + wide * hw, y1], [x0 - wide * hw, y1]], np.float32)
        pts = np.vstack([pts, rect])
    hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
    hull_img = np.zeros((H, W), np.uint8)
    cv2.fillPoly(hull_img, [np.round(hull).astype(np.int32)], 255)
    r = max(4, int(dil_frac * (box[2] - box[0])))
    dil = cv2.dilate(hull_img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return hull_img, dil


def body_core(kp: np.ndarray, box, W: int, H: int, thr: float):
    """Skeleton-driven body prior: (core, front) uint8 masks.  core = torso quad + head ellipse +
    limb capsules + hand hulls; front = arms + hands only (they win overlaps, being nearer the camera)."""
    core = np.zeros((H, W), np.uint8)
    front = np.zeros((H, W), np.uint8)
    if box is None:
        return core, front
    bw = max(20.0, float(box[2] - box[0]))
    ok = kp[:, 2] >= thr
    P = kp[:, :2]

    def pt(i):
        return int(round(P[i, 0])), int(round(P[i, 1]))

    def capsule(img, i, j, r):
        if ok[i] and ok[j]:
            cv2.line(img, pt(i), pt(j), 255, max(1, int(2 * r)))
            cv2.circle(img, pt(i), int(r), 255, -1)
            cv2.circle(img, pt(j), int(r), 255, -1)

    def grown(layer, r):
        return cv2.dilate(layer, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(r) + 1, 2 * int(r) + 1)))

    # torso: shoulders down to the hips, or to the box bottom when the hips are out of frame
    if ok[5] and ok[6]:
        lo_l = P[11] if ok[11] else np.array([P[5, 0], min(H - 1, box[3])])
        lo_r = P[12] if ok[12] else np.array([P[6, 0], min(H - 1, box[3])])
        quad = np.array([P[5], P[6], lo_r, lo_l], np.float32)
        layer = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(layer, np.round(quad).astype(np.int32), 255)
        core |= grown(layer, 0.06 * bw)
    # head: ellipse around the face points, pushed up for forehead / hair
    face = kp[[0, 1, 2, 3, 4] + list(range(23, 91))]
    face = face[face[:, 2] >= thr][:, :2]
    if len(face) >= 8:
        x0, y0 = face.min(0)
        x1, y1 = face.max(0)
        fw, fh = max(x1 - x0, 10.0), max(y1 - y0, 10.0)
        cv2.ellipse(core, (int((x0 + x1) / 2), int((y0 + y1) / 2 - 0.12 * fh)), (int(0.52 * fw), int(0.62 * fh)), 0, 0, 360, 255, -1)
    for a, b in ((5, 7), (7, 9), (6, 8), (8, 10)):
        capsule(front, a, b, 0.07 * bw)
    for a, b in ((11, 13), (13, 15), (12, 14), (14, 16)):
        capsule(core, a, b, 0.08 * bw)
    for lo, hi in ((91, 112), (112, 133)):
        hand = kp[lo:hi]
        hand = hand[hand[:, 2] >= thr][:, :2]
        if len(hand) >= 8:
            layer = np.zeros((H, W), np.uint8)
            cv2.fillConvexPoly(layer, np.round(cv2.convexHull(hand.astype(np.float32)).reshape(-1, 2)).astype(np.int32), 255)
            front |= grown(layer, 0.035 * bw)
    core |= front
    return core, front


def fill_small_holes(binary: np.ndarray, max_area: int) -> np.ndarray:
    inv = (~binary).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, 4)
    out = binary.copy()
    H, W = binary.shape
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a <= max_area and x > 0 and y > 0 and x + w < W and y + h < H:
            out[lab == i] = True
    return out


def make_masks(alpha: np.ndarray, kps: dict, boxes: dict, remove: np.ndarray | None, W: int, H: int,
               thr: float, dil_frac: float, feather: float, hole_px: int) -> dict:
    hull, dil, dist, front, m = {}, {}, {}, {}, {}
    for s in SIDES:
        hull[s], dil[s] = person_region(kps[s], boxes[s], s, W, H, thr, dil_frac) if kps[s] is not None else (None, None)
        if dil[s] is None:
            m[s], front[s], dist[s] = np.zeros((H, W), np.float32), np.zeros((H, W), bool), None
            continue
        core, fr = body_core(kps[s], boxes[s], W, H, thr)
        m[s] = np.maximum(alpha * (dil[s] > 0), core.astype(np.float32) / 255.0)
        front[s] = fr > 0
        dist[s] = cv2.distanceTransform((hull[s] == 0).astype(np.uint8), cv2.DIST_L2, 3)
    # overlap: arms/hands in front win, otherwise the nearer hull (ties -> L)
    if dist["L"] is not None and dist["R"] is not None:
        both = (m["L"] > 0.5) & (m["R"] > 0.5)
        only_l, only_r = front["L"] & ~front["R"], front["R"] & ~front["L"]
        win_l = only_l | (~only_r & (dist["L"] <= dist["R"]))
        m["L"][both & ~win_l] = 0.0
        m["R"][both & win_l] = 0.0
    out = {}
    for s in SIDES:
        x = m[s]
        b = fill_small_holes(x > 0.5, hole_px)
        x = np.maximum(x, b.astype(np.float32))
        if remove is not None:
            x[remove > 0] = 0.0
        x = cv2.GaussianBlur(x, (0, 0), feather / 3.0)
        out[s] = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    out["both"] = np.maximum(out["L"], out["R"])
    return out


# ----------------------------------------------------------------------------- io helpers
def read_frames(video: Path, idx: list[int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    out = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if ok:
            out.append(fr)
    cap.release()
    return out


def shot_ranges(cuts: list[int], n: int) -> list[tuple[int, int]]:
    edges = [0] + sorted(c for c in cuts if 0 < c < n) + [n]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def frame_entry(f: int, present: bool, held: bool, box, kp_xy, conf) -> dict:
    if kp_xy is None:
        return {"f": f, "present": False, "held": False, "bbox": None, "kps": None}
    kps = np.concatenate([kp_xy, conf[:, None]], 1)
    return {"f": f, "present": bool(present), "held": bool(held),
            "bbox": [round(float(v), 1) for v in box],
            "kps": [[round(float(x), 1), round(float(y), 1), round(float(c), 3)] for x, y, c in kps]}


def entry_kps(e: dict):
    return np.array(e["kps"], np.float32) if e.get("kps") else None


# ----------------------------------------------------------------------------- stages
def run_pose(args, log, W, H, n, fps, cuts):
    det, pose = build_models(args.device, log)
    probe = read_frames(args.video, [min(300, n - 1)])[0]
    provider = args.device
    if args.device == "dml" and not validate_dml(probe, det, pose, log):
        log("DML unreliable for these models -> falling back to CPU")
        det, pose = build_models("cpu", log)
        provider = "cpu"
    from rtmlib.visualization.skeleton import coco133
    kp_names = [coco133["keypoint_info"][i]["name"] for i in range(133)]

    for s in SIDES:
        (WORK / f"pose_{s}").mkdir(parents=True, exist_ok=True)
    filt = {s: OneEuro(fps, args.min_cutoff, args.beta) for s in SIDES}
    prev = {s: None for s in SIDES}
    tracks = {s: [] for s in SIDES}
    stats = {s: {"held": 0, "absent": 0} for s in SIDES}
    t_det = t_pose = t_draw = 0.0
    t0 = time.time()
    cap = cv2.VideoCapture(str(args.video))
    for f in range(n):
        ok, frame = cap.read()
        if not ok:
            log(f"frame {f}: read failed, stopping")
            n = f
            break
        if f in cuts:
            for s in SIDES:
                filt[s].reset()
                prev[s] = None
        ta = time.time()
        dets = detect_persons(det, frame)
        boxes = select_two(dets, prev, W, H)
        tb = time.time()
        t_det += tb - ta
        for s in SIDES:
            present, held, box = boxes[s] is not None, False, boxes[s]
            if box is None and prev[s] is not None:
                box, held = prev[s], True
            if box is None:
                filt[s].reset()
                prev[s] = None
                stats[s]["absent"] += 1
                tracks[s].append(frame_entry(f, False, False, None, None, None))
                canvas = np.zeros((H, W, 3), np.uint8)
            else:
                kp, sc = pose(frame, [box])
                kp, sc = kp[0].astype(np.float64), sc[0].astype(np.float32)
                sc = np.clip(sc / CONF_SCALE, 0, 1)
                sc[(kp[:, 0] < 0) | (kp[:, 0] >= W) | (kp[:, 1] < 0) | (kp[:, 1] >= H)] = 0.0
                if held:
                    stats[s]["held"] += 1
                    nb = kps_bbox(kp, sc, W, H)
                    if nb is not None:
                        box = nb
                sm = filt[s](kp)
                prev[s] = np.asarray(box, np.float32)
                tracks[s].append(frame_entry(f, present, held, box, sm, sc))
                td = time.time()
                canvas = draw_wan_pose(np.zeros((H, W, 3), np.uint8),
                                       np.concatenate([sm, sc[:, None]], 1), args.kpt_thr, args.stick, args.face)
                t_draw += time.time() - td
            cv2.imwrite(str(WORK / f"pose_{s}" / f"{f:05d}.png"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        t_pose += time.time() - tb
        if f % 50 == 0 or f == n - 1:
            el = time.time() - t0
            log(f"  pose f{f:04d}/{n}  {el / (f + 1):.3f} s/frame  det {t_det / (f + 1):.3f}  pose+draw {t_pose / (f + 1):.3f}"
                f"  held L/R {stats['L']['held']}/{stats['R']['held']}  absent L/R {stats['L']['absent']}/{stats['R']['absent']}")
    cap.release()
    el = time.time() - t0
    data = {"fps": fps, "n": n, "width": W, "height": H, "cuts": cuts, "kp_names": kp_names,
            "provider": provider, "smoothing": {"min_cutoff": args.min_cutoff, "beta": args.beta},
            "conf_scale": CONF_SCALE, "conf_note": "conf = min(1, raw_simcc_logit / conf_scale); 0 = outside the frame",
            "L": tracks["L"], "R": tracks["R"]}
    with open(WORK / "pose_body.json", "w", encoding="utf-8") as fh:
        json.dump(data, fh, separators=(",", ":"))
    log(f"pose stage: {el:.1f}s total = {el / max(1, n):.3f} s/frame (provider {provider}); "
        f"held L {stats['L']['held'] / n:.3f} R {stats['R']['held'] / n:.3f}; absent L {stats['L']['absent']} R {stats['R']['absent']}")
    return data


def run_masks(args, log, data):
    W, H, n, cuts = data["width"], data["height"], data["n"], data["cuts"]
    for d in ("mask_L", "mask_R", "mask_both", "matte_isnet"):
        (WORK / d).mkdir(parents=True, exist_ok=True)
    isnet = ISNet(ISNET, log)

    # per-shot hanging mic (stationary bright-grey blob touching the top of the frame, top-centre band)
    mic = {}
    for a, b in shot_ranges(cuts, n):
        samp = read_frames(args.video, list(np.linspace(a, b - 1, min(8, b - a)).astype(int)))
        m = static_grey_blob(samp, x0=int(0.35 * W), x1=int(0.65 * W), y0=0, y1=int(0.6 * H), sat=80, val=110,
                             frac=0.75, touch_top=True, wmin=6, wmax=120, hmin=30, amax=int(0.05 * W * H), grow=5)
        if m is None:
            log(f"  shot {a}-{b - 1}: no mic blob found (no mic removal)")
        else:
            ys, xs = np.where(m > 0)
            log(f"  shot {a}-{b - 1}: mic blob x {xs.min()}-{xs.max()} y {ys.min()}-{ys.max()} ({len(xs)} px)")
        mic[(a, b)] = m
    # burned-in logo: stationary white blob in the bottom-left corner across the whole video
    samp = read_frames(args.video, list(np.linspace(0, n - 1, 24).astype(int)))
    logo = static_grey_blob(samp, x0=0, x1=int(0.25 * W), y0=int(0.7 * H), y1=H, sat=40, val=190,
                            frac=0.9, touch_top=False, wmin=10, wmax=120, hmin=10, amax=6000, grow=4)
    if logo is None:
        log("  no static logo blob found")
    else:
        ys, xs = np.where(logo > 0)
        log(f"  logo blob x {xs.min()}-{xs.max()} y {ys.min()}-{ys.max()}")

    t_matte = t_mask = 0.0
    t0 = time.time()
    cap = cv2.VideoCapture(str(args.video))
    shot = None
    for f in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        for rng in mic:
            if rng[0] <= f < rng[1]:
                shot = rng
        remove = mic[shot]
        if logo is not None:
            remove = logo if remove is None else np.maximum(remove, logo)
        ta = time.time()
        cache = WORK / "matte_isnet" / f"{f:05d}.png"
        if args.reuse_matte and cache.exists():
            alpha = cv2.imread(str(cache), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        else:
            alpha = isnet(frame)
            cv2.imwrite(str(cache), (alpha * 255).astype(np.uint8))
        tb = time.time()
        t_matte += tb - ta
        kps = {s: entry_kps(data[s][f]) for s in SIDES}
        boxes = {s: data[s][f]["bbox"] for s in SIDES}
        masks = make_masks(alpha, kps, boxes, remove, W, H, args.kpt_thr_mask, args.dilate, args.feather, args.hole_px)
        for k, v in masks.items():
            cv2.imwrite(str(WORK / f"mask_{k}" / f"{f:05d}.png"), v)
        t_mask += time.time() - tb
        if f % 50 == 0 or f == n - 1:
            log(f"  mask f{f:04d}/{n}  {(time.time() - t0) / (f + 1):.3f} s/frame  matte {t_matte / (f + 1):.3f}  masks {t_mask / (f + 1):.3f}")
    cap.release()
    el = time.time() - t0
    log(f"mask stage: {el:.1f}s = {el / max(1, n):.3f} s/frame")


def run_sheet(args, log, data):
    W, H = data["width"], data["height"]
    frames = [f for f in args.sheet_frames if f < data["n"]]
    imgs = read_frames(args.video, frames)
    rows = []
    for f, fr in zip(frames, imgs):
        sk = np.zeros_like(fr)
        for s in SIDES:
            p = WORK / f"pose_{s}" / f"{f:05d}.png"
            if p.exists():
                sk = np.maximum(sk, cv2.imread(str(p)))
        ov = cv2.addWeighted(fr, 0.45, sk, 1.0, 0)
        for s, col in (("L", (0, 255, 255)), ("R", (255, 200, 0))):
            e = data[s][f]
            if e["bbox"]:
                x0, y0, x1, y1 = [int(v) for v in e["bbox"]]
                cv2.rectangle(ov, (x0, y0), (x1, y1), col, 1)
                tag = s + (" held" if e["held"] else "") + ("" if e["present"] else " !")
                cv2.putText(ov, tag, (x0 + 3, max(14, y0 + 16)), 0, 0.55, col, 2)
        mv = fr.copy().astype(np.float32)
        mL = cv2.imread(str(WORK / "mask_L" / f"{f:05d}.png"), 0)
        mR = cv2.imread(str(WORK / "mask_R" / f"{f:05d}.png"), 0)
        if mL is not None and mR is not None:
            aL, aR = mL.astype(np.float32) / 255.0, mR.astype(np.float32) / 255.0
            tint = np.zeros_like(mv)
            tint[..., 2] = 255 * aL
            tint[..., 0] = 255 * aR
            mv = mv * (1 - 0.55 * np.maximum(aL, aR)[..., None]) + 0.55 * tint
            for a, col in ((aL, (0, 0, 255)), (aR, (255, 0, 0))):
                cnt, _ = cv2.findContours((a > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(mv, cnt, -1, col, 1)
        mv = np.clip(mv, 0, 255).astype(np.uint8)
        cv2.putText(fr, f"f{f}", (8, 24), 0, 0.8, (0, 255, 255), 2)
        cells = [cv2.resize(x, (W // 2, H // 2), interpolation=cv2.INTER_AREA) for x in (fr, ov, mv)]
        rows.append(np.hstack(cells))
    sheet = np.vstack(rows)
    out = WORK / "round4" / "pose_sheet.jpg"
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    log(f"sheet -> {out} ({sheet.shape[1]}x{sheet.shape[0]})")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, default=WORK / "cut.mp4")
    ap.add_argument("--stage", choices=["all", "pose", "masks", "sheet"], default="all")
    ap.add_argument("--device", choices=["dml", "cpu"], default="dml")
    ap.add_argument("--cuts", default=",".join(map(str, DEFAULT_CUTS)))
    ap.add_argument("--min-cutoff", type=float, default=1.5)
    ap.add_argument("--beta", type=float, default=0.03)
    ap.add_argument("--kpt-thr", type=float, default=0.5, help="draw threshold (Wan-Animate default 0.5)")
    ap.add_argument("--stick", choices=["v1", "v2"], default="v2")
    ap.add_argument("--face", action="store_true", help="also draw the 68 face points")
    ap.add_argument("--kpt-thr-mask", type=float, default=0.5)
    ap.add_argument("--dilate", type=float, default=0.35, help="hull dilation as a fraction of the box width")
    ap.add_argument("--feather", type=float, default=6.0)
    ap.add_argument("--hole-px", type=int, default=2500)
    ap.add_argument("--reuse-matte", action="store_true")
    ap.add_argument("--sheet-frames", default=",".join(map(str, SHEET_FRAMES)))
    ap.add_argument("--end", type=int, default=0, help="smoke test: only process frames [0, end)")
    args = ap.parse_args()
    args.sheet_frames = [int(v) for v in args.sheet_frames.split(",")]
    cuts = [int(v) for v in args.cuts.split(",") if v]

    log = Tee(WORK / "round4" / "pose_run.log")
    log(f"=== pose_body {time.strftime('%Y-%m-%d %H:%M:%S')} stage={args.stage} device={args.device} args={vars(args)}")
    cap = cv2.VideoCapture(str(args.video))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if args.end:
        n = min(n, args.end)
    log(f"video {args.video}: {W}x{H} {n} frames @ {fps:.3f} fps, cuts {cuts}")

    data = None
    if args.stage in ("all", "pose"):
        data = run_pose(args, log, W, H, n, fps, cuts)
    if data is None:
        data = json.load(open(WORK / "pose_body.json", encoding="utf-8"))
    if args.stage in ("all", "masks"):
        run_masks(args, log, data)
    if args.stage in ("all", "masks", "sheet"):
        run_sheet(args, log, data)
    n = data["n"]
    for s in SIDES:
        held = sum(1 for e in data[s] if e["held"])
        absent = sum(1 for e in data[s] if not e["present"] and not e["held"])
        log(f"summary {s}: held {held}/{n} = {held / n:.3f}, absent {absent}")
    log("done")


if __name__ == "__main__":
    main()

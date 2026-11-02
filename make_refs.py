#!/usr/bin/env python
"""
Reference images for the Wan2.2-Animate Replacement recast (round 4, step 17):
one character per image, on the flat COLORS set orange (refs/cat_meta.json set_bgr).

  make_refs.py user --photos faces/user/*.PNG --out refs/user_ref.png
      buffalo_l face + ISNet person matte per photo, crop to the person with a 6 %
      margin, 1024 px tall plate, shadows warmed toward the set.  Photos are ranked
      by sunglasses / full-body coverage / face sharpness / frontal-ness; the best
      one becomes --out, every candidate lands on <out>_sheet.jpg with its scores.
      FAILS (exit 2) unless the best photo shows sunglasses (--no-sunglasses-check
      for stand-in tests).

  make_refs.py calibrate
      prints the eye-band dark/white fraction on work/cut.mp4 frames (both rappers
      wear sunglasses) and faces/user/selected/*.jpg (no glasses) -> threshold.

  make_refs.py cat --photo faces/cat/IMG_9318.PNG --out refs/cat_body_base.png [--costume]
      two-pass ISNet matte of the whole seated cat, eyes located by template
      matching against the IMG_9325 hero head, cat at 70 % of a 1024 px plate.
      Also refs/cat_body_base_hq.png = same with the sharper hero head pasted over
      the small head (aligned by the eye centres).  --costume adds the manual
      costume layers (assets/cat_costume/*.png: stool, cap bill, flannel, glasses,
      cap crown) -> refs/cat_ref_costume.png and refs/cat_ref_sheet.jpg.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from make_cat_ref import (OUT as G512, FRAME_BGR, clean_alpha, colour_match, draw_glasses,  # noqa: E402
                          ensure_isnet, label, over)

REFS = ROOT / "refs"
ASSETS = ROOT / "assets"
WORK = ROOT / "work"
MODELS = ROOT / "models"
COSTUME_DIR = ASSETS / "cat_costume"
PLATE_H = 1024
DEFAULT_SET_BGR = (5, 54, 192)
# Calibrated with `make_refs.py calibrate` (see work/round4/refs_run.log):
#   cut.mp4 sunglasses faces (n=44): min 0.158 p10 0.359 median 0.492 ; no-glasses (n=10, incl. clear-lens
#   black frames IMG_3595 / IMG_9330): max 0.150 -> geometric midpoint 0.232
SUNGLASSES_THR = 0.23
HERO_512 = REFS / "cat_src_512.png"
HERO_ALPHA_512 = REFS / "cat_alpha_512.png"


# ----------------------------------------------------------------------------- shared helpers
def set_colour() -> tuple[int, int, int]:
    meta = REFS / "cat_meta.json"
    if meta.exists():
        return tuple(int(v) for v in json.load(open(meta))["set_bgr"])
    return DEFAULT_SET_BGR


def read_image(path: str) -> np.ndarray:
    """Photo -> BGR, EXIF-rotated (phone JPEGs are stored sideways; cv2.imread ignores the tag).
    HEIC/HEIF through pillow-heif."""
    from PIL import Image, ImageOps
    if Path(path).suffix.lower() in (".heic", ".heif"):
        import pillow_heif
        pillow_heif.register_heif_opener()
    try:
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    except Exception as e:  # noqa: BLE001
        sys.exit(f"cannot read {path}: {e}")
    return np.asarray(im)[:, :, ::-1].copy()


class ISNet:
    """ISNet general-use on an arbitrary image: resized to 1024x1024 (non-uniform,
    like rembg), matte resized back to the input size, min-max normalised."""

    def __init__(self, model: Path):
        import onnxruntime as ort
        model = ensure_isnet(model)
        providers = [p for p in ("DmlExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.sess = ort.InferenceSession(str(model), providers=providers)
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(bgr, (1024, 1024), interpolation=cv2.INTER_AREA if max(h, w) > 1024 else cv2.INTER_CUBIC),
                           cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - 0.5)
        x = np.transpose(x, (2, 0, 1))[None]
        m = self.sess.run(None, {self.inp: x})[0][0, 0].astype(np.float32)
        m = (m - m.min()) / max(1e-6, float(m.max() - m.min()))
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)


def bbox_of(alpha: np.ndarray, thr: float = 0.5):
    ys, xs = np.where(alpha > thr)
    if not len(ys):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def largest_component(alpha: np.ndarray, thr: float = 0.5) -> np.ndarray:
    core = (alpha > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(core, 8)
    if n <= 1:
        return alpha
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    keep = cv2.dilate((lab == big).astype(np.uint8), np.ones((5, 5), np.uint8), iterations=2)
    return alpha * keep


def make_plate(bgr: np.ndarray, alpha: np.ndarray, set_bgr, fill: float, side_margin: float = 0.06,
               min_aspect: float = 0.0, warm: float = 0.07, feather: float = 4.0):
    """Crop the subject bbox (alpha > 0.5) and scale it so its height is `fill` * PLATE_H,
    centred on a PLATE_H-tall set-orange plate whose width = bbox width + 2*side_margin
    (at least min_aspect * PLATE_H).  Returns plate, alpha_out, (scale, ox, oy) so that
    plate_xy = src_xy * scale + (ox, oy)."""
    x0, y0, x1, y1 = bbox_of(alpha)
    bw, bh = x1 - x0, y1 - y0
    scale = fill * PLATE_H / bh
    W = int(round(max(bw * scale * (1 + 2 * side_margin), min_aspect * PLATE_H)))
    W += W % 2
    ox = (W - bw * scale) / 2 - x0 * scale
    oy = (PLATE_H - bh * scale) / 2 - y0 * scale
    M = np.float32([[scale, 0, ox], [0, scale, oy]])
    fill_col = tuple(int(v) for v in set_bgr)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4
    img = cv2.warpAffine(bgr, M, (W, PLATE_H), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=fill_col)
    a = cv2.warpAffine(alpha, M, (W, PLATE_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    a = clean_alpha(a, feather)
    warmed = colour_match(img, a, set_bgr, warm)
    plate = np.empty_like(img)
    plate[:] = np.array(fill_col, np.uint8)
    a3 = a[..., None]
    plate = np.clip(warmed.astype(np.float32) * a3 + plate.astype(np.float32) * (1 - a3), 0, 255).astype(np.uint8)
    return plate, a, (scale, ox, oy)


def fit_tile(img: np.ndarray, h: int) -> np.ndarray:
    s = h / img.shape[0]
    return cv2.resize(img, (max(1, int(round(img.shape[1] * s))), h), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)


def text_block(img: np.ndarray, lines: list[str], y0: int = 40, scale: float = 0.55) -> np.ndarray:
    d = img.copy()
    for i, t in enumerate(lines):
        y = y0 + i * int(26 * scale / 0.55)
        cv2.putText(d, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(d, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return d


# ----------------------------------------------------------------------------- sunglasses check
def eye_band_fraction(bgr: np.ndarray, kps: np.ndarray, dark_v: int = 70, white_s: int = 60, white_v: int = 170,
                      ax_iod: float = 1.25, ay_iod: float = 0.45) -> dict:
    """Fraction of the eye-band ellipse (swap.eyewear_mask geometry: centred on the
    5-point eye midpoint, half axes 1.25 x 0.45 iod, rotated by the eye-line roll)
    that is black lens (V < dark_v) or white frame (S < white_s and V > white_v)."""
    k = np.asarray(kps, np.float32)
    le, re = k[0], k[1]
    v = re - le
    iod = float(np.hypot(*v))
    ang = float(np.degrees(np.arctan2(v[1], v[0])))
    mid = (le + re) / 2
    H, W = bgr.shape[:2]
    ell = np.zeros((H, W), np.uint8)
    cv2.ellipse(ell, (int(round(mid[0])), int(round(mid[1]))),
                (max(1, int(round(ax_iod * iod))), max(1, int(round(ay_iod * iod)))), ang, 0, 360, 255, -1)
    x0, y0, x1, y1 = bbox_of(ell.astype(np.float32) / 255.0)
    hsv = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    e = ell[y0:y1, x0:x1] > 0
    S, V = hsv[..., 1], hsv[..., 2]
    n = max(1, int(e.sum()))
    dark = float(((V < dark_v) & e).sum()) / n
    white = float(((S < white_s) & (V > white_v) & e).sum()) / n
    return {"frac": dark + white, "dark": dark, "white": white, "iod": iod, "ellipse_px": n}


def face_app(det_size: int = 640):
    from track import build_app
    return build_app(det_size)


def biggest_face(app, img: np.ndarray, min_det: float = 0.5):
    from track import detect
    faces = [f for f in detect(app, img, min_det=min_det)]
    if not faces:
        # phone photos: try the detector at a larger input size
        faces = [f for f in detect(app, img, input_size=(1280, 1280), min_det=min_det)]
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def cmd_calibrate(args) -> float:
    app = face_app()
    glasses, none = [], []
    cap = cv2.VideoCapture(str(WORK / "cut.mp4"))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    from track import detect
    for fi in range(0, n, args.every):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if not ok:
            break
        for f in detect(app, fr, min_det=0.6):
            if f.bbox[3] - f.bbox[1] < 40:       # wide shots: too small to judge
                continue
            r = eye_band_fraction(fr, f.kps)
            glasses.append(r["frac"])
            print(f"  cut.mp4 f{fi:4d} face_h={f.bbox[3]-f.bbox[1]:5.0f} iod={r['iod']:5.1f} dark={r['dark']:.3f} white={r['white']:.3f} frac={r['frac']:.3f}")
    cap.release()
    for p in sorted(glob.glob(str(ROOT / "faces/user/selected/*.jpg"))) + [str(ROOT / p) for p in args.extra_none]:
        img = read_image(p)
        f = biggest_face(app, img)
        if f is None:
            print(f"  {p}: no face")
            continue
        r = eye_band_fraction(img, f.kps)
        none.append(r["frac"])
        print(f"  {Path(p).name:28s} iod={r['iod']:5.1f} dark={r['dark']:.3f} white={r['white']:.3f} frac={r['frac']:.3f}")
    g, z = np.array(glasses), np.array(none)
    print(f"CALIBRATION sunglasses(n={len(g)}): min={g.min():.3f} p10={np.percentile(g,10):.3f} median={np.median(g):.3f} max={g.max():.3f}")
    print(f"CALIBRATION no-glasses (n={len(z)}): min={z.min():.3f} median={np.median(z):.3f} p90={np.percentile(z,90):.3f} max={z.max():.3f}")
    thr = float(np.sqrt(max(z.max(), 1e-3) * np.percentile(g, 10)))   # geometric midpoint of the gap
    print(f"CALIBRATION threshold (geometric mid of no-glasses max and sunglasses p10) = {thr:.3f}; "
          f"module default SUNGLASSES_THR = {SUNGLASSES_THR}")
    return thr


# ----------------------------------------------------------------------------- user reference
def cmd_user(args) -> int:
    set_bgr = set_colour()
    app = face_app()
    isnet = ISNet(Path(args.isnet))
    photos = []
    for pat in args.photos:
        hits = sorted(glob.glob(str(ROOT / pat))) or sorted(glob.glob(pat))
        photos += hits or [pat]
    cands = []
    for p in photos:
        img = read_image(p)
        H, W = img.shape[:2]
        f = biggest_face(app, img)
        if f is None:
            print(f"{p}: NO FACE -> skipped")
            continue
        bx0, by0, bx1, by1 = [float(v) for v in f.bbox]
        fh = by1 - by0
        sg = eye_band_fraction(img, f.kps)
        sg_ok = sg["frac"] >= args.sunglasses_thr
        # sharpness: Laplacian variance on the face crop normalised to 200 px wide
        face = img[int(max(0, by0)):int(min(H, by1)), int(max(0, bx0)):int(min(W, bx1))]
        face = cv2.resize(face, (200, int(200 * face.shape[0] / max(1, face.shape[1]))), interpolation=cv2.INTER_AREA)
        sharp = float(cv2.Laplacian(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        pose = getattr(f, "pose", None)
        yaw = float(pose[1]) if pose is not None else 0.0
        # person matte: whole image, largest component that contains the face
        raw = isnet(img)
        alpha = largest_component(raw)
        alpha = clean_alpha(alpha, max(2.0, 0.004 * H))
        mb = bbox_of(alpha)
        if mb is None:
            print(f"{p}: empty matte -> skipped")
            continue
        mx0, my0, mx1, my1 = mb
        matte_h_frac = (my1 - my0) / H
        body_ratio = (my1 - by0) / fh                       # face heights from the brow line to the matte bottom
        touches_bottom = my1 >= H - max(2, 0.01 * H)
        coverage = min(1.0, body_ratio / 5.0)               # head to the knees is enough (the pose drives the legs)
        feet = 1.0 if (not touches_bottom and body_ratio >= 5.5) else (0.5 if not touches_bottom else 0.0)
        frontal = max(0.0, 1.0 - abs(yaw) / 45.0)
        face_res = min(1.0, fh / 200.0)                     # identity detail available in the source
        cands.append(dict(path=p, img=img, alpha=alpha, sg=sg, sg_ok=sg_ok, sharp=sharp, yaw=yaw, body_ratio=body_ratio,
                          matte_h_frac=matte_h_frac, touches_bottom=touches_bottom, coverage=coverage, feet=feet,
                          frontal=frontal, face_h=fh, face_res=face_res, kps=f.kps))
    if not cands:
        print("no usable photo")
        return 2
    smax = max(c["sharp"] for c in cands)
    # score = 3 sunglasses + 1.5 coverage + 0.5 feet + 1.5 sharpness (sqrt of the Laplacian variance,
    # ~linear in edge contrast, relative to the sharpest candidate) + 1 face resolution + 1 frontal
    for c in cands:
        c["sharp_n"] = math.sqrt(c["sharp"] / max(1e-6, smax))
        c["score"] = (3.0 * c["sg_ok"] + 1.5 * c["coverage"] + 0.5 * c["feet"] + 1.5 * c["sharp_n"]
                      + 1.0 * c["face_res"] + 1.0 * c["frontal"])
    cands.sort(key=lambda c: -c["score"])
    print(f"{'photo':30s} {'score':>5s} {'sungl':>6s} ok {'body':>5s} {'matteH':>6s} feet {'sharp':>7s} {'faceH':>5s} {'yaw':>6s}")
    for c in cands:
        print(f"{Path(c['path']).name:30s} {c['score']:5.2f} {c['sg']['frac']:6.3f} {int(c['sg_ok'])}  {c['body_ratio']:5.2f} "
              f"{c['matte_h_frac']:6.2f} {c['feet']:4.1f} {c['sharp']:7.1f} {c['face_h']:5.0f} {c['yaw']:6.1f}")
    best = cands[0]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tiles = []
    for i, c in enumerate(cands):
        plate, a, _ = make_plate(c["img"], c["alpha"], set_bgr, fill=1.0 / (1 + 2 * args.margin), side_margin=args.margin,
                                 warm=args.warm)
        c["plate"] = plate
        t = fit_tile(plate, 640)
        t = text_block(t, [f"#{i+1} {Path(c['path']).name}", f"score {c['score']:.2f}",
                           f"sunglasses {c['sg']['frac']:.3f} {'OK' if c['sg_ok'] else 'NONE'}",
                           f"body {c['body_ratio']:.1f} faceH  matte {c['matte_h_frac']:.2f}",
                           f"feet {c['feet']:.1f}  sharp {c['sharp']:.0f}  faceH {c['face_h']:.0f}", f"yaw {c['yaw']:+.1f}"],
                       y0=26, scale=0.5)
        tiles.append(t)
    sheet = np.hstack(tiles)
    sheet_path = out.with_name(out.stem + "_sheet.jpg")
    cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not best["sg_ok"] and not args.no_sunglasses_check:
        print(f"ERROR: no sunglasses detected in {best['path']} (eye-band fraction {best['sg']['frac']:.3f} < "
              f"{args.sunglasses_thr}); best candidate rejected. Wrote {sheet_path.as_posix()} only.")
        return 2
    cv2.imwrite(str(out), best["plate"])
    print(f"wrote {out.as_posix()} ({best['plate'].shape[1]}x{best['plate'].shape[0]}) from {best['path']} and {sheet_path.as_posix()}"
          + ("  [sunglasses check SKIPPED]" if not best["sg_ok"] else ""))
    return 0


# ----------------------------------------------------------------------------- cat reference
def hero_eye_template():
    """Eye+nose band of the upright hero head (refs/cat_src_512.png) with its landmarks."""
    meta = json.load(open(REFS / "cat_meta.json"))
    hero = cv2.imread(str(HERO_512), cv2.IMREAD_COLOR)
    (lx, ly), (rx, ry) = meta["eye_l"], meta["eye_r"]
    iod = meta["iod"]
    x0, x1 = int(lx - 0.75 * iod), int(rx + 0.75 * iod)
    y0, y1 = int(ly - 0.55 * iod), int(ly + 0.85 * iod)
    tmpl = hero[y0:y1, x0:x1]
    return tmpl, ((lx - x0, ly - y0), (rx - x0, ry - y0)), meta


def find_cat_eyes(img: np.ndarray, roi, scales):
    """Multi-scale TM_CCOEFF_NORMED of the hero eye band inside roi (x0,y0,x1,y1) of img.
    Returns eye_l, eye_r (image px), best score and scale."""
    tmpl, (tl, tr), _ = hero_eye_template()
    x0, y0, x1, y1 = roi
    g = cv2.GaussianBlur(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), (0, 0), 1.0)
    tg = cv2.GaussianBlur(cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY), (0, 0), 1.0)
    best = None
    for s in scales:
        tw, th = int(round(tg.shape[1] * s)), int(round(tg.shape[0] * s))
        if tw < 12 or th < 8 or tw >= g.shape[1] or th >= g.shape[0]:
            continue
        t = cv2.resize(tg, (tw, th), interpolation=cv2.INTER_AREA)
        r = cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(r)
        if best is None or mv > best[0]:
            best = (mv, s, ml)
    mv, s, (mx, my) = best
    el = (x0 + mx + tl[0] * s, y0 + my + tl[1] * s)
    er = (x0 + mx + tr[0] * s, y0 + my + tr[1] * s)
    return el, er, mv, s


def rgba_layer(W: int, H: int) -> np.ndarray:
    return np.zeros((H, W, 4), np.uint8)


def paste_rgba(layer: np.ndarray, patch: np.ndarray, x: int, y: int) -> None:
    """Alpha-over `patch` (BGRA) onto `layer` (BGRA) at (x, y), clipped."""
    H, W = layer.shape[:2]
    ph, pw = patch.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + pw), min(H, y + ph)
    if x1 <= x0 or y1 <= y0:
        return
    src = patch[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)
    dst = layer[y0:y1, x0:x1].astype(np.float32)
    a = src[..., 3:4] / 255.0
    da = dst[..., 3:4] / 255.0
    oa = a + da * (1 - a)
    col = (src[..., :3] * a + dst[..., :3] * da * (1 - a)) / np.maximum(oa, 1e-3)
    layer[y0:y1, x0:x1, :3] = np.clip(col, 0, 255).astype(np.uint8)
    layer[y0:y1, x0:x1, 3] = np.clip(oa[..., 0] * 255, 0, 255).astype(np.uint8)


def plaid_texture(W: int, H: int, cell: float, base=(30, 30, 190), dark=(15, 15, 25), angle_deg: float = 0.0) -> np.ndarray:
    """Red/black tartan: two sets of stripes (horizontal + vertical) multiplied, plus a
    thin bright pin stripe, rotated by angle_deg."""
    S = int(math.hypot(W, H)) + 4
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)

    def stripes(u):
        t = (u / cell) % 1.0
        band = np.where((t > 0.30) & (t < 0.70), 1.0, 0.0)                # wide black band
        pin = np.where(np.abs(t - 0.05) < 0.02, 1.0, 0.0)                  # thin light pin stripe
        return band, pin

    bh, ph = stripes(yy)
    bv, pv = stripes(xx)
    darkness = 1.0 - (1.0 - 0.55 * bh) * (1.0 - 0.55 * bv)               # multiplied stripe sets
    base_f = np.array(base, np.float32)[None, None]
    dark_f = np.array(dark, np.float32)[None, None]
    col = base_f * (1 - darkness[..., None]) + dark_f * darkness[..., None]
    pin = np.clip(ph + pv, 0, 1)[..., None]
    col = col * (1 - 0.5 * pin) + np.array((60, 60, 230), np.float32)[None, None] * 0.5 * pin
    weave = 0.92 + 0.08 * (((xx + yy) % 2.0) < 1.0)                       # faint twill
    col = col * weave[..., None]
    M = cv2.getRotationMatrix2D((S / 2, S / 2), angle_deg, 1.0)
    M[0, 2] += W / 2 - S / 2
    M[1, 2] += H / 2 - S / 2
    return cv2.warpAffine(np.clip(col, 0, 255).astype(np.uint8), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def shade(layer_rgb: np.ndarray, mask: np.ndarray, light_dir=(-0.6, -0.8), strength: float = 0.35) -> np.ndarray:
    """Cheap lighting: brighten toward the light direction, darken away from it, using the
    mask's distance-transform gradient as a fake normal."""
    dt = cv2.distanceTransform((mask > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    gx = cv2.Sobel(dt, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(dt, cv2.CV_32F, 0, 1, ksize=5)
    n = np.sqrt(gx * gx + gy * gy) + 1e-3
    l = (gx / n) * light_dir[0] + (gy / n) * light_dir[1]
    l = cv2.GaussianBlur(l, (0, 0), 3)
    edge = np.clip(dt / (0.25 * dt.max() + 1e-3), 0, 1)                    # flat in the middle
    f = 1.0 + strength * (-l) * (1 - edge) - 0.15 * (1 - edge)
    return np.clip(layer_rgb.astype(np.float32) * f[..., None], 0, 255).astype(np.uint8)


def draw_costume(W: int, H: int, alpha: np.ndarray, eye_l, eye_r, args) -> dict[str, np.ndarray]:
    """Procedural costume layers (all BGRA W x H).  Geometry is driven by the eye
    positions / iod and the cat's alpha silhouette."""
    (lx, ly), (rx, ry) = eye_l, eye_r
    iod = math.hypot(rx - lx, ry - ly)
    ang = math.degrees(math.atan2(ry - ly, rx - lx))
    ex, ey = (lx + rx) / 2, (ly + ry) / 2
    x0, y0, x1, y1 = bbox_of(alpha)
    layers: dict[str, np.ndarray] = {}
    ss = 2                                     # supersampling for the vector parts
    Ws, Hs = W * ss, H * ss

    def P(x, y):
        return (int(round(x * ss)), int(round(y * ss)))

    def finish(rgb_s: np.ndarray, a_s: np.ndarray) -> np.ndarray:
        rgb = cv2.resize(rgb_s, (W, H), interpolation=cv2.INTER_AREA)
        a = cv2.resize(a_s, (W, H), interpolation=cv2.INTER_AREA)
        return np.dstack([rgb, a]).astype(np.uint8)

    # ---- head geometry from the alpha: head width on the eye row, top of the head, chin
    row = int(np.clip(round(ey), 0, H - 1))
    cols = np.where(alpha[row] > 0.5)[0]
    head_w = float(cols.max() - cols.min() + 1) if len(cols) else 2.6 * iod
    head_cx = float((cols.max() + cols.min()) / 2) if len(cols) else ex
    top_y = float(y0)
    chin_y = ey + args.chin_iod * iod

    # ---- (1) glasses: draw_glasses works in a 512 canvas -> draw around a local origin, then paste
    g_iod = iod
    gl_meta = {"eye_l": [G512 / 2 - g_iod / 2 * math.cos(math.radians(ang)), G512 / 2 - g_iod / 2 * math.sin(math.radians(ang))],
               "eye_r": [G512 / 2 + g_iod / 2 * math.cos(math.radians(ang)), G512 / 2 + g_iod / 2 * math.sin(math.radians(ang))]}
    gl = draw_glasses(gl_meta, g_iod, args.glasses_r, max(2.0, 7.0 * g_iod / 124.88))
    layer = rgba_layer(W, H)
    paste_rgba(layer, gl, int(round(ex - G512 / 2)), int(round(ey - G512 / 2)))
    layers["glasses"] = layer

    # ---- (2) backwards red cap: crown in front of the head (between the ears), bill behind the head
    red = (25, 25, 200)
    red_d = (15, 15, 120)
    crown_rgb = np.zeros((Hs, Ws, 3), np.uint8)
    crown_a = np.zeros((Hs, Ws), np.uint8)
    cw = args.cap_w * head_w / 2                                   # half width of the crown
    c_top = ey - args.cap_top_iod * iod                            # dome apex (above the skull, like a real crown)
    c_bot = ey - args.cap_brow_iod * iod                           # lower edge above the brow
    hh = c_bot - c_top
    # half-ellipse dome, slightly flattened on top and bulging at the sides
    dome = [[ex - cw, c_bot]]
    for t in np.linspace(math.pi, 0, 24):
        dome.append([ex + cw * math.cos(t) * (1.0 + 0.05 * math.sin(t)), c_bot - hh * math.sin(t) ** 0.85])
    dome.append([ex + cw, c_bot])
    dome = np.array(dome, np.float32)
    R = cv2.getRotationMatrix2D((ex, ey), -ang, 1.0)               # follow the head roll
    dome_r = cv2.transform(dome[None], R)[0]
    cv2.fillPoly(crown_a, [np.round(dome_r * ss).astype(np.int32)], 255, cv2.LINE_AA)
    crown_rgb[:] = red
    # panel seams + button
    for k in (-0.5, 0.0, 0.5):
        p0 = cv2.transform(np.float32([[[ex + k * cw * 0.9, c_bot]]]), R)[0, 0]
        p1 = cv2.transform(np.float32([[[ex + k * cw * 0.15, c_top + 0.02 * iod]]]), R)[0, 0]
        cv2.line(crown_rgb, P(*p0), P(*p1), red_d, max(1, ss), cv2.LINE_AA)
    btn = cv2.transform(np.float32([[[ex, c_top + 0.02 * iod]]]), R)[0, 0]
    cv2.circle(crown_rgb, P(*btn), int(0.05 * iod * ss), red_d, -1, cv2.LINE_AA)
    # sweatband strip along the lower edge
    band_a = np.zeros((Hs, Ws), np.uint8)
    cv2.line(band_a, P(*dome_r[0]), P(*dome_r[-1]), 255, int(max(2, 0.10 * iod * ss)), cv2.LINE_AA)
    band = (band_a > 0) & (crown_a > 0)
    crown_rgb[band] = red_d
    crown = finish(crown_rgb, crown_a)
    crown[..., :3] = shade(crown[..., :3], crown[..., 3] / 255.0, strength=0.4)
    layers["cap_crown"] = crown

    bill_rgb = np.zeros((Hs, Ws, 3), np.uint8)
    bill_a = np.zeros((Hs, Ws), np.uint8)
    bill_rgb[:] = red
    # the bill sticks out at the back: for a frontal head it peeks out on the side the head
    # turns away from (args.bill_side: -1 image-left, +1 image-right) as a flat curved wedge
    side = args.bill_side
    b_len = args.bill_len * iod
    b_y = c_bot - 0.30 * hh                                        # the bill leaves the crown at its lower third
    bill = np.array([[ex + side * cw * 0.5, b_y - 0.16 * iod], [ex + side * (cw + b_len * 0.55), b_y - 0.30 * iod],
                     [ex + side * (cw + b_len), b_y - 0.20 * iod], [ex + side * (cw + b_len * 1.02), b_y - 0.02 * iod],
                     [ex + side * (cw + b_len * 0.8), b_y + 0.14 * iod], [ex + side * cw * 0.5, b_y + 0.16 * iod]], np.float32)
    bill_r = cv2.transform(bill[None], R)[0]
    cv2.fillPoly(bill_a, [np.round(bill_r * ss).astype(np.int32)], 255, cv2.LINE_AA)
    cv2.polylines(bill_rgb, [np.round(bill_r * ss).astype(np.int32)], True, red_d, max(1, ss), cv2.LINE_AA)
    billL = finish(bill_rgb, bill_a)
    billL[..., :3] = shade(billL[..., :3], billL[..., 3] / 255.0, strength=0.3)
    layers["cap_bill"] = billL

    # ---- (3) plaid flannel: shoulders + chest below the chin, V opening, collar lapels
    sh_y = chin_y + args.shirt_neck_iod * iod                       # shoulder line
    body_cols = np.where(alpha[int(np.clip(sh_y + 0.6 * iod, 0, H - 1))] > 0.5)[0]
    body_w = float(body_cols.max() - body_cols.min() + 1) if len(body_cols) else head_w * 1.3
    body_cx = float((body_cols.max() + body_cols.min()) / 2) if len(body_cols) else ex
    sw = max(body_w, head_w) * args.shirt_w / 2                       # shoulder half width (padded)
    bot = min(y1 - 0.3 * iod, sh_y + args.shirt_len_iod * iod)        # hem at mid torso, paws stay fur
    shirt = np.array([[body_cx - sw * 0.35, sh_y - 0.35 * iod], [body_cx - sw, sh_y + 0.35 * iod],
                      [body_cx - sw * 1.02, sh_y + 1.4 * iod], [body_cx - sw * 0.90, bot - 0.35 * iod],
                      [body_cx - sw * 0.55, bot], [body_cx, bot + 0.08 * iod], [body_cx + sw * 0.55, bot],
                      [body_cx + sw * 0.90, bot - 0.35 * iod], [body_cx + sw * 1.02, sh_y + 1.4 * iod],
                      [body_cx + sw, sh_y + 0.35 * iod], [body_cx + sw * 0.35, sh_y - 0.35 * iod]], np.float32)
    shirt_a = np.zeros((Hs, Ws), np.uint8)
    cv2.fillPoly(shirt_a, [np.round(shirt * ss).astype(np.int32)], 255, cv2.LINE_AA)
    # keep the shirt on the body: body alpha dilated by ~4 % of the shoulder width (padded shoulders)
    body = cv2.resize((alpha > 0.5).astype(np.uint8), (Ws, Hs), interpolation=cv2.INTER_NEAREST)
    kd = int(max(3, 0.04 * sw * ss)) | 1
    body = cv2.dilate(body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd)))
    shirt_a = cv2.bitwise_and(shirt_a, body * 255)
    # V opening down the chest (fur shows) and the lapels
    v_depth = args.shirt_v_iod * iod
    vpoly = np.array([[body_cx - 0.28 * sw, sh_y - 0.4 * iod], [body_cx + 0.28 * sw, sh_y - 0.4 * iod],
                      [body_cx, sh_y + v_depth]], np.float32)
    cv2.fillPoly(shirt_a, [np.round(vpoly * ss).astype(np.int32)], 0, cv2.LINE_AA)
    tex = plaid_texture(Ws, Hs, cell=args.plaid_cell * iod * ss, angle_deg=-6.0)
    shirt_rgb = tex.copy()
    # lapels: two folded triangles beside the V in slightly darker plaid, with a seam line
    lap_a = np.zeros((Hs, Ws), np.uint8)
    for sgn in (-1, 1):
        lap = np.array([[body_cx + sgn * 0.28 * sw, sh_y - 0.4 * iod], [body_cx + sgn * 0.62 * sw, sh_y - 0.05 * iod],
                        [body_cx + sgn * 0.30 * sw, sh_y + v_depth * 0.55], [body_cx + sgn * 0.02 * sw, sh_y + v_depth * 0.85]],
                       np.float32)
        cv2.fillPoly(lap_a, [np.round(lap * ss).astype(np.int32)], 255, cv2.LINE_AA)
        cv2.polylines(shirt_rgb, [np.round(lap * ss).astype(np.int32)], True, (10, 10, 40), max(1, ss), cv2.LINE_AA)
    lap_tex = plaid_texture(Ws, Hs, cell=args.plaid_cell * iod * ss, angle_deg=40.0)
    lm = (lap_a > 0)
    shirt_rgb[lm] = (lap_tex[lm].astype(np.float32) * 0.85).astype(np.uint8)
    shirt_a = np.maximum(shirt_a, cv2.bitwise_and(lap_a, body * 255))
    # a couple of buttons down the placket below the V
    for k in range(3):
        cy = sh_y + v_depth + (0.5 + 0.55 * k) * iod
        if cy < bot - 0.2 * iod:
            cv2.circle(shirt_rgb, P(body_cx, cy), int(0.045 * iod * ss), (40, 40, 40), -1, cv2.LINE_AA)
    shirtL = finish(shirt_rgb, shirt_a)
    shirtL[..., :3] = shade(shirtL[..., :3], shirtL[..., 3] / 255.0, strength=0.45)
    layers["flannel"] = shirtL

    # ---- (4) wooden stool: seat ellipse under the rump (behind the cat) + legs down to the plate bottom
    seat_cy = y1 - args.stool_sink * iod
    rump_w = float(max(int((alpha[y] > 0.5).sum()) for y in range(int(y0 + 0.5 * (y1 - y0)), y1)))
    seat_rx = rump_w * args.stool_w / 2
    seat_ry = seat_rx * 0.28
    wood = (60, 110, 175)
    wood_d = (35, 65, 110)
    st_rgb = np.zeros((Hs, Ws, 3), np.uint8)
    st_a = np.zeros((Hs, Ws), np.uint8)
    leg_w = int(max(2, 0.10 * iod * ss))
    leg_bot = H - 0.02 * H
    for k in (-0.8, 0.0, 0.8):
        lx0 = body_cx + k * seat_rx * 0.9
        lx1 = body_cx + k * seat_rx * 1.15
        cv2.line(st_a, P(lx0, seat_cy), P(lx1, leg_bot), 255, leg_w, cv2.LINE_AA)
        cv2.line(st_rgb, P(lx0, seat_cy), P(lx1, leg_bot), wood_d if k == 0 else wood, leg_w, cv2.LINE_AA)
    # rung
    cv2.line(st_a, P(body_cx - seat_rx * 1.0, (seat_cy + leg_bot) / 2), P(body_cx + seat_rx * 1.0, (seat_cy + leg_bot) / 2), 255, leg_w // 2 + 1, cv2.LINE_AA)
    cv2.line(st_rgb, P(body_cx - seat_rx * 1.0, (seat_cy + leg_bot) / 2), P(body_cx + seat_rx * 1.0, (seat_cy + leg_bot) / 2), wood_d, leg_w // 2 + 1, cv2.LINE_AA)
    # seat: thick disc = top ellipse + rim
    rim = int(0.18 * iod)
    cv2.ellipse(st_a, P(body_cx, seat_cy + rim), (int(seat_rx * ss), int(seat_ry * ss)), 0, 0, 360, 255, -1, cv2.LINE_AA)
    cv2.ellipse(st_rgb, P(body_cx, seat_cy + rim), (int(seat_rx * ss), int(seat_ry * ss)), 0, 0, 360, wood_d, -1, cv2.LINE_AA)
    cv2.rectangle(st_a, P(body_cx - seat_rx, seat_cy), P(body_cx + seat_rx, seat_cy + rim), 255, -1)
    cv2.rectangle(st_rgb, P(body_cx - seat_rx, seat_cy), P(body_cx + seat_rx, seat_cy + rim), wood_d, -1)
    cv2.ellipse(st_a, P(body_cx, seat_cy), (int(seat_rx * ss), int(seat_ry * ss)), 0, 0, 360, 255, -1, cv2.LINE_AA)
    # top with a light-to-dark gradient and a few grain lines
    top = np.zeros((Hs, Ws), np.uint8)
    cv2.ellipse(top, P(body_cx, seat_cy), (int(seat_rx * ss), int(seat_ry * ss)), 0, 0, 360, 255, -1, cv2.LINE_AA)
    yy = np.arange(Hs, dtype=np.float32)[:, None]
    grad = np.clip((yy - (seat_cy - seat_ry) * ss) / max(1.0, 2 * seat_ry * ss), 0, 1)
    top_col = np.array((90, 150, 210), np.float32)[None, None] * (1 - grad[..., None] * 0.35) + 0
    tm = top > 0
    st_rgb[tm] = np.clip(np.broadcast_to(top_col, (Hs, Ws, 3))[tm], 0, 255).astype(np.uint8)
    for k in range(-3, 4):
        yk = seat_cy + k * seat_ry * 0.22
        cv2.ellipse(st_rgb, P(body_cx, seat_cy), (int(seat_rx * ss * (1 - abs(k) * 0.06)), int(seat_ry * ss * (1 - abs(k) * 0.06))),
                    0, 0, 360, (70, 120, 180), max(1, ss // 2), cv2.LINE_AA) if k else None
    layers["stool"] = finish(st_rgb, st_a)
    return layers


def paste_hero_head(base: np.ndarray, alpha: np.ndarray, eye_l, eye_r, set_bgr, args):
    """Replace the small/soft head with the hero head (refs/cat_src_512 + alpha), aligned by
    the eye centres (similarity), old head above the chin blanked to set orange first."""
    meta = json.load(open(REFS / "cat_meta.json"))
    hero = cv2.imread(str(HERO_512), cv2.IMREAD_COLOR)
    halpha = cv2.imread(str(HERO_ALPHA_512), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    # cat_src_512 is already composited over its set orange: un-premultiply so the faded neck
    # does not get the orange blended in twice
    hero_set = np.array(meta["set_bgr"], np.float32)[None, None]
    ha0 = np.maximum(halpha, 0.02)[..., None]
    hero = np.clip((hero.astype(np.float32) - hero_set * (1 - ha0)) / ha0, 0, 255).astype(np.uint8)
    H, W = base.shape[:2]
    ex, ey = (eye_l[0] + eye_r[0]) / 2, (eye_l[1] + eye_r[1]) / 2
    iod = math.hypot(eye_r[0] - eye_l[0], eye_r[1] - eye_l[1])
    ang = math.atan2(eye_r[1] - eye_l[1], eye_r[0] - eye_l[0])
    if args.hq_align == "width":
        # closed eyes make the iod estimate soft: match the head width on the eye row instead
        row = int(np.clip(round(ey), 0, H - 1))
        cc = np.where(alpha[row] > 0.5)[0]
        head_w = float(cc.max() - cc.min() + 1) if len(cc) else iod * meta["head_w"] / meta["iod"]
        s = head_w / meta["head_w"]
        ex = float((cc.max() + cc.min()) / 2) if len(cc) else ex
    else:
        s = iod / meta["iod"]
    M = cv2.getRotationMatrix2D(tuple(meta["eye_c"]), -math.degrees(ang), s)
    tx, ty = M @ np.array([meta["eye_c"][0], meta["eye_c"][1], 1.0])
    M[0, 2] += ex - tx
    M[1, 2] += ey - ty
    hero_eyes = {k: tuple(float(v) for v in (M @ np.array([meta[k][0], meta[k][1], 1.0]))) for k in ("eye_l", "eye_r")}
    hw = cv2.warpAffine(hero, M, (W, H), flags=cv2.INTER_LANCZOS4 if s > 1 else cv2.INTER_AREA,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=tuple(int(v) for v in set_bgr))
    ha = cv2.warpAffine(halpha, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    ha = np.clip(ha * args.hq_head_scale_alpha, 0, 1)
    ex, ey = (eye_l[0] + eye_r[0]) / 2, (eye_l[1] + eye_r[1]) / 2
    iod = math.hypot(eye_r[0] - eye_l[0], eye_r[1] - eye_l[1])
    chin_y = ey + args.chin_iod * iod
    yy = np.arange(H, dtype=np.float32)[:, None]
    # dissolve the hero neck into the body's chest over ~0.5 iod below the chin
    ha = ha * np.clip((chin_y + 0.55 * iod - yy) / (0.5 * iod), 0, 1)
    # blank the old head above the chin line (soft ramp) so nothing peeks around the new head
    ramp = np.clip((chin_y + 0.10 * iod - yy) / (0.3 * iod), 0, 1)     # 1 above the chin, 0 just below it
    # ... but only within the old head's horizontal extent, so the shoulders beside the neck survive
    xx = np.arange(W, dtype=np.float32)[None, :]
    half = 0.5 * (meta["head_w"] * s) if args.hq_align == "width" else 0.5 * iod * meta["head_w"] / meta["iod"]
    win = np.clip((1.12 * half - np.abs(xx - ex)) / (0.15 * half), 0, 1)
    hard = cv2.dilate((alpha > 0.15).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32)
    old_head = cv2.GaussianBlur(hard * ramp * win, (0, 0), 1.5)
    plate = np.empty_like(base)
    plate[:] = np.array(set_bgr, np.uint8)
    base_wo = np.clip(base.astype(np.float32) * (1 - old_head[..., None]) + plate.astype(np.float32) * old_head[..., None], 0, 255)
    new_alpha = np.clip(alpha * (1 - ramp) + ha, 0, 1)
    out = np.clip(base_wo * (1 - ha[..., None]) + hw.astype(np.float32) * ha[..., None], 0, 255).astype(np.uint8)
    info = {"hero_scale": round(s, 4), "hero_iod_px": round(meta["iod"] * s, 1), "align": args.hq_align,
            "eye_l": [round(v, 1) for v in hero_eyes["eye_l"]], "eye_r": [round(v, 1) for v in hero_eyes["eye_r"]]}
    return out, new_alpha, info


def cmd_cat(args) -> int:
    set_bgr = set_colour()
    isnet = ISNet(Path(args.isnet))
    img = read_image(args.photo)
    # trim letterbox bars (screenshot)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows = np.where(g.max(1) > 20)[0]
    cols = np.where(g.max(0) > 20)[0]
    img = img[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    H, W = img.shape[:2]
    print(f"photo {args.photo}: content {W}x{H} (bars trimmed at rows {rows.min()}..{rows.max()})")
    # pass 1: coarse matte on the whole photo -> cat bbox (or --roi)
    if args.roi:
        x0, y0, x1, y1 = [int(v) for v in args.roi.split(",")]
    else:
        coarse = largest_component(isnet(img))
        x0, y0, x1, y1 = bbox_of(coarse)
        print(f"  coarse cat bbox {x0},{y0},{x1},{y1}")
    # pass 2: matte on a 2x bbox window (cat fills the ISNet input)
    cw, ch = x1 - x0, y1 - y0
    px0, py0 = int(max(0, x0 - 0.5 * cw)), int(max(0, y0 - 0.5 * ch))
    px1, py1 = int(min(W, x1 + 0.5 * cw)), int(min(H, y1 + 0.5 * ch))
    win = img[py0:py1, px0:px1]
    fine = largest_component(isnet(win))
    alpha = np.zeros((H, W), np.float32)
    alpha[py0:py1, px0:px1] = fine
    alpha = clean_alpha(alpha, 2.0)
    bx0, by0, bx1, by1 = bbox_of(alpha)
    print(f"  fine cat bbox {bx0},{by0},{bx1},{by1} ({bx1-bx0}x{by1-by0} px)")
    # eyes by template matching against the hero eye band.  Prior: the head width (widest
    # alpha row in the top quarter of the cat, below the ear tips) relates to the iod as in the
    # hero (head_w 265 / iod 124.88) -> template scale searched at +-25 % of that, in the
    # band 8..32 % down the cat.
    hmeta = json.load(open(REFS / "cat_meta.json"))
    bh = by1 - by0
    rows_w = [(int((alpha[y] > 0.5).sum()), y) for y in range(int(by0 + 0.12 * bh), int(by0 + 0.30 * bh))]
    head_w0 = float(max(rows_w)[0]) if rows_w else 0.6 * (bx1 - bx0)
    iod_prior = head_w0 * hmeta["iod"] / hmeta["head_w"]
    s_prior = iod_prior / hmeta["iod"]
    roi = (max(0, bx0 - 10), max(0, int(by0 + 0.05 * bh)), min(W, bx1 + 10), int(by0 + 0.40 * bh))
    if args.eyes:
        v = [float(t) for t in args.eyes.split(",")]
        el, er, score, tscale = (v[0], v[1]), (v[2], v[3]), 1.0, 0.0
    else:
        el, er, score, tscale = find_cat_eyes(img, roi, np.linspace(0.75 * s_prior, 1.25 * s_prior, 21))
    print(f"  head width prior {head_w0:.0f} px -> iod prior {iod_prior:.1f} px (template scale {s_prior:.3f})")
    iod0 = math.hypot(er[0] - el[0], er[1] - el[1])
    print(f"  eyes (photo px) L=({el[0]:.0f},{el[1]:.0f}) R=({er[0]:.0f},{er[1]:.0f}) iod={iod0:.1f} tm_score={score:.3f} tm_scale={tscale:.3f}")
    # upright: rotate so the eye line is level (only if the roll is meaningful)
    roll = math.degrees(math.atan2(er[1] - el[1], er[0] - el[0]))
    if abs(roll) > args.max_roll:
        R = cv2.getRotationMatrix2D(((el[0] + er[0]) / 2, (el[1] + er[1]) / 2), roll, 1.0)
        img = cv2.warpAffine(img, R, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
        alpha = cv2.warpAffine(alpha, R, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        el = tuple(cv2.transform(np.float32([[el]]), R)[0, 0])
        er = tuple(cv2.transform(np.float32([[er]]), R)[0, 0])
        print(f"  levelled the eye line: roll {roll:.1f} deg")
    plate, a, (s, ox, oy) = make_plate(img, alpha, set_bgr, fill=args.fill, side_margin=args.margin,
                                       min_aspect=args.min_aspect, warm=args.warm, feather=3.0)
    EL = (el[0] * s + ox, el[1] * s + oy)
    ER = (er[0] * s + ox, er[1] * s + oy)
    iod = math.hypot(ER[0] - EL[0], ER[1] - EL[1])
    ey = (EL[1] + ER[1]) / 2
    row = int(np.clip(round(ey), 0, PLATE_H - 1))
    cc = np.where(a[row] > 0.5)[0]
    head_w = float(cc.max() - cc.min() + 1) if len(cc) else 0
    pb = bbox_of(a)
    print(f"  plate {plate.shape[1]}x{plate.shape[0]}: cat bbox {pb}, cat height {pb[3]-pb[1]} px "
          f"({(pb[3]-pb[1])/PLATE_H:.2f} of H); FACE: iod={iod:.1f} px, head width on eye row={head_w:.0f} px; "
          f"photo-space head width ~{head_w/s:.0f} px (scale {s:.3f})")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), plate)
    cv2.imwrite(str(out.with_name(out.stem + "_alpha.png")), (a * 255).round().astype(np.uint8))
    dbg = plate.copy()
    for p in (EL, ER):
        cv2.circle(dbg, (int(p[0]), int(p[1])), 5, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(WORK / "round4" / "cat_base_eyes.jpg"), dbg, [cv2.IMWRITE_JPEG_QUALITY, 90])
    meta = {"eye_l": [round(EL[0], 1), round(EL[1], 1)], "eye_r": [round(ER[0], 1), round(ER[1], 1)], "iod": round(iod, 1),
            "head_w": round(head_w, 1), "bbox": list(pb), "plate": [plate.shape[1], plate.shape[0]], "set_bgr": list(set_bgr),
            "photo": args.photo, "tm_score": round(float(score), 3), "photo_to_plate_scale": round(s, 4)}

    # ---- HQ head
    hq, hq_alpha, info = None, None, {}
    if not args.no_hq:
        hq, hq_alpha, info = paste_hero_head(plate, a, EL, ER, set_bgr, args)
        if hq is not None:
            cv2.imwrite(str(out.with_name(out.stem + "_hq.png")), hq)
            cv2.imwrite(str(out.with_name(out.stem + "_hq_alpha.png")), (hq_alpha * 255).round().astype(np.uint8))
            print(f"  wrote {out.with_name(out.stem + '_hq.png').as_posix()} {info}")
            meta["hq"] = info

    # ---- costume
    if args.costume:
        use_img, use_a = (hq, hq_alpha) if (hq is not None and not args.costume_on_base) else (plate, a)
        cEL, cER = (tuple(info["eye_l"]), tuple(info["eye_r"])) if use_img is hq else (EL, ER)
        layers = draw_costume(plate.shape[1], PLATE_H, use_a, cEL, cER, args)
        COSTUME_DIR.mkdir(parents=True, exist_ok=True)
        order_behind = ["stool", "cap_bill"]
        order_front = ["flannel", "glasses", "cap_crown"]
        for k, L in layers.items():
            cv2.imwrite(str(COSTUME_DIR / f"{k}.png"), L)
        bg = np.empty_like(use_img)
        bg[:] = np.array(set_bgr, np.uint8)
        comp = bg.copy()
        for k in order_behind:
            comp = over(comp, layers[k])
        cat_rgba = np.dstack([use_img, (use_a * 255).round().astype(np.uint8)])
        comp = over(comp, cat_rgba)
        for k in order_front:
            comp = over(comp, layers[k])
        cv2.imwrite(str(REFS / "cat_ref_costume.png"), comp)
        with open(COSTUME_DIR / "layers.json", "w", newline="\n") as f:
            json.dump({"order": order_behind + ["<cat>"] + order_front, "eye_l": list(cEL), "eye_r": list(cER),
                       "plate": meta["plate"], "base": ("hq" if use_img is hq else "base")}, f, indent=2)
        ex = cv2.imread(str(WORK / "example" / "ex_2.jpg"))
        tiles = [label(fit_tile(plate, 768), "base"), label(fit_tile(hq if hq is not None else plate, 768), "hq head" if hq is not None else "(no hq)"),
                 label(fit_tile(comp, 768), "costume"), label(fit_tile(ex, 768), "example ex_2")]
        cv2.imwrite(str(REFS / "cat_ref_sheet.jpg"), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"  wrote refs/cat_ref_costume.png refs/cat_ref_sheet.jpg and {len(layers)} layers in assets/cat_costume/")
    with open(out.with_suffix(".json"), "w", newline="\n") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {out.as_posix()} + {out.with_suffix('.json').name}")
    return 0


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--isnet", default=str(MODELS / "isnet-general-use.onnx"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    u = sub.add_parser("user")
    u.add_argument("--photos", nargs="+", required=True)
    u.add_argument("--out", default=str(REFS / "user_ref.png"))
    u.add_argument("--margin", type=float, default=0.06)
    u.add_argument("--warm", type=float, default=0.07)
    u.add_argument("--sunglasses-thr", type=float, default=SUNGLASSES_THR)
    u.add_argument("--no-sunglasses-check", action="store_true", help="stand-in tests only")

    c = sub.add_parser("calibrate")
    c.add_argument("--every", type=int, default=30)
    c.add_argument("--extra-none", nargs="*", default=[])

    k = sub.add_parser("cat")
    k.add_argument("--photo", default="faces/cat/IMG_9318.PNG")
    k.add_argument("--out", default=str(REFS / "cat_body_base.png"))
    k.add_argument("--roi", default=None, help="x0,y0,x1,y1 cat bbox in the bar-trimmed photo (skips the coarse pass)")
    k.add_argument("--eyes", default=None, help="lx,ly,rx,ry in the bar-trimmed photo (skips template matching)")
    k.add_argument("--fill", type=float, default=0.70)
    k.add_argument("--margin", type=float, default=0.10)
    k.add_argument("--min-aspect", type=float, default=0.62)
    k.add_argument("--warm", type=float, default=0.07)
    k.add_argument("--max-roll", type=float, default=3.0)
    k.add_argument("--no-hq", action="store_true")
    k.add_argument("--hq-head-scale-alpha", type=float, default=1.0)
    k.add_argument("--costume", action="store_true")
    k.add_argument("--costume-on-base", action="store_true", help="draw the costume on the base instead of the hq head")
    k.add_argument("--chin-iod", type=float, default=1.15, help="chin below the eye line, in iod")
    k.add_argument("--glasses-r", type=float, default=0.44)
    k.add_argument("--cap-w", type=float, default=1.08, help="crown width / head width")
    k.add_argument("--cap-top-iod", type=float, default=1.2, help="crown apex above the eyes, in iod")
    k.add_argument("--cap-brow-iod", type=float, default=0.42, help="crown lower edge above the eyes, in iod")
    k.add_argument("--bill-side", type=int, default=-1)
    k.add_argument("--bill-len", type=float, default=0.85)
    k.add_argument("--shirt-neck-iod", type=float, default=0.25)
    k.add_argument("--shirt-w", type=float, default=1.15)
    k.add_argument("--shirt-v-iod", type=float, default=1.1)
    k.add_argument("--shirt-len-iod", type=float, default=3.2, help="shirt hem below the shoulder line, in iod")
    k.add_argument("--plaid-cell", type=float, default=0.42)
    k.add_argument("--stool-w", type=float, default=1.0, help="seat width / rump width")
    k.add_argument("--stool-sink", type=float, default=0.9, help="seat centre above the paw bottom, in iod")
    k.add_argument("--hq-align", choices=["width", "iod"], default="width", help="hero head scale from the head width or the iod")
    args = ap.parse_args()
    (WORK / "round4").mkdir(parents=True, exist_ok=True)
    if args.cmd == "user":
        sys.exit(cmd_user(args))
    if args.cmd == "calibrate":
        cmd_calibrate(args)
        sys.exit(0)
    sys.exit(cmd_cat(args))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Build the LivePortrait SOURCE reference for the cat head (interface contract B).

Inputs
  faces/cat/IMG_9325.PNG   hero photo (frontal, sharpest; head tilted ~15 deg, a
                           gloved hand holds the cat under the chin, steel table)
  work/cut.mp4             to sample the COLORS set orange near the RIGHT head
  models/isnet-general-use.onnx   rembg ISNet matting model (downloaded once)

Outputs (all 512x512, same coordinate space)
  refs/cat_src_512.png          cat head upright on set orange (LivePortrait source)
  refs/cat_src_glasses_512.png  same + procedural blue round glasses
  refs/cat_alpha_512.png        8-bit soft alpha of head+neck (255 = cat)
  refs/cat_meta.json            landmarks in 512 coords + set colour
  assets/glasses_blue_512.png   the glasses alone, RGBA
  work/cat_ref_sheet.jpg        [crop | alpha | src | src+glasses]

Usage
  python make_cat_ref.py                       # defaults = tuned hero landmarks
  python make_cat_ref.py --points 432,449,762,360,632,600,650,706
  python make_cat_ref.py --no-matte            # skip ISNet, alpha = keep-region only
  python make_cat_ref.py --side-factor 4.0 --eye-y 0.42
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
REFS = ROOT / "refs"
ASSETS = ROOT / "assets"
WORK = ROOT / "work"
MODELS = ROOT / "models"

HERO = "faces/cat/IMG_9325.PNG"
# Hero-photo landmarks in full-resolution pixel coords (1125x1484), placed by eye
# on a gridded copy and refined on the reference sheet.
# order: eye_l (image-left), eye_r (image-right), nose tip, mouth cleft
HERO_POINTS = {
    "eye_l": (432.0, 449.0),
    "eye_r": (762.0, 360.0),
    "nose": (632.0, 600.0),
    "mouth": (650.0, 706.0),
}

ISNET_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx"
ISNET_BYTES = 178_648_008
OUT = 512
FRAME_BGR = (200, 120, 40)   # blue round frames


# ----------------------------------------------------------------------------- helpers
def parse_points(s: str | None) -> dict[str, tuple[float, float]]:
    if not s:
        return dict(HERO_POINTS)
    v = [float(x) for x in s.split(",")]
    if len(v) != 8:
        sys.exit("--points needs 8 numbers: eye_l_x,eye_l_y,eye_r_x,eye_r_y,nose_x,nose_y,mouth_x,mouth_y")
    return {"eye_l": (v[0], v[1]), "eye_r": (v[2], v[3]), "nose": (v[4], v[5]), "mouth": (v[6], v[7])}


def apply_affine(M: np.ndarray, p) -> tuple[float, float]:
    x, y = p
    return (float(M[0, 0] * x + M[0, 1] * y + M[0, 2]), float(M[1, 0] * x + M[1, 1] * y + M[1, 2]))


def upright_affine(pts: dict, side_factor: float, eye_y: float, cx_shift: float) -> tuple[np.ndarray, float, float]:
    """Similarity transform: rotate so the eye line is level, scale so a crop of
    side_factor * iod maps to OUT px, and place the eye midpoint at
    (OUT/2 + cx_shift*OUT, eye_y*OUT). Returns (2x3 matrix, roll_deg, scale)."""
    (lx, ly), (rx, ry) = pts["eye_l"], pts["eye_r"]
    dx, dy = rx - lx, ry - ly
    roll = math.degrees(math.atan2(dy, dx))         # > 0 : right eye lower than left
    iod = math.hypot(dx, dy)
    scale = OUT / (side_factor * iod)
    ex, ey = (lx + rx) / 2.0, (ly + ry) / 2.0
    M = cv2.getRotationMatrix2D((ex, ey), roll, scale)   # positive = CCW in image coords -> levels the eyes
    tx, ty = apply_affine(M, (ex, ey))
    M[0, 2] += OUT / 2.0 + cx_shift * OUT - tx
    M[1, 2] += eye_y * OUT - ty
    return M, roll, scale


def sample_set_colour(video: Path, frame_idx: int, analysis: Path) -> tuple[list[int], tuple[int, int]]:
    """Median BGR of a 21x21 backdrop patch ~40 px above the RIGHT performer's
    head bbox at frame_idx (fixed fallback point if analysis.json is missing)."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"cannot read frame {frame_idx} of {video}")
    fps = 30.0
    px, py = 700, 30
    if analysis.exists():
        a = json.load(open(analysis))
        t = frame_idx / fps
        R = [r for r in a["tracks"]["R"] if r]
        r = min(R, key=lambda r: abs(r["t"] - t))
        px = int(r["cx"] + 0.8 * r["w"])                 # beside the head, away from the mic
        py = int(max(12, r["cy"] - r["h"] / 2 - 40))
    h, w = fr.shape[:2]
    px, py = int(np.clip(px, 12, w - 13)), int(np.clip(py, 12, h - 13))
    patch = fr[py - 10:py + 11, px - 10:px + 11].reshape(-1, 3)
    bgr = [int(v) for v in np.median(patch, axis=0)]
    return bgr, (px, py)


# ----------------------------------------------------------------------------- matting
def ensure_isnet(path: Path) -> Path:
    if path.exists() and path.stat().st_size == ISNET_BYTES:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print("downloading isnet-general-use.onnx ...")
    urllib.request.urlretrieve(ISNET_URL, path)
    if path.stat().st_size != ISNET_BYTES:
        sys.exit(f"isnet download size mismatch: {path.stat().st_size} != {ISNET_BYTES}")
    return path


def isnet_matte(bgr512: np.ndarray, model: Path) -> np.ndarray:
    """ISNet general-use on the 512 crop upscaled to 1024. Returns float32 alpha
    in [0,1] at 512x512 (min-max normalised, no extra sigmoid)."""
    import onnxruntime as ort

    providers = [p for p in ("DmlExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    sess = ort.InferenceSession(str(model), providers=providers)
    inp = sess.get_inputs()[0].name
    rgb = cv2.cvtColor(cv2.resize(bgr512, (1024, 1024), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = (x - 0.5) / 1.0
    x = np.transpose(x, (2, 0, 1))[None]
    out = sess.run(None, {inp: x})[0]
    m = out[0, 0].astype(np.float32)
    m = (m - m.min()) / max(1e-6, float(m.max() - m.min()))
    print(f"  isnet provider={sess.get_providers()[0]} raw range=({out.min():.3f},{out.max():.3f})")
    return cv2.resize(m, (OUT, OUT), interpolation=cv2.INTER_AREA)


# Hand-traced head+neck outline in hero-photo pixel coords (clockwise from the
# left ear tip). Tight where the gloved hand / table touch the fur (left cheek,
# under the jaw, right temple), loose where the ISNet matte does the work (ear
# tips, forehead against the dark scrubs). The clipped right ear is tapered to a
# point at the photo edge.
HERO_KEEP_POLY = [
    (35, 70), (95, 215), (150, 330), (198, 400), (203, 480),            # left ear + temple
    (212, 560), (225, 640), (268, 720), (335, 780),                     # left cheek + jaw (glove thumb outside)
    (445, 800), (505, 850), (530, 905), (620, 925),                     # neck left (left-hand fingers outside)
    (700, 890), (760, 850), (875, 790),                                 # neck right + right jaw
    (945, 700), (985, 600), (945, 520), (915, 440),                     # right cheek (palm outside)
    (908, 360), (903, 250), (893, 130), (888, 60), (858, 5), (812, 12),  # right temple + tapered clipped ear
    (752, 60), (748, 120), (725, 165), (680, 185),                      # right ear inner edge (glove fingers above)
    (560, 180), (450, 172), (330, 170), (150, 40),                      # forehead (tight) + loose left ear top
]


def keep_region(M: np.ndarray, meta512: dict, iod: float, poly=HERO_KEEP_POLY) -> np.ndarray:
    """Soft 'keep' mask in 512 space from the hero-coordinate outline mapped
    through the crop affine, with a vertical fade over the neck so the cutout
    dissolves into the performer's collar. Everything outside (glove, paws,
    table, human) is dropped even if the matte keeps it."""
    pts = np.array([apply_affine(M, p) for p in poly], np.float32)
    m = np.zeros((OUT, OUT), np.float32)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1.0, cv2.LINE_AA)
    chin = meta512["chin_y"]
    y0, y1 = chin + 0.12 * iod, float(pts[:, 1].max())
    yy = np.arange(OUT, dtype=np.float32)[:, None]
    fade = np.clip((y1 - yy) / max(1.0, y1 - y0), 0, 1) ** 0.8
    m = m * fade
    m = cv2.GaussianBlur(m, (0, 0), 0.025 * iod)
    return np.clip(m, 0, 1)


def clean_alpha(a: np.ndarray, feather_px: float) -> np.ndarray:
    a8 = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    a8 = cv2.morphologyEx(a8, cv2.MORPH_CLOSE, k, iterations=2)
    a8 = cv2.morphologyEx(a8, cv2.MORPH_OPEN, k, iterations=1)
    # keep the largest connected component of the solid core, re-admit the soft edge around it
    core = (a8 > 128).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(core, 8)
    if n > 1:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        core = (lab == big).astype(np.uint8)
    core = cv2.dilate(core, k, iterations=3)
    a = a8.astype(np.float32) / 255.0 * core
    a = cv2.GaussianBlur(a, (0, 0), feather_px / 2.0)
    return np.clip(a, 0, 1)


def mirror_clipped_ear(crop: np.ndarray, alpha: np.ndarray, valid: np.ndarray, meta512: dict, iod: float,
                       feather_px: float = 6.0) -> tuple[np.ndarray, np.ndarray, dict]:
    """IMG_9325 clips the image-right ear at the photo's top edge. Rebuild its tip
    from the image-left ear: mirror crop+alpha about the head axis (eye_c x),
    align the mirrored ear's outer edge to the intact lower part of the right
    ear, and paste it (colour + alpha) only where the photo is missing / the
    alpha is empty above the clip line, with a soft feather."""
    H, W = alpha.shape
    axis = meta512["eye_c"][0]
    ey = meta512["eye_c"][1]
    m_crop = cv2.flip(crop, 1)
    m_alpha = cv2.flip(alpha, 1)
    # mirror about x=axis: flipped x' = W-1-x ; want x'' = 2*axis - x  -> shift by 2*axis-(W-1)
    shift0 = 2.0 * axis - (W - 1)
    # rows where the right ear is intact (between the clip line and the eye line)
    clip_y = None
    right = (np.arange(W) > axis + 0.3 * iod)[None, :]
    ear_zone = right & (np.arange(W) < axis + 1.2 * iod)[None, :] & (np.arange(H) < ey - 0.6 * iod)[:, None]
    inv = (valid < 0.5) & ear_zone
    ys_inv = np.where(inv.any(1))[0]
    if len(ys_inv):
        clip_y = float(ys_inv.max())          # lowest row still outside the photo on the right
    if clip_y is None or clip_y < 4:
        return crop, alpha, {"mirrored_ear": False}
    r0, r1 = int(clip_y + 4), int(min(H - 1, ey - 0.6 * iod))
    if r1 - r0 < 6:
        return crop, alpha, {"mirrored_ear": False}
    # per-row outer edge alignment
    shifts = []
    for y in range(r0, r1):
        a_row = np.where((alpha[y] > 0.5) & right[0])[0]
        m_row = np.where(m_alpha[y] > 0.5)[0]
        m_row = m_row[(m_row + shift0) > axis + 0.3 * iod]
        if len(a_row) and len(m_row):
            shifts.append(float(a_row.max()) - float(m_row.max() + shift0))
    shift = shift0 + (float(np.median(shifts)) if shifts else 0.0)
    Mx = np.float32([[1, 0, shift], [0, 1, 0]])
    m_crop = cv2.warpAffine(m_crop, Mx, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    m_alpha = cv2.warpAffine(m_alpha, Mx, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # zone: right ear, above the clip line (+ a small overlap), soft edges
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    zone = np.clip((clip_y + 10 - yy) / 10.0, 0, 1) * np.clip((xx - (axis + 0.25 * iod)) / (0.15 * iod), 0, 1)
    zone = cv2.GaussianBlur(zone, (0, 0), feather_px)
    # take the mirrored tip only where the original alpha is missing (clipped)
    take = zone * np.clip(m_alpha - alpha, 0, 1)
    take = cv2.GaussianBlur(take, (0, 0), feather_px / 2.0)
    new_alpha = np.maximum(alpha, m_alpha * zone)
    t3 = take[..., None]
    new_crop = np.clip(crop.astype(np.float32) * (1 - t3) + m_crop.astype(np.float32) * t3, 0, 255).astype(np.uint8)
    return new_crop, new_alpha, {"mirrored_ear": True, "clip_y": round(clip_y, 1), "shift_px": round(shift, 1),
                                 "added_alpha_px": int((new_alpha - alpha > 0.5).sum())}


# ----------------------------------------------------------------------------- compositing
def colour_match(bgr: np.ndarray, alpha: np.ndarray, set_bgr, amount: float) -> np.ndarray:
    """Warm the cat toward the set spill: blend `amount` of the set colour into
    the shadows (weighted by 1-luminance) so the whites pick up the orange
    bounce like everything else on that stage."""
    f = bgr.astype(np.float32)
    lum = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    w = amount * (1.0 - lum) ** 0.7
    w = w[..., None] * alpha[..., None]
    set_f = np.array(set_bgr, np.float32)[None, None, :]
    out = f * (1 - w) + set_f * w
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_glasses(meta512: dict, iod: float, r_factor: float, stroke: float,
                 colour_bgr=FRAME_BGR, tint: float = 0.10, ss: int = 4) -> np.ndarray:
    """Procedural blue round-frame glasses, BGRA 512x512, drawn at ss x
    supersampling with LINE_AA and downsampled with INTER_AREA."""
    S = OUT * ss
    rgb = np.zeros((S, S, 3), np.float32)
    a = np.zeros((S, S), np.float32)
    (lx, ly), (rx, ry) = meta512["eye_l"], meta512["eye_r"]
    r = r_factor * iod
    dy = 0.06 * iod                       # lenses sit slightly below the pupils
    L = (lx, ly + dy)
    Rr = (rx, ry + dy)

    def P(p):
        return (int(round(p[0] * ss)), int(round(p[1] * ss)))

    rs, sw = int(round(r * ss)), max(1, int(round(stroke * ss)))
    col = np.array(colour_bgr, np.float32) / 255.0
    dark = col * 0.55
    light = np.clip(col * 1.35 + 0.08, 0, 1)

    # translucent lens tint
    tint_layer = np.zeros((S, S), np.float32)
    for c in (L, Rr):
        cv2.circle(tint_layer, P(c), rs - sw // 2, 1.0, -1, cv2.LINE_AA)
    lens_col = np.array((235, 190, 120), np.float32) / 255.0     # pale blue glass (BGR)
    a = np.maximum(a, tint_layer * tint)
    rgb += tint_layer[..., None] * lens_col

    frame = np.zeros((S, S), np.float32)
    # bridge: arc between the inner rims, slightly above the lens centres
    gap = (Rr[0] - L[0]) - 2 * r
    mid = ((L[0] + Rr[0]) / 2, (L[1] + Rr[1]) / 2 - 0.12 * r)
    cv2.ellipse(frame, P(mid), (int(round((gap / 2 + stroke * 0.8) * ss)), int(round(max(gap, 0.15 * r) * 0.75 * ss))),
                0, 200, 340, 1.0, int(sw * 0.85), cv2.LINE_AA)
    # temple stubs: short symmetric hinges from the outer rim toward the ears (<= 0.15 iod)
    stub = min(0.15 * iod, 0.30 * r)
    for c, sgn in ((L, -1), (Rr, 1)):
        p0 = (c[0] + sgn * (r - stroke * 0.3), c[1] - 0.12 * r)
        p1 = (c[0] + sgn * (r + stub), c[1] - 0.12 * r - 0.30 * stub)
        cv2.line(frame, P(p0), P(p1), 1.0, int(sw * 0.8), cv2.LINE_AA)
        cv2.circle(frame, P(p0), int(sw * 0.7), 1.0, -1, cv2.LINE_AA)
    # rims
    for c in (L, Rr):
        cv2.circle(frame, P(c), rs, 1.0, sw, cv2.LINE_AA)

    # simple top-left highlight on the rims for a bit of depth
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    shade = np.zeros((S, S), np.float32)
    for c in (L, Rr):
        d = np.hypot(xx - c[0] * ss, yy - c[1] * ss)
        ring = np.clip(1 - np.abs(d - rs) / (sw * 0.75), 0, 1)
        ang = np.arctan2(yy - c[1] * ss, xx - c[0] * ss)
        hl = np.clip(np.cos(ang + math.radians(135)), 0, 1) ** 3
        shade = np.maximum(shade, ring * hl)
    base = frame[..., None] * (col * 0.85 + dark * 0.15)[None, None, :]
    base = base * (1 - shade[..., None] * 0.7) + light[None, None, :] * (shade * 0.7 * frame)[..., None]
    rgb = rgb * (1 - frame[..., None]) + base
    a = np.maximum(a, frame)
    # thin dark outline just outside/inside the rim for definition on orange
    outline = np.zeros((S, S), np.float32)
    for c in (L, Rr):
        cv2.circle(outline, P(c), rs + sw // 2 + ss, 1.0, max(1, ss), cv2.LINE_AA)
        cv2.circle(outline, P(c), rs - sw // 2 - ss, 1.0, max(1, ss), cv2.LINE_AA)
    outline = outline * (1 - frame)
    rgb = rgb * (1 - outline[..., None]) + dark[None, None, :] * outline[..., None]
    a = np.maximum(a, outline * 0.6)

    rgba = np.dstack([np.clip(rgb, 0, 1), np.clip(a, 0, 1)])
    rgba = cv2.resize(rgba, (OUT, OUT), interpolation=cv2.INTER_AREA)
    # un-premultiply the colour where alpha > 0 so the PNG composites cleanly
    al = rgba[..., 3:4]
    rgba[..., :3] = np.where(al > 1e-3, np.clip(rgba[..., :3] / np.maximum(al, 1e-3), 0, 1), 0)
    return (rgba * 255).round().astype(np.uint8)


def over(bg_bgr: np.ndarray, fg_bgra: np.ndarray) -> np.ndarray:
    a = fg_bgra[..., 3:4].astype(np.float32) / 255.0
    out = bg_bgr.astype(np.float32) * (1 - a) + fg_bgra[..., :3].astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def label(img: np.ndarray, text: str) -> np.ndarray:
    d = img.copy()
    cv2.putText(d, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(d, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return d


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=HERO)
    ap.add_argument("--points", default=None,
                    help="eye_l_x,eye_l_y,eye_r_x,eye_r_y,nose_x,nose_y,mouth_x,mouth_y in hero px")
    ap.add_argument("--side-factor", type=float, default=4.1, help="crop side = factor * inter-ocular distance")
    ap.add_argument("--eye-y", type=float, default=0.42, help="eye midpoint y as a fraction of 512")
    ap.add_argument("--cx-shift", type=float, default=0.0, help="eye midpoint x shift from centre, fraction of 512")
    ap.add_argument("--no-matte", action="store_true", help="skip ISNet; alpha = keep region only")
    ap.add_argument("--feather", type=float, default=4.0, help="alpha feather in px")
    ap.add_argument("--no-mirror-ear", action="store_true", help="do not rebuild the clipped right ear tip from the left ear")
    ap.add_argument("--warm", type=float, default=0.07, help="set-colour blend into shadows (0-1)")
    ap.add_argument("--glasses-r", type=float, default=0.44, help="lens radius as a fraction of iod")
    ap.add_argument("--glasses-stroke", type=float, default=7.0, help="frame stroke px at 512")
    ap.add_argument("--set-frame", type=int, default=300)
    ap.add_argument("--video", default=str(WORK / "cut.mp4"))
    ap.add_argument("--isnet", default=str(MODELS / "isnet-general-use.onnx"))
    ap.add_argument("--sheet", default=str(WORK / "cat_ref_sheet.jpg"))
    ap.add_argument("--debug-dir", default=None, help="write intermediate images here")
    args = ap.parse_args()

    REFS.mkdir(exist_ok=True)
    ASSETS.mkdir(exist_ok=True)
    dbg = Path(args.debug_dir) if args.debug_dir else None
    if dbg:
        dbg.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(ROOT / args.src), cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"cannot read {args.src}")
    pts = parse_points(args.points)

    # (1) set colour
    set_bgr, (spx, spy) = sample_set_colour(Path(args.video), args.set_frame, WORK / "analysis.json")
    print(f"set colour BGR={set_bgr} sampled at frame {args.set_frame} px=({spx},{spy})")

    # (2) upright similarity crop -> 512
    M, roll, scale = upright_affine(pts, args.side_factor, args.eye_y, args.cx_shift)
    crop = cv2.warpAffine(img, M, (OUT, OUT), flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=tuple(int(v) for v in set_bgr))
    valid = cv2.warpAffine(np.ones(img.shape[:2], np.float32), M, (OUT, OUT), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    p512 = {k: apply_affine(M, v) for k, v in pts.items()}
    eye_c = ((p512["eye_l"][0] + p512["eye_r"][0]) / 2, (p512["eye_l"][1] + p512["eye_r"][1]) / 2)
    iod = math.hypot(p512["eye_r"][0] - p512["eye_l"][0], p512["eye_r"][1] - p512["eye_l"][1])
    chin_y = p512["mouth"][1] + 0.30 * iod
    meta = {k: [round(v[0], 2), round(v[1], 2)] for k, v in p512.items()}
    meta["eye_c"] = [round(eye_c[0], 2), round(eye_c[1], 2)]
    meta["chin_y"] = round(chin_y, 2)
    print(f"roll={roll:.2f} deg scale={scale:.4f} iod512={iod:.1f} eye_c512={meta['eye_c']}")

    # (3) matte
    keep = keep_region(M, meta, iod)
    if args.no_matte:
        alpha = keep
    else:
        model = ensure_isnet(Path(args.isnet))
        raw = isnet_matte(crop, model)
        if dbg:
            cv2.imwrite(str(dbg / "matte_raw.png"), (raw * 255).astype(np.uint8))
            cv2.imwrite(str(dbg / "keep.png"), (keep * 255).astype(np.uint8))
        alpha = raw * keep
    alpha = clean_alpha(alpha * valid, args.feather)
    ear_info = {"mirrored_ear": False}
    if not args.no_mirror_ear:
        crop, alpha, ear_info = mirror_clipped_ear(crop, alpha, valid, meta, iod)
        alpha = clean_alpha(alpha, args.feather)
        print("mirrored clipped ear:", ear_info)
    alpha8 = (alpha * 255).round().astype(np.uint8)

    # (4) source on set orange (+ mild warm spill)
    warm = colour_match(crop, alpha, set_bgr, args.warm)
    bgfill = np.empty_like(crop)
    bgfill[:] = np.array(set_bgr, np.uint8)
    a3 = alpha[..., None]
    src = np.clip(warm.astype(np.float32) * a3 + bgfill.astype(np.float32) * (1 - a3), 0, 255).astype(np.uint8)

    # (5) glasses
    glasses = draw_glasses(meta, iod, args.glasses_r, args.glasses_stroke)
    src_gl = over(src, glasses)

    # (6) meta
    ys, xs = np.where(alpha > 0.5)
    head_top_y = float(ys.min()) if len(ys) else 0.0
    row = int(round(eye_c[1]))
    cols = np.where(alpha[row] > 0.5)[0]
    head_w = float(cols.max() - cols.min() + 1) if len(cols) else 0.0
    meta_out = {
        "eye_l": meta["eye_l"], "eye_r": meta["eye_r"], "nose": meta["nose"], "mouth": meta["mouth"],
        "eye_c": meta["eye_c"], "iod": round(iod, 2),
        "head_top_y": round(head_top_y, 1), "chin_y": round(chin_y, 1), "head_w": round(head_w, 1),
        "set_bgr": set_bgr, "source": args.src,
        "glasses": {"r_px": round(args.glasses_r * iod, 1), "stroke_px": args.glasses_stroke,
                    "colour_bgr": list(FRAME_BGR), "asset": "assets/glasses_blue_512.png"},
        "hero_points_px": {k: list(v) for k, v in pts.items()},
        "hero_roll_deg": round(roll, 2), "hero_to_512_scale": round(scale, 5), "ear_fix": ear_info,
        "affine_hero_to_512": [[round(float(x), 6) for x in r] for r in M],
        "notes": ("512x512 LivePortrait source space. Hero photo rotated upright by the eye line, "
                  f"cropped to {args.side_factor} x iod, eyes at y={args.eye_y}*512. Alpha = ISNet general-use "
                  "matte x hand-traced head+neck outline (glove/paws/table excluded, neck faded), "
                  f"feathered {args.feather} px. Background and clipped-ear fill = set orange sampled from "
                  f"cut.mp4 frame {args.set_frame}; shadows warmed {int(args.warm * 100)}% toward it. "
                  "Clipped right ear tip rebuilt by mirroring the left ear. Glasses: procedural blue round frames, short symmetric temple stubs, 4x supersampled. Mouth closed."),
    }

    cv2.imwrite(str(REFS / "cat_src_512.png"), src)
    cv2.imwrite(str(REFS / "cat_src_glasses_512.png"), src_gl)
    cv2.imwrite(str(REFS / "cat_alpha_512.png"), alpha8)
    cv2.imwrite(str(ASSETS / "glasses_blue_512.png"), glasses)
    with open(REFS / "cat_meta.json", "w", newline="\n") as f:
        json.dump(meta_out, f, indent=2)
        f.write("\n")

    # (7) sheet with landmark overlay on the crop panel
    crop_d = crop.copy()
    for k, (x, y) in p512.items():
        cv2.circle(crop_d, (int(x), int(y)), 4, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.line(crop_d, (0, int(chin_y)), (OUT, int(chin_y)), (0, 200, 255), 1)
    cv2.line(crop_d, (0, int(head_top_y)), (OUT, int(head_top_y)), (255, 200, 0), 1)
    sheet = np.hstack([label(crop_d, "crop+lmk"), label(cv2.cvtColor(alpha8, cv2.COLOR_GRAY2BGR), "alpha"),
                       label(src, "src"), label(src_gl, "src+glasses")])
    cv2.imwrite(args.sheet, sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("wrote refs/cat_src_512.png refs/cat_src_glasses_512.png refs/cat_alpha_512.png refs/cat_meta.json "
          f"assets/glasses_blue_512.png {Path(args.sheet).as_posix()}")
    keys = ("eye_l", "eye_r", "nose", "mouth", "eye_c", "iod", "head_top_y", "chin_y", "head_w", "set_bgr")
    print(json.dumps({k: meta_out[k] for k in keys}))


if __name__ == "__main__":
    main()

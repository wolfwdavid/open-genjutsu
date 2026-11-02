#!/usr/bin/env python
"""
judge_take.py -- automatic QA of a round-4 block render (todo step 15, phase 1a).

Runs buffalo_l (track.py's build_app / detect: SCRFD det + 106/68 landmarks + ArcFace) on every
--every-th frame of a render and classifies every face by its x-centre (< width/2 = LEFT).

  LEFT  (the user):  face detected fraction; SUNGLASSES PRESENT fraction = fraction of the detected
                     LEFT faces whose eye band (ellipse on the 5-kps eye line, swap.py eyewear_mask
                     geometry: half-axes 1.25 x 0.45 iod, rotated by the eye-line roll) is dominated
                     by dark (V < 70) or white-frame (S < 60, V > 170) pixels above a threshold
                     calibrated on work/cut.mp4 (both rappers wear sunglasses -> ~1.0) vs
                     faces/user/selected (no glasses -> ~0); identity = ArcFace cosine of the LEFT
                     face vs the mean embedding of faces/user/selected (informational).
  RIGHT (the cat):   number of frames with the RIGHT performer's human face in the right half:
                     det score > --right-det AND ArcFace cosine vs the performer's mean embedding
                     (learned from work/cut.mp4 during calibration) > --right-cos. SCRFD fires on the
                     cat's glasses-face too (det 0.6-0.8 on ~19 % of the round-3 cat_R.mp4 frames) but
                     those embeddings sit at cos < 0.06 vs the performer (his own frames: > 0.5), so
                     the identity gate is what makes this ~0 after the cat pass and ~all on cut.mp4.
                     The det-only count is reported as information.
  sheet.jpg:         one row per blocks.sheet_times: render frame | same-time frame of blocks.ref |
                     example still (work/example/ex_2, ex_3, ex_6 cycled).
Writes <out>/sheet.jpg, <out>/report.json, <out>/report.txt. The sunglasses threshold calibration is
cached in work/round4/sunglasses_calib.json (--recalib recomputes it).

Usage
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe judge_take.py --input work/render.mp4 \\
      --blocks work/blocks.json --side both --out work/round4/judge_render/
  ... --cut-domain     # the input is an 872-frame cut-domain clip (cat_R.mp4, cut.mp4): only the
                       # block frame ranges are judged and mapped to the 661-frame block domain
Exit code 0 (informational tool: the numbers are the verdict).
"""
from __future__ import annotations

import os
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
WORK = Path(os.environ["VG_WORK"]).resolve() if os.environ.get("VG_WORK") else ROOT / "work"
sys.path.insert(0, str(ROOT))
from track import build_app, detect  # noqa: E402  (buffalo_l on DirectML, same setup as the tracker)

IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
EXAMPLES = ["work/example/ex_2.jpg", "work/example/ex_3.jpg", "work/example/ex_6.jpg"]


# ----------------------------------------------------------------------------- eye band
def eyeband_frac(img: np.ndarray, kps: np.ndarray, ax_iod: float = 1.25, ay_iod: float = 0.45,
                 dark_v: int = 70, white_s: int = 60, white_v: int = 170) -> float | None:
    """Fraction of the eye-band ellipse (swap.py eyewear_mask geometry) that is black-lens or
    white-frame coloured. None when the ellipse is degenerate / off-frame."""
    k = np.asarray(kps, np.float32)
    le, re = k[0], k[1]
    v = re - le
    iod = float(np.hypot(*v))
    if iod < 2:
        return None
    ang = float(np.degrees(np.arctan2(v[1], v[0])))
    mid = (le + re) / 2
    H, W = img.shape[:2]
    x0 = int(max(0, mid[0] - 1.6 * iod)); x1 = int(min(W, mid[0] + 1.6 * iod + 1))
    y0 = int(max(0, mid[1] - 1.0 * iod)); y1 = int(min(H, mid[1] + 1.0 * iod + 1))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    roi = img[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    S, V = hsv[..., 1], hsv[..., 2]
    ell = np.zeros(roi.shape[:2], np.uint8)
    cv2.ellipse(ell, (int(round(mid[0] - x0)), int(round(mid[1] - y0))),
                (max(1, int(round(ax_iod * iod))), max(1, int(round(ay_iod * iod)))), ang, 0, 360, 255, -1)
    inside = ell > 0
    n = int(inside.sum())
    if n < 16:
        return None
    m = ((V < dark_v) | ((S < white_s) & (V > white_v))) & inside
    return float(m.sum()) / n


def largest(faces: list):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def cx_of(face) -> float:
    return float(face.bbox[0] + face.bbox[2]) / 2


# ----------------------------------------------------------------------------- calibration
def user_refs(app, folder: Path, min_det: float) -> tuple[np.ndarray, list[float], list[str]]:
    """Mean ArcFace embedding (unit norm) + eye-band fractions of the reference photos."""
    files = sorted({Path(f).resolve() for ext in IMAGE_EXTS for f in glob.glob(str(folder / ext))})
    files = [str(f) for f in files]  # de-duplicated: Windows matches *.jpg and *.JPG alike
    embs, fracs, names = [], [], []
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        faces = detect(app, img, min_det=min_det)
        if not faces:
            print("  ref %s: no face" % Path(f).name)
            continue
        face = largest(faces)
        embs.append(face.normed_embedding)
        fr = eyeband_frac(img, face.kps)
        fracs.append(fr if fr is not None else float("nan"))
        names.append(Path(f).name)
    if not embs:
        sys.exit("no usable faces in %s" % folder)
    E = np.stack(embs).mean(0)
    return (E / np.linalg.norm(E)).astype(np.float32), fracs, names


def calibrate(app, cut: Path, blocks: dict, every: int, min_det: float, ref_fracs: list[float]) -> dict:
    """Eye-band fractions of every face (both rappers, sunglasses on) in the block frame ranges of
    the source cut, sampled every `every` frames, vs the no-glasses reference photos."""
    wanted = {}
    for b in blocks["blocks"]:
        for i in range(b["cut_start"], b["cut_end"], every):
            wanted[i] = b["name"]
    cap = cv2.VideoCapture(str(cut))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fr_glasses, i = [], 0
    embs = {"L": [], "R": []}
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i in wanted:
            for face in detect(app, img, min_det=min_det):
                side = "L" if cx_of(face) < W / 2 else "R"
                embs[side].append(face.normed_embedding)
                fr = eyeband_frac(img, face.kps)
                if fr is not None:
                    fr_glasses.append((i, side, fr))
        i += 1
    cap.release()

    def mean_emb(v):
        if not v:
            return None
        m = np.stack(v).mean(0)
        return [float(x) for x in m / np.linalg.norm(m)]

    g = np.array([x[2] for x in fr_glasses], float)
    n = np.array([x for x in ref_fracs if x == x], float)
    p_g = float(np.percentile(g, 5)) if len(g) else float("nan")
    p_n = float(np.percentile(n, 95)) if len(n) else float("nan")
    if len(g) and len(n) and p_g > p_n:
        thr = 0.5 * (p_g + p_n)
    else:  # overlap: pick the threshold that best separates the two sets
        cands = np.unique(np.concatenate([g, n]))
        best, thr = -1.0, 0.35
        for c in cands:
            acc = (np.mean(g >= c) if len(g) else 0) + (np.mean(n < c) if len(n) else 0)
            if acc > best:
                best, thr = acc, float(c)
    return {
        "threshold": float(thr),
        "cut": str(cut), "every": every, "min_det": min_det,
        "glasses_n": int(len(g)), "glasses_min": float(g.min()) if len(g) else None,
        "glasses_p05": p_g, "glasses_median": float(np.median(g)) if len(g) else None,
        "glasses_above_thr": float(np.mean(g >= thr)) if len(g) else None,
        "glasses_L_above_thr": float(np.mean([x[2] >= thr for x in fr_glasses if x[1] == "L"])) if len(g) else None,
        "glasses_R_above_thr": float(np.mean([x[2] >= thr for x in fr_glasses if x[1] == "R"])) if len(g) else None,
        "noglasses_n": int(len(n)), "noglasses_max": float(n.max()) if len(n) else None,
        "noglasses_p95": p_n, "noglasses_median": float(np.median(n)) if len(n) else None,
        "noglasses_above_thr": float(np.mean(n >= thr)) if len(n) else None,
        "noglasses_fracs": [float(x) for x in n],
        "L_perf_n": len(embs["L"]), "L_perf_embedding": mean_emb(embs["L"]),
        "R_perf_n": len(embs["R"]), "R_perf_embedding": mean_emb(embs["R"]),
    }


# ----------------------------------------------------------------------------- sheet
def tile(img: np.ndarray, h: int, label: str) -> np.ndarray:
    s = h / img.shape[0]
    t = cv2.resize(img, (int(round(img.shape[1] * s)), h), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(t, (6, 6), (6 + tw + 12, 6 + th + 14), (30, 30, 30), -1)
    cv2.putText(t, label, (12, 6 + th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return t


def build_sheet(render_frames: dict[int, np.ndarray], ref_frames: dict[int, np.ndarray], sheet_idx: list[int],
                fps: float, per_frame: dict[int, dict], examples: list[str], out: Path) -> None:
    ex_imgs = [cv2.imread(str(ROOT / e)) for e in examples]
    ex_imgs = [e for e in ex_imgs if e is not None]
    rows = []
    for r, idx in enumerate(sheet_idx):
        rf = render_frames.get(idx)
        if rf is None:
            continue
        h = rf.shape[0]
        pf = per_frame.get(idx, {})
        lab = "render f%d t=%.1fs" % (idx, idx / fps)
        if "L_frac" in pf and pf["L_frac"] is not None:
            lab += "  L eyeband %.2f%s" % (pf["L_frac"], " SG" if pf.get("L_sunglasses") else " no-SG")
        if "L_cos" in pf and pf["L_cos"] is not None:
            lab += " cos %.2f" % pf["L_cos"]
        if "R_face" in pf:
            lab += "  R face:%s" % ("YES det %.2f cos %.2f" % (pf["R_det"], pf["R_cos_perf"] or 0) if pf["R_face"]
                                    else ("cat? det %.2f cos %.2f" % (pf["R_det"], pf["R_cos_perf"] or 0) if pf["R_det_face"] else "none"))
        cols = [tile(rf.copy(), h, lab)]
        ref = ref_frames.get(idx)
        cols.append(tile(ref.copy() if ref is not None else np.zeros_like(rf), h, "ref f%d" % idx))
        if ex_imgs:
            e = ex_imgs[r % len(ex_imgs)]
            cols.append(tile(e.copy(), h, "example %s" % Path(examples[r % len(examples)]).stem))
        rows.append(np.concatenate(cols, 1))
    if not rows:
        print("sheet: nothing to write")
        return
    w = max(r.shape[1] for r in rows)
    rows = [r if r.shape[1] == w else np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
    sheet = np.concatenate(rows, 0)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote %s %dx%d" % (out, sheet.shape[1], sheet.shape[0]))


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="render to judge (661-frame block domain unless --cut-domain)")
    ap.add_argument("--blocks", default=str(WORK / "blocks.json"))
    ap.add_argument("--side", choices=["L", "R", "both"], default="both")
    ap.add_argument("--out", required=True, help="output directory (sheet.jpg, report.json, report.txt)")
    ap.add_argument("--every", type=int, default=2, help="judge every Nth frame")
    ap.add_argument("--cut-domain", action="store_true", help="input frames are indexed like work/cut.mp4")
    ap.add_argument("--refs", default=str(ROOT / "faces" / "user" / "selected"))
    ap.add_argument("--examples", default=",".join(EXAMPLES))
    ap.add_argument("--det-size", type=int, default=640)
    ap.add_argument("--min-det", type=float, default=0.45, help="detector floor for the LEFT face / calibration")
    ap.add_argument("--right-det", type=float, default=0.6, help="det score that counts as a face on the RIGHT")
    ap.add_argument("--right-cos", type=float, default=0.3, help="cos vs the RIGHT performer's mean embedding that makes it HIS face")
    ap.add_argument("--thr", type=float, default=None, help="sunglasses eye-band threshold (default: calibrated)")
    ap.add_argument("--calib", default=str(WORK / "round4" / "sunglasses_calib.json"))
    ap.add_argument("--calib-every", type=int, default=6)
    ap.add_argument("--recalib", action="store_true")
    args = ap.parse_args()

    blocks = json.loads(Path(args.blocks).read_text())
    fps = float(blocks["fps"])
    n_total = int(blocks["n_total"])
    rel = lambda p: Path(p) if Path(p).is_absolute() else ROOT / p
    ref_path = rel(blocks["ref"])
    cut_path = rel(blocks.get("cut", "work/cut.mp4"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not Path(args.input).exists():
        sys.exit("missing input %s" % args.input)
    t0 = time.time()

    # block-domain index -> input index
    if args.cut_domain:
        in_of = {}
        for b in blocks["blocks"]:
            for i in range(b["n"]):
                in_of[b["out_start"] + i] = b["cut_start"] + i
    else:
        in_of = {i: i for i in range(n_total)}
    cut_of = {}
    for b in blocks["blocks"]:
        for i in range(b["n"]):
            cut_of[b["out_start"] + i] = b["cut_start"] + i
    block_of = {}
    for b in blocks["blocks"]:
        for i in range(b["n"]):
            block_of[b["out_start"] + i] = b["name"]

    sheet_idx = [int(round(t * fps)) for t in blocks.get("sheet_times", [])]
    judge_idx = sorted(set(range(0, n_total, args.every)) | {i for i in sheet_idx if i < n_total})
    want_in = {in_of[i]: i for i in judge_idx}

    app = build_app(args.det_size)
    print("refs          %s" % args.refs)
    user_emb, ref_fracs, ref_names = user_refs(app, Path(args.refs), args.min_det)
    for nm, fr in zip(ref_names, ref_fracs):
        print("  %-12s eye-band frac %.3f (no glasses)" % (nm, fr))

    calib_path = Path(args.calib)
    if args.thr is None:
        calib = json.loads(calib_path.read_text()) if calib_path.exists() and not args.recalib else None
        if calib is not None and calib.get("R_perf_embedding"):
            print("calibration   cached %s" % calib_path)
        else:
            tc = time.time()
            calib = calibrate(app, cut_path, blocks, args.calib_every, args.min_det, ref_fracs)
            calib_path.parent.mkdir(parents=True, exist_ok=True)
            calib_path.write_text(json.dumps(calib, indent=2) + "\n")
            print("calibration   %s (%.1f s)" % (calib_path, time.time() - tc))
        thr = float(calib["threshold"])
        print("calibration   threshold %.3f: cut.mp4 faces above thr %.3f (L %.3f, R %.3f; n=%d, min %.3f, p05 %.3f, median %.3f) | "
              "user refs above thr %.3f (n=%d, max %.3f, median %.3f)" % (
                  thr, calib["glasses_above_thr"], calib["glasses_L_above_thr"], calib["glasses_R_above_thr"],
                  calib["glasses_n"], calib["glasses_min"], calib["glasses_p05"], calib["glasses_median"],
                  calib["noglasses_above_thr"], calib["noglasses_n"], calib["noglasses_max"], calib["noglasses_median"]))
    else:
        calib = {"threshold": args.thr, "source": "--thr"}
        thr = args.thr
        print("calibration   threshold %.3f (manual)" % thr)
    perf_R = np.asarray(calib["R_perf_embedding"], np.float32) if calib.get("R_perf_embedding") else None
    perf_L = np.asarray(calib["L_perf_embedding"], np.float32) if calib.get("L_perf_embedding") else None
    if perf_R is None:
        print("WARNING       no RIGHT performer embedding (manual --thr): RIGHT check is det-only")
    cos_or_none = lambda emb, ref: float(np.dot(emb, ref)) if ref is not None else None

    # ---- main pass
    cap = cv2.VideoCapture(args.input)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_in = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("input         %s %dx%d, %d frames (%s domain), judging %d frames (every %d + sheet frames)" % (
        args.input, W, H, n_in, "cut" if args.cut_domain else "block", len(judge_idx), args.every))
    do_L = args.side in ("L", "both")
    do_R = args.side in ("R", "both")
    per_frame: dict[int, dict] = {}
    render_frames: dict[int, np.ndarray] = {}
    i = 0
    tm = time.time()
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i in want_in:
            idx = want_in[i]
            rec: dict = {"idx": idx, "cut": cut_of[idx], "block": block_of[idx]}
            faces = detect(app, img, min_det=min(args.min_det, args.right_det))
            left = [f for f in faces if cx_of(f) < W / 2 and f.det_score >= args.min_det]
            right = [f for f in faces if cx_of(f) >= W / 2]
            if do_L:
                if left:
                    f = max(left, key=lambda x: x.det_score)
                    fr = eyeband_frac(img, f.kps)
                    rec.update(L_det=float(f.det_score), L_w=float(f.bbox[2] - f.bbox[0]), L_frac=fr,
                               L_sunglasses=(fr is not None and fr >= thr),
                               L_cos=float(np.dot(f.normed_embedding, user_emb)),
                               L_cos_perf=cos_or_none(f.normed_embedding, perf_L))
                else:
                    rec.update(L_det=None, L_w=None, L_frac=None, L_sunglasses=None, L_cos=None, L_cos_perf=None)
            if do_R:
                if right:
                    f = max(right, key=lambda x: x.det_score)
                    rmax = float(f.det_score)
                    cp = cos_or_none(f.normed_embedding, perf_R)
                    rec.update(R_det=rmax, R_det_face=rmax > args.right_det, R_w=float(f.bbox[2] - f.bbox[0]),
                               R_cos_perf=cp, R_cos_user=float(np.dot(f.normed_embedding, user_emb)),
                               R_face=rmax > args.right_det and (cp is None or cp > args.right_cos))
                else:
                    rec.update(R_det=0.0, R_det_face=False, R_w=None, R_cos_perf=None, R_cos_user=None, R_face=False)
            per_frame[idx] = rec
            if idx in sheet_idx:
                render_frames[idx] = img.copy()
        i += 1
    cap.release()
    print("judged        %d frames in %.1f s (%.2f s/frame)" % (len(per_frame), time.time() - tm,
                                                              (time.time() - tm) / max(1, len(per_frame))))
    if n_in != (872 if args.cut_domain else n_total) and n_in > 0:
        print("WARNING       input has %d frames, expected %d" % (n_in, 872 if args.cut_domain else n_total))

    # ---- summarise
    recs = [per_frame[k] for k in sorted(per_frame)]
    rep: dict = {"input": args.input, "blocks": args.blocks, "side": args.side, "every": args.every,
                 "cut_domain": args.cut_domain, "n_judged": len(recs), "n_total": n_total, "n_input_frames": n_in,
                 "threshold": thr, "calibration": calib, "user_refs": ref_names}
    lines = ["judge_take %s (%d of %d frames, every %d)" % (args.input, len(recs), n_total, args.every)]
    if do_L:
        det = [r for r in recs if r["L_det"] is not None]
        sg = [r for r in det if r["L_frac"] is not None]
        n_sg = sum(1 for r in sg if r["L_sunglasses"])
        cos = np.array([r["L_cos"] for r in det], float) if det else np.array([])
        no_sg = [r for r in sg if not r["L_sunglasses"]]
        by_block = {}
        for b in blocks["blocks"]:
            d = [r for r in det if r["block"] == b["name"]]
            s = [r for r in d if r["L_frac"] is not None]
            by_block[b["name"]] = {
                "judged": sum(1 for r in recs if r["block"] == b["name"]), "detected": len(d),
                "sunglasses_frac": (sum(1 for r in s if r["L_sunglasses"]) / len(s)) if s else None,
                "cos_mean": float(np.mean([r["L_cos"] for r in d])) if d else None}
        rep["L"] = {
            "detected": len(det), "detected_frac": len(det) / max(1, len(recs)),
            "sunglasses_frac": (n_sg / len(sg)) if sg else None, "sunglasses_n": n_sg, "eyeband_n": len(sg),
            "eyeband_median": float(np.median([r["L_frac"] for r in sg])) if sg else None,
            "identity_cos_mean": float(cos.mean()) if len(cos) else None,
            "identity_cos_median": float(np.median(cos)) if len(cos) else None,
            "identity_cos_p10": float(np.percentile(cos, 10)) if len(cos) else None,
            "identity_cos_min": float(cos.min()) if len(cos) else None,
            "cos_vs_L_performer_mean": float(np.mean([r["L_cos_perf"] for r in det if r["L_cos_perf"] is not None])) if det and perf_L is not None else None,
            "face_w_median": float(np.median([r["L_w"] for r in det])) if det else None,
            "no_sunglasses_frames": [(r["idx"], r["cut"], round(r["L_frac"], 3)) for r in no_sg],
            "missing_frames": [(r["idx"], r["cut"]) for r in recs if r["L_det"] is None],
            "by_block": by_block}
        lines.append("LEFT   face detected %d/%d (%.3f); sunglasses present %d/%d (%.3f, thr %.3f, eye-band median %s); "
                     "identity cos vs user mean %s / median %s / p10 %s (vs original L performer %s); face w median %s px" % (
                         len(det), len(recs), rep["L"]["detected_frac"], n_sg, len(sg),
                         rep["L"]["sunglasses_frac"] if sg else float("nan"), thr,
                         "%.3f" % rep["L"]["eyeband_median"] if sg else "n/a",
                         "%.3f" % rep["L"]["identity_cos_mean"] if len(cos) else "n/a",
                         "%.3f" % rep["L"]["identity_cos_median"] if len(cos) else "n/a",
                         "%.3f" % rep["L"]["identity_cos_p10"] if len(cos) else "n/a",
                         "%.3f" % rep["L"]["cos_vs_L_performer_mean"] if rep["L"]["cos_vs_L_performer_mean"] is not None else "n/a",
                         "%.0f" % rep["L"]["face_w_median"] if det else "n/a"))
        for name, bb in by_block.items():
            lines.append("       block %s: judged %d, detected %d, sunglasses %s, cos mean %s" % (
                name, bb["judged"], bb["detected"],
                "%.3f" % bb["sunglasses_frac"] if bb["sunglasses_frac"] is not None else "n/a",
                "%.3f" % bb["cos_mean"] if bb["cos_mean"] is not None else "n/a"))
        if no_sg:
            lines.append("       no-sunglasses frames (block idx / cut idx / eye-band): %s%s" % (
                ", ".join("%d/%d/%.2f" % x for x in rep["L"]["no_sunglasses_frames"][:25]), " ..." if len(no_sg) > 25 else ""))
        if rep["L"]["missing_frames"]:
            mf = rep["L"]["missing_frames"]
            lines.append("       LEFT face missing on %d frames (block/cut): %s%s" % (
                len(mf), ", ".join("%d/%d" % x for x in mf[:25]), " ..." if len(mf) > 25 else ""))
    if do_R:
        rf = [r for r in recs if r["R_face"]]
        rd = [r for r in recs if r["R_det_face"]]
        by_block = {b["name"]: {"judged": sum(1 for r in recs if r["block"] == b["name"]),
                                "faces": sum(1 for r in rf if r["block"] == b["name"]),
                                "det_only": sum(1 for r in rd if r["block"] == b["name"])} for b in blocks["blocks"]}
        cps = [r["R_cos_perf"] for r in rd if r["R_cos_perf"] is not None]
        fmt = lambda r: (r["idx"], r["cut"], round(r["R_det"], 3), round(r["R_cos_perf"], 3) if r["R_cos_perf"] is not None else None)
        rep["R"] = {"human_face_frames": len(rf), "human_face_frac": len(rf) / max(1, len(recs)),
                    "det_only_frames": len(rd), "det_only_frac": len(rd) / max(1, len(recs)),
                    "det_threshold": args.right_det, "cos_threshold": args.right_cos,
                    "det_max_median": float(np.median([r["R_det"] for r in recs])) if recs else None,
                    "cos_perf_of_detections_max": float(max(cps)) if cps else None,
                    "cos_perf_of_detections_median": float(np.median(cps)) if cps else None,
                    "face_frames": [fmt(r) for r in rf],
                    "det_only_face_frames": [fmt(r) for r in rd],
                    "by_block": by_block}
        lines.append("RIGHT  performer's human face (det > %.2f & cos vs performer-R > %.2f) on %d/%d judged frames (%.3f); "
                     "det-only detections %d/%d (%.3f), their cos vs performer-R median %s max %s" % (
                         args.right_det, args.right_cos, len(rf), len(recs), rep["R"]["human_face_frac"],
                         len(rd), len(recs), rep["R"]["det_only_frac"],
                         "%.3f" % rep["R"]["cos_perf_of_detections_median"] if cps else "n/a",
                         "%.3f" % rep["R"]["cos_perf_of_detections_max"] if cps else "n/a"))
        for name, bb in by_block.items():
            lines.append("       block %s: judged %d, human faces %d (det-only %d)" % (name, bb["judged"], bb["faces"], bb["det_only"]))
        if rf:
            lines.append("       right-face frames (block/cut/det/cos): %s%s" % (
                ", ".join("%d/%d/%.2f/%s" % (a, b, c, "%.2f" % d if d is not None else "-") for a, b, c, d in rep["R"]["face_frames"][:25]),
                " ..." if len(rf) > 25 else ""))

    # ---- sheet
    ref_frames: dict[int, np.ndarray] = {}
    if ref_path.exists():
        cap = cv2.VideoCapture(str(ref_path))
        j = 0
        want = set(sheet_idx)
        while want:
            ok, img = cap.read()
            if not ok:
                break
            if j in want:
                ref_frames[j] = img
                want.discard(j)
            j += 1
        cap.release()
    examples = [e.strip() for e in args.examples.split(",") if e.strip()]
    build_sheet(render_frames, ref_frames, sheet_idx, fps, per_frame, examples, out_dir / "sheet.jpg")

    rep["per_frame"] = recs
    rep["wall_s"] = time.time() - t0
    (out_dir / "report.json").write_text(json.dumps(rep, indent=1, default=float) + "\n")
    lines.append("wall %.1f s; sheet %s" % (rep["wall_s"], out_dir / "sheet.jpg"))
    (out_dir / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())

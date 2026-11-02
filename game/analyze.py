#!/usr/bin/env python
"""
analyze.py -- find the characters and the moments of decision in a clip (CPU, a few seconds).

  frames   sampled at --fps (default 4) with ffmpeg
  cuts     shot cuts = big mean-abs-diff jumps between consecutive samples
  moments  the top --moments peaks of motion energy, at least --gap seconds apart, never on a cut
  characters per moment: YOLOv10m (COCO) detections of people and animals, each with a square 1:1
           crop centred on the box (the "eyes" of that character) and a thumbnail

Writes <out>/analysis.json plus <out>/frames/mNN.png and <out>/crops/mNN_cK.png.

  python game/analyze.py --video clip.mp4 --out work_ex/game/<id>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
YOLO = ROOT / "models" / "detection" / "yolov10m.onnx"
COCO = {0: "person", 14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
        20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe"}


def ffprobe(video: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,nb_frames:format=duration", "-of", "json", str(video)],
                         capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    st = j["streams"][0]
    num, den = st["r_frame_rate"].split("/")
    return dict(width=int(st["width"]), height=int(st["height"]), fps=float(num) / float(den),
                duration=float(j["format"]["duration"]))


def sample_frames(video: Path, fps: float, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("s*.png"):
        p.unlink()
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vf", f"fps={fps:g}",
                    "-start_number", "0", str(out / "s%05d.png")], check=True)
    return sorted(out.glob("s*.png"))


class Yolo:
    def __init__(self, path: Path):
        import onnxruntime as ort
        provs = [p for p in ("DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
                 if p in ort.get_available_providers()]
        self.s = ort.InferenceSession(str(path), providers=provs)
        self.inp = self.s.get_inputs()[0].name

    def __call__(self, im: np.ndarray, thr: float = 0.35) -> list[dict]:
        h, w = im.shape[:2]
        sz = 640
        r = min(sz / h, sz / w)
        nh, nw = int(h * r), int(w * r)
        can = np.full((sz, sz, 3), 114, np.uint8)
        can[:nh, :nw] = cv2.resize(im, (nw, nh))
        x = can[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255
        det = self.s.run(None, {self.inp: x})[0][0]
        res = []
        for x1, y1, x2, y2, sc, c in det:
            c = int(c)
            if sc < thr or c not in COCO:
                continue
            box = [float(v / r) for v in (x1, y1, x2, y2)]
            box = [max(0.0, box[0]), max(0.0, box[1]), min(float(w), box[2]), min(float(h), box[3])]
            if (box[2] - box[0]) * (box[3] - box[1]) < 0.002 * w * h:
                continue
            res.append(dict(cls=COCO[c], score=round(float(sc), 3), box=[round(v, 1) for v in box]))
        res.sort(key=lambda d: -(d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))
        return res


def square_crop(im: np.ndarray, box: list[float], scale: float = 1.6, size: int = 512) -> np.ndarray:
    """1:1 crop centred on the box, side = scale x the longer box side, replicate-padded at borders."""
    h, w = im.shape[:2]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    side = max(box[2] - box[0], box[3] - box[1]) * scale
    side = max(64.0, min(side, max(h, w)))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    s = int(round(side))
    pad = s
    padded = cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    crop = padded[y0 + pad:y0 + pad + s, x0 + pad:x0 + pad + s]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--moments", type=int, default=5)
    ap.add_argument("--gap", type=float, default=2.0, help="min seconds between moments")
    ap.add_argument("--cut-corr", type=float, default=0.5, help="HSV-histogram correlation below which a sample is a shot cut")
    a = ap.parse_args()
    video, out = Path(a.video), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    info = ffprobe(video)
    samples = sample_frames(video, a.fps, out / "samples")
    print(f"{video.name}: {info['width']}x{info['height']} {info['duration']:.1f}s -> {len(samples)} samples", flush=True)

    imgs = [cv2.imread(str(p)) for p in samples]
    small = [cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (160, int(160 * im.shape[0] / im.shape[1]))).astype(np.float32)
             for im in imgs]
    hists = []
    for im in imgs:
        h = cv2.calcHist([cv2.cvtColor(im, cv2.COLOR_BGR2HSV)], [0, 1], None, [24, 16], [0, 180, 0, 256])
        hists.append(cv2.normalize(h, None).flatten())
    diff = np.array([0.0] + [float(np.abs(small[i] - small[i - 1]).mean()) for i in range(1, len(small))])
    corr = np.array([1.0] + [float(cv2.compareHist(hists[i - 1], hists[i], cv2.HISTCMP_CORREL)) for i in range(1, len(hists))])
    cuts = [i for i in range(1, len(corr)) if corr[i] < a.cut_corr]
    energy = diff.copy()
    for c in cuts:
        energy[max(0, c - 1):c + 2] = 0.0  # never pick the cut itself
    energy[0] = 0.0
    yolo = Yolo(YOLO)
    order = [int(i) for i in np.argsort(-energy) if energy[i] > 0]
    gap = int(round(a.gap * a.fps))
    cand: list[int] = []
    for i in order:
        if all(abs(i - j) >= gap for j in cand):
            cand.append(i)
        if len(cand) >= a.moments * 3:
            break
    if not cand:
        cand = [len(samples) // 2]
    dets = {i: yolo(imgs[i])[:6] for i in cand}
    # rank: a peak with someone in it beats an empty peak; then by energy
    cand.sort(key=lambda i: (-(min(len(dets[i]), 2)), -energy[i]))
    chosen = sorted(cand[:a.moments])
    print(f"cuts at samples {cuts}; moments {chosen}", flush=True)
    (out / "frames").mkdir(exist_ok=True)
    (out / "crops").mkdir(exist_ok=True)
    moments = []
    for mi, si in enumerate(chosen):
        im = imgs[si]
        fpath = out / "frames" / f"m{mi:02d}.png"
        cv2.imwrite(str(fpath), im)
        chars = []
        for ci, d in enumerate(dets[si]):
            crop = square_crop(im, d["box"])
            cpath = out / "crops" / f"m{mi:02d}_c{ci}.png"
            cv2.imwrite(str(cpath), crop)
            chars.append(dict(id=f"m{mi:02d}_c{ci}", cls=d["cls"], score=d["score"], box=d["box"],
                              crop=str(cpath.relative_to(out)).replace("\\", "/"),
                              label=f"{d['cls']} {ci + 1}"))
        h, w = im.shape[:2]
        side = min(h, w)
        whole = square_crop(im, [w / 2 - side / 2, h / 2 - side / 2, w / 2 + side / 2, h / 2 + side / 2], scale=1.0)
        cpath = out / "crops" / f"m{mi:02d}_any.png"
        cv2.imwrite(str(cpath), whole)
        chars.append(dict(id=f"m{mi:02d}_any", cls="custom", score=0, box=[0, 0, w, h],
                          crop=str(cpath.relative_to(out)).replace("\\", "/"), label="someone else here"))
        moments.append(dict(id=f"m{mi:02d}", t=round(si / a.fps, 2), sample=si, energy=round(float(energy[si]), 1),
                            frame=str(fpath.relative_to(out)).replace("\\", "/"), characters=chars))
        print(f"  moment {mi} @ {si / a.fps:.1f}s: {[c['label'] for c in chars]}", flush=True)
    json.dump(dict(video=str(video), **info, sample_fps=a.fps, cuts=cuts, moments=moments),
              open(out / "analysis.json", "w", encoding="utf-8"), indent=1)
    print(f"wrote {out / 'analysis.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""
mouth_curve.py -- per-frame mouth-open amount (0..1) from an isolated vocal track.

  vocals wav -> mono -> band-pass 150-4000 Hz -> RMS in 10 ms hops -> dB
  -> noise gate (below --gate-db under the loud level = closed)
  -> normalise to 0..1 between the gate and the loud level (p95)
  -> asymmetric smoothing: fast attack (--attack-ms), slower release (--release-ms), like a jaw
  -> sampled at the video frame times (frame i covers [i/fps, (i+1)/fps): mean of that window)

Writes <out>.json {"fps", "n", "mouth": [...]} and <out>.png (vocal waveform + curve).

Usage
  .venv/Scripts/python.exe mouth_curve.py --vocals "work_ex/audio_sep/song_(Vocals)_UVR-MDX-NET-Voc_FT.wav" \
      --fps 24 --n 361 --out work_ex/mouth
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt


def read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    dt = {2: np.int16, 4: np.int32}[sw]
    x = np.frombuffer(raw, dt).astype(np.float32) / float(np.iinfo(dt).max)
    return x.reshape(-1, ch).mean(1), sr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vocals", required=True)
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate-db", type=float, default=30.0, help="closed where the local peak is this far below the loud level")
    ap.add_argument("--attack-ms", type=float, default=15.0)
    ap.add_argument("--win-ms", type=float, default=250.0, help="syllable window")
    ap.add_argument("--gamma", type=float, default=1.3, help=">1 keeps small bumps closed")
    ap.add_argument("--release-ms", type=float, default=70.0)
    ap.add_argument("--offset-ms", type=float, default=0.0, help="shift the curve (+ = mouth later)")
    a = ap.parse_args()

    x, sr = read_wav(a.vocals)
    sos = butter(4, [150, 4000], btype="band", fs=sr, output="sos")
    x = sosfiltfilt(sos, x)
    hop = int(sr * 0.010)
    nh = len(x) // hop
    rms = np.sqrt(np.mean(x[: nh * hop].reshape(nh, hop) ** 2, axis=1) + 1e-12)
    # syllable envelope: how far the level rises above the quiet points of the surrounding
    # --win-ms window, relative to the local peak (rap is dense: a plain level never closes)
    from scipy.ndimage import minimum_filter1d, maximum_filter1d, uniform_filter1d
    wh = max(3, int(a.win_ms / 10))
    lin = uniform_filter1d(rms, 2)
    base = uniform_filter1d(minimum_filter1d(lin, wh), wh)
    peak = uniform_filter1d(maximum_filter1d(lin, wh), wh)
    env = np.clip((lin - base) / (peak - base + 1e-6), 0, 1) ** a.gamma
    db = 20 * np.log10(peak + 1e-9)
    env[db < np.percentile(20 * np.log10(rms + 1e-9), 95) - a.gate_db] = 0.0   # silence gate
    # jaw-like smoothing: open fast, close slower
    aa = np.exp(-10.0 / a.attack_ms)
    ar = np.exp(-10.0 / a.release_ms)
    sm = np.zeros_like(env)
    for i in range(1, len(env)):
        c = aa if env[i] > sm[i - 1] else ar
        sm[i] = c * sm[i - 1] + (1 - c) * env[i]
    t_h = (np.arange(nh) + 0.5) * 0.010 + a.offset_ms / 1000.0
    mouth = []
    for i in range(a.n):
        t0, t1 = i / a.fps, (i + 1) / a.fps
        sel = (t_h >= t0) & (t_h < t1)
        mouth.append(float(sm[sel].mean()) if sel.any() else 0.0)
    mouth = np.array(mouth)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"fps": a.fps, "n": a.n, "mouth": [round(v, 4) for v in mouth.tolist()],
               "params": vars(a)}, open(str(out) + ".json", "w"), indent=0)
    # quick plot
    import cv2
    W, H = 1600, 360
    img = np.full((H, W, 3), 255, np.uint8)
    xs = np.linspace(0, len(x) - 1, W).astype(int)
    amp = np.abs(x)
    wv = [amp[max(0, xs[k] - 400): xs[k] + 400].max() for k in range(W)]
    wv = np.array(wv) / (max(wv) + 1e-9)
    for k in range(W):
        cv2.line(img, (k, H // 2 - int(wv[k] * 150)), (k, H // 2 + int(wv[k] * 150)), (200, 200, 200), 1)
    pts = np.array([[int(i / (a.n - 1) * (W - 1)), int(H - 20 - m * (H - 40))] for i, m in enumerate(mouth)], np.int32)
    cv2.polylines(img, [pts], False, (0, 0, 220), 2)
    for s in range(0, int(a.n / a.fps) + 1):
        xx = int(s * a.fps / (a.n - 1) * (W - 1))
        cv2.line(img, (xx, H - 12), (xx, H), (0, 0, 0), 1)
        cv2.putText(img, f"{s}s", (xx + 2, H - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.imwrite(str(out) + ".png", img)
    open_frac = float((mouth > 0.25).mean())
    print(f"{a.n} frames: mouth mean {mouth.mean():.2f}, open (>0.25) {open_frac:.0%} of frames; "
          f"wrote {out}.json and {out}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())

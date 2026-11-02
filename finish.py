#!/usr/bin/env python
"""
finish.py -- last step of the cat-duet pipeline (interface contract F).

  1. re-mux the ORIGINAL audio (work/audio.m4a) over the composited video
     work/cat_R.mp4  ->  work/cat_R_audio.mp4      (-c:v copy -c:a copy -shortest)
     The video is delayed by the source clip's own video/audio start offset
     (work/cut.mp4: video pts 0.033 s, audio 0.000 s) so frame i lands at the
     same instant of the song as it did in the source cut (--av-offset auto).
  2. compose.py  ->  work/final.mp4  (720x1280 9:16, caption above the video band)
  3. contact sheet work/final_sheet.jpg (6 frames at 1, 5, 12, 17, 21.8, 27 s)
     + ffprobe duration / size / fps of final.mp4
  4. sync checks
       audio: cross-correlate final.mp4's audio against work/audio.m4a (and the
              source cut) -> lag must be < --max-offset-ms (default 40)
       video: cross-correlate the per-frame motion-energy series of final.mp4
              against work/cut.mp4 -> lag must be 0 frames (the compose/mux
              chain must not drop or duplicate a frame)

Block mode (round 4, --blocks work/blocks.json from prep_blocks.py): the render covers only
the rapping blocks (cut frames 104..580 + 688..871, 661 frames starting at pts 0). Unless
given explicitly, --audio / --ref / --sheet-times come from the json (audio_blocks.m4a,
blocks_ref.mp4), --av-offset is 0, --muxed / --sheet sit beside --out, the audio check runs
against blocks.audio AND the song trimmed to the same frame ranges (never against cut.mp4),
and the video check runs against blocks.ref with the same frame count required.

Usage
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe finish.py
  ... --caption "the cat killed it ngl" --bg white --caption-color black
  ... --av-offset 0            # do not delay the video (frame 0 at t=0)
  ... --no-video-check         # skip the frame-alignment check
  ... --blocks work/blocks.json --input work/render.mp4 --out work/final_v2.mp4   # block mode
Exit code 0 = all checks passed, 2 = a sync check failed.
"""
from __future__ import annotations

import os
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
WORK = Path(os.environ["VG_WORK"]).resolve() if os.environ.get("VG_WORK") else ROOT / "work"


# ----------------------------------------------------------------------------- ffprobe
def probe(path: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration,size,bit_rate:stream=index,codec_type,codec_name,width,height,"
         "r_frame_rate,avg_frame_rate,nb_frames,duration,start_time,sample_rate,channels",
         "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def stream(info: dict, kind: str) -> dict | None:
    for s in info.get("streams", []):
        if s.get("codec_type") == kind:
            return s
    return None


def count_frames(path: str) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True).stdout.strip()
    return int(out.split(",")[0]) if out else -1


def first_pts(path: str, kind: str) -> float:
    sel = "v:0" if kind == "video" else "a:0"
    # %+#N counts packets of ALL streams, so read a generous number and keep the first
    # frame of the selected stream only.
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", sel, "-show_entries", "frame=pts_time",
         "-of", "csv=p=0", "-read_intervals", "%+#120", path],
        capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        tok = line.strip().split(",")[0]
        if tok and tok != "N/A":
            return float(tok)
    return float("nan")


def fps_of(s: dict) -> float:
    num, den = s.get("r_frame_rate", "30/1").split("/")
    return float(num) / float(den or 1)


def describe(path: str) -> dict:
    info = probe(path)
    v = stream(info, "video") or {}
    a = stream(info, "audio")
    d = {
        "path": path,
        "duration": float(info["format"]["duration"]),
        "size_bytes": int(info["format"]["size"]),
        "width": int(v.get("width", 0)), "height": int(v.get("height", 0)),
        "fps": fps_of(v), "nb_frames": count_frames(path),
        "v_start": float(v.get("start_time", 0.0)),
        "v_dur": float(v.get("duration", 0.0)),
        "audio": None,
    }
    if a:
        d["audio"] = {"codec": a.get("codec_name"), "sr": int(a.get("sample_rate", 0)),
                      "ch": int(a.get("channels", 0)), "dur": float(a.get("duration", 0.0)),
                      "start": float(a.get("start_time", 0.0))}
    return d


def print_desc(tag: str, d: dict) -> None:
    a = d["audio"]
    print("%-14s %dx%d @ %.3f fps, %d frames, video %.3f s (start %.3f), file %.3f s, %.2f MB" % (
        tag, d["width"], d["height"], d["fps"], d["nb_frames"], d["v_dur"], d["v_start"],
        d["duration"], d["size_bytes"] / 1e6))
    if a:
        print("%-14s audio %s %d Hz %dch %.3f s (start %.3f)" % ("", a["codec"], a["sr"], a["ch"], a["dur"], a["start"]))


# ----------------------------------------------------------------------------- steps
def av_offset_from_ref(ref: str) -> float:
    """Video start minus audio start of the reference clip (seconds)."""
    return first_pts(ref, "video") - first_pts(ref, "audio")


def remux(video: str, audio: str, out: str, offset: float) -> None:
    cmd = ["ffmpeg", "-y", "-v", "error"]
    if abs(offset) > 1e-6:
        cmd += ["-itsoffset", "%.6f" % offset]
    cmd += ["-i", video, "-i", audio,
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "copy",
            "-shortest", "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True)


def compose(inp: str, out: str, args) -> None:
    cmd = [sys.executable, str(ROOT / "compose.py"), "--input", inp, "--out", out,
           "--caption", args.caption, "--crf", str(args.crf), "--bg", args.bg,
           "--caption-color", args.caption_color, "--font-size", str(args.font_size)]
    if args.font:
        cmd += ["--font", args.font]
    if args.video_y is not None:
        cmd += ["--video-y", str(args.video_y)]
    subprocess.run(cmd, check=True)


def build_sheet(video: str, times: list[float], out: str, fps: float, per_row: int = 3) -> None:
    cap = cv2.VideoCapture(video)
    tiles = []
    for t in times:
        f = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if not ok:
            print("sheet: could not read frame %d (t=%.2f s)" % (f, t))
            continue
        label = "f%d  t=%.1fs" % (f, t)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(fr, (8, 8), (8 + tw + 16, 8 + th + 18), (40, 40, 40), -1)  # dark box: the pad is white
        cv2.putText(fr, label, (16, 8 + th + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(fr)
    cap.release()
    if not tiles:
        print("sheet: nothing to write")
        return
    rows = []
    for i in range(0, len(tiles), per_row):
        row = tiles[i:i + per_row]
        while len(row) < per_row:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.concatenate(row, 1))
    sheet = np.concatenate(rows, 0)
    cv2.imwrite(out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("wrote %s %dx%d" % (out, sheet.shape[1], sheet.shape[0]))


# ----------------------------------------------------------------------------- sync checks
def decode_audio(path: str, sr: int, afilter: str | None = None) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vn"]
    if afilter:
        cmd += ["-filter_complex", afilter, "-map", "[a]"]
    cmd += ["-ac", "1", "-ar", str(sr), "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return x


def xcorr_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Lag (samples) that maximises the normalised cross-correlation of a vs b.
    Positive lag = a is DELAYED relative to b (a[t] ~ b[t - lag])."""
    a = a - a.mean()
    b = b - b.mean()
    n = len(a) + len(b) - 1
    nfft = 1 << (n - 1).bit_length()
    fa = np.fft.rfft(a, nfft)
    fb = np.fft.rfft(b, nfft)
    cc = np.fft.irfft(fa * np.conj(fb), nfft)
    cc = np.concatenate([cc[-max_lag:], cc[:max_lag + 1]])  # lags -max_lag .. +max_lag
    k = int(np.argmax(cc))
    denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b))) or 1.0
    return k - max_lag, float(cc[k] / denom)


def audio_check(final: str, ref_audio: str, ref_video: str, sr: int, max_ms: float,
                refs: list[tuple[str, str, str | None]] | None = None) -> tuple[bool, float]:
    """refs = [(tag, path, ffmpeg audio filter producing [a] or None)]; default = the original
    audio and the source cut, unfiltered."""
    fa = decode_audio(final, sr)
    ok_all = True
    worst = 0.0
    if refs is None:
        refs = [("audio.m4a", ref_audio, None), ("cut.mp4", ref_video, None)]
    for tag, other, afilter in refs:
        ra = decode_audio(other, sr, afilter)
        lag, peak = xcorr_lag(fa, ra, max_lag=sr)  # search +-1 s
        ms = 1000.0 * lag / sr
        worst = max(worst, abs(ms))
        verdict = "OK" if abs(ms) < max_ms else "FAIL"
        print("audio sync final.mp4 vs %-10s lag %+7.2f ms (%+d samples @ %d Hz), peak corr %.3f  [%s]" % (
            tag, ms, lag, sr, peak, verdict))
        ok_all &= abs(ms) < max_ms
    return ok_all, worst


def motion_series(path: str, size=(96, 54)) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    prev = None
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA).astype(np.float32)
        out.append(0.0 if prev is None else float(np.mean(np.abs(g - prev))))
        prev = g
    cap.release()
    return np.asarray(out, dtype=np.float64)


def video_check(final: str, ref: str, max_lag: int = 15, ref_tag: str = "cut.mp4") -> tuple[bool, int]:
    """Frame alignment: cross-correlate motion-energy series (mean |frame_t - frame_t-1|).
    Faces differ between the two clips but hands / mic / camera push produce the same motion
    peaks, and the shot cuts are huge spikes in both series."""
    mf = motion_series(final)
    mr = motion_series(ref)
    n = min(len(mf), len(mr))
    lag, peak = xcorr_lag(mf[:n], mr[:n], max_lag)
    verdict = "OK" if lag == 0 else "FAIL"
    if len(mf) != len(mr):
        verdict = "FAIL"
    print("video sync final.mp4 vs %s: %d / %d frames, motion-series lag %+d frames, peak corr %.3f  [%s]" % (
        ref_tag, len(mf), len(mr), lag, peak, verdict))
    return lag == 0 and len(mf) == len(mr), lag


# ----------------------------------------------------------------------------- block mode
def song_trim_filter(blocks: dict) -> str:
    """ffmpeg filter that trims the song to the block frame ranges (what prep_blocks.py wrote)."""
    fps = float(blocks["fps"])
    off = float(blocks.get("song_offset", 0.0))
    parts, labels = [], []
    for i, b in enumerate(blocks["blocks"]):
        parts.append("[0:a]atrim=start=%.6f:end=%.6f,asetpts=PTS-STARTPTS[t%d]" % (
            b["cut_start"] / fps + off, b["cut_end"] / fps + off, i))
        labels.append("[t%d]" % i)
    return ";".join(parts) + ";" + "".join(labels) + "concat=n=%d:v=0:a=1[a]" % len(labels)


def apply_blocks(args, ap: argparse.ArgumentParser) -> dict:
    """Block-mode defaults: only arguments still at their parser default are replaced."""
    blocks = json.loads(Path(args.blocks).read_text())
    rel = lambda p: str(p) if Path(p).is_absolute() else str(ROOT / p)  # json paths are repo-relative
    at_default = lambda name: getattr(args, name) == ap.get_default(name)
    if at_default("audio"):
        args.audio = rel(blocks["audio"])
    if at_default("ref"):
        args.ref = rel(blocks["ref"])
    if at_default("av_offset"):
        args.av_offset = "0"
    if at_default("sheet_times") and blocks.get("sheet_times"):
        args.sheet_times = ",".join(str(t) for t in blocks["sheet_times"])
    out = Path(args.out)
    if at_default("muxed"):
        args.muxed = str(out.with_name(out.stem + "_muxed.mp4"))
    if at_default("sheet"):
        args.sheet = str(out.with_name(out.stem + "_sheet.jpg"))
    blocks["_song"] = rel(blocks["song"]) if blocks.get("song") else None
    print("block mode    %s: %d blocks, %d frames; audio %s, ref %s, av-offset %s, sheet times %s" % (
        args.blocks, len(blocks["blocks"]), blocks["n_total"], args.audio, args.ref, args.av_offset, args.sheet_times))
    return blocks


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(WORK / "cat_R.mp4"), help="composited clip (contract E)")
    ap.add_argument("--audio", default=str(WORK / "audio.m4a"), help="original audio to re-mux")
    ap.add_argument("--ref", default=str(WORK / "cut.mp4"), help="source cut: A/V offset reference + sync checks")
    ap.add_argument("--muxed", default=str(WORK / "cat_R_audio.mp4"))
    ap.add_argument("--out", default=str(WORK / "final.mp4"))
    ap.add_argument("--sheet", default=str(WORK / "final_sheet.jpg"))
    ap.add_argument("--sheet-times", default="1,5,12,17,21.8,27")
    ap.add_argument("--caption", default="the cat killed it ngl")
    ap.add_argument("--font", default=None)
    ap.add_argument("--font-size", type=int, default=44)
    ap.add_argument("--bg", default="white")
    ap.add_argument("--caption-color", default="black")
    ap.add_argument("--video-y", type=int, default=440, help="top of the video band in the 1280-px frame (meme layout: band high, caption ~90 px above it)")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--av-offset", default="auto",
                    help="seconds to delay the video before muxing; 'auto' = ref video start - ref audio start")
    ap.add_argument("--max-offset-ms", type=float, default=40.0)
    ap.add_argument("--sr", type=int, default=16000, help="analysis sample rate for the audio cross-correlation")
    ap.add_argument("--no-video-check", action="store_true")
    ap.add_argument("--skip-build", action="store_true", help="only run the checks + sheet on an existing --out")
    ap.add_argument("--blocks", default=None, help="work/blocks.json from prep_blocks.py: block mode (see above)")
    args = ap.parse_args()
    blocks = apply_blocks(args, ap) if args.blocks else None

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit("%s not on PATH" % tool)
    for p in (args.input, args.audio, args.ref):
        if not Path(p).exists():
            sys.exit("missing input: %s" % p)

    t0 = time.time()
    print_desc("input", describe(args.input))
    ainfo = probe(args.audio)
    a = stream(ainfo, "audio") or {}
    print("%-14s %s %s Hz %sch %.3f s" % ("audio", a.get("codec_name"), a.get("sample_rate"), a.get("channels"),
                                           float(ainfo["format"]["duration"])))

    if args.av_offset == "auto":
        offset = av_offset_from_ref(args.ref)
        if offset != offset:  # nan: could not probe the reference
            sys.exit("could not read the first video/audio pts of %s; pass --av-offset explicitly" % args.ref)
        print("av offset     auto from %s: video starts %.6f s after the audio -> -itsoffset on the video" % (
            args.ref, offset))
    else:
        offset = float(args.av_offset)
        print("av offset     %.6f s (manual)" % offset)

    if not args.skip_build:
        t1 = time.time()
        remux(args.input, args.audio, args.muxed, offset)
        print("step 1 remux  -> %s  (%.1f s)" % (args.muxed, time.time() - t1))
        print_desc("muxed", describe(args.muxed))

        t2 = time.time()
        compose(args.muxed, args.out, args)
        print("step 2 compose -> %s  caption=%r  (%.1f s)" % (args.out, args.caption, time.time() - t2))

    final = describe(args.out)
    print_desc("final", final)
    print("final.mp4: duration %.3f s, %dx%d, %.3f fps, %d frames, %.2f MB" % (
        final["duration"], final["width"], final["height"], final["fps"], final["nb_frames"],
        final["size_bytes"] / 1e6))
    print("first pts     final video %.6f s / audio %.6f s ; ref video %.6f s / audio %.6f s" % (
        first_pts(args.out, "video"), first_pts(args.out, "audio"),
        first_pts(args.ref, "video"), first_pts(args.ref, "audio")))

    times = [float(x) for x in args.sheet_times.split(",") if x.strip()]
    build_sheet(args.out, times, args.sheet, final["fps"] or 30.0)

    if blocks is None:
        ok_a, worst_ms = audio_check(args.out, args.audio, args.ref, args.sr, args.max_offset_ms)
    else:
        refs = [(Path(args.audio).name, args.audio, None)]
        if blocks["_song"] and Path(blocks["_song"]).exists():
            refs.append(("song(trim)", blocks["_song"], song_trim_filter(blocks)))
        ok_a, worst_ms = audio_check(args.out, args.audio, args.ref, args.sr, args.max_offset_ms, refs)
        n_exp = int(blocks["n_total"])
        if final["nb_frames"] != n_exp:
            print("frame count   %d != %d expected from %s  [FAIL]" % (final["nb_frames"], n_exp, args.blocks))
            ok_a = False
    ok_v = True
    if not args.no_video_check:
        ok_v, _ = video_check(args.out, args.ref, ref_tag=Path(args.ref).name if blocks else "cut.mp4")

    ok = ok_a and ok_v
    print("checks        audio %s (worst |lag| %.2f ms < %.0f ms), video %s ; total wall %.1f s" % (
        "PASS" if ok_a else "FAIL", worst_ms, args.max_offset_ms,
        "PASS" if ok_v else ("FAIL" if not args.no_video_check else "skipped"), time.time() - t0))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())

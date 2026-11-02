#!/usr/bin/env python
"""
prep_blocks.py -- round-4 block prep (todo step 15, phase 1a).

The round-4 final is the two rapping blocks of work/cut.mp4 only:
  block A = cut frames 104..580 (477 frames)  = the existing work/gj_A.mp4
  block B = cut frames 688..871 (184 frames)  -> work/gj_B2.mp4 (the old gj_B.mp4 starts
            one frame early: its frame 0 is cut frame 687, the last wide-shot frame)

Writes
  work/gj_B2.mp4          cut frames 688..871, libx264 crf 16, 30 fps, no audio
  work/blocks_ref.mp4     gj_A frames + gj_B2 frames (661 frames, video only): the frame
                          reference every block render is checked against
  work/audio_blocks.m4a   the song for exactly those frame ranges (sample-accurate atrim on
                          the decoded work/audio.m4a, concat, aac 192k 44.1 kHz stereo)
  work/blocks.json        block table read by finish.py --blocks and judge_take.py --blocks

Frame i of block X sits at song time (cut_start + i) / fps + song_offset, where song_offset =
video first pts - audio first pts of work/cut.mp4 (+0.033 s: cut.mp4 carries its video on a
33 ms edit list). The block clips start at pts 0, so finish.py muxes them with --av-offset 0.

Every output is verified: frame counts, pixel match of gj_B2 frame 0 / last vs cut frames
688 / 871 (mean abs diff < 1), gj_A frame 0 vs cut frame 104, blocks_ref frame 0 / 477 / 660,
and the audio duration (22.033 s within 25 ms).

Usage
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe prep_blocks.py
Exit code 0 = all verifications passed, 2 = a verification failed.
"""
from __future__ import annotations

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
WORK = ROOT / "work"
FPS = 30


# ----------------------------------------------------------------------------- helpers
def run(cmd: list[str]) -> None:
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def first_pts(path: Path, kind: str) -> float:
    sel = "v:0" if kind == "video" else "a:0"
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", sel, "-show_entries", "frame=pts_time",
         "-of", "csv=p=0", "-read_intervals", "%+#120", str(path)],
        capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        tok = line.strip().split(",")[0]
        if tok and tok != "N/A":
            return float(tok)
    return float("nan")


def probe_duration(path: Path, kind: str | None = None) -> float:
    if kind is None:
        ent, sel = "format=duration", []
    else:
        ent, sel = "stream=duration", ["-select_streams", "v:0" if kind == "video" else "a:0"]
    out = subprocess.run(["ffprobe", "-v", "error", *sel, "-show_entries", ent, "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip()
    return float(out.split(",")[0])


def grab(path: Path, wanted: set[int]) -> tuple[dict[int, np.ndarray], int]:
    """Sequential decode (frame-accurate): returns {index: frame} for the wanted indices + frame count."""
    cap = cv2.VideoCapture(str(path))
    out, i = {}, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in wanted:
            out[i] = fr
        i += 1
    cap.release()
    return out, i


def mad(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


class Verifier:
    def __init__(self) -> None:
        self.fails = 0

    def check(self, ok: bool, msg: str) -> None:
        print("  [%s] %s" % ("OK" if ok else "FAIL", msg))
        if not ok:
            self.fails += 1


# ----------------------------------------------------------------------------- steps
def cut_video(src: Path, out: Path, start: int, end: int, crf: int) -> None:
    """Frames start..end inclusive (0-based decode order), re-timed from pts 0, no audio."""
    run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
         "-vf", "select='between(n\\,%d\\,%d)',setpts=N/%d/TB" % (start, end, FPS),
         "-fps_mode", "passthrough", "-an",
         "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(out)])


def concat_video(parts: list[Path], out: Path, crf: int) -> None:
    cmd = ["ffmpeg", "-y", "-v", "error"]
    for p in parts:
        cmd += ["-i", str(p)]
    chain = "".join("[%d:v]setpts=PTS-STARTPTS[v%d];" % (i, i) for i in range(len(parts)))
    chain += "".join("[v%d]" % i for i in range(len(parts))) + "concat=n=%d:v=1:a=0,setpts=N/%d/TB[v]" % (len(parts), FPS)
    cmd += ["-filter_complex", chain, "-map", "[v]", "-fps_mode", "passthrough", "-an",
            "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(out)]
    run(cmd)


def cut_audio(song: Path, out: Path, ranges: list[tuple[float, float]], bitrate: str) -> None:
    """Sample-accurate trims of the decoded song, concatenated, re-encoded aac."""
    chain = ""
    for i, (t0, t1) in enumerate(ranges):
        chain += "[0:a]atrim=start=%.6f:end=%.6f,asetpts=PTS-STARTPTS[a%d];" % (t0, t1, i)
    chain += "".join("[a%d]" % i for i in range(len(ranges))) + "concat=n=%d:v=0:a=1[a]" % len(ranges)
    run(["ffmpeg", "-y", "-v", "error", "-i", str(song), "-filter_complex", chain, "-map", "[a]",
         "-c:a", "aac", "-b:a", bitrate, "-ar", "44100", "-ac", "2", "-movflags", "+faststart", str(out)])


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cut", default=str(WORK / "cut.mp4"))
    ap.add_argument("--song", default=str(WORK / "audio.m4a"), help="original audio of the cut (bit-exact copy)")
    ap.add_argument("--gj-a", default=str(WORK / "gj_A.mp4"), help="existing block A clip (cut frames 104..580)")
    ap.add_argument("--gj-b2", default=str(WORK / "gj_B2.mp4"))
    ap.add_argument("--ref", default=str(WORK / "blocks_ref.mp4"))
    ap.add_argument("--audio", default=str(WORK / "audio_blocks.m4a"))
    ap.add_argument("--json", default=str(WORK / "blocks.json"))
    ap.add_argument("--a-range", default="104,581", help="block A cut frames [start,end) ")
    ap.add_argument("--b-range", default="688,872", help="block B cut frames [start,end) ")
    ap.add_argument("--sheet-times", default="1,5,9,13,17.5,21")
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--abr", default="192k")
    ap.add_argument("--max-audio-err-ms", type=float, default=25.0)
    args = ap.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit("%s not on PATH" % tool)
    cut, song, gj_a = Path(args.cut), Path(args.song), Path(args.gj_a)
    for p in (cut, song, gj_a):
        if not p.exists():
            sys.exit("missing input: %s" % p)
    gj_b2, ref, audio, js = Path(args.gj_b2), Path(args.ref), Path(args.audio), Path(args.json)
    a0, a1 = (int(x) for x in args.a_range.split(","))
    b0, b1 = (int(x) for x in args.b_range.split(","))
    n_a, n_b = a1 - a0, b1 - b0
    v = Verifier()
    t0 = time.time()

    # song offset: cut frame i sits at song time i/fps + (video first pts - audio first pts)
    off = first_pts(cut, "video") - first_pts(cut, "audio")
    if off != off:
        sys.exit("could not probe the first pts of %s" % cut)
    print("song offset   cut video first pts - audio first pts = %+.6f s" % off)

    # 1. gj_B2
    print("step 1 gj_B2  cut frames %d..%d -> %s" % (b0, b1 - 1, gj_b2))
    cut_video(cut, gj_b2, b0, b1 - 1, args.crf)

    # 2. blocks_ref
    print("step 2 ref    %s + %s -> %s" % (gj_a.name, gj_b2.name, ref))
    concat_video([gj_a, gj_b2], ref, args.crf)

    # 3. audio
    ranges = [(a0 / FPS + off, a1 / FPS + off), (b0 / FPS + off, b1 / FPS + off)]
    print("step 3 audio  song %.6f..%.6f + %.6f..%.6f -> %s" % (*ranges[0], *ranges[1], audio))
    cut_audio(song, audio, ranges, args.abr)

    # 4. verify
    print("step 4 verify")
    cf, n_cut = grab(cut, {a0, a1 - 1, b0, b1 - 1})
    bf, n_b2 = grab(gj_b2, {0, n_b - 1})
    af, n_ga = grab(gj_a, {0, n_a - 1})
    rf, n_ref = grab(ref, {0, n_a - 1, n_a, n_a + n_b - 1})
    v.check(n_b2 == n_b, "gj_B2 frame count %d (expect %d)" % (n_b2, n_b))
    v.check(n_ga == n_a, "gj_A frame count %d (expect %d)" % (n_ga, n_a))
    v.check(n_ref == n_a + n_b, "blocks_ref frame count %d (expect %d)" % (n_ref, n_a + n_b))
    if n_b2 == n_b and n_ga == n_a and n_ref == n_a + n_b:
        d = mad(bf[0], cf[b0]); v.check(d < 1.0, "gj_B2 frame 0 vs cut frame %d: mean abs diff %.3f" % (b0, d))
        d = mad(bf[n_b - 1], cf[b1 - 1]); v.check(d < 1.0, "gj_B2 frame %d vs cut frame %d: mean abs diff %.3f" % (n_b - 1, b1 - 1, d))
        d = mad(af[0], cf[a0]); v.check(d < 1.0, "gj_A frame 0 vs cut frame %d: mean abs diff %.3f" % (a0, d))
        d = mad(af[n_a - 1], cf[a1 - 1]); v.check(d < 1.0, "gj_A frame %d vs cut frame %d: mean abs diff %.3f" % (n_a - 1, a1 - 1, d))
        d = mad(rf[0], cf[a0]); v.check(d < 1.0, "blocks_ref frame 0 vs cut frame %d: mean abs diff %.3f" % (a0, d))
        d = mad(rf[n_a - 1], cf[a1 - 1]); v.check(d < 1.0, "blocks_ref frame %d vs cut frame %d: mean abs diff %.3f" % (n_a - 1, a1 - 1, d))
        d = mad(rf[n_a], cf[b0]); v.check(d < 1.0, "blocks_ref frame %d vs cut frame %d: mean abs diff %.3f" % (n_a, b0, d))
        d = mad(rf[n_a + n_b - 1], cf[b1 - 1]); v.check(d < 1.0, "blocks_ref frame %d vs cut frame %d: mean abs diff %.3f" % (n_a + n_b - 1, b1 - 1, d))
    vp = first_pts(ref, "video")
    v.check(abs(vp) < 1e-6, "blocks_ref video first pts %.6f s (expect 0)" % vp)
    expect = (n_a + n_b) / FPS
    adur = probe_duration(audio, "audio")
    fdur = probe_duration(audio)
    err_ms = 1000.0 * (adur - expect)
    v.check(abs(err_ms) <= args.max_audio_err_ms,
            "audio_blocks stream %.3f s / container %.3f s vs %.3f s expected: err %+.1f ms (limit %.0f)" % (
                adur, fdur, expect, err_ms, args.max_audio_err_ms))
    vdur = probe_duration(ref, "video")
    v.check(abs(vdur - expect) < 0.5 / FPS, "blocks_ref video %.3f s vs %.3f s" % (vdur, expect))

    # 5. blocks.json
    rel = lambda p: Path(p).resolve().relative_to(ROOT).as_posix() if Path(p).resolve().is_relative_to(ROOT) else str(p)
    blocks = {
        "fps": FPS,
        "blocks": [
            {"name": "A", "src": rel(gj_a), "cut_start": a0, "cut_end": a1, "n": n_a, "out_start": 0},
            {"name": "B", "src": rel(gj_b2), "cut_start": b0, "cut_end": b1, "n": n_b, "out_start": n_a},
        ],
        "n_total": n_a + n_b,
        "audio": rel(audio),
        "ref": rel(ref),
        "song": rel(song),
        "cut": rel(cut),
        "song_offset": round(off, 6),
        "sheet_times": [float(x) for x in args.sheet_times.split(",") if x.strip()],
    }
    js.write_text(json.dumps(blocks, indent=2) + "\n")
    print("wrote %s: %d blocks, %d frames, %.3f s" % (js, len(blocks["blocks"]), blocks["n_total"], blocks["n_total"] / FPS))
    print("prep_blocks   %d verification failure(s), total wall %.1f s" % (v.fails, time.time() - t0))
    return 2 if v.fails else 0


if __name__ == "__main__":
    sys.exit(main())

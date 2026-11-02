#!/usr/bin/env python
"""
Compose the swapped clip into the 9:16 TikTok layout used by the viral repost:
white canvas 720x1280, video letterboxed in the middle, caption above.

Usage:
  python compose.py --input work/swapped.mp4 --out work/final.mp4 --caption "sam killed it ngl"
  python compose.py ... --bg black --caption-color white
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

W, H = 720, 1280


def find_font() -> str:
    for c in [
        r"C:/Windows/Fonts/segoeuib.ttf",
        r"C:/Windows/Fonts/arialbd.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        if Path(c).exists():
            return c
    sys.exit("no font found; pass --font")


def ffescape(s: str) -> str:
    # drawtext needs : \ ' escaped
    return s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\\\\\'")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--caption", default="")
    ap.add_argument("--font", default=None)
    ap.add_argument("--font-size", type=int, default=34)
    ap.add_argument("--bg", default="white")
    ap.add_argument("--caption-color", default="black")
    ap.add_argument("--video-y", type=int, default=None, help="top of video band (default: centered)")
    ap.add_argument("--crf", type=int, default=18)
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not on PATH")
    font = args.font or find_font()

    # scale to 720 wide, keep aspect, even height
    vf = [f"scale={W}:-2:flags=lanczos"]
    y = "(oh-ih)/2" if args.video_y is None else str(args.video_y)
    vf.append(f"pad={W}:{H}:0:{y}:color={args.bg}")
    if args.caption:
        # caption sits ~120px above the video band; compute band position from input size
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
             "-of", "csv=p=0", args.input], capture_output=True, text=True, check=True
        ).stdout.strip().split(",")
        iw, ih = int(probe[0]), int(probe[1])
        band_h = int(round(ih * W / iw))
        band_h -= band_h % 2
        top = (H - band_h) // 2 if args.video_y is None else args.video_y
        cap_y = str(max(top - 120, 20))
        vf.append(
            "drawtext=fontfile='{f}':text='{t}':fontsize={s}:fontcolor={c}:x=(w-text_w)/2:y={y}".format(
                f=font.replace(":", "\\:"), t=ffescape(args.caption), s=args.font_size,
                c=args.caption_color, y=cap_y
            )
        )

    cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", args.input,
        "-vf", ",".join(vf),
        "-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf), "-pix_fmt", "yuv420p",
        "-c:a", "copy", "-movflags", "+faststart", args.out,
    ]
    subprocess.run(cmd, check=True)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
splice_frames.py -- replace a range of rendered frames with a re-render.

Copies SRC/00000.png.. into DST starting at index --start (strided frame numbering, the same
numbering wan_animate.py uses for work/wan/<name>/frames). The frames being replaced are first
backed up to DST/../frames_backup/ (once; an existing backup is kept), so a splice can be undone.

Usage
  .venv/Scripts/python.exe splice_frames.py --src work/wan/cat_A_close/frames --dst work/wan/cat_A/frames --start 179
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--start", type=int, required=True)
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    new = sorted(src.glob("*.png"))
    old = sorted(dst.glob("*.png"))
    if not new:
        sys.exit(f"no frames in {src}")
    if a.start + len(new) > len(old):
        sys.exit(f"splice {a.start}+{len(new)} runs past the {len(old)} frames in {dst}")
    backup = dst.parent / "frames_backup"
    if not backup.exists():
        shutil.copytree(dst, backup)
        print(f"backed up {len(old)} frames -> {backup}")
    for i, p in enumerate(new):
        shutil.copy(p, dst / f"{a.start + i:05d}.png")
    print(f"spliced {len(new)} frames into {dst} at {a.start}..{a.start + len(new) - 1}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

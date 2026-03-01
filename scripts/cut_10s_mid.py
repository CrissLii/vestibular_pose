#!/usr/bin/env python3
# coding: utf-8
"""
Cut a video to a 10-second clip centered around the middle.
- Prefer the mid segment.
- If duration < 10s, do NOT cut; just copy to output (re-encode to mp4 for consistency).

Usage:
  python scripts/cut_10s_mid.py --in data/raw/input.mp4 --out data/raw/input_10s.mp4
  python scripts/cut_10s_mid.py --in data/raw/input.mov --out data/raw/input_10s.mp4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from moviepy.editor import VideoFileClip


def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def cut_10s_mid(in_path: Path, out_path: Path, target_sec: float = 10.0) -> dict:
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    ensure_parent(out_path)

    with VideoFileClip(str(in_path)) as clip:
        dur = float(clip.duration or 0.0)

        # If too short -> no cut, keep as is (write out to mp4)
        if dur <= 0:
            raise RuntimeError("Could not read video duration.")

        if dur < target_sec:
            # No cut
            sub = clip
            start, end = 0.0, dur
            mode = "no_cut"
        else:
            mid = dur / 2.0
            start = max(0.0, mid - target_sec / 2.0)
            end = min(dur, start + target_sec)
            # Safety: if end-start < target_sec due to boundaries, shift left
            start = max(0.0, end - target_sec)
            sub = clip.subclip(start, end)
            mode = "cut_mid"

        # Write mp4
        # - audio codec aac (if audio exists)
        # - codec libx264 is standard
        # - remove temp audio file automatically
        sub.write_videofile(
            str(out_path),
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=str(out_path.with_suffix(".temp-audio.m4a")),
            remove_temp=True,
            threads=max(1, os.cpu_count() or 1),
            verbose=False,
            logger=None,
        )

    return {
        "mode": mode,
        "input": str(in_path),
        "output": str(out_path),
        "duration_in": dur,
        "start": start,
        "end": end,
        "duration_out": end - start,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="input video path")
    ap.add_argument("--out", dest="out_path", required=True, help="output mp4 path")
    ap.add_argument("--sec", type=float, default=10.0, help="target clip length seconds (default 10)")
    args = ap.parse_args()

    info = cut_10s_mid(Path(args.in_path), Path(args.out_path), target_sec=args.sec)
    print("✅ Done")
    print(info)

#python scripts/cut_10s_mid.py --in data/raw/IMG_1276.MOV --out data/raw/jump_10s.mp4
if __name__ == "__main__":
    main()

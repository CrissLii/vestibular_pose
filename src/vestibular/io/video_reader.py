from __future__ import annotations
import cv2
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple

@dataclass(frozen=True)
class VideoMeta:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

def get_video_meta(video_path: str | Path) -> VideoMeta:
    p = Path(video_path)
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {p}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return VideoMeta(path=p, fps=fps, frame_count=frame_count, width=width, height=height)

def iter_frames(video_path: str | Path) -> Iterator[Tuple[int, "cv2.Mat"]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield idx, frame
        idx += 1
    cap.release()

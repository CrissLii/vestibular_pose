from __future__ import annotations
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ..io.video_reader import get_video_meta, iter_frames
from ..pose.keypoints import KeypointsFrame


SKELETON = [
    (5, 6), (5, 11), (6, 12), (11, 12),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

_CACHED_FONT: Optional[ImageFont.FreeTypeFont] = None


def _get_cn_font(font_size: int = 28) -> ImageFont.FreeTypeFont:
    global _CACHED_FONT
    if _CACHED_FONT is not None:
        return _CACHED_FONT
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for fp in candidates:
        try:
            _CACHED_FONT = ImageFont.truetype(fp, font_size)
            return _CACHED_FONT
        except Exception:
            continue
    _CACHED_FONT = ImageFont.load_default()
    return _CACHED_FONT


def _make_label_overlay(text: str, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Pre-render a label overlay image and its alpha mask once.

    Returns (overlay_bgr, alpha_mask) — both with shape (h, w, 3) / (h, w).
    These get blitted onto each frame with a simple array copy.
    """
    x, y = 10, 10
    pad_x, pad_y = 12, 8
    font = _get_cn_font(font_size=28)

    tmp = Image.new("RGBA", (width, 80), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tmp)
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = draw.textsize(text, font=font)

    box_w = min(width - 20, tw + pad_x * 2)
    box_h = th + pad_y * 2
    overlay_h = y + box_h + 4

    img = Image.new("RGBA", (width, overlay_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([x, y, x + box_w, y + box_h], fill=(0, 0, 0, 220))
    draw.text((x + pad_x, y + pad_y), text, font=font, fill=(255, 255, 255, 255))

    arr = np.asarray(img)
    bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
    alpha = arr[:, :, 3]
    return bgr, alpha


def _blit_label(frame: np.ndarray, overlay_bgr: np.ndarray, alpha: np.ndarray):
    """Fast per-frame label composite using pre-rendered overlay."""
    h = overlay_bgr.shape[0]
    if frame.shape[0] < h:
        return
    roi = frame[:h, :overlay_bgr.shape[1]]
    mask = alpha > 0
    roi[mask] = overlay_bgr[mask]


def _draw_kpts(
    img,
    xy: np.ndarray,
    conf: np.ndarray,
    conf_thresh: float = 0.2,
    weak_indices: List[int] | None = None,
):
    weak_set = set(weak_indices or [])
    for i in range(len(conf)):
        if conf[i] < conf_thresh:
            continue
        px, py = int(xy[i][0]), int(xy[i][1])
        if i in weak_set:
            cv2.circle(img, (px, py), 6, (0, 0, 255), -1)
            cv2.circle(img, (px, py), 8, (0, 0, 255), 2)
        else:
            cv2.circle(img, (px, py), 3, (0, 255, 0), -1)

    for a, b in SKELETON:
        if a >= len(conf) or b >= len(conf):
            continue
        if conf[a] < conf_thresh or conf[b] < conf_thresh:
            continue
        ax, ay = int(xy[a][0]), int(xy[a][1])
        bx, by = int(xy[b][0]), int(xy[b][1])
        color = (0, 0, 255) if (a in weak_set or b in weak_set) else (0, 255, 255)
        cv2.line(img, (ax, ay), (bx, by), color, 2)


def _draw_cop_dot(
    img,
    xy: np.ndarray,
    conf: np.ndarray,
    conf_thresh: float = 0.2,
    trail: List[tuple] | None = None,
):
    LHP, RHP = 11, 12
    if max(LHP, RHP) >= len(conf):
        return
    if conf[LHP] < conf_thresh or conf[RHP] < conf_thresh:
        return
    hip = ((xy[LHP] + xy[RHP]) / 2.0).astype(int)
    hx, hy = int(hip[0]), int(hip[1])

    if trail is not None:
        trail.append((hx, hy))
        if len(trail) > 30:
            trail.pop(0)
        for i, (tx, ty) in enumerate(trail):
            alpha = (i + 1) / len(trail)
            r = max(1, int(3 * alpha))
            color = (
                int(255 * (1 - alpha)),
                int(100 * alpha),
                int(255 * alpha),
            )
            cv2.circle(img, (tx, ty), r, color, -1)

    cv2.circle(img, (hx, hy), 7, (0, 200, 255), -1)
    cv2.circle(img, (hx, hy), 10, (0, 200, 255), 2)


def _identify_weak_side(
    kpt_frames: List[KeypointsFrame],
    grading: dict | None,
    conf_thresh: float = 0.2,
) -> List[int]:
    if not grading:
        return []
    reasons = grading.get("reasons", {})

    asym = reasons.get("asym_limb")
    ai = reasons.get("ai_hand")
    si = reasons.get("si_load")

    LEFT_JOINTS = [5, 7, 9, 11, 13, 15]
    RIGHT_JOINTS = [6, 8, 10, 12, 14, 16]

    if asym is not None and not np.isnan(asym):
        left_range, right_range = 0.0, 0.0
        for f in kpt_frames:
            for j in LEFT_JOINTS:
                if j < len(f.conf) and f.conf[j] >= conf_thresh:
                    left_range += abs(float(f.xy[j][1]))
            for j in RIGHT_JOINTS:
                if j < len(f.conf) and f.conf[j] >= conf_thresh:
                    right_range += abs(float(f.xy[j][1]))
        return LEFT_JOINTS if left_range < right_range else RIGHT_JOINTS

    if ai is not None and not np.isnan(ai):
        return LEFT_JOINTS if ai > 0 else RIGHT_JOINTS

    if si is not None and not np.isnan(si):
        return LEFT_JOINTS if si > 0 else RIGHT_JOINTS

    return []


def _remux_to_h264(src: Path) -> Path:
    """Re-encode mp4v video to H.264 for browser compatibility."""
    import subprocess
    dst = src.with_suffix(".h264.mp4")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(src),
                "-c:v", "libx264", "-preset", "ultrafast",
                "-crf", "23", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-an", str(dst),
            ],
            check=True,
            capture_output=True,
        )
        src.unlink(missing_ok=True)
        dst.rename(src)
        return src
    except (subprocess.CalledProcessError, FileNotFoundError):
        if dst.exists():
            dst.unlink(missing_ok=True)
        return src


def render_annotated_video(
    video_path: str | Path,
    kpt_frames: List[KeypointsFrame],
    out_path: str | Path,
    label: str,
    conf_thresh: float = 0.2,
    grading: dict | None = None,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = get_video_meta(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = meta.fps if meta.fps and meta.fps > 0 else 25.0
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (meta.width, meta.height))

    # Pre-render label overlay once (instead of per-frame PIL conversion)
    label_bgr, label_alpha = _make_label_overlay(label, meta.width)

    kmap = {f.frame_idx: f for f in kpt_frames}
    weak_joints = _identify_weak_side(kpt_frames, grading, conf_thresh)
    cop_trail: List[tuple] = []

    for idx, frame in iter_frames(video_path):
        f = kmap.get(idx)
        if f is not None:
            _draw_kpts(frame, f.xy, f.conf, conf_thresh=conf_thresh,
                       weak_indices=weak_joints)
            _draw_cop_dot(frame, f.xy, f.conf, conf_thresh=conf_thresh,
                          trail=cop_trail)
        _blit_label(frame, label_bgr, label_alpha)
        writer.write(frame)

    writer.release()

    _remux_to_h264(out_path)
    return out_path

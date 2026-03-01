from __future__ import annotations
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ..io.video_reader import get_video_meta, iter_frames
from ..pose.keypoints import KeypointsFrame


# COCO-17 skeleton (common)
SKELETON = [
    (5, 6), (5, 11), (6, 12), (11, 12),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]
def _get_cn_font(font_size: int = 28) -> ImageFont.FreeTypeFont:
    """
    Try common Chinese fonts on macOS. Fallback to default if not found.
    """
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",                 # macOS (new)
        "/System/Library/Fonts/STHeiti Medium.ttc",           # macOS (older)
        "/System/Library/Fonts/Hiragino Sans GB.ttc",         # some macs
        "/Library/Fonts/Arial Unicode.ttf",                   # sometimes exists
    ]
    for fp in candidates:
        try:
            return ImageFont.truetype(fp, font_size)
        except Exception:
            continue
    # fallback (will not render Chinese well, but avoids crash)
    return ImageFont.load_default()


def _draw_label_box_pil(img_bgr: np.ndarray, text: str) -> np.ndarray:
    """
    Draw a black box + Chinese text using PIL, return BGR image.
    """
    h, w = img_bgr.shape[:2]
    x, y = 10, 10
    pad_x, pad_y = 12, 8
    font = _get_cn_font(font_size=28)

    # BGR -> RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)

    # measure text size
    # PIL>=8: textbbox is preferred
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = draw.textsize(text, font=font)

    box_w = min(w - 20, tw + pad_x * 2)
    box_h = th + pad_y * 2

    # background box
    draw.rectangle([x, y, x + box_w, y + box_h], fill=(0, 0, 0))

    # text
    draw.text((x + pad_x, y + pad_y), text, font=font, fill=(255, 255, 255))

    # RGB -> BGR
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)

def _draw_label_box(img, text: str):
    h, w = img.shape[:2]
    x, y = 10, 10
    box_w, box_h = min(720, w - 20), 46
    cv2.rectangle(img, (x, y), (x + box_w, y + box_h), (0, 0, 0), -1)
    # 注意：OpenCV 默认不支持中文字体渲染（会显示方块/问号）。
    # 这里用英文/符号也能看；若要中文完美显示，需要 PIL+字体文件（后续可加）。
    # 我们先保证 UI/JSON 是中文，视频角标可以先用简短中文（部分系统可显示）。
    cv2.putText(img, text, (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

def _draw_kpts(
    img,
    xy: np.ndarray,
    conf: np.ndarray,
    conf_thresh: float = 0.2,
    weak_indices: List[int] | None = None,
):
    """Draw keypoints and skeleton. Highlight weak-side joints in red."""
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
    """Draw COP (hip midpoint) as a glowing dot with fading trail."""
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
    """Heuristic: find the weaker side based on asymmetry metrics or motion range."""
    if not grading:
        return []
    reasons = grading.get("reasons", {})

    asym = reasons.get("asym_limb")
    ai = reasons.get("ai_hand")
    si = reasons.get("si_load")

    # COCO-17: Left = odd indices (5,7,9,11,13,15), Right = even (6,8,10,12,14,16)
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
        frame = _draw_label_box_pil(frame, label)
        writer.write(frame)

    writer.release()
    return out_path

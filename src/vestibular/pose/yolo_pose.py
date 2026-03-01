from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from ultralytics import YOLO

from .keypoints import KeypointsFrame, pick_first_person

@dataclass
class YoloPoseConfig:
    model_path: str
    conf: float = 0.25
    imgsz: int = 640
    device: Optional[str] = None  # e.g. "cpu" or "mps" (Mac) if supported by your install

class YoloPoseEstimator:
    """Pose-only wrapper around Ultralytics YOLO Pose."""

    def __init__(self, cfg: YoloPoseConfig):
        self.cfg = cfg
        self.model = YOLO(cfg.model_path)

    def predict_video(self, video_path: str | Path, verbose: bool = False):
        # Ultralytics returns a generator/list of Results (one per frame)
        return self.model.predict(
            source=str(video_path),
            conf=self.cfg.conf,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
            verbose=verbose,
            stream=True,  # streaming yields results per frame without loading all into memory
        )

    def results_to_keypoints(self, results_stream):
        """
        Convert Ultralytics Results stream into List[KeypointsFrame].
        Robust to frames with no detections (skip).
        """
        import numpy as np
        from .keypoints import KeypointsFrame

        frames = []
        for i, r in enumerate(results_stream):
            # r.keypoints may be None if no person detected
            kpts = getattr(r, "keypoints", None)
            if kpts is None:
                continue

            xy = getattr(kpts, "xy", None)  # shape: (n, k, 2)
            conf = getattr(kpts, "conf", None)  # shape: (n, k)
            if xy is None or conf is None:
                continue

            # Convert to numpy
            if hasattr(xy, "cpu"):
                xy = xy.cpu()
            if hasattr(conf, "cpu"):
                conf = conf.cpu()

            try:
                xy_np = np.asarray(xy)
                conf_np = np.asarray(conf)
            except Exception:
                continue

            # ✅ 핵心：这一帧没有任何人 -> 跳过
            if xy_np.ndim < 3 or xy_np.shape[0] == 0:
                continue

            # ✅ 多人时：取置信度最高的那个人（更稳）
            # person_score: mean keypoint conf per person
            person_scores = conf_np.mean(axis=1)  # (n,)
            best_idx = int(np.argmax(person_scores))

            frames.append(
                KeypointsFrame(
                    frame_idx=i,
                    xy=xy_np[best_idx].astype(np.float32),
                    conf=conf_np[best_idx].astype(np.float32),
                )
            )

        return frames


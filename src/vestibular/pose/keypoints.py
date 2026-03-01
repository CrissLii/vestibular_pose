from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class KeypointsFrame:
    frame_idx: int
    xy: np.ndarray      # (K, 2) float32
    conf: np.ndarray    # (K,) float32
    person_idx: int = 0

    def valid_mask(self, thresh: float) -> np.ndarray:
        return self.conf >= thresh

def pick_first_person(xy_n: np.ndarray, conf_n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Select the first detected person.
    xy_n: (N, K, 2), conf_n: (N, K)
    returns: (K,2), (K,)
    """
    if xy_n.ndim != 3:
        raise ValueError(f"Expected xy shape (N,K,2), got {xy_n.shape}")
    if conf_n.ndim != 2:
        raise ValueError(f"Expected conf shape (N,K), got {conf_n.shape}")
    if xy_n.shape[0] == 0:
        raise ValueError("No persons detected")
    return xy_n[0], conf_n[0]

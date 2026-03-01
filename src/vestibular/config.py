from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data: Path
    raw: Path
    processed: Path
    outputs: Path
    keypoints: Path
    reports: Path
    videos: Path

def get_paths(project_root: str | Path | None = None) -> ProjectPaths:
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    data = root / "data"
    outputs = data / "outputs"
    return ProjectPaths(
        root=root,
        data=data,
        raw=data / "raw",
        processed=data / "processed",
        outputs=outputs,
        keypoints=outputs / "keypoints",
        reports=outputs / "reports",
        videos=outputs / "videos",
    )

# Default thresholds (can be tuned)
DEFAULT_CONF_KPT = 0.20

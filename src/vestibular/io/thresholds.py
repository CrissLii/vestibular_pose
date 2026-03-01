from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional

def load_thresholds(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Thresholds file not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))

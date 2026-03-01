from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt

def plot_series(
    y: Sequence[float],
    title: str,
    out_path: str | Path,
    x_label: str = "frame",
    y_label: str = "value",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(list(y))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path

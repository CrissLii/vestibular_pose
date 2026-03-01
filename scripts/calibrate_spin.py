#标定脚本

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from vestibular.pipeline.run_one_video import run_spin_on_video

def parse_args():
    p = argparse.ArgumentParser("Calibrate thresholds for spin_in_place using good videos.")
    p.add_argument("--videos-dir", required=True, help="Directory containing good videos (mp4/mov).")
    p.add_argument("--model", required=True, help="YOLO pose model path, e.g. yolo11n-pose.pt")
    p.add_argument("--out", default="data/thresholds.json", help="Output thresholds json path")
    p.add_argument("--device", default=None)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--kpt-conf", type=float, default=0.20)
    return p.parse_args()

def main():
    args = parse_args()
    videos_dir = Path(args.videos_dir)
    out_path = Path(args.out)

    video_paths = sorted([p for p in videos_dir.rglob("*") if p.suffix.lower() in {".mp4", ".mov", ".m4v"}])
    if not video_paths:
        raise FileNotFoundError(f"No videos found in: {videos_dir}")

    rows = []
    for vp in video_paths:
        res = run_spin_on_video(
            video_path=vp,
            model_path=args.model,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            kpt_conf_thresh=args.kpt_conf,
            out_stem=f"calib_{vp.stem}",
        )
        m = res["metrics"]
        rows.append({
            "video": str(vp),
            "trunk_angle_std_deg": m["trunk_angle_std_deg"],
            "trunk_angle_mean_deg": m["trunk_angle_mean_deg"],
            "frames_used": m["frames_used"],
            "frames_total": m["frames_total"],
        })

    stds = np.array([r["trunk_angle_std_deg"] for r in rows], dtype=float)
    means = np.array([r["trunk_angle_mean_deg"] for r in rows], dtype=float)

    # 分位数：用“good样本”的分布去定义边界
    # 经验：P80 作为轻度上界，P95 作为中度上界，P99 作为重度上界
    def q(x, p):
        return float(np.quantile(x, p))

    def make_thresholds_from_good(values: np.ndarray):
        # 如果样本少，用倍率margin；样本够，再用分位数
        if len(values) < 5:
            base = float(np.median(values))
            return {
                "mild_max": base * 1.20,  # 允许比“标准”差一点
                "moderate_max": base * 1.50,
                "severe_max": base * 2.00,
                "method": "margin_from_median",
                "base": base,
            }
        else:
            return {
                "mild_max": float(np.quantile(values, 0.80)),
                "moderate_max": float(np.quantile(values, 0.95)),
                "severe_max": float(np.quantile(values, 0.99)),
                "method": "quantiles",
            }

    std_t = make_thresholds_from_good(stds)
    mean_t = make_thresholds_from_good(means)

    thresholds = {
        "spin_in_place": {
            "source": "calibration_from_good_videos",
            "n_videos": int(len(rows)),
            "std_deg": std_t,
            "mean_deg": mean_t,
            "notes": "If n_videos<5, thresholds use margin from median; with more videos, use quantiles."
        },
        "samples": rows,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(thresholds, ensure_ascii=False, indent=2), encoding="utf-8")
    print("✅ Calibration done.")
    print("Saved thresholds to:", out_path)

if __name__ == "__main__":
    main()

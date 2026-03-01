from __future__ import annotations
import argparse
from pathlib import Path

from vestibular.pipeline.run_one_video import run_spin_on_video

def parse_args():
    p = argparse.ArgumentParser(description="Run spin_in_place (原地旋转) evaluation on a video.")
    p.add_argument("--video", required=True, help="Path to input video, e.g. data/raw/spin.mp4")
    p.add_argument("--model", required=True, help="Path to YOLO pose model, e.g. yolo11n-pose.pt")
    p.add_argument("--device", default=None, help="Device string for ultralytics (optional).")
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    p.add_argument("--kpt-conf", type=float, default=0.20, help="Keypoint confidence threshold.")
    p.add_argument("--thresholds", default=None, help="Path to thresholds.json (optional)")

    return p.parse_args()

def main():
    args = parse_args()
    res = run_spin_on_video(
        video_path=Path(args.video),
        model_path=args.model,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        kpt_conf_thresh=args.kpt_conf,
        thresholds_path=args.thresholds,
    )
    print("✅ Done")
    print("Report:", res["report_path"])
    print("Metrics:", res["metrics"])

if __name__ == "__main__":
    main()

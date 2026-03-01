"""Batch-test all 6 videos in data/raw/ and print summary."""
import time
from vestibular.pipeline.run_auto import run_auto_on_video

VIDEOS = [
    ("data/raw/spin.mp4", "spin_in_place"),
    ("data/raw/jump_10s.mp4", "jump_in_place"),
    ("data/raw/wheelbarrow_10s.mp4", "wheelbarrow_walk"),
    ("data/raw/run_10s.mp4", "run_straight"),
    ("data/raw/forward_roll_10s.mp4", "forward_roll"),
    ("data/raw/head_up_10s.mp4", "head_up_prone"),
]

def main():
    for vpath, expected in VIDEOS:
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"Running: {vpath} (expected: {expected})")
        t0 = time.time()
        try:
            res = run_auto_on_video(
                vpath, "yolo11n-pose.pt",
                thresholds_path="data/thresholds.json",
            )
            dt = time.time() - t0
            detected = res["action_detected"]
            sev = res["grading"]["severity"]
            passed = res["grading"]["pass"]
            tag = "OK" if detected == expected else "MISMATCH"
            print(f"  Detected: {detected} [{tag}]")
            print(f"  Severity: {sev} | Pass: {passed}")
            print(f"  Time: {dt:.1f}s")
            print(f"  Report: {res['report_path']}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

import argparse
from vestibular.pipeline.run_auto import run_auto_on_video

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--thresholds", default=None)
    p.add_argument("--view", default="unknown", choices=["front", "side", "unknown"],
                   help="Camera view angle (front/side/unknown)")
    args = p.parse_args()

    res = run_auto_on_video(
        args.video, args.model,
        thresholds_path=args.thresholds,
        view=args.view,
    )
    print("✅ Done")
    print("Report:", res["report_path"])
    print("Detected:", res["action_detected"])
    print("Severity:", res["grading"]["severity"], "Pass:", res["grading"]["pass"])

if __name__ == "__main__":
    main()

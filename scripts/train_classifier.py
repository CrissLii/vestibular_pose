#!/usr/bin/env python
"""Train an action classifier on the labeled dataset videos.

Usage:
    python scripts/train_classifier.py

Processes all videos in dataset/{action}/front|side/subject_*/
through pose estimation → feature extraction → RandomForest training.
Saves:
    data/action_classifier.pkl   — trained model
    data/action_features.csv     — extracted features for inspection
"""
from __future__ import annotations

import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vestibular.pose.yolo_pose import YoloPoseConfig, YoloPoseEstimator
from vestibular.io.video_reader import get_video_meta
from vestibular.actions.feature_extractor import (
    extract_features,
    features_to_vector,
    FEATURE_NAMES,
)

# Action directory name → internal action label
DIR_TO_ACTION = {
    "1.原地旋转": "spin_in_place",
    "2.小推车": "wheelbarrow_walk",
    "3.原地纵跳": "jump_in_place",
    "4.前滚翻": "forward_roll",
    "5.超人飞": "head_up_prone",
    "6.直线加速跑": "run_straight",
}

# Old unclassified directories (flat structure, no front/side info)
OLD_DIR_TO_ACTION = {
    "1.原地旋转5-10圈": "spin_in_place",
    "2.原地向上跳跃": "jump_in_place",
    "3.小推车": "wheelbarrow_walk",
    "4.直线加速跑": "run_straight",
    "5.前滚翻": "forward_roll",
    "6.抬头向上": "head_up_prone",
}

DATASET_DIR = PROJECT_ROOT / "dataset"
MODEL_PATH = PROJECT_ROOT / "data" / "action_classifier.pkl"
CSV_PATH = PROJECT_ROOT / "data" / "action_features.csv"
KPT_CACHE = PROJECT_ROOT / "data" / "kpt_cache.pkl"
MODEL_WEIGHTS = "yolo11n-pose.pt"


def find_videos(include_old: bool = True,
                max_old_per_class: int = 999) -> list[tuple[str, Path, str]]:
    """Scan dataset directories for video files.

    Returns list of (action_label, video_path, view).
    """
    videos = []

    # Primary: classified dirs with front/side/subject structure
    for dir_name, action in DIR_TO_ACTION.items():
        action_dir = DATASET_DIR / dir_name
        if not action_dir.exists():
            print(f"  [SKIP] {action_dir} not found")
            continue
        for view_dir in sorted(action_dir.iterdir()):
            if not view_dir.is_dir():
                continue
            view_name = view_dir.name
            if view_name not in ("front", "side"):
                continue
            for subj_dir in sorted(view_dir.iterdir()):
                if not subj_dir.is_dir():
                    continue
                for vf in sorted(subj_dir.iterdir()):
                    if vf.suffix.lower() in (".mov", ".mp4", ".avi"):
                        videos.append((action, vf, view_name))

    # Supplementary: old unclassified directories (flat structure)
    if include_old:
        classified_files = {v[1].name for v in videos}
        for dir_name, action in OLD_DIR_TO_ACTION.items():
            action_dir = DATASET_DIR / dir_name
            if not action_dir.exists():
                continue
            count = 0
            for vf in sorted(action_dir.iterdir()):
                if vf.suffix.lower() not in (".mov", ".mp4", ".avi"):
                    continue
                if vf.name in classified_files:
                    continue
                if count >= max_old_per_class:
                    break
                videos.append((action, vf, "unknown"))
                count += 1

    return videos


def main():
    print("=" * 60)
    print("  Action Classifier Training")
    print("=" * 60)

    videos = find_videos()
    print(f"\nFound {len(videos)} labeled videos:")
    action_counts: dict[str, int] = {}
    for action, vf, view in videos:
        action_counts[action] = action_counts.get(action, 0) + 1
    for action, cnt in sorted(action_counts.items()):
        print(f"  {action}: {cnt} videos")

    if len(videos) < 6:
        print("\nERROR: Not enough labeled videos. Need at least 1 per class.")
        sys.exit(1)

    VID_STRIDE = 3

    # Try to load cached keypoints
    kpt_cache: dict[str, tuple] = {}
    if KPT_CACHE.exists():
        try:
            with open(KPT_CACHE, "rb") as f:
                kpt_cache = pickle.load(f)
            print(f"\nLoaded keypoint cache ({len(kpt_cache)} entries)")
        except Exception:
            kpt_cache = {}

    estimator = None

    # Feature extraction
    all_features: list[dict] = []
    all_labels: list[str] = []
    skipped = 0
    cache_updated = False

    for i, (action, vpath, view) in enumerate(videos):
        tag = f"[{i+1}/{len(videos)}]"
        vkey = str(vpath.relative_to(PROJECT_ROOT))
        print(f"\n{tag} {action} ({view}) — {vpath.name}")

        if vkey in kpt_cache:
            kpt_frames, fps = kpt_cache[vkey]
            print(f"  Cache hit: {len(kpt_frames)} frames")
        else:
            if estimator is None:
                print(f"\nLoading YOLO model: {MODEL_WEIGHTS} (vid_stride={VID_STRIDE})")
                estimator = YoloPoseEstimator(
                    YoloPoseConfig(model_path=MODEL_WEIGHTS, conf=0.25, imgsz=640)
                )

            t0 = time.time()
            try:
                meta = get_video_meta(str(vpath))
                fps = (meta.fps or 30.0) / VID_STRIDE
                results = estimator.predict_video(str(vpath), vid_stride=VID_STRIDE)
                kpt_frames = estimator.results_to_keypoints(results)
            except Exception as e:
                print(f"  ERROR pose inference: {e}")
                skipped += 1
                continue

            print(f"  Pose: {len(kpt_frames)} frames in {time.time()-t0:.1f}s")
            kpt_cache[vkey] = (kpt_frames, fps)
            cache_updated = True

        feat = extract_features(kpt_frames, fps=fps)
        if feat is None:
            print(f"  SKIP: not enough valid frames")
            skipped += 1
            continue

        feat["_label"] = action
        feat["_view"] = view
        feat["_file"] = vkey
        all_features.append(feat)
        all_labels.append(action)
        print(f"  OK — {int(feat['n_frames'])} active frames, "
              f"trunk_mean={feat['trunk_mean']:.1f}° "
              f"hip_disp={feat['hip_disp']:.0f}px")

    if cache_updated:
        KPT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(KPT_CACHE, "wb") as f:
            pickle.dump(kpt_cache, f)
        print(f"\nSaved keypoint cache → {KPT_CACHE}")

    print(f"\n{'='*60}")
    print(f"Extracted features from {len(all_features)} videos "
          f"({skipped} skipped)")

    if len(all_features) < 6:
        print("ERROR: not enough successful extractions to train.")
        sys.exit(1)

    # Save CSV
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    csv_cols = FEATURE_NAMES + ["_label", "_view", "_file"]
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
        writer.writeheader()
        for feat in all_features:
            writer.writerow(feat)
    print(f"\nSaved features → {CSV_PATH}")

    # Build arrays
    X = np.array([features_to_vector(f) for f in all_features])
    y = np.array(all_labels)

    unique_labels = sorted(set(y))
    print(f"\nClasses: {unique_labels}")
    for lbl in unique_labels:
        print(f"  {lbl}: {np.sum(y == lbl)} samples")

    # Try multiple classifiers and pick the best one
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score, cross_val_predict, LeaveOneOut

    loo = LeaveOneOut()

    candidates_clf = {
        "RF": RandomForestClassifier(
            n_estimators=200, max_depth=4, min_samples_leaf=1,
            class_weight="balanced", random_state=42,
        ),
        "SVM-rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10, gamma="scale",
                        probability=True, class_weight="balanced")),
        ]),
        "SVM-linear": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="linear", C=1.0,
                        probability=True, class_weight="balanced")),
        ]),
        "KNN-3": Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=3, weights="distance")),
        ]),
        "LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=2000,
                                      class_weight="balanced")),
        ]),
        "GBM": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, min_samples_leaf=1,
            learning_rate=0.1, random_state=42,
        ),
    }

    print("\n--- Classifier Comparison (LOO CV) ---")
    best_name, best_acc, best_clf = "", 0.0, None
    for name, clf in candidates_clf.items():
        clf.fit(X, y)
        scores = cross_val_score(clf, X, y, cv=loo)
        acc = scores.mean()
        print(f"  {name:12s}  {acc:.1%}  ({scores.sum():.0f}/{len(scores)})")
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_clf = clf

    print(f"\nBest: {best_name} ({best_acc:.1%})")

    # Per-class analysis for the best
    best_clf.fit(X, y)
    y_pred_loo = cross_val_predict(best_clf, X, y, cv=loo)
    print(f"\nPer-class LOO accuracy ({best_name}):")
    for lbl in unique_labels:
        mask = y == lbl
        acc = np.mean(y_pred_loo[mask] == lbl)
        wrong = y_pred_loo[mask][y_pred_loo[mask] != lbl]
        misclass = ", ".join(wrong) if len(wrong) > 0 else "none"
        print(f"  {lbl}: {acc:.0%}  misclassified→ {misclass}")

    # Feature importance (if RF or GBM)
    if hasattr(best_clf, "feature_importances_"):
        importances = best_clf.feature_importances_
        top_k = 10
        top_idx = np.argsort(importances)[::-1][:top_k]
        print(f"\nTop {top_k} features:")
        for idx in top_idx:
            print(f"  {FEATURE_NAMES[idx]:25s} {importances[idx]:.4f}")

    # Re-train best on full data and save
    best_clf.fit(X, y)
    model_data = {
        "classifier": best_clf,
        "feature_names": FEATURE_NAMES,
        "labels": unique_labels,
        "n_samples": len(X),
        "loo_accuracy": float(best_acc),
        "classifier_name": best_name,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)
    print(f"\nSaved model ({best_name}) → {MODEL_PATH}")
    print(f"LOO accuracy: {best_acc:.1%}")
    print("\nDone!")


if __name__ == "__main__":
    main()

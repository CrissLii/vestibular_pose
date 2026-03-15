#!/usr/bin/env python
"""视频预处理工具：裁剪视频中的动作核心片段，去除前摇和结束动作。

使用方法：
    python scripts/preprocess_videos.py --input dataset/raw/ --output dataset/processed/
    
功能：
1. 自动检测动作开始和结束时间
2. 裁剪视频保留核心动作片段
3. 批量处理整个目录
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vestibular.pose.yolo_pose import YoloPoseConfig, YoloPoseEstimator
from vestibular.io.video_reader import get_video_meta


def detect_action_boundaries(
    keypoints: list[np.ndarray],
    fps: float,
    action_type: str = "auto",
    min_duration: float = 3.0,
    max_duration: float = 30.0,
) -> tuple[int, int]:
    """检测动作的开始和结束帧。
    
    策略：
    1. 计算重心(hip)的速度和位移
    2. 找到持续运动的区间
    3. 去除前后的静止/准备阶段
    
    Args:
        keypoints: 关键点序列 (n_frames, 17, 3)
        fps: 帧率
        action_type: 动作类型（用于特定策略）
        min_duration: 最小动作时长（秒）
        max_duration: 最大动作时长（秒）
    
    Returns:
        (start_frame, end_frame)
    """
    if len(keypoints) == 0:
        return 0, 0
    
    # 提取hip中点作为重心
    hip_positions = []
    for kpt in keypoints:
        if kpt is None:
            hip_positions.append(None)
            continue
        # COCO-17: left_hip=11, right_hip=12
        # KeypointsFrame has .xy (K,2) and .conf (K,)
        xy = kpt.xy      # (17, 2)
        conf = kpt.conf  # (17,)
        
        l_hip = xy[11] if conf[11] > 0.3 else None
        r_hip = xy[12] if conf[12] > 0.3 else None
        
        if l_hip is not None and r_hip is not None:
            hip_positions.append((l_hip + r_hip) / 2)
        elif l_hip is not None:
            hip_positions.append(l_hip)
        elif r_hip is not None:
            hip_positions.append(r_hip)
        else:
            hip_positions.append(None)
    
    # 填充缺失值
    valid_positions = [p for p in hip_positions if p is not None]
    if len(valid_positions) < 10:
        return 0, len(keypoints)
    
    for i in range(len(hip_positions)):
        if hip_positions[i] is None:
            hip_positions[i] = valid_positions[0]
    
    positions = np.array(hip_positions)
    
    # 计算速度（像素/帧）
    velocities = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    velocities = np.concatenate([[0], velocities])
    
    # 平滑速度曲线
    window = int(fps * 0.5)  # 0.5秒窗口
    if window > 1:
        velocities = np.convolve(velocities, np.ones(window)/window, mode='same')
    
    # 计算速度阈值（使用中位数的倍数）
    velocity_threshold = np.median(velocities) * 1.5
    velocity_threshold = max(velocity_threshold, 5.0)  # 最小阈值5像素/帧
    
    # 找到持续运动的区间
    is_moving = velocities > velocity_threshold
    
    # 找到第一个持续运动的开始
    min_frames = int(fps * min_duration)
    max_frames = int(fps * max_duration)
    
    start_frame = 0
    for i in range(len(is_moving) - min_frames):
        if np.sum(is_moving[i:i+min_frames]) > min_frames * 0.6:
            start_frame = max(0, i - int(fps * 0.5))  # 往前留0.5秒
            break
    
    # 找到最后一个持续运动的结束
    end_frame = len(keypoints)
    for i in range(len(is_moving) - 1, min_frames, -1):
        if np.sum(is_moving[max(0, i-min_frames):i]) > min_frames * 0.6:
            end_frame = min(len(keypoints), i + int(fps * 0.5))  # 往后留0.5秒
            break
    
    # 限制最大时长
    if end_frame - start_frame > max_frames:
        # 保留中间部分
        mid = (start_frame + end_frame) // 2
        start_frame = mid - max_frames // 2
        end_frame = mid + max_frames // 2
    
    return start_frame, end_frame


def trim_video(
    input_path: Path,
    output_path: Path,
    start_frame: int,
    end_frame: int,
) -> bool:
    """裁剪视频。
    
    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径
        start_frame: 开始帧
        end_frame: 结束帧
    
    Returns:
        是否成功
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if start_frame <= frame_idx < end_frame:
            out.write(frame)
        
        frame_idx += 1
        if frame_idx >= end_frame:
            break
    
    cap.release()
    out.release()
    
    return True


def process_video(
    input_path: Path,
    output_path: Path,
    estimator: YoloPoseEstimator,
    action_type: str = "auto",
    dry_run: bool = False,
) -> dict:
    """处理单个视频。
    
    Returns:
        处理结果字典
    """
    print(f"\n处理: {input_path.name}")
    
    # 姿态估计
    try:
        meta = get_video_meta(str(input_path))
        fps = meta.fps or 30.0
        
        results = estimator.predict_video(str(input_path), vid_stride=3)
        keypoints = estimator.results_to_keypoints(results)
        
        if len(keypoints) < 30:
            return {
                "status": "error",
                "message": "视频太短或姿态检测失败",
            }
        
        print(f"  检测到 {len(keypoints)} 帧")
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"姿态估计失败: {e}",
        }
    
    # 检测动作边界
    start_frame, end_frame = detect_action_boundaries(
        keypoints, fps / 3, action_type
    )
    
    # 转换回原始帧数（因为用了vid_stride=3）
    start_frame_orig = start_frame * 3
    end_frame_orig = end_frame * 3
    
    duration_orig = (end_frame_orig - start_frame_orig) / fps
    
    print(f"  检测到动作区间: {start_frame_orig}-{end_frame_orig} "
          f"({duration_orig:.1f}秒)")
    
    if dry_run:
        return {
            "status": "dry_run",
            "start_frame": start_frame_orig,
            "end_frame": end_frame_orig,
            "duration": duration_orig,
        }
    
    # 裁剪视频
    success = trim_video(input_path, output_path, start_frame_orig, end_frame_orig)
    
    if success:
        print(f"  ✓ 保存到: {output_path}")
        return {
            "status": "success",
            "start_frame": start_frame_orig,
            "end_frame": end_frame_orig,
            "duration": duration_orig,
            "output": str(output_path),
        }
    else:
        return {
            "status": "error",
            "message": "视频裁剪失败",
        }


def main():
    parser = argparse.ArgumentParser(description="视频预处理：裁剪动作核心片段")
    parser.add_argument("--input", type=str, required=True,
                        help="输入目录或视频文件")
    parser.add_argument("--output", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--model", type=str, default="yolo11n-pose.pt",
                        help="YOLO模型路径")
    parser.add_argument("--action", type=str, default="auto",
                        help="动作类型（用于特定策略）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只检测不裁剪")
    parser.add_argument("--recursive", action="store_true",
                        help="递归处理子目录")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"错误: 输入路径不存在: {input_path}")
        sys.exit(1)
    
    # 加载YOLO模型
    print(f"加载YOLO模型: {args.model}")
    estimator = YoloPoseEstimator(
        YoloPoseConfig(model_path=args.model, conf=0.25, imgsz=640)
    )
    
    # 收集视频文件
    video_files = []
    if input_path.is_file():
        video_files = [input_path]
    else:
        pattern = "**/*" if args.recursive else "*"
        for ext in [".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI"]:
            video_files.extend(input_path.glob(f"{pattern}{ext}"))
    
    print(f"\n找到 {len(video_files)} 个视频文件")
    
    # 处理每个视频
    results = []
    for i, vpath in enumerate(sorted(video_files)):
        print(f"\n[{i+1}/{len(video_files)}]")
        
        # 保持目录结构
        if input_path.is_dir():
            rel_path = vpath.relative_to(input_path)
            out_path = output_path / rel_path.parent / f"{vpath.stem}_trimmed.mp4"
        else:
            out_path = output_path / f"{vpath.stem}_trimmed.mp4"
        
        result = process_video(vpath, out_path, estimator, args.action, args.dry_run)
        result["input"] = str(vpath)
        results.append(result)
    
    # 统计
    print("\n" + "="*60)
    print("处理完成")
    print("="*60)
    
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = sum(1 for r in results if r["status"] == "error")
    
    print(f"成功: {success_count}")
    print(f"失败: {error_count}")
    
    if args.dry_run:
        print("\n(dry-run模式，未实际裁剪)")
    
    # 显示失败的
    if error_count > 0:
        print("\n失败的视频:")
        for r in results:
            if r["status"] == "error":
                print(f"  {Path(r['input']).name}: {r.get('message', 'unknown')}")


if __name__ == "__main__":
    main()

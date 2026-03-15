#!/usr/bin/env python3
"""手动视频裁剪工具

功能：
- 打开视频文件
- 通过滑块选择开始和结束时间
- 实时预览
- 一键保存到指定目录

使用方法：
    python scripts/video_trimmer.py
    python scripts/video_trimmer.py --input dataset/1.原地旋转5-10圈

依赖：
    pip install opencv-python pillow
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2

# --- GUI ---
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from PIL import Image, ImageTk
except ImportError:
    print("请安装 Pillow: pip install pillow")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 动作目录映射
ACTION_DIRS = {
    "1.原地旋转": "spin_in_place",
    "2.小推车": "wheelbarrow_walk",
    "3.原地纵跳": "jump_in_place",
    "4.前滚翻": "forward_roll",
    "5.超人飞": "head_up_prone",
    "6.直线加速跑": "run_straight",
}

OLD_DIR_MAP = {
    "1.原地旋转5-10圈": "1.原地旋转",
    "2.原地向上跳跃": "3.原地纵跳",
    "3.小推车": "2.小推车",
    "4.直线加速跑": "6.直线加速跑",
    "5.前滚翻": "4.前滚翻",
    "6.抬头向上": "5.超人飞",
}


class VideoTrimmer:
    """视频裁剪工具主界面"""

    PREVIEW_W = 800
    PREVIEW_H = 450

    def __init__(self, root: tk.Tk, initial_dir: str | None = None):
        self.root = root
        self.root.title("视频裁剪工具 - 前庭功能评估数据集")
        self.root.configure(bg="#1e1e2e")
        self.root.resizable(True, True)

        # 状态变量
        self.cap: cv2.VideoCapture | None = None
        self.video_path: Path | None = None
        self.total_frames: int = 0
        self.fps: float = 30.0
        self.current_frame: int = 0
        self.start_frame: int = 0
        self.end_frame: int = 0
        self.is_playing: bool = False
        self._play_thread: threading.Thread | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._video_files: list[Path] = []
        self._current_file_idx: int = 0

        self._build_ui()

        if initial_dir:
            self._load_directory(Path(initial_dir))

    # ------------------------------------------------------------------ UI --

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TScale", background="#1e1e2e", troughcolor="#313244",
                         sliderthickness=18)
        style.configure("TButton", background="#cba6f7", foreground="#1e1e2e",
                         font=("Helvetica", 11, "bold"), padding=6)
        style.map("TButton", background=[("active", "#a6e3a1")])
        style.configure("Accent.TButton", background="#a6e3a1",
                         foreground="#1e1e2e", font=("Helvetica", 12, "bold"))
        style.configure("TCombobox", fieldbackground="#313244",
                         background="#313244", foreground="#cdd6f4")

        C = "#1e1e2e"   # base
        C2 = "#313244"  # surface
        FG = "#cdd6f4"  # text
        ACC = "#cba6f7" # purple

        # ── 顶部工具栏 ─────────────────────────────────────────────────────
        top = tk.Frame(self.root, bg=C2, pady=6)
        top.pack(fill=tk.X, padx=0, pady=0)

        tk.Button(top, text="📂 打开视频", command=self._open_file,
                  bg=ACC, fg=C, font=("Helvetica", 11, "bold"),
                  relief=tk.FLAT, padx=10).pack(side=tk.LEFT, padx=8)

        tk.Button(top, text="📁 打开目录", command=self._open_directory,
                  bg="#89b4fa", fg=C, font=("Helvetica", 11, "bold"),
                  relief=tk.FLAT, padx=10).pack(side=tk.LEFT, padx=4)

        self._file_label = tk.Label(top, text="未选择文件", bg=C2, fg=FG,
                                     font=("Helvetica", 10))
        self._file_label.pack(side=tk.LEFT, padx=16)

        # 目录导航按钮（右侧）
        tk.Button(top, text="◀ 上一个", command=self._prev_file,
                  bg="#585b70", fg=FG, font=("Helvetica", 10),
                  relief=tk.FLAT, padx=8).pack(side=tk.RIGHT, padx=4)
        tk.Button(top, text="下一个 ▶", command=self._next_file,
                  bg="#585b70", fg=FG, font=("Helvetica", 10),
                  relief=tk.FLAT, padx=8).pack(side=tk.RIGHT, padx=4)
        self._nav_label = tk.Label(top, text="", bg=C2, fg="#a6adc8",
                                    font=("Helvetica", 9))
        self._nav_label.pack(side=tk.RIGHT, padx=8)

        # ── 预览区域 ───────────────────────────────────────────────────────
        preview_frame = tk.Frame(self.root, bg=C)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        self._canvas = tk.Canvas(preview_frame, width=self.PREVIEW_W,
                                  height=self.PREVIEW_H,
                                  bg="#11111b", highlightthickness=0)
        self._canvas.pack(expand=True)
        self._canvas.create_text(
            self.PREVIEW_W // 2, self.PREVIEW_H // 2,
            text="请打开视频文件", fill="#585b70",
            font=("Helvetica", 20))

        # ── 时间信息 ───────────────────────────────────────────────────────
        info_row = tk.Frame(self.root, bg=C)
        info_row.pack(fill=tk.X, padx=12)

        self._time_label = tk.Label(
            info_row, text="当前: 0.0s | 开始: 0.0s | 结束: 0.0s | 时长: 0.0s",
            bg=C, fg="#a6e3a1", font=("Consolas", 11))
        self._time_label.pack(side=tk.LEFT)

        self._frame_label = tk.Label(
            info_row, text="帧: 0 / 0",
            bg=C, fg="#89b4fa", font=("Consolas", 10))
        self._frame_label.pack(side=tk.RIGHT)

        # ── 当前帧滑块 ─────────────────────────────────────────────────────
        slider_frame = tk.Frame(self.root, bg=C, pady=4)
        slider_frame.pack(fill=tk.X, padx=12)

        tk.Label(slider_frame, text="当前帧", bg=C, fg=FG,
                 font=("Helvetica", 9)).grid(row=0, column=0, padx=4)
        self._seek_var = tk.IntVar(value=0)
        self._seek_slider = tk.Scale(
            slider_frame, from_=0, to=100,
            orient=tk.HORIZONTAL, variable=self._seek_var,
            command=self._on_seek,
            bg=C, fg=FG, troughcolor=C2, highlightthickness=0,
            sliderrelief=tk.FLAT, length=700, showvalue=False)
        self._seek_slider.grid(row=0, column=1, sticky=tk.EW, padx=4)
        slider_frame.columnconfigure(1, weight=1)

        # ── 开始/结束滑块 ──────────────────────────────────────────────────
        trim_frame = tk.Frame(self.root, bg=C, pady=2)
        trim_frame.pack(fill=tk.X, padx=12)

        tk.Label(trim_frame, text="开始帧", bg=C, fg="#f38ba8",
                 font=("Helvetica", 9)).grid(row=0, column=0, padx=4)
        self._start_var = tk.IntVar(value=0)
        self._start_slider = tk.Scale(
            trim_frame, from_=0, to=100,
            orient=tk.HORIZONTAL, variable=self._start_var,
            command=self._on_start_change,
            bg=C, fg="#f38ba8", troughcolor=C2, highlightthickness=0,
            sliderrelief=tk.FLAT, length=700, showvalue=False)
        self._start_slider.grid(row=0, column=1, sticky=tk.EW, padx=4)

        tk.Label(trim_frame, text="结束帧", bg=C, fg="#89b4fa",
                 font=("Helvetica", 9)).grid(row=1, column=0, padx=4)
        self._end_var = tk.IntVar(value=100)
        self._end_slider = tk.Scale(
            trim_frame, from_=0, to=100,
            orient=tk.HORIZONTAL, variable=self._end_var,
            command=self._on_end_change,
            bg=C, fg="#89b4fa", troughcolor=C2, highlightthickness=0,
            sliderrelief=tk.FLAT, length=700, showvalue=False)
        self._end_slider.grid(row=1, column=1, sticky=tk.EW, padx=4)
        trim_frame.columnconfigure(1, weight=1)

        # ── 播放控制 ───────────────────────────────────────────────────────
        ctrl_frame = tk.Frame(self.root, bg=C2, pady=8)
        ctrl_frame.pack(fill=tk.X, padx=0, pady=4)

        btn_cfg = dict(bg="#585b70", fg=FG, font=("Helvetica", 11),
                       relief=tk.FLAT, padx=10, pady=4)

        tk.Button(ctrl_frame, text="⏮ 跳到开始",
                  command=self._jump_to_start, **btn_cfg).pack(side=tk.LEFT, padx=6)
        tk.Button(ctrl_frame, text="◀ 后退1s",
                  command=lambda: self._step(-int(self.fps)), **btn_cfg).pack(side=tk.LEFT, padx=4)
        self._play_btn = tk.Button(ctrl_frame, text="▶ 播放",
                                   command=self._toggle_play,
                                   bg="#a6e3a1", fg=C,
                                   font=("Helvetica", 12, "bold"),
                                   relief=tk.FLAT, padx=14, pady=4)
        self._play_btn.pack(side=tk.LEFT, padx=4)
        tk.Button(ctrl_frame, text="前进1s ▶",
                  command=lambda: self._step(int(self.fps)), **btn_cfg).pack(side=tk.LEFT, padx=4)
        tk.Button(ctrl_frame, text="⏭ 跳到结束",
                  command=self._jump_to_end, **btn_cfg).pack(side=tk.LEFT, padx=6)

        # 设定开始/结束按钮
        tk.Button(ctrl_frame, text="📍 当前帧设为开始",
                  command=self._set_start_here,
                  bg="#f38ba8", fg=C,
                  font=("Helvetica", 10, "bold"),
                  relief=tk.FLAT, padx=10).pack(side=tk.LEFT, padx=12)
        tk.Button(ctrl_frame, text="📍 当前帧设为结束",
                  command=self._set_end_here,
                  bg="#89b4fa", fg=C,
                  font=("Helvetica", 10, "bold"),
                  relief=tk.FLAT, padx=10).pack(side=tk.LEFT, padx=4)

        # ── 保存区域 ───────────────────────────────────────────────────────
        save_frame = tk.Frame(self.root, bg=C, pady=6)
        save_frame.pack(fill=tk.X, padx=12, pady=4)

        tk.Label(save_frame, text="动作类别:", bg=C, fg=FG,
                 font=("Helvetica", 10)).grid(row=0, column=0, padx=4, sticky=tk.W)
        self._action_var = tk.StringVar(value=list(ACTION_DIRS.keys())[0])
        action_cb = ttk.Combobox(save_frame, textvariable=self._action_var,
                                  values=list(ACTION_DIRS.keys()), width=18,
                                  state="readonly")
        action_cb.grid(row=0, column=1, padx=4)

        tk.Label(save_frame, text="视角:", bg=C, fg=FG,
                 font=("Helvetica", 10)).grid(row=0, column=2, padx=4, sticky=tk.W)
        self._view_var = tk.StringVar(value="front")
        view_cb = ttk.Combobox(save_frame, textvariable=self._view_var,
                                values=["front", "side"], width=8,
                                state="readonly")
        view_cb.grid(row=0, column=3, padx=4)

        tk.Label(save_frame, text="受试者:", bg=C, fg=FG,
                 font=("Helvetica", 10)).grid(row=0, column=4, padx=4, sticky=tk.W)
        self._subject_var = tk.StringVar(value="01")
        subj_cb = ttk.Combobox(save_frame, textvariable=self._subject_var,
                                values=[f"{i:02d}" for i in range(1, 31)],
                                width=5, state="readonly")
        subj_cb.grid(row=0, column=5, padx=4)

        self._save_path_label = tk.Label(
            save_frame, text="保存路径: -",
            bg=C, fg="#a6adc8", font=("Consolas", 9))
        self._save_path_label.grid(row=1, column=0, columnspan=6,
                                    sticky=tk.W, pady=2)

        # 更新保存路径预览
        for var in (self._action_var, self._view_var, self._subject_var):
            var.trace_add("write", lambda *_: self._update_save_path())

        tk.Button(save_frame, text="💾 保存裁剪视频",
                  command=self._save,
                  bg="#a6e3a1", fg="#1e1e2e",
                  font=("Helvetica", 13, "bold"),
                  relief=tk.FLAT, padx=20, pady=6).grid(
                  row=0, column=6, rowspan=2, padx=16, sticky=tk.NS)

        self._status = tk.Label(
            self.root, text="就绪",
            bg="#181825", fg="#a6e3a1",
            font=("Consolas", 10), anchor=tk.W, padx=8)
        self._status.pack(fill=tk.X, side=tk.BOTTOM)

    # --------------------------------------------------------------- LOAD --

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="选择视频文件",
            initialdir=str(PROJECT_ROOT / "dataset"),
            filetypes=[("视频", "*.mp4 *.MP4 *.mov *.MOV *.avi *.AVI"),
                       ("所有文件", "*.*")])
        if path:
            self._load_video(Path(path))

    def _open_directory(self):
        d = filedialog.askdirectory(
            title="选择视频目录",
            initialdir=str(PROJECT_ROOT / "dataset"))
        if d:
            self._load_directory(Path(d))

    def _load_directory(self, d: Path):
        exts = {".mp4", ".mov", ".avi", ".MP4", ".MOV", ".AVI"}
        self._video_files = sorted(
            [f for f in d.iterdir() if f.suffix in exts])
        if not self._video_files:
            messagebox.showwarning("提示", "该目录中没有视频文件")
            return
        self._current_file_idx = 0
        # 尝试从目录名推断动作类别
        dir_name = d.name
        for old_name, new_name in OLD_DIR_MAP.items():
            if old_name in dir_name or dir_name == old_name:
                self._action_var.set(new_name)
                break
        for action_name in ACTION_DIRS:
            if action_name in dir_name or dir_name == action_name:
                self._action_var.set(action_name)
                break
        self._load_video(self._video_files[0])
        self._update_nav_label()

    def _prev_file(self):
        if not self._video_files or self._current_file_idx <= 0:
            return
        self._current_file_idx -= 1
        self._load_video(self._video_files[self._current_file_idx])
        self._update_nav_label()

    def _next_file(self):
        if not self._video_files or self._current_file_idx >= len(self._video_files) - 1:
            return
        self._current_file_idx += 1
        self._load_video(self._video_files[self._current_file_idx])
        self._update_nav_label()

    def _update_nav_label(self):
        if self._video_files:
            self._nav_label.config(
                text=f"{self._current_file_idx + 1} / {len(self._video_files)}")

    def _load_video(self, path: Path):
        self._stop_play()
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            messagebox.showerror("错误", f"无法打开视频: {path.name}")
            return
        self.video_path = path
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.start_frame = 0
        self.end_frame = self.total_frames - 1
        self.current_frame = 0

        for sl in (self._seek_slider, self._start_slider, self._end_slider):
            sl.config(to=self.total_frames - 1)
        self._seek_var.set(0)
        self._start_var.set(0)
        self._end_var.set(self.total_frames - 1)

        self._file_label.config(text=path.name)
        self._update_save_path()
        self._show_frame(0)
        self._set_status(f"已加载: {path.name}  ({self.total_frames} 帧, {self.fps:.1f}fps)")

    # ------------------------------------------------------------- PLAYBACK --

    def _show_frame(self, idx: int):
        if not self.cap:
            return
        idx = max(0, min(idx, self.total_frames - 1))
        self.current_frame = idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            return
        # 标注开始/结束区域
        h, w = frame.shape[:2]
        # 绿色竖线标记开始帧（当前帧到达开始帧时）
        cv2.putText(frame, f"Frame: {idx}  |  Start: {self.start_frame}  |  End: {self.end_frame}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2)
        # 区域外变暗
        if idx < self.start_frame or idx > self.end_frame:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            frame = cv2.addWeighted(frame, 0.4, overlay, 0.6, 0)
            cv2.putText(frame, "区域外", (w//2 - 40, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 255), 2)

        # 转换显示
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        pil_img = pil_img.resize((self.PREVIEW_W, self.PREVIEW_H), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(pil_img)
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)

        # 更新标签
        t = idx / self.fps
        ts = self.start_frame / self.fps
        te = self.end_frame / self.fps
        dur = max(0, te - ts)
        self._time_label.config(
            text=f"当前: {t:.1f}s | 开始: {ts:.1f}s | 结束: {te:.1f}s | 裁剪时长: {dur:.1f}s")
        self._frame_label.config(text=f"帧: {idx} / {self.total_frames - 1}")
        self._seek_var.set(idx)

    def _toggle_play(self):
        if self.is_playing:
            self._stop_play()
        else:
            self._start_play()

    def _start_play(self):
        if not self.cap:
            return
        self.is_playing = True
        self._play_btn.config(text="⏸ 暂停", bg="#fab387")
        self._play_after_id = None
        self._schedule_next_frame()

    def _stop_play(self):
        self.is_playing = False
        if hasattr(self, "_play_after_id") and self._play_after_id:
            self.root.after_cancel(self._play_after_id)
            self._play_after_id = None
        self._play_btn.config(text="▶ 播放", bg="#a6e3a1")

    def _schedule_next_frame(self):
        if not self.is_playing:
            return
        next_frame = self.current_frame + 1
        if next_frame > self.total_frames - 1:
            self._stop_play()
            return
        self._show_frame(next_frame)
        # 计算下一帧延迟（毫秒），限制最低16ms（约60fps上限）
        delay = max(16, int(1000.0 / self.fps))
        self._play_after_id = self.root.after(delay, self._schedule_next_frame)

    def _step(self, delta: int):
        self._stop_play()
        self._show_frame(self.current_frame + delta)

    def _jump_to_start(self):
        self._stop_play()
        self._show_frame(self.start_frame)

    def _jump_to_end(self):
        self._stop_play()
        self._show_frame(self.end_frame)

    def _on_seek(self, val):
        self._stop_play()
        self._show_frame(int(float(val)))

    def _on_start_change(self, val):
        self.start_frame = int(float(val))
        if self.start_frame > self.end_frame:
            self.end_frame = self.start_frame
            self._end_var.set(self.end_frame)
        self._show_frame(self.start_frame)
        self._update_save_path()

    def _on_end_change(self, val):
        self.end_frame = int(float(val))
        if self.end_frame < self.start_frame:
            self.start_frame = self.end_frame
            self._start_var.set(self.start_frame)
        self._show_frame(self.end_frame)
        self._update_save_path()

    def _set_start_here(self):
        self.start_frame = self.current_frame
        self._start_var.set(self.start_frame)
        self._update_save_path()
        self._show_frame(self.current_frame)
        self._set_status(f"开始帧设为: {self.start_frame} ({self.start_frame/self.fps:.1f}s)")

    def _set_end_here(self):
        self.end_frame = self.current_frame
        self._end_var.set(self.end_frame)
        self._update_save_path()
        self._show_frame(self.current_frame)
        self._set_status(f"结束帧设为: {self.end_frame} ({self.end_frame/self.fps:.1f}s)")

    # -------------------------------------------------------------- SAVE --

    def _get_save_path(self) -> Path:
        action = self._action_var.get()
        view = self._view_var.get()
        subject = f"subject_{self._subject_var.get()}"
        out_dir = PROJECT_ROOT / "dataset" / action / view / subject
        stem = self.video_path.stem if self.video_path else "clip"
        return out_dir / f"{stem}_trimmed.mp4"

    def _update_save_path(self, *_):
        if self.video_path:
            p = self._get_save_path()
            dur = max(0, (self.end_frame - self.start_frame) / self.fps)
            self._save_path_label.config(
                text=f"保存路径: {p}  |  裁剪时长: {dur:.1f}s")

    def _save(self):
        if not self.cap or not self.video_path:
            messagebox.showwarning("提示", "请先打开视频文件")
            return
        if self.start_frame >= self.end_frame:
            messagebox.showwarning("提示", "开始帧必须小于结束帧")
            return

        save_path = self._get_save_path()
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 检查是否已存在
        if save_path.exists():
            if not messagebox.askyesno("覆盖确认",
                                        f"文件已存在:\n{save_path.name}\n是否覆盖？"):
                return

        self._set_status("正在保存...")
        self.root.update()

        # 读取视频信息
        cap = cv2.VideoCapture(str(self.video_path))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
        total_to_write = self.end_frame - self.start_frame + 1

        # macOS 上优先用 avc1，失败则降级到 mp4v
        tmp_path = save_path.with_suffix(".tmp.mp4")
        written = 0
        out = None
        for fourcc_str in ("avc1", "mp4v", "XVID"):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            out = cv2.VideoWriter(str(tmp_path), fourcc, fps, (w, h))
            if out.isOpened():
                print(f"[保存] 使用编码: {fourcc_str}  尺寸: {w}x{h}  fps: {fps:.1f}")
                break
            out.release()
            out = None

        if out is None or not out.isOpened():
            messagebox.showerror("保存失败", "无法创建视频写入器，请检查 OpenCV 安装")
            self._set_status("❌ 保存失败：无法创建写入器")
            return

        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        for i in range(self.start_frame, self.end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                print(f"[保存] 第 {i} 帧读取失败，提前结束")
                break
            out.write(frame)
            written += 1
            if written % 30 == 0:
                self._set_status(f"正在保存... {written}/{total_to_write} 帧")
                self.root.update_idletasks()

        cap.release()
        out.release()

        # 验证文件大小
        file_size = tmp_path.stat().st_size if tmp_path.exists() else 0
        print(f"[保存] 写入 {written} 帧，文件大小: {file_size/1024:.1f} KB")

        if file_size < 1024 or written == 0:
            tmp_path.unlink(missing_ok=True)
            messagebox.showerror("保存失败",
                                  f"文件写入异常（大小: {file_size} bytes）\n"
                                  f"写入帧数: {written}/{total_to_write}")
            self._set_status("❌ 保存失败：文件大小异常")
            return

        # 重命名临时文件为最终文件
        tmp_path.replace(save_path)

        dur = written / fps
        print(f"[保存] 成功保存到: {save_path}")
        self._set_status(f"✓ 已保存: {save_path.name}  ({dur:.1f}s, {written}帧)")
        messagebox.showinfo("保存成功",
                             f"视频已保存到:\n{save_path}\n时长: {dur:.1f}秒  ({written}帧)")

        # 自动跳到下一个文件
        if self._video_files and self._current_file_idx < len(self._video_files) - 1:
            if messagebox.askyesno("继续", "是否跳到下一个视频？"):
                self._next_file()

    def _set_status(self, msg: str):
        self._status.config(text=msg)
        self.root.update_idletasks()


# ============================================================== MAIN ==

def main():
    parser = argparse.ArgumentParser(description="手动视频裁剪工具")
    parser.add_argument("--input", type=str, default=None,
                        help="初始打开的目录")
    args = parser.parse_args()

    root = tk.Tk()
    root.geometry("920x780")
    app = VideoTrimmer(root, initial_dir=args.input)
    root.mainloop()


if __name__ == "__main__":
    main()
 
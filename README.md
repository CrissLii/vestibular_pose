# vestibular_pose

基于视频姿态估计的**儿童前庭/感觉统合训练评估系统**。

通过 YOLO Pose 提取人体关键点，自动识别训练动作类型，计算可解释性指标，并给出严重程度分级（正常 / 轻度 / 中度 / 重度）与训练建议。

## 支持的动作

| 动作 ID | 中文名称 | 主要评估指标 |
|---------|---------|------------|
| `spin_in_place` | 原地旋转 5-10 圈 | 躯干角度标准差 / 均值 |
| `jump_in_place` | 原地向上跳跃 | 跳跃幅度、着地稳定性 |
| `wheelbarrow_walk` | 小推车 | 手腕低于臀部比例、身体下沉度 |
| `run_straight` | 直线加速跑 | 臀部位移、加速比、路径摆动 |
| `forward_roll` | 前滚翻 | 躯干角度范围、鼻子垂直范围 |
| `head_up_prone` | 抬头向上（俯卧撑） | 躯干水平比例、抬头比例 |

## 项目结构

```
vestibular_pose/
├── scripts/                        # 入口脚本
│   ├── run_auto.py                # 全自动评估流水线（推荐）
│   ├── run_spin.py                # 单一动作（旋转）评估
│   ├── run_ui.py                  # Gradio 网页 UI
│   ├── calibrate_spin.py          # 旋转阈值标定
│   └── cut_10s_mid.py             # 视频裁剪（取中间 10 秒）
├── src/vestibular/                # 核心库
│   ├── pose/                      # 姿态估计（YOLO Pose 封装）
│   ├── actions/                   # 6 种动作的评估器 + 动作检测
│   ├── features/                  # 特征提取（躯干角度等）
│   ├── io/                        # 视频读取 & 阈值加载
│   ├── pipeline/                  # 流水线编排
│   ├── viz/                       # 可视化（骨骼叠加 / 折线图）
│   └── ui/                        # Gradio 界面
├── data/
│   └── thresholds.json            # 标定阈值（旋转动作）
└── tests/
    └── test_smoke.py
```

## 安装

要求 Python >= 3.9，推荐使用虚拟环境。

```bash
# 克隆仓库
git clone git@github.com:CrissLii/vestibular_pose.git
cd vestibular_pose

# 安装（可编辑模式，含所有依赖）
pip install -e .

# 或仅安装依赖
pip install -r requirements.txt
```

## 使用方式

### 1. 全自动评估（推荐）

自动检测动作类型并评估：

```bash
python scripts/run_auto.py \
  --video data/raw/your_video.mp4 \
  --model yolo11n-pose.pt \
  --thresholds data/thresholds.json   # 可选
```

输出：
- 检测到的动作类型
- 严重程度分级与建议
- JSON 报告（`data/outputs/reports/`）
- 骨骼叠加可视化视频（`data/outputs/videos/`）

### 2. Gradio 网页界面

提供上传视频 → 自动评估 → 查看结果的一站式体验：

```bash
python scripts/run_ui.py
```

浏览器访问 `http://127.0.0.1:7860`。

### 3. 单一动作评估（旋转）

```bash
python scripts/run_spin.py \
  --video data/raw/spin.mp4 \
  --model yolo11n-pose.pt
```

### 4. 阈值标定

使用"正常"旋转视频标定评估阈值：

```bash
python scripts/calibrate_spin.py \
  --videos-dir data/raw/good_spins/ \
  --model yolo11n-pose.pt \
  --out data/thresholds.json
```

### 5. 视频裁剪工具

截取视频中间 10 秒片段：

```bash
python scripts/cut_10s_mid.py \
  --in data/raw/input.MOV \
  --out data/raw/output_10s.mp4
```

## 评估流水线

```
输入视频
   ↓
YOLO Pose → 提取 COCO-17 关键点
   ↓
规则检测 → 自动识别动作类型
   ↓
动作评估器 → 计算指标 + 严重程度分级
   ↓
输出：JSON 报告 / 骨骼叠加视频 / 折线图
```

## 数据目录

将视频放入 `data/raw/` 目录。所有视频文件和数据输出均已通过 `.gitignore` 排除，不会被提交到仓库。

```
data/
├── raw/              # 输入视频（gitignored）
├── processed/        # 中间处理数据（gitignored）
├── outputs/
│   ├── keypoints/    # 提取的关键点 .npz
│   ├── reports/      # JSON 报告 + 折线图
│   └── videos/       # 骨骼叠加可视化视频
└── thresholds.json   # 标定阈值配置
```

## 技术栈

- [Ultralytics YOLO Pose](https://docs.ultralytics.com/) — 人体姿态估计
- [OpenCV](https://opencv.org/) + [NumPy](https://numpy.org/) — 视频处理与数值计算
- [Matplotlib](https://matplotlib.org/) — 数据可视化
- [Gradio](https://www.gradio.app/) — Web UI
- [MoviePy](https://zulko.github.io/moviepy/) — 视频裁剪

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest

# 代码检查
ruff check src/ scripts/ tests/
```

## License

MIT

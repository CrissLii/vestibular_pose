# vestibular_pose

基于视频姿态估计的**儿童前庭/感觉统合训练评估系统**。

通过 YOLO Pose 提取人体关键点，自动识别训练动作类型，计算可解释性指标，并给出严重程度分级（正常 / 轻度 / 中度 / 重度）与训练建议。

## 支持的动作

| 动作 ID | 中文名称 | 主要评估指标 |
|---------|---------|------------|
| `spin_in_place` | 原地旋转 5-10 圈 | 旋转角速度、角速度变异系数、头部漂移 |
| `jump_in_place` | 原地向上跳跃 | 跳跃高度、膝关节着地角度、肢体对称性、节奏稳定性 |
| `wheelbarrow_walk` | 小推车 | 躯干下沉度、步长对称性、手交替指数、侧向摆动 |
| `run_straight` | 直线加速跑 | 峰值速度、加速时间、跑偏距离、步频 |
| `forward_roll` | 前滚翻 | 滚翻耗时、角度覆盖、动作平滑度 (jerk)、偏航角 |
| `head_up_prone` | 抬头向上（俯卧撑） | 躯干角度、保持时间、头部晃动、承重对称指数 |

## 项目结构

```
vestibular_pose/
├── scripts/                        # 入口脚本
│   ├── run_auto.py                # 全自动评估流水线（CLI）
│   ├── run_api.py                 # FastAPI 后端服务
│   ├── run_ui.py                  # Gradio 网页 UI（旧版）
│   ├── start.sh                   # 一键启动前后端
│   ├── batch_test.py              # 批量测试 6 个动作
│   ├── run_spin.py                # 单一动作（旋转）评估
│   ├── calibrate_spin.py          # 旋转阈值标定
│   └── cut_10s_mid.py             # 视频裁剪（取中间 10 秒）
├── frontend/                       # React 前端（Vite + TypeScript）
│   ├── src/
│   │   ├── App.tsx                # 主应用
│   │   ├── api.ts                 # API 调用封装
│   │   ├── types.ts               # TypeScript 类型定义
│   │   └── components/
│   │       ├── Header.tsx         # 页头
│   │       ├── UploadPanel.tsx    # 拖拽上传 + 参数设置
│   │       ├── ResultPanel.tsx    # 评估结果总览
│   │       ├── VideoPlayer.tsx    # 自定义视频播放器（变速）
│   │       ├── RadarChart.tsx     # 交互式雷达图
│   │       ├── CopTrajectory.tsx  # COP 重心轨迹散点图
│   │       ├── SymmetryChart.tsx  # 对称性柱状图
│   │       ├── MetricTable.tsx    # 详细指标表格
│   │       └── SeverityBadge.tsx  # 严重度徽章 + 星级
│   └── package.json
├── src/vestibular/                # 核心库
│   ├── pose/                      # 姿态估计（YOLO Pose 封装）
│   ├── actions/                   # 6 种动作的评估器 + 动作检测
│   │   ├── detectors.py          # 规则式动作分类器
│   │   ├── context.py            # EvalContext / Severity / ViewAngle
│   │   ├── registry.py           # 动作注册表
│   │   └── [action].py           # 各动作评估器
│   ├── features/                  # 特征提取（躯干角度、关节角、速度等）
│   ├── io/                        # 视频读取 & 阈值加载
│   ├── pipeline/                  # 流水线编排
│   ├── api/                       # FastAPI 后端
│   │   └── server.py             # API 路由 + 会话管理
│   ├── viz/                       # 可视化
│   │   ├── overlay_pose.py       # 骨骼叠加（含 COP 轨迹 + 弱侧高亮）
│   │   ├── charts.py             # 雷达图 / COP / 对称性图表生成
│   │   └── plot_series.py        # 时间序列折线图
│   └── ui/                        # Gradio 界面（旧版）
├── data/
│   └── thresholds.json            # 标定阈值（旋转动作）
└── tests/
    └── test_smoke.py
```

## 安装

要求 Python >= 3.9，Node.js >= 18（用于前端），推荐使用 conda 环境。

```bash
# 克隆仓库
git clone git@github.com:CrissLii/vestibular_pose.git
cd vestibular_pose

# 创建 conda 环境（推荐）
conda create -n vestibular_pose python=3.10 -y
conda activate vestibular_pose

# 安装 Python 依赖（可编辑模式）
pip install -e .

# 安装前端依赖
cd frontend && npm install && cd ..

# （可选）安装 Node.js 到 conda 环境
conda install -c conda-forge nodejs -y
```

## 使用方式

### 1. React 前端 + FastAPI 后端（推荐）

现代化 Web 界面，支持拖拽上传、交互式图表、变速播放：

```bash
# 一键启动（同时启动前后端）
bash scripts/start.sh

# 或分别启动
python scripts/run_api.py              # 后端: http://127.0.0.1:8000
cd frontend && npm run dev             # 前端: http://localhost:5173
```

打开浏览器访问 `http://localhost:5173`。

**前端功能：**
- 拖拽上传视频（MP4 / MOV / AVI）
- 自动检测动作类型 + 手动切换重新评估
- 五星评级 + 严重度徽章
- 交互式雷达图（各指标 1-5 分）
- COP 重心轨迹图（时间渐变色）
- 对称性分析柱状图
- 详细指标表格（数值 + 等级）
- 视频播放器（0.25x / 0.5x / 1x / 1.5x / 2x 变速）
- 标注视频中叠加 COP 跟踪光点 + 弱侧肢体红色高亮
- JSON 报告下载

### 2. 全自动评估（CLI）

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

### 3. Gradio 网页界面（旧版）

```bash
python scripts/run_ui.py
# 浏览器访问 http://127.0.0.1:7860
```

### 4. 批量测试

对 `data/raw/` 中的 6 个测试视频逐一评估并汇总结果：

```bash
python scripts/batch_test.py
```

### 5. 阈值标定

使用"正常"旋转视频标定评估阈值：

```bash
python scripts/calibrate_spin.py \
  --videos-dir data/raw/good_spins/ \
  --model yolo11n-pose.pt \
  --out data/thresholds.json
```

### 6. 视频裁剪工具

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
规则检测 → 自动识别动作类型（6 个候选评分）
   ↓
动作评估器 → 计算指标 + 严重程度分级（正常/轻度/中度/重度）
   ↓
视频渲染 → 骨骼叠加 + COP 光点 + 弱侧高亮
   ↓
输出：JSON 报告 / 标注视频 / 交互式图表
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/evaluate` | 上传视频，运行完整评估流水线 |
| POST | `/api/re-evaluate` | 手动选择动作，重新评估（无需重跑姿态推理） |
| GET  | `/api/actions` | 列出支持的 6 个动作 |
| GET  | `/static/videos/{name}` | 获取标注视频 |
| GET  | `/static/reports/{name}` | 获取 JSON 报告 |

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
- [SciPy](https://scipy.org/) — 信号处理（峰值检测、卷积平滑）
- [Matplotlib](https://matplotlib.org/) — 数据可视化
- [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) — 后端 API
- [React](https://react.dev/) + [Vite](https://vite.dev/) + [TypeScript](https://www.typescriptlang.org/) — 前端框架
- [Recharts](https://recharts.org/) — 交互式图表（雷达图、散点图、柱状图）
- [Lucide React](https://lucide.dev/) — 图标库
- [MoviePy](https://zulko.github.io/moviepy/) — 视频裁剪

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest

# 代码检查
ruff check src/ scripts/ tests/

# 前端开发（热重载）
cd frontend && npm run dev
```

## License

MIT

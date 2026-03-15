#!/bin/bash
# 数据清洗和组织脚本
# 功能：
# 1. 清洗旧数据（去除前摇和结束动作）
# 2. 按照新的目录结构组织数据（front/side/subject_XX）
# 3. 合并到统一的训练集

set -e

PROJECT_ROOT="/Users/criss/Desktop/机器学习学习资料/石若含毕业项目/vestibular_pose"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  视频数据清洗和组织流程"
echo "=========================================="

# 激活conda环境
echo ""
echo "请确保已激活 vestibular_pose 环境："
echo "  conda activate vestibular_pose"
echo ""
read -p "按Enter继续..."

# 步骤1：创建临时目录
echo ""
echo "[步骤1] 创建临时目录..."
mkdir -p dataset_cleaned
mkdir -p dataset_organized

# 步骤2：清洗旧数据（去除前摇和结束动作）
echo ""
echo "[步骤2] 清洗旧数据视频..."
echo "这一步会比较耗时（预计30-60分钟）"
echo ""

OLD_DIRS=(
    "1.原地旋转5-10圈"
    "2.原地向上跳跃"
    "3.小推车"
    "4.直线加速跑"
    "5.前滚翻"
    "6.抬头向上"
)

for dir in "${OLD_DIRS[@]}"; do
    if [ -d "dataset/$dir" ]; then
        echo "处理: $dir"
        python scripts/preprocess_videos.py \
            --input "dataset/$dir" \
            --output "dataset_cleaned/$dir" \
            --model yolo11n-pose.pt
    fi
done

echo ""
echo "✓ 视频清洗完成！"

# 步骤3：组织目录结构
echo ""
echo "[步骤3] 组织目录结构..."
echo "请手动将清洗后的视频按以下结构组织："
echo ""
echo "dataset_organized/"
echo "├── 1.原地旋转/"
echo "│   ├── front/"
echo "│   │   ├── subject_01/  # 受试者1的正面视频"
echo "│   │   ├── subject_02/  # 受试者2的正面视频"
echo "│   │   └── ..."
echo "│   └── side/"
echo "│       ├── subject_01/  # 受试者1的侧面视频"
echo "│       └── ..."
echo "├── 2.小推车/"
echo "│   ├── front/"
echo "│   └── side/"
echo "└── ..."
echo ""
echo "清洗后的视频在: dataset_cleaned/"
echo ""
read -p "组织完成后按Enter继续..."

# 步骤4：合并新旧数据
echo ""
echo "[步骤4] 合并数据集..."

# 备份原始dataset
if [ ! -d "dataset_backup" ]; then
    echo "备份原始dataset..."
    cp -r dataset dataset_backup
fi

# 复制已组织的新数据到dataset
echo "复制组织好的数据..."
cp -r dataset_organized/* dataset/

echo ""
echo "✓ 数据集组织完成！"

# 步骤5：验证数据集
echo ""
echo "[步骤5] 验证数据集..."
echo ""
echo "数据集统计："
for action_dir in dataset/[1-6].*; do
    if [ -d "$action_dir" ]; then
        action_name=$(basename "$action_dir")
        video_count=$(find "$action_dir" -type f \( -name "*.MOV" -o -name "*.mp4" -o -name "*.avi" \) | wc -l | tr -d ' ')
        echo "  $action_name: $video_count 个视频"
    fi
done

echo ""
echo "=========================================="
echo "  数据准备完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "  1. 删除旧的关键点缓存: rm data/kpt_cache.pkl"
echo "  2. 运行训练: python scripts/train_classifier.py"
echo ""

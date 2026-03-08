#!/bin/bash
# 在服务器上首次部署时执行（需已安装 git / python3 / node / npm）。
# 在项目根目录执行: bash scripts/setup_server.sh
# 完成后仍需配置 Nginx 和 systemd，见 docs/服务器部署说明.md

set -e
cd "$(dirname "$0")/.."
ROOT="$PWD"

echo "==> 创建 Python 虚拟环境并安装后端依赖"
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

echo "==> 创建数据目录"
mkdir -p data/raw data/outputs/videos data/outputs/reports

echo "==> 安装前端依赖并构建"
cd frontend
npm ci
npm run build
cd "$ROOT"

echo "==> 完成。请按 docs/服务器部署说明.md 配置 systemd 并放行 8000 端口。"

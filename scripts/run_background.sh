#!/bin/bash
# 24 小时运行：同时启动 FastAPI 后端 + Vite 前端，供 launchd 或手动后台使用。
# 用法: bash scripts/run_background.sh

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"

# 若有虚拟环境则激活（可选）
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
  source "$PROJECT_ROOT/.venv/bin/activate"
fi

cd "$PROJECT_ROOT"
mkdir -p logs

# 后端 8000
python scripts/run_api.py >> logs/backend.log 2>&1 &
BACKEND_PID=$!

# 前端 5173（需先等一两秒再起前端，避免端口冲突）
sleep 2
(cd frontend && npm run dev) >> logs/frontend.log 2>&1 &
FRONTEND_PID=$!

echo "Backend PID: $BACKEND_PID (port 8000)"
echo "Frontend PID: $FRONTEND_PID (port 5173)"
echo "Logs: $PROJECT_ROOT/logs/backend.log, logs/frontend.log"
echo "其他设备访问: http://$(ipconfig getifaddr en0 2>/dev/null || echo '本机IP'):5173"

# 保持脚本不退出，方便 launchd 监控
wait $BACKEND_PID $FRONTEND_PID

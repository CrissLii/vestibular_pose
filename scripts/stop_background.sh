#!/bin/bash
# 停止 24 小时运行的后台服务（按端口结束 8000 和 5173）
for port in 8000 5173; do
  pids=$(lsof -ti:"$port" 2>/dev/null)
  if [ -n "$pids" ]; then
    echo "Stopping process(es) on port $port: $pids"
    echo "$pids" | xargs kill -9 2>/dev/null
  fi
done
echo "Done. Ports 8000 and 5173 should be free."

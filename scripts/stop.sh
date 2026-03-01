#!/bin/bash
# Stop FastAPI backend (port 8000) and React frontend dev server (port 5173).
# Usage: bash scripts/stop.sh

killed=0

for port in 8000 5173; do
  pids=$(lsof -ti:"$port" 2>/dev/null)
  if [ -n "$pids" ]; then
    echo "Stopping processes on port $port: $pids"
    echo "$pids" | xargs kill -9 2>/dev/null
    killed=$((killed + 1))
  fi
done

if [ "$killed" -eq 0 ]; then
  echo "No running services found on ports 8000 / 5173."
else
  echo "Done. All services stopped."
fi

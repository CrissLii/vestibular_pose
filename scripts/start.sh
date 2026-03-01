#!/bin/bash
# Start both FastAPI backend and React frontend dev servers.
# Usage: bash scripts/start.sh

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Starting FastAPI backend on http://127.0.0.1:8000 ..."
cd "$PROJECT_ROOT" && python scripts/run_api.py &
BACKEND_PID=$!

echo "Starting React frontend on http://localhost:5173 ..."
cd "$PROJECT_ROOT/frontend" && npm run dev &
FRONTEND_PID=$!

echo ""
echo "  Backend PID:  $BACKEND_PID"
echo "  Frontend PID: $FRONTEND_PID"
echo ""
echo "  Open http://localhost:5173 in your browser."
echo "  Press Ctrl+C to stop both servers."
echo ""

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait

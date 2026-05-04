#!/usr/bin/env bash
# Run from the project root: ./start.sh

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ── kill any stale server on port 8000 ───────────────────────────────────────
stale_pid=$(lsof -ti tcp:8000 -sTCP:LISTEN 2>/dev/null || true)
if [ -n "$stale_pid" ]; then
    echo "Killing stale process on port 8000 (PID $stale_pid)..."
    kill -9 $stale_pid 2>/dev/null || true
    sleep 1
fi

# ── find python ───────────────────────────────────────────────────────────────
if [ -x "$ROOT/.venv/bin/python" ]; then
    python="$ROOT/.venv/bin/python"
elif [ -x "$ROOT/venv/bin/python" ]; then
    python="$ROOT/venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    python="python3"
elif command -v python >/dev/null 2>&1; then
    python="python"
else
    echo "Error: python not found. Install Python (e.g. 'sudo apt install python3') then re-run." >&2
    exit 1
fi

# ── find npm ──────────────────────────────────────────────────────────────────
if ! command -v npm >/dev/null 2>&1; then
    echo "Error: npm not found. Install Node.js from https://nodejs.org then re-run." >&2
    exit 1
fi
npm_cmd="$(command -v npm)"

echo "python: $python"
echo "npm:    $($npm_cmd --version)"
echo ""

# ── start FastAPI server ──────────────────────────────────────────────────────
echo "Starting FastAPI server on http://127.0.0.1:8000 ..."
"$python" -m uvicorn server:app --host 127.0.0.1 --port 8000 &
server_pid=$!

# ── start Plasmo extension dev server ────────────────────────────────────────
echo "Starting extension dev server..."
(cd "$ROOT/extension" && "$npm_cmd" run dev) &
ext_pid=$!

# ── cleanup on exit ──────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $server_pid 2>/dev/null || true
    kill $ext_pid 2>/dev/null || true
    # also kill any child processes (npm spawns subprocesses)
    pkill -P $server_pid 2>/dev/null || true
    pkill -P $ext_pid 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Both servers running. Press Ctrl+C to stop."
wait $server_pid
#!/bin/zsh
set -euo pipefail

APP_DIR="${0:A:h}"
LOG_DIR="$APP_DIR/logs"
mkdir -p "$LOG_DIR"
cd "$APP_DIR"

detector_pid=""
web_pid=""

stop_processes() {
  if [[ -n "$web_pid" ]]; then
    kill "$web_pid" 2>/dev/null || true
  fi
  if [[ -n "$detector_pid" ]]; then
    kill "$detector_pid" 2>/dev/null || true
  fi
}
trap stop_processes EXIT INT TERM

"$APP_DIR/ai-detector.command" >>"$LOG_DIR/detector.log" 2>&1 &
detector_pid="$!"

"$APP_DIR/ai-detector-web.command" >>"$LOG_DIR/web.log" 2>&1 &
web_pid="$!"

for _attempt in {1..80}; do
  if curl --fail --silent --max-time 1 http://127.0.0.1/ >/dev/null; then
    open http://localhost
    break
  fi
  kill -0 "$web_pid" 2>/dev/null || break
  sleep 0.25
done

wait "$web_pid"

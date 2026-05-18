#!/usr/bin/env bash
# Batch runner for the road quality pipeline.
# Processes multiple videos sequentially, logging each run.
#
# Usage:
#   ./scripts/batch_run.sh
#
# Configure VIDEOS and WEIGHTS below before running.
set -u -o pipefail

VIDEOS=(
    "/path/to/video1.mp4"
    "/path/to/video2.mp4"
)
WEIGHTS="best_11.pt"
LOG_DIR="logs_batch"
mkdir -p "$LOG_DIR"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE="$SCRIPT_DIR/run_pipeline.py"

start_ts=$(date +%s)
ok_videos=()
fail_videos=()

for v in "${VIDEOS[@]}"; do
  if [[ ! -f "$v" ]]; then
    echo "[WARN] File not found: $v"
    continue
  fi

  base=$(basename "$v")
  log_file="$LOG_DIR/${base%.mp4}.log"

  echo
  echo "==============================="
  echo "[START] $(date)  Video: $base"
  echo "Log: $log_file"
  echo "==============================="

  if python "$PIPELINE" \
        --video "$v" \
        --weights "$WEIGHTS" \
        >"$log_file" 2>&1; then
    echo "[OK]   $base finished successfully"
    ok_videos+=("$base")
  else
    echo "[FAIL] $base finished with error"
    fail_videos+=("$base")
  fi
done

end_ts=$(date +%s)
total_sec=$(( end_ts - start_ts ))

echo
echo "======================================"
echo "              SUMMARY"
echo "======================================"
echo "Total time: ${total_sec} seconds"
hours=$(python3 -c "print(round($total_sec / 3600, 2))")
echo "Approx: ${hours} hours"
echo

echo "Successful: ${#ok_videos[@]}"
for v in "${ok_videos[@]}"; do
  echo "  + $v"
done
echo
echo "Failed: ${#fail_videos[@]}"
for v in "${fail_videos[@]}"; do
  echo "  - $v"
  log_file="$LOG_DIR/${v%.mp4}.log"
  if [[ -f "$log_file" ]]; then
    echo "    Last 10 lines:"
    tail -n 10 "$log_file" | sed 's/^/      /'
  fi
done
echo
echo "Full logs: $LOG_DIR"

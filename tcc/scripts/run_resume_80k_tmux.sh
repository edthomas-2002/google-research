#!/usr/bin/env bash
set -euo pipefail

SESSION="${TMUX_SESSION:-resume80k}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG="${RESUME_LOG:-/home/ec2-user/tennis/outputs/resume_80k.log}"

tmux kill-session -t "$SESSION" 2>/dev/null || true

CMD="cd '$ROOT' && source tcc-env/bin/activate"
CMD="$CMD && SCHEDULE_SHUTDOWN=1 bash tcc/scripts/run_resume_80k_align_s3.sh 2>&1 | tee -a '$LOG'"

tmux new-session -d -s "$SESSION" bash -lc "$CMD"

echo "Started tmux: $SESSION"
echo "Log: $LOG"
echo "Attach: tmux attach -t $SESSION"

#!/usr/bin/env bash
# Run milestone train+align pipeline in tmux (persistent disk, shutdown on exit).
set -euo pipefail

SESSION="${TMUX_SESSION:-align}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ec2-user/tennis/outputs}"
LOG="${ALIGN_LOG:-$OUTPUT_ROOT/pipeline.log}"

export OUTPUT_ROOT
export LOGDIR="${LOGDIR:-$OUTPUT_ROOT/logs/tennis_forehand_rear}"
export TFRECORD_DIR="${TFRECORD_DIR:-$OUTPUT_ROOT/tfrecords/tennis_forehand_rear_tfrecords}"
export ALIGN_DIR="${ALIGN_DIR:-$OUTPUT_ROOT/alignments}"
export TCC_FOREHAND_CONFIG="${TCC_FOREHAND_CONFIG:-$ROOT/tcc/configs/tennis_forehand_rear_persistent.yml}"
export MAX_VIDEOS="${MAX_VIDEOS:-0}"
export MILESTONES="${MILESTONES:-20000 30000 40000}"
export ALIGN_SEED="${ALIGN_SEED:-42}"
export PREPARE_SEED="${PREPARE_SEED:-42}"
export SKIP_PREPARE="${SKIP_PREPARE:-0}"
export SCHEDULE_SHUTDOWN="${SCHEDULE_SHUTDOWN:-1}"

if ! command -v tmux >/dev/null; then
  echo "tmux is not installed."
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
tmux kill-session -t "$SESSION" 2>/dev/null || true

CMD="cd '$ROOT' && source tcc-env/bin/activate && export TF_CPP_MIN_LOG_LEVEL=1"
CMD="$CMD && export OUTPUT_ROOT='$OUTPUT_ROOT' LOGDIR='$LOGDIR' TFRECORD_DIR='$TFRECORD_DIR'"
CMD="$CMD && export ALIGN_DIR='$ALIGN_DIR' TCC_FOREHAND_CONFIG='$TCC_FOREHAND_CONFIG'"
CMD="$CMD && export MAX_VIDEOS='$MAX_VIDEOS' MILESTONES='$MILESTONES' ALIGN_SEED='$ALIGN_SEED'"
CMD="$CMD && export PREPARE_SEED='$PREPARE_SEED' SKIP_PREPARE='$SKIP_PREPARE'"
CMD="$CMD && export SCHEDULE_SHUTDOWN='$SCHEDULE_SHUTDOWN'"
CMD="$CMD; bash tcc/scripts/run_train_align_milestones.sh 2>&1 | tee -a '$LOG'"

tmux new-session -d -s "$SESSION" bash -lc "$CMD"

echo "Started tmux session: $SESSION"
echo "  Persistent outputs: $OUTPUT_ROOT"
echo "  Log: $LOG"
echo ""
echo "  Checkpoints:  $LOGDIR"
echo "  Alignments:   $ALIGN_DIR/iter_{20000,30000,40000}_val_seed${ALIGN_SEED}.mp4"
echo ""
echo "Attach:  tmux attach -t $SESSION"
echo "Tail:    tail -f $LOG"

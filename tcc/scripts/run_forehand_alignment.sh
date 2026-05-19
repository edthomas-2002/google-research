#!/usr/bin/env bash
# Full pipeline: TFRecords -> train -> random-pair alignment video.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
source tcc-env/bin/activate
export TF_CPP_MIN_LOG_LEVEL=1

VIDEO_DIR="${VIDEO_DIR:-$ROOT/tcc/Forehands/Rear View}"
LOGDIR="${LOGDIR:-/tmp/tennis_forehand_rear_logs}"
OUTPUT="${OUTPUT:-/tmp/tennis_forehand_aligned.gif}"
SEED="${SEED:-}"
MAX_VIDEOS="${MAX_VIDEOS:-0}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

echo "=== Forehand TCC alignment pipeline ==="
echo "Video dir: $VIDEO_DIR"
echo "Log dir:   $LOGDIR"
echo "Output:    $OUTPUT"

if [[ "$SKIP_PREPARE" != "1" ]]; then
  echo ""
  echo "=== Step 1/3: Build TFRecords ==="
  PREP_ARGS=(--video_dir "$VIDEO_DIR")
  if [[ "$MAX_VIDEOS" != "0" ]]; then
    PREP_ARGS+=(--max_videos "$MAX_VIDEOS")
  fi
  python tcc/scripts/prepare_forehand_tfrecords.py "${PREP_ARGS[@]}"
fi

if [[ "$SKIP_TRAIN" != "1" ]]; then
  echo ""
  echo "=== Step 2/3: Train TCC ==="
  LOGDIR="$LOGDIR" bash tcc/scripts/train_forehand.sh
fi

echo ""
echo "=== Step 3/3: Align random pair ==="
ALIGN_ARGS=(--video_dir "$VIDEO_DIR" --logdir "$LOGDIR" --output "$OUTPUT")
if [[ -n "$SEED" ]]; then
  ALIGN_ARGS+=(--seed "$SEED")
fi
python tcc/scripts/align_random_pair.py "${ALIGN_ARGS[@]}"

echo ""
echo "Finished. Open: $OUTPUT"

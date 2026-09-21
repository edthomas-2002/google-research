#!/usr/bin/env bash
# Build deterministic train/val TFRecords with clips sampled at nominal 24 FPS.
# Reads videos from disk (not /tmp). Converted clips are deleted only after all
# TFRecords are written. Staging copy for training: /tmp/%s_tfrecords/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
source tcc-env/bin/activate

VIDEO_DIR="${FOREHAND_VIDEO_DIR:-/home/ec2-user/tennis/videos/Forehands/Rear View}"
OUTPUT_ROOT="${TCC_OUTPUT_ROOT:-/home/ec2-user/tennis/outputs}"
TFRECORD_DIR="${FOREHAND_TFRECORD_DIR:-$OUTPUT_ROOT/tennis_forehand_rear_tfrecords}"
TMP_TFRECORD_DIR="/tmp/tennis_forehand_rear_tfrecords"

python -m tcc.dataset_preparation.videos_to_tfrecords \
  --input_dir "$VIDEO_DIR" \
  --output_dir "$TFRECORD_DIR" \
  --name tennis_forehand_rear \
  --fps 24 \
  --drop_videos_below_fps \
  --val_fraction 0.1 \
  --splits_json tcc/data/tennis_forehand_rear_splits.json \
  --action_label -1 \
  --atomic_delete_videos

rsync -a --info=progress2 "$TFRECORD_DIR/" "$TMP_TFRECORD_DIR/"
echo "TFRecords: $TFRECORD_DIR (copy at $TMP_TFRECORD_DIR)"

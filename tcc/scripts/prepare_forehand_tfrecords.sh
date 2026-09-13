#!/usr/bin/env bash
# Build deterministic train/val TFRecords with clips sampled at nominal 30 FPS.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
source tcc-env/bin/activate

python -m tcc.dataset_preparation.videos_to_tfrecords \
  --input_dir "/tmp/Forehands/Rear View" \
  --name tennis_forehand_rear \
  --fps 30 \
  --drop_videos_below_fps \
  --val_fraction 0.1 \
  --splits_json tcc/data/tennis_forehand_rear_splits.json \
  --action_label -1

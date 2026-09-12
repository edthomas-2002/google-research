#!/usr/bin/env bash
# Align two video clips with a trained checkpoint using stock extract_embeddings + visualize_alignment.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 CLIP_A CLIP_B [OUTPUT_GIF]" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

CLIP_A="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
CLIP_B="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
OUTPUT="${3:-/tmp/aligned.gif}"
LOGDIR="${TCC_OUTPUT_ROOT:-/home/ec2-user/tennis/outputs}/logs/tennis_forehand_rear"
PAIR_DIR="/tmp/tennis_forehand_rear_pair_tfrecords"
STAGING="${PAIR_DIR}_staging"
EMB_PATH="$(dirname "$OUTPUT")/$(basename "$OUTPUT" .gif)_embeddings.npy"

rm -rf "$STAGING" "$PAIR_DIR"
mkdir -p "$STAGING" "$(dirname "$OUTPUT")"
ln -s "$CLIP_A" "$STAGING/$(basename "$CLIP_A")"
ln -s "$CLIP_B" "$STAGING/$(basename "$CLIP_B")"

python -m tcc.dataset_preparation.videos_to_tfrecords \
  --input_dir "$STAGING" \
  --name tennis_forehand_rear_pair_val \
  --output_dir "$PAIR_DIR" \
  --file_pattern '*' \
  --files_per_shard 2 \
  --action_label -1

rm -rf "$STAGING"

python -m tcc.extract_embeddings \
  --alsologtostderr \
  --logdir "$LOGDIR" \
  --dataset tennis_forehand_rear_pair \
  --split val \
  --path_to_tfrecords '/tmp/%s_tfrecords/' \
  --save_path "$EMB_PATH" \
  --keep_data \
  --keep_labels=false \
  --max_embs 2 \
  --sample_all_stride 2 \
  --frames_per_batch 64

python -m tcc.visualize_alignment \
  --alsologtostderr \
  --video_path "$OUTPUT" \
  --embs_path "$EMB_PATH" \
  --reference_video 0 \
  --candidate_video 1 \
  --use_dtw \
  --interval 50

echo "Alignment GIF: $OUTPUT"
echo "Embeddings:    $EMB_PATH"

#!/usr/bin/env bash
# Run a compact first-round TCC encoder sweep, sequentially.
#
# All runs inherit configs/tennis_forehand_rear.yml and change one variable
# from the baseline. Set EXPERIMENTS to a space-separated subset to run fewer:
#   EXPERIMENTS="baseline context_short" bash tcc/scripts/run_forehand_experiments.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

BASE_CONFIG="${TCC_FOREHAND_BASE_CONFIG:-$ROOT/tcc/configs/tennis_forehand_rear.yml}"
EXPERIMENT_ROOT="${TCC_EXPERIMENT_ROOT:-/tmp/tcc_forehand_experiments}"
OUTPUT_ROOT="${TCC_OUTPUT_ROOT:-/home/ec2-user/tennis/outputs}"
PERSISTENT_ROOT="${TCC_EXPERIMENT_OUTPUT_ROOT:-$OUTPUT_ROOT/logs/tennis_forehand_rear_experiments}"
EXPERIMENTS="${EXPERIMENTS:-baseline context_short context_long temperature_sharp temperature_soft backbone_frozen}"

mkdir -p "$EXPERIMENT_ROOT/configs" "$PERSISTENT_ROOT"

write_config() {
  local output_path="$1"
  local frame_stride="$2"
  local temperature="$3"
  local train_base="$4"
  local batch_size="$5"
  local num_frames="$6"

  awk \
    -v frame_stride="$frame_stride" \
    -v temperature="$temperature" \
    -v train_base="$train_base" \
    -v batch_size="$batch_size" \
    -v num_frames="$num_frames" '
      /^[[:space:]]+MAX_ITERS:/ {
        sub(/MAX_ITERS:.*/, "MAX_ITERS: 10000")
      }
      /^[[:space:]]+FRAME_STRIDE:/ {
        sub(/FRAME_STRIDE:.*/, "FRAME_STRIDE: " frame_stride)
      }
      /^[[:space:]]+SOFTMAX_TEMPERATURE:/ {
        sub(/SOFTMAX_TEMPERATURE:.*/,
            "SOFTMAX_TEMPERATURE: " temperature)
      }
      /^[[:space:]]+TRAIN_BASE:/ {
        sub(/TRAIN_BASE:.*/, "TRAIN_BASE: " train_base)
      }
      /^[[:space:]]+BATCH_SIZE:/ {
        sub(/BATCH_SIZE:.*/, "BATCH_SIZE: " batch_size)
      }
      /^[[:space:]]+NUM_FRAMES:/ {
        sub(/NUM_FRAMES:.*/, "NUM_FRAMES: " num_frames)
      }
      { print }
    ' "$BASE_CONFIG" > "$output_path"
}

for experiment in $EXPERIMENTS; do
  batch_size=2
  num_frames=20
  case "$experiment" in
    baseline)
      frame_stride=5
      temperature=0.1
      train_base=only_bn
      ;;
    context_short)
      frame_stride=2
      temperature=0.1
      train_base=only_bn
      ;;
    context_long)
      frame_stride=10
      temperature=0.1
      train_base=only_bn
      ;;
    temperature_sharp)
      frame_stride=5
      temperature=0.05
      train_base=only_bn
      ;;
    temperature_soft)
      frame_stride=5
      temperature=0.2
      train_base=only_bn
      ;;
    backbone_frozen)
      frame_stride=5
      temperature=0.1
      train_base=frozen
      ;;
    *)
      echo "Unknown experiment: $experiment" >&2
      exit 2
      ;;
  esac

  config_path="$EXPERIMENT_ROOT/configs/$experiment.yml"
  logdir="$EXPERIMENT_ROOT/$experiment"
  persistent_dir="$PERSISTENT_ROOT/$experiment"
  write_config \
    "$config_path" "$frame_stride" "$temperature" "$train_base" \
    "$batch_size" "$num_frames"

  echo
  echo "=== $experiment: batch=$batch_size frames=$num_frames stride=$frame_stride temperature=$temperature train_base=$train_base max_iters=10000 ==="
  if [[ "${TCC_EXPERIMENT_DRY_RUN:-0}" == "1" ]]; then
    continue
  fi
  TCC_FOREHAND_CONFIG="$config_path" \
  TCC_LOGDIR="$logdir" \
  TCC_PERSISTENT_DIR="$persistent_dir" \
  WANDB_RUN_NAME="forehand-tcc-$experiment" \
    bash tcc/scripts/train_forehand.sh
done

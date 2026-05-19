#!/usr/bin/env bash
# Full forehand dataset: train to 20k / 30k / 40k with val alignment MP4s at each milestone.
# All artifacts under OUTPUT_ROOT on persistent disk. Shuts down instance on exit (success or fail).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
source tcc-env/bin/activate
export TF_CPP_MIN_LOG_LEVEL=1

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ec2-user/tennis/outputs}"
LOGDIR="${LOGDIR:-$OUTPUT_ROOT/logs/tennis_forehand_rear}"
TFRECORD_DIR="${TFRECORD_DIR:-$OUTPUT_ROOT/tfrecords/tennis_forehand_rear_tfrecords}"
ALIGN_DIR="${ALIGN_DIR:-$OUTPUT_ROOT/alignments}"
PAIR_TFRECORD_ROOT="${PAIR_TFRECORD_ROOT:-$OUTPUT_ROOT/pair_tfrecords}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$OUTPUT_ROOT/weights}"
PIPELINE_LOG="${PIPELINE_LOG:-$OUTPUT_ROOT/pipeline.log}"
TCC_FOREHAND_CONFIG="${TCC_FOREHAND_CONFIG:-$ROOT/tcc/configs/tennis_forehand_rear_persistent.yml}"
VIDEO_DIR="${VIDEO_DIR:-$ROOT/tcc/Forehands/Rear View}"
MILESTONES="${MILESTONES:-20000 30000 40000}"
ALIGN_SEED="${ALIGN_SEED:-42}"
PREPARE_SEED="${PREPARE_SEED:-42}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
MAX_VIDEOS="${MAX_VIDEOS:-0}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
SCHEDULE_SHUTDOWN="${SCHEDULE_SHUTDOWN:-1}"

mkdir -p "$OUTPUT_ROOT" "$LOGDIR" "$ALIGN_DIR" "$WEIGHTS_DIR" "$PAIR_TFRECORD_ROOT"

shutdown_when_done() {
  local code=$?
  if [[ "$SCHEDULE_SHUTDOWN" == "1" ]]; then
    echo ""
    echo "Pipeline finished (exit $code). Scheduling shutdown in 1 minute..."
    sudo shutdown -h +1 || true
  fi
}
trap shutdown_when_done EXIT

RESNET="$WEIGHTS_DIR/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5"
if [[ ! -f "$RESNET" ]]; then
  echo "Downloading ResNet50v2 weights to $RESNET ..."
  wget -q -P "$WEIGHTS_DIR" \
    https://github.com/keras-team/keras-applications/releases/download/resnet/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5
fi

python - <<'PY' || true
import tensorflow as tf
for g in tf.config.list_physical_devices('GPU'):
  tf.config.experimental.set_memory_growth(g, True)
print('GPUs:', tf.config.list_physical_devices('GPU'))
PY

has_tfrecords() {
  compgen -G "$TFRECORD_DIR/tennis_forehand_rear_train-"'*.tfrecord' > /dev/null
}

if [[ "$SKIP_PREPARE" != "1" ]] && ! has_tfrecords; then
  echo "=== Building TFRecords -> $TFRECORD_DIR ==="
  PREP=(python tcc/scripts/prepare_forehand_tfrecords.py
        --video_dir "$VIDEO_DIR"
        --output_dir "$TFRECORD_DIR")
  if [[ "$MAX_VIDEOS" != "0" ]]; then
    PREP+=(--max_videos "$MAX_VIDEOS")
  fi
  "${PREP[@]}"
elif has_tfrecords; then
  echo "=== Using existing TFRecords in $TFRECORD_DIR ==="
else
  echo "ERROR: No TFRecords in $TFRECORD_DIR and SKIP_PREPARE=1" >&2
  exit 1
fi

# Path pattern for pair dataset: .../pair_tfrecords/%s_tfrecords/ -> tennis_forehand_rear_pair_tfrecords
PATH_TO_TFRECORDS_PATTERN="${PAIR_TFRECORD_ROOT}/%s_tfrecords/"

train_until() {
  local target="$1"
  echo ""
  echo "=== Training until iter $target ==="
  mkdir -p "$LOGDIR"
  export LOGDIR TCC_FOREHAND_CONFIG
  export TARGET_ITERS="$target"
  python - <<'PY'
import os
import sys
import yaml

logdir = os.environ["LOGDIR"]
src = os.environ["TCC_FOREHAND_CONFIG"]
target = int(os.environ["TARGET_ITERS"])
config_path = os.path.join(logdir, "config.yml")

if os.path.isfile(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
else:
    with open(src) as f:
        cfg = yaml.safe_load(f)

cfg.setdefault("TRAIN", {})
cfg["TRAIN"]["MAX_ITERS"] = target
with open(config_path, "w") as f:
    yaml.safe_dump(cfg, f, default_flow_style=False)
print("LOGDIR config TRAIN.MAX_ITERS -> %d" % target)
PY
  python -m tcc.train --alsologtostderr --logdir="$LOGDIR" --force_train
}

run_alignment() {
  local iter="$1"
  local out_mp4="$ALIGN_DIR/iter_${iter}_val_seed${ALIGN_SEED}.mp4"
  echo ""
  echo "=== Alignment at iter $iter -> $out_mp4 ==="
  python tcc/scripts/align_random_pair.py \
    --video_dir "$VIDEO_DIR" \
    --logdir "$LOGDIR" \
    --output "$out_mp4" \
    --seed "$ALIGN_SEED" \
    --from_val \
    --prepare_seed "$PREPARE_SEED" \
    --val_fraction "$VAL_FRACTION" \
    --max_videos "$MAX_VIDEOS" \
    --path_to_tfrecords "$PATH_TO_TFRECORDS_PATTERN" \
    --pair_tfrecord_dir "${PAIR_TFRECORD_ROOT}/tennis_forehand_rear_pair_tfrecords"
}

echo "=== Forehand milestone pipeline ===" | tee -a "$PIPELINE_LOG"
echo "OUTPUT_ROOT=$OUTPUT_ROOT" | tee -a "$PIPELINE_LOG"
echo "LOGDIR=$LOGDIR" | tee -a "$PIPELINE_LOG"
echo "MILESTONES=$MILESTONES" | tee -a "$PIPELINE_LOG"

for target in $MILESTONES; do
  train_until "$target"
  run_alignment "$target"
done

echo "" | tee -a "$PIPELINE_LOG"
echo "=== All milestones complete ===" | tee -a "$PIPELINE_LOG"
echo "Alignment videos:" | tee -a "$PIPELINE_LOG"
ls -lh "$ALIGN_DIR"/iter_*_val_seed"${ALIGN_SEED}".mp4 | tee -a "$PIPELINE_LOG"
echo "Checkpoints: $LOGDIR" | tee -a "$PIPELINE_LOG"
echo "Compare MP4s and pick the smallest iter with good phase lock." | tee -a "$PIPELINE_LOG"

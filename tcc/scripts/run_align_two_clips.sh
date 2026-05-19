#!/usr/bin/env bash
# End-to-end: TFRecords -> TCC train -> side-by-side aligned MP4 for two random forehands.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
source tcc-env/bin/activate
export TF_CPP_MIN_LOG_LEVEL=1

VIDEO_DIR="${VIDEO_DIR:-$ROOT/tcc/Forehands/Rear View}"
LOGDIR="${LOGDIR:-/tmp/tennis_forehand_rear_logs}"
OUTPUT="${OUTPUT:-/tmp/tennis_forehand_aligned.mp4}"
SEED="${SEED:-$(date +%s)}"
MAX_VIDEOS="${MAX_VIDEOS:-0}"   # 0 = all clips in folder
SKIP_PREPARE="${SKIP_PREPARE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
RESUME_TRAIN="${RESUME_TRAIN:-0}"
FROM_VAL="${FROM_VAL:-0}"
PREPARE_SEED="${PREPARE_SEED:-42}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
TCC_FOREHAND_CONFIG="${TCC_FOREHAND_CONFIG:-$ROOT/tcc/configs/tennis_forehand_rear.yml}"

RESNET="/tmp/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5"
if [[ ! -f "$RESNET" ]]; then
  echo "Downloading ResNet50v2 weights..."
  wget -q -P /tmp/ https://github.com/keras-team/keras-applications/releases/download/resnet/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5
fi

python - <<'PY' || true
import tensorflow as tf
for g in tf.config.list_physical_devices('GPU'):
  tf.config.experimental.set_memory_growth(g, True)
print('GPUs:', tf.config.list_physical_devices('GPU'))
PY

if [[ "$SKIP_PREPARE" != "1" ]]; then
  echo "=== Building TFRecords from $VIDEO_DIR ==="
  PREP=(python tcc/scripts/prepare_forehand_tfrecords.py --video_dir "$VIDEO_DIR")
  if [[ "$MAX_VIDEOS" != "0" ]]; then
    PREP+=(--max_videos "$MAX_VIDEOS")
  fi
  "${PREP[@]}"
fi

if [[ "$SKIP_TRAIN" != "1" ]]; then
  echo "=== Training TCC (GPU) ==="
  mkdir -p "$LOGDIR"
  if [[ "$RESUME_TRAIN" == "1" ]]; then
    echo "Resuming from latest checkpoint in $LOGDIR"
    export LOGDIR TCC_FOREHAND_CONFIG
    python - <<PY
import os
import sys
import yaml

logdir = os.environ["LOGDIR"]
src = os.environ["TCC_FOREHAND_CONFIG"]
config_path = os.path.join(logdir, "config.yml")
if not os.path.isfile(config_path):
    sys.exit("No config.yml in logdir; cannot resume.")
with open(src) as f:
    src_cfg = yaml.safe_load(f)
with open(config_path) as f:
    cfg = yaml.safe_load(f)
cfg.setdefault("TRAIN", {})
cfg["TRAIN"]["MAX_ITERS"] = src_cfg["TRAIN"]["MAX_ITERS"]
with open(config_path, "w") as f:
    yaml.safe_dump(cfg, f, default_flow_style=False)
print("Updated %s TRAIN.MAX_ITERS -> %d" % (
    config_path, cfg["TRAIN"]["MAX_ITERS"]))
PY
  else
    echo "Config: $TCC_FOREHAND_CONFIG"
    cp "$TCC_FOREHAND_CONFIG" "$LOGDIR/config.yml"
  fi
  python -m tcc.train --alsologtostderr --logdir="$LOGDIR" --force_train
fi

echo "=== Aligning two random clips -> $OUTPUT (seed=$SEED) ==="
ALIGN_ARGS=(
  --video_dir "$VIDEO_DIR"
  --logdir "$LOGDIR"
  --output "$OUTPUT"
  --seed "$SEED"
  --prepare_seed "$PREPARE_SEED"
  --val_fraction "$VAL_FRACTION"
  --max_videos "$MAX_VIDEOS"
)
if [[ "$FROM_VAL" == "1" ]]; then
  ALIGN_ARGS+=(--from_val)
  echo "Sampling pair from validation split only."
fi
python tcc/scripts/align_random_pair.py "${ALIGN_ARGS[@]}"

echo ""
echo "Done: $OUTPUT"
echo "Metadata: ${OUTPUT%.*}_pair.json"

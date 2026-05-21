#!/usr/bin/env bash
# Train TCC on Forehands/Rear View (TFRecords in /tmp; checkpoints on persistent disk).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
source tcc-env/bin/activate

export TF_CPP_MIN_LOG_LEVEL=1
# Use GPU memory growth to avoid grabbing all VRAM on T4.
python - <<'PY' || true
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
print('GPUs:', gpus)
PY

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ec2-user/tennis/outputs}"
LOGDIR="${LOGDIR:-$OUTPUT_ROOT/logs/tennis_forehand_rear}"
TFRECORD_DIR="${TFRECORD_DIR:-/tmp/tennis_forehand_rear_tfrecords}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$OUTPUT_ROOT/weights}"
CONFIG_SRC="${TCC_FOREHAND_CONFIG:-$(dirname "$0")/../configs/tennis_forehand_rear_persistent.yml}"
RESNET="$WEIGHTS_DIR/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5"

mkdir -p "$OUTPUT_ROOT" "$LOGDIR" "$WEIGHTS_DIR"

if [[ ! -f "$RESNET" ]]; then
  echo "Downloading ResNet50v2 weights to $WEIGHTS_DIR ..."
  wget -q -P "$WEIGHTS_DIR" \
    https://github.com/keras-team/keras-applications/releases/download/resnet/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5
fi

if ! compgen -G "$TFRECORD_DIR/tennis_forehand_rear_train-"'*.tfrecord' > /dev/null; then
  echo "TFRecords not found in $TFRECORD_DIR" >&2
  echo "Run: python tcc/scripts/prepare_forehand_tfrecords.py  # syncs videos to /tmp and builds TFRecords" >&2
  exit 1
fi

FOREHAND_VIDEO_DIR="${FOREHAND_VIDEO_DIR:-/tmp/Forehands/Rear View}"
echo "Forehand videos (if needed for prep/align): $FOREHAND_VIDEO_DIR"

mkdir -p "$LOGDIR"
cp "$CONFIG_SRC" "$LOGDIR/config.yml"
echo "Installed config: $LOGDIR/config.yml"
echo "LOGDIR=$LOGDIR"
echo "TFRECORD_DIR=$TFRECORD_DIR"

EXTRA_TRAIN_FLAGS="${EXTRA_TRAIN_FLAGS:-}"
python -m tcc.train --alsologtostderr --logdir="$LOGDIR" $EXTRA_TRAIN_FLAGS

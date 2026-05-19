#!/usr/bin/env bash
# Train TCC on Forehands/Rear View TFRecords.
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

LOGDIR="${LOGDIR:-/tmp/tennis_forehand_rear_logs}"
CONFIG_SRC="${TCC_FOREHAND_CONFIG:-$(dirname "$0")/../configs/tennis_forehand_rear.yml}"
RESNET="/tmp/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5"

if [[ ! -f "$RESNET" ]]; then
  echo "Downloading ResNet50v2 weights to /tmp/ ..."
  wget -q -P /tmp/ \
    https://github.com/keras-team/keras-applications/releases/download/resnet/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5
fi

if [[ ! -d "/tmp/tennis_forehand_rear_tfrecords" ]]; then
  echo "TFRecords not found. Run prepare_forehand_tfrecords.py first."
  exit 1
fi

mkdir -p "$LOGDIR"
cp "$CONFIG_SRC" "$LOGDIR/config.yml"
echo "Installed config: $LOGDIR/config.yml"

EXTRA_TRAIN_FLAGS="${EXTRA_TRAIN_FLAGS:-}"
python -m tcc.train --alsologtostderr --logdir="$LOGDIR" $EXTRA_TRAIN_FLAGS

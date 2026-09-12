#!/usr/bin/env bash
# Train TCC on tennis_forehand_rear.
# TFRecords + ImageNet weights: /tmp  Checkpoints: $TCC_OUTPUT_ROOT
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

OUTPUT_ROOT="${TCC_OUTPUT_ROOT:-/home/ec2-user/tennis/outputs}"
LOGDIR="$OUTPUT_ROOT/logs/tennis_forehand_rear"
CONFIG_SRC="${TCC_FOREHAND_CONFIG:-$ROOT/tcc/configs/tennis_forehand_rear.yml}"
RESNET="/tmp/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5"
TFRECORD_DIR="/tmp/tennis_forehand_rear_tfrecords"

mkdir -p "$LOGDIR"

# Same as tcc/run.sh: Keras Applications ResNet50v2 (no top) into /tmp.
if [[ ! -f "$RESNET" ]]; then
  wget -P /tmp/ \
    https://github.com/keras-team/keras-applications/releases/download/resnet/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5
fi

if ! compgen -G "$TFRECORD_DIR/tennis_forehand_rear_train-"'*.tfrecord' > /dev/null; then
  echo "TFRecords not found in $TFRECORD_DIR" >&2
  echo "Run: python -m tcc.dataset_preparation.videos_to_tfrecords --input_dir \"/tmp/Forehands/Rear View\" --name tennis_forehand_rear --val_fraction 0.1 --splits_json tcc/data/tennis_forehand_rear_splits.json --action_label -1" >&2
  exit 1
fi

cp "$CONFIG_SRC" "$LOGDIR/config.yml"
python - "$LOGDIR/config.yml" "$LOGDIR" <<'PY'
import sys
path, logdir = sys.argv[1:3]
text = open(path).read()
out = []
for line in text.splitlines(True):
  if line.startswith('LOGDIR:'):
    line = 'LOGDIR: %s\n' % logdir
  out.append(line)
open(path, 'w').writelines(out)
PY

echo "TCC_OUTPUT_ROOT=$OUTPUT_ROOT"
echo "checkpoints=$LOGDIR"

EXTRA_TRAIN_FLAGS="${EXTRA_TRAIN_FLAGS:-}"
python -m tcc.train --alsologtostderr --logdir="$LOGDIR" $EXTRA_TRAIN_FLAGS

#!/usr/bin/env bash
# Train TCC on tennis_forehand_rear.
# TFRecords, ImageNet weights, and normal training output: /tmp.
# One last and one best checkpoint are mirrored to $TCC_OUTPUT_ROOT.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

source tcc-env/bin/activate
OUTPUT_ROOT="${TCC_OUTPUT_ROOT:-/home/ec2-user/tennis/outputs}"
LOGDIR="/tmp/alignment_logs"
PERSISTENT_DIR="$OUTPUT_ROOT/logs/tennis_forehand_rear"
WANDB_ENTITY="${WANDB_ENTITY:-edthomas-2002-private}"
WANDB_PROJECT="${WANDB_PROJECT:-tennis-forehand-tcc}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-forehand-tcc}"
CONFIG_SRC="${TCC_FOREHAND_CONFIG:-$ROOT/tcc/configs/tennis_forehand_rear.yml}"
RESNET="/tmp/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5"
TFRECORD_DIR="/tmp/tennis_forehand_rear_tfrecords"

mkdir -p "$LOGDIR" "$PERSISTENT_DIR/last" "$PERSISTENT_DIR/best"

# Same as tcc/run.sh: Keras Applications ResNet50v2 (no top) into /tmp.
if [[ ! -f "$RESNET" ]]; then
  wget -P /tmp/ \
    https://github.com/keras-team/keras-applications/releases/download/resnet/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5
fi

if ! compgen -G "$TFRECORD_DIR/tennis_forehand_rear_train-"'*.tfrecord' > /dev/null; then
  echo "TFRecords not found in $TFRECORD_DIR" >&2
  echo "Run: bash tcc/scripts/prepare_forehand_tfrecords.sh" >&2
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
cp "$LOGDIR/config.yml" "$PERSISTENT_DIR/config.yml"
cp "$LOGDIR/config.yml" "$PERSISTENT_DIR/last/config.yml"
cp "$LOGDIR/config.yml" "$PERSISTENT_DIR/best/config.yml"

echo "TCC_OUTPUT_ROOT=$OUTPUT_ROOT"
echo "local checkpoints=$LOGDIR"
echo "persistent last/best=$PERSISTENT_DIR"

EXTRA_TRAIN_FLAGS="${EXTRA_TRAIN_FLAGS:-}"
python -m tcc.train \
  --alsologtostderr \
  --logdir="$LOGDIR" \
  --persistent_checkpoint_dir="$PERSISTENT_DIR" \
  --wandb_entity="$WANDB_ENTITY" \
  --wandb_project="$WANDB_PROJECT" \
  --wandb_run_name="$WANDB_RUN_NAME" \
  $EXTRA_TRAIN_FLAGS

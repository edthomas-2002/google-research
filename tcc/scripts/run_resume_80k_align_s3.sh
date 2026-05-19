#!/usr/bin/env bash
# Resume 40k checkpoint -> train to 80k, val alignment MP4, S3 upload, shutdown on exit.
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
PIPELINE_LOG="${PIPELINE_LOG:-$OUTPUT_ROOT/resume_80k.log}"
TCC_FOREHAND_CONFIG="${TCC_FOREHAND_CONFIG:-$ROOT/tcc/configs/tennis_forehand_rear_persistent.yml}"
VIDEO_DIR="${VIDEO_DIR:-$ROOT/tcc/Forehands/Rear View}"
TARGET_ITERS="${TARGET_ITERS:-80000}"
ALIGN_SEED="${ALIGN_SEED:-42}"
PREPARE_SEED="${PREPARE_SEED:-42}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
S3_BUCKET_PREFIX="${S3_BUCKET_PREFIX:-s3://tennis-models/demo_videos/swing_alignment}"
OUTPUT_BASENAME="${OUTPUT_BASENAME:-iter_80000_val_seed${ALIGN_SEED}.mp4}"
SCHEDULE_SHUTDOWN="${SCHEDULE_SHUTDOWN:-1}"

shutdown_when_done() {
  local code=$?
  if [[ "$SCHEDULE_SHUTDOWN" == "1" ]]; then
    echo ""
    echo "Pipeline finished (exit $code). Scheduling shutdown in 1 minute..."
    sudo shutdown -h +1 || true
  fi
}
trap shutdown_when_done EXIT

mkdir -p "$OUTPUT_ROOT" "$LOGDIR" "$ALIGN_DIR" "$WEIGHTS_DIR" "$PAIR_TFRECORD_ROOT"

RESNET="$WEIGHTS_DIR/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5"
if [[ ! -f "$RESNET" ]]; then
  echo "Downloading ResNet50v2 weights..."
  wget -q -P "$WEIGHTS_DIR" \
    https://github.com/keras-team/keras-applications/releases/download/resnet/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5
fi

if ! compgen -G "$TFRECORD_DIR/tennis_forehand_rear_train-"'*.tfrecord' > /dev/null; then
  echo "ERROR: Missing TFRecords in $TFRECORD_DIR" >&2
  exit 1
fi

if [[ ! -f "$LOGDIR/checkpoint" ]]; then
  echo "ERROR: No checkpoint in $LOGDIR — cannot resume." >&2
  exit 1
fi

echo "=== Resume training to $TARGET_ITERS ===" | tee "$PIPELINE_LOG"
echo "LOGDIR=$LOGDIR" | tee -a "$PIPELINE_LOG"
cat "$LOGDIR/checkpoint" | tee -a "$PIPELINE_LOG"

export LOGDIR TCC_FOREHAND_CONFIG TARGET_ITERS
python - <<'PY'
import os
import yaml

logdir = os.environ["LOGDIR"]
src = os.environ["TCC_FOREHAND_CONFIG"]
target = int(os.environ["TARGET_ITERS"])
config_path = os.path.join(logdir, "config.yml")

with open(src) as f:
    src_cfg = yaml.safe_load(f)
if os.path.isfile(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
else:
    cfg = src_cfg.copy()

cfg.setdefault("TRAIN", {})
cfg["TRAIN"]["MAX_ITERS"] = target
# Keep persistent paths in logdir config.
for key in ("LOGDIR", "PATH_TO_TFRECORDS", "MODEL"):
    if key in src_cfg:
        cfg[key] = src_cfg[key]
if "MODEL" in src_cfg and "RESNET_PRETRAINED_WEIGHTS" in src_cfg.get("MODEL", {}):
    cfg.setdefault("MODEL", {})["RESNET_PRETRAINED_WEIGHTS"] = src_cfg["MODEL"]["RESNET_PRETRAINED_WEIGHTS"]

with open(config_path, "w") as f:
    yaml.safe_dump(cfg, f, default_flow_style=False)
print("Updated %s TRAIN.MAX_ITERS -> %d" % (config_path, target))
PY

python -m tcc.train --alsologtostderr --logdir="$LOGDIR" --force_train 2>&1 | tee -a "$PIPELINE_LOG"

OUT_MP4="$ALIGN_DIR/$OUTPUT_BASENAME"
echo "" | tee -a "$PIPELINE_LOG"
echo "=== Alignment -> $OUT_MP4 ===" | tee -a "$PIPELINE_LOG"
python tcc/scripts/align_random_pair.py \
  --video_dir "$VIDEO_DIR" \
  --logdir "$LOGDIR" \
  --output "$OUT_MP4" \
  --seed "$ALIGN_SEED" \
  --from_val \
  --prepare_seed "$PREPARE_SEED" \
  --val_fraction "$VAL_FRACTION" \
  --path_to_tfrecords "${PAIR_TFRECORD_ROOT}/%s_tfrecords/" \
  --pair_tfrecord_dir "${PAIR_TFRECORD_ROOT}/tennis_forehand_rear_pair_tfrecords" \
  2>&1 | tee -a "$PIPELINE_LOG"

S3_URI="${S3_BUCKET_PREFIX}/${OUTPUT_BASENAME}"
echo "" | tee -a "$PIPELINE_LOG"
echo "=== Uploading to $S3_URI ===" | tee -a "$PIPELINE_LOG"
aws s3 cp "$OUT_MP4" "$S3_URI" 2>&1 | tee -a "$PIPELINE_LOG"

PAIR_JSON="${OUT_MP4%.mp4}_pair.json"
if [[ -f "$PAIR_JSON" ]]; then
  aws s3 cp "$PAIR_JSON" "${S3_BUCKET_PREFIX}/${OUTPUT_BASENAME%.mp4}_pair.json" \
    2>&1 | tee -a "$PIPELINE_LOG" || true
fi

echo "" | tee -a "$PIPELINE_LOG"
echo "=== Done ===" | tee -a "$PIPELINE_LOG"
echo "  Local MP4: $OUT_MP4" | tee -a "$PIPELINE_LOG"
echo "  S3:        $S3_URI" | tee -a "$PIPELINE_LOG"
ls -lh "$OUT_MP4" | tee -a "$PIPELINE_LOG"

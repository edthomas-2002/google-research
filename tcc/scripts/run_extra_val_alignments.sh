#!/usr/bin/env bash
# Generate N extra val-pair alignment MP4s from an existing checkpoint (default: 40k model).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
source tcc-env/bin/activate
export TF_CPP_MIN_LOG_LEVEL=1

LOGDIR="${LOGDIR:-/home/ec2-user/tennis/outputs/logs/tennis_forehand_rear}"
ALIGN_DIR="${ALIGN_DIR:-/home/ec2-user/tennis/outputs/alignments}"
VIDEO_DIR="${VIDEO_DIR:-$ROOT/tcc/Forehands/Rear View}"
PAIR_TFRECORD_ROOT="${PAIR_TFRECORD_ROOT:-/home/ec2-user/tennis/outputs/pair_tfrecords}"
PATH_TO_TFRECORDS="${PAIR_TFRECORD_ROOT}/%s_tfrecords/"
PAIR_TFRECORD_DIR="${PAIR_TFRECORD_ROOT}/tennis_forehand_rear_pair_tfrecords"
# Space-separated seeds (avoid 42 if that was the milestone demo seed).
SEEDS="${SEEDS:-100 101 102 103 104 105 106 107 108 109}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-iter_40000_extra_val}"

mkdir -p "$ALIGN_DIR"
MANIFEST="$ALIGN_DIR/${OUTPUT_PREFIX}_manifest.json"

for seed in $SEEDS; do
  out="$ALIGN_DIR/${OUTPUT_PREFIX}_seed${seed}.mp4"
  if [[ -f "$out" ]]; then
    echo "Skip seed=$seed (exists): $out"
    continue
  fi
  echo ""
  echo "=== Aligning val pair seed=$seed -> $out ==="
  python tcc/scripts/align_random_pair.py \
    --video_dir "$VIDEO_DIR" \
    --logdir "$LOGDIR" \
    --output "$out" \
    --seed "$seed" \
    --from_val \
    --prepare_seed 42 \
    --path_to_tfrecords "$PATH_TO_TFRECORDS" \
    --pair_tfrecord_dir "$PAIR_TFRECORD_DIR"
done

export ALIGN_DIR OUTPUT_PREFIX
python - <<'PY'
import glob
import json
import os

align_dir = os.environ["ALIGN_DIR"]
prefix = os.environ["OUTPUT_PREFIX"]
paths = sorted(glob.glob(os.path.join(align_dir, prefix + "_seed*_pair.json")))
entries = []
for p in paths:
    with open(p) as f:
        entries.append(json.load(f))
manifest = os.path.join(align_dir, prefix + "_manifest.json")
with open(manifest, "w") as f:
    json.dump(entries, f, indent=2)
print("Wrote manifest:", manifest, "(%d pairs)" % len(entries))
PY

echo ""
echo "Done."
ls -lh "$ALIGN_DIR"/${OUTPUT_PREFIX}_seed*.mp4

#!/usr/bin/env bash
# Smoke-test TCC on the published pouring dataset (validates env + GPU).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
source tcc-env/bin/activate

export TF_CPP_MIN_LOG_LEVEL=1

RESNET="/tmp/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5"
if [[ ! -f "$RESNET" ]]; then
  wget -q -P /tmp/ https://github.com/keras-team/keras-applications/releases/download/resnet/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5
fi

if ! bash tcc/dataset_preparation/download_pouring_data.sh; then
  echo "WARNING: Pouring TFRecords download failed (bucket often returns 403)."
  echo "Skipping pouring demo. Proceed with forehand scripts instead."
  exit 0
fi

rm -rf /tmp/alignment_logs
mkdir -p /tmp/alignment_logs
cp tcc/configs/demo.yml /tmp/alignment_logs/config.yml

python -m tcc.train --alsologtostderr --logdir=/tmp/alignment_logs --debug --force_train
python -m tcc.extract_embeddings --alsologtostderr \
  --logdir=/tmp/alignment_logs \
  --dataset=pouring \
  --split=val \
  --max_embs=4 \
  --keep_data \
  --keep_labels=false \
  --frames_per_batch=50 \
  --sample_all_stride=1 \
  --save_path=/tmp/pouring_embeddings.npy
python -m tcc.visualize_alignment --alsologtostderr \
  --video_path=/tmp/pouring_aligned.gif \
  --embs_path=/tmp/pouring_embeddings.npy \
  --reference_video=0 \
  --candidate_video=1

echo "Pouring demo done: /tmp/pouring_aligned.gif"

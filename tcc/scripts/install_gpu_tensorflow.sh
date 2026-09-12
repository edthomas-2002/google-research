#!/usr/bin/env bash
# Install TensorFlow with bundled CUDA/cuDNN (T4 / CUDA 12 compatible).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
source tcc-env/bin/activate
pip install --upgrade pip
pip install "tensorflow[and-cuda]>=2.15,<2.16"
python -c "import tensorflow as tf; print('TF', tf.__version__); print('GPU', tf.config.list_physical_devices('GPU'))"

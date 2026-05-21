# Forehand rear-view TCC (GPU)

Data: `tcc/Forehands/Rear View/` (~1257 clips)

Artifacts default under **`/home/ec2-user/tennis/outputs/`** (override with `OUTPUT_ROOT` for prepare/train; `TCC_OUTPUT_ROOT` for alignment — set both to the same path if you relocate).

## Scripts in `tcc/scripts/`

| Script | Purpose |
|--------|---------|
| `prepare_forehand_tfrecords.py` | Build train/val TFRecords + split counts JSON |
| `train_forehand.sh` | Train TCC on GPU (checkpoints on persistent disk) |
| `align_random_pair.py` | Embed two clips and write side-by-side alignment MP4 |
| `install_gpu_tensorflow.sh` | Reinstall TensorFlow with CUDA if GPU not detected |

## One-time setup

```bash
cd /home/ec2-user/tennis/google-research
python3 -m venv tcc-env && source tcc-env/bin/activate
pip install -r tcc/requirements.txt
bash tcc/scripts/install_gpu_tensorflow.sh   # if GPU not detected

mkdir -p /home/ec2-user/tennis/outputs/weights
wget -P /home/ec2-user/tennis/outputs/weights/ \
  https://github.com/keras-team/keras-applications/releases/download/resnet/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5
```

Verify GPU:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## 1. Build full forehand TFRecords (~1–3 hours)

```bash
python tcc/scripts/prepare_forehand_tfrecords.py
# Writes $OUTPUT_ROOT/tfrecords/tennis_forehand_rear_tfrecords/
#       and tcc/data/tennis_forehand_rear_splits.json
```

## 2. Train TCC on GPU

```bash
bash tcc/scripts/train_forehand.sh
# Config: tcc/configs/tennis_forehand_rear_persistent.yml
#         -> $OUTPUT_ROOT/logs/tennis_forehand_rear/config.yml
# Checkpoints: $OUTPUT_ROOT/logs/tennis_forehand_rear/
```

Resume:

```bash
EXTRA_TRAIN_FLAGS="--force_train" bash tcc/scripts/train_forehand.sh
```

Tune `TRAIN.MAX_ITERS` in `tcc/configs/tennis_forehand_rear_persistent.yml` (paper uses 150000).

## 3. Align two clips

```bash
export TCC_OUTPUT_ROOT="${TCC_OUTPUT_ROOT:-$OUTPUT_ROOT}"
python tcc/scripts/align_random_pair.py --seed 42
# Output: $OUTPUT_ROOT/alignments/tennis_forehand_aligned.mp4
#         $OUTPUT_ROOT/alignments/tennis_forehand_aligned_pair.json
```

Specific clips:

```bash
python tcc/scripts/align_random_pair.py \
  --clip_a "tcc/Forehands/Rear View/NADAL_FH (10).mp4" \
  --clip_b "tcc/Forehands/Rear View/FRITZ_FH (9).mp4" \
  --output "$OUTPUT_ROOT/alignments/my_alignment.mp4"
```

Validation-only pairs (same split as TFRecord prep):

```bash
python tcc/scripts/align_random_pair.py --seed 42 --from_val
```

## Full pipeline (manual)

```bash
python tcc/scripts/prepare_forehand_tfrecords.py
bash tcc/scripts/train_forehand.sh
python tcc/scripts/align_random_pair.py --seed 42 --from_val
```

Skip prepare if TFRecords already exist; resume training with `EXTRA_TRAIN_FLAGS="--force_train"`.

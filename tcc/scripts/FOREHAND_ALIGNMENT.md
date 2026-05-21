# Forehand rear-view TCC (GPU)

Forehand clips (~1257) are synced from **`s3://tennis-swing-data/Forehands/`** into **`/tmp/Forehands/Rear View/`** (ephemeral; re-sync after instance restart).

**TFRecords** and **pair TFRecords** also live in **`/tmp/`**. **Checkpoints**, **weights**, and **alignment MP4s** stay under **`/home/ec2-user/tennis/outputs/`** (`OUTPUT_ROOT` / `TCC_OUTPUT_ROOT`).

Override paths with `FOREHAND_S3_URI`, `FOREHAND_VIDEO_DIR`, `TFRECORD_DIR`.

## Scripts in `tcc/scripts/`

| Script | Purpose |
|--------|---------|
| `prepare_forehand_tfrecords.py` | S3 sync (if needed) → TFRecords + split JSON |
| `train_forehand.sh` | Train TCC on GPU (checkpoints on persistent disk) |
| `align_random_pair.py` | S3 sync (if needed) → embed two clips → alignment MP4 |
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

Downloads clips to `/tmp` when missing, then builds TFRecords:

```bash
python tcc/scripts/prepare_forehand_tfrecords.py
# Videos:  /tmp/Forehands/Rear View/
# TFRecords: /tmp/tennis_forehand_rear_tfrecords/
# Splits:  tcc/data/tennis_forehand_rear_splits.json
```

Manual sync only:

```bash
aws s3 sync s3://tennis-swing-data/Forehands/ /tmp/Forehands/
python tcc/scripts/prepare_forehand_tfrecords.py --skip_download
```

## 2. Train TCC on GPU

```bash
bash tcc/scripts/train_forehand.sh
# Config: tcc/configs/tennis_forehand_rear_persistent.yml
#         -> $OUTPUT_ROOT/logs/tennis_forehand_rear/config.yml
# Checkpoints: $OUTPUT_ROOT/logs/tennis_forehand_rear/last.* and best.*
```

Resume:

```bash
EXTRA_TRAIN_FLAGS="--force_train" bash tcc/scripts/train_forehand.sh
```

## 3. Align two clips

```bash
export TCC_OUTPUT_ROOT="${TCC_OUTPUT_ROOT:-/home/ec2-user/tennis/outputs}"
python tcc/scripts/align_random_pair.py --seed 42
```

Specific clips (under `/tmp` after sync):

```bash
python tcc/scripts/align_random_pair.py \
  --clip_a "/tmp/Forehands/Rear View/NADAL_FH (10).mp4" \
  --clip_b "/tmp/Forehands/Rear View/FRITZ_FH (9).mp4" \
  --output "$OUTPUT_ROOT/alignments/my_alignment.mp4"
```

Validation-only pairs:

```bash
python tcc/scripts/align_random_pair.py --seed 42 --from_val
```

## Full pipeline (manual)

```bash
python tcc/scripts/prepare_forehand_tfrecords.py
bash tcc/scripts/train_forehand.sh
python tcc/scripts/align_random_pair.py --seed 42 --from_val
```

After a restart: re-run `prepare_forehand_tfrecords.py` (re-syncs `/tmp` and rebuilds TFRecords).

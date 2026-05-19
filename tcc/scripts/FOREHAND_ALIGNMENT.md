# Forehand rear-view TCC (GPU)

Data: `tcc/Forehands/Rear View/` (~1257 clips)

## One-time setup

```bash
cd /home/ec2-user/tennis/google-research
python3 -m venv tcc-env && source tcc-env/bin/activate
pip install -r tcc/requirements.txt
bash tcc/scripts/install_gpu_tensorflow.sh   # if GPU not detected

wget -P /tmp/ https://github.com/keras-team/keras-applications/releases/download/resnet/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5
```

Verify GPU:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## 1. Pouring demo (optional)

```bash
bash tcc/scripts/run_pouring_demo.sh
# Output: /tmp/pouring_aligned.gif (if download succeeds; bucket may 403)
```

## 2. Build full forehand TFRecords (~1–3 hours)

```bash
python tcc/scripts/prepare_forehand_tfrecords.py
# Writes /tmp/tennis_forehand_rear_tfrecords/ and tcc/data/tennis_forehand_rear_splits.json
```

Monitor: `tail -f /tmp/prepare_forehand.log`

## 3. Train TCC on GPU (~hours; 20k iters default)

```bash
bash tcc/scripts/train_forehand.sh
# Config: tcc/configs/tennis_forehand_rear.yml → /tmp/tennis_forehand_rear_logs/config.yml
# Checkpoints: /tmp/tennis_forehand_rear_logs/
```

Resume:

```bash
EXTRA_TRAIN_FLAGS="--force_train" bash tcc/scripts/train_forehand.sh
```

Tune `TRAIN.MAX_ITERS` in config (paper uses 150000).

## 4. Align two random clips

```bash
python tcc/scripts/align_random_pair.py --seed 42
# Output: /tmp/tennis_forehand_aligned.gif
#         /tmp/tennis_forehand_aligned_pair.json (which clips were used)
```

Specific clips:

```bash
python tcc/scripts/align_random_pair.py \
  --clip_a "tcc/Forehands/Rear View/NADAL_FH (10).mp4" \
  --clip_b "tcc/Forehands/Rear View/FRITZ_FH (9).mp4" \
  --output /tmp/my_alignment.gif
```

## Full pipeline

```bash
bash tcc/scripts/run_forehand_alignment.sh
```

Skip steps if already done:

```bash
SKIP_PREPARE=1 SKIP_TRAIN=1 bash tcc/scripts/run_forehand_alignment.sh
```

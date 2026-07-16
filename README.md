# Tennis TCC (Temporal Cycle Consistency)

Forehand rear-view alignment based on Google Research TCC. Full train/align docs: [`tcc/scripts/FOREHAND_ALIGNMENT.md`](tcc/scripts/FOREHAND_ALIGNMENT.md).

## Download the trained model (for inference)

The 80k-iter TCC checkpoint lives in **`s3://tennis-models/phase_alignment/forehand_rear/`**. Inference (`align_random_pair.py`, `extract_embeddings`) expects it under:

```
/home/ec2-user/tennis/outputs/logs/tennis_forehand_rear/
```

Download:

```bash
mkdir -p /home/ec2-user/tennis/outputs/logs/tennis_forehand_rear
aws s3 sync s3://tennis-models/phase_alignment/forehand_rear/ \
  /home/ec2-user/tennis/outputs/logs/tennis_forehand_rear/
```

That directory should contain at least:

| File | Role |
|------|------|
| `80k_iter.data-00000-of-00001` | Weights |
| `80k_iter.index` | TF index |
| `checkpoint` | Points TF at `80k_iter` |
| `config.yml` | Model/training config |

Override the root with `TCC_OUTPUT_ROOT` if needed (default: `/home/ec2-user/tennis/outputs`).

## Run alignment (inference)

```bash
cd /home/ec2-user/tennis/google-research
source tcc-env/bin/activate   # if using the project venv
export TCC_OUTPUT_ROOT="${TCC_OUTPUT_ROOT:-/home/ec2-user/tennis/outputs}"
python tcc/scripts/align_random_pair.py --seed 42
```

See [`tcc/scripts/FOREHAND_ALIGNMENT.md`](tcc/scripts/FOREHAND_ALIGNMENT.md) for setup, TFRecords, and training.

---

Upstream: [Google Research](https://research.google). Source under Apache 2.0; datasets under CC BY 4.0.

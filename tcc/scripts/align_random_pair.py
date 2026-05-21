#!/usr/bin/env python3
"""Pick two random forehand clips, embed them, and render an alignment video."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import random
import shutil
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
  sys.path.insert(0, _REPO_ROOT)

from tcc import forehand_data

DATASET_NAME = 'tennis_forehand_rear'
PAIR_DATASET_NAME = 'tennis_forehand_rear_pair'
DEFAULT_VIDEO_DIR = forehand_data.default_video_dir()
_OUTPUT_ROOT = os.environ.get(
    'TCC_OUTPUT_ROOT', '/home/ec2-user/tennis/outputs')
_DEFAULT_TMP = '/tmp'
DEFAULT_LOGDIR = os.path.join(_OUTPUT_ROOT, 'logs', 'tennis_forehand_rear')
PAIR_TFRECORD_DIR = os.path.join(
    _DEFAULT_TMP, 'tennis_forehand_rear_pair_tfrecords')
DEFAULT_OUTPUT = os.path.join(
    _OUTPUT_ROOT, 'alignments', 'tennis_forehand_aligned.mp4')
DEFAULT_PATH_TO_TFRECORDS = os.path.join(_DEFAULT_TMP, '%s_tfrecords/')


def list_videos(video_dir):
  return sorted(
      os.path.join(video_dir, f)
      for f in os.listdir(video_dir)
      if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')))


def val_videos(video_dir, prepare_seed, val_fraction, max_videos):
  """Match train/val split in prepare_forehand_tfrecords.py."""
  videos = list_videos(video_dir)
  rng = random.Random(prepare_seed)
  videos = list(videos)
  rng.shuffle(videos)
  if max_videos > 0:
    videos = videos[:max_videos]
  n_val = max(1, int(round(len(videos) * val_fraction)))
  if n_val >= len(videos):
    n_val = len(videos) - 1
  return videos[:n_val]


def symlink_pair(paths, staging_dir):
  if os.path.exists(staging_dir):
    shutil.rmtree(staging_dir)
  os.makedirs(staging_dir)
  for src in paths:
    os.symlink(os.path.abspath(src),
               os.path.join(staging_dir, os.path.basename(src)))


def run(cmd, cwd):
  print('\n>>>', ' '.join(cmd), flush=True)
  subprocess.check_call(cmd, cwd=cwd)


def main():
  root = os.path.dirname(os.path.dirname(__file__))
  repo_root = os.path.dirname(root)

  parser = argparse.ArgumentParser()
  parser.add_argument('--video_dir', default=DEFAULT_VIDEO_DIR)
  parser.add_argument(
      '--skip_download',
      action='store_true',
      help='Do not run aws s3 sync; fail if videos are missing.')
  parser.add_argument('--logdir', default=DEFAULT_LOGDIR)
  parser.add_argument('--output', default=DEFAULT_OUTPUT)
  parser.add_argument('--seed', type=int, default=None,
                      help='Random seed (default: random each run).')
  parser.add_argument('--clip_a', default=None, help='Optional path to first clip.')
  parser.add_argument('--clip_b', default=None, help='Optional path to second clip.')
  parser.add_argument('--use_dtw', action='store_true', default=True)
  parser.add_argument('--no_dtw', action='store_false', dest='use_dtw')
  parser.add_argument('--sample_stride', type=int, default=2,
                      help='Frame stride when embedding (1 = every frame).')
  parser.add_argument(
      '--from_val',
      action='store_true',
      help='Sample only from the validation split (same shuffle as TFRecord prep).')
  parser.add_argument(
      '--prepare_seed',
      type=int,
      default=42,
      help='Seed used for train/val split when --from_val is set.')
  parser.add_argument(
      '--val_fraction',
      type=float,
      default=0.1,
      help='Validation fraction when --from_val is set.')
  parser.add_argument(
      '--max_videos',
      type=int,
      default=0,
      help='Cap clips before split (0 = all); must match TFRecord prep.')
  parser.add_argument(
      '--path_to_tfrecords',
      default=DEFAULT_PATH_TO_TFRECORDS,
      help='TFRecord glob pattern for extract_embeddings (%%s = dataset name).')
  parser.add_argument(
      '--pair_tfrecord_dir',
      default=PAIR_TFRECORD_DIR,
      help='Directory for two-clip pair TFRecords.')
  args = parser.parse_args()

  if args.clip_a and args.clip_b:
    pair = [os.path.abspath(args.clip_a), os.path.abspath(args.clip_b)]
  else:
    video_dir = forehand_data.ensure_forehand_videos(
        args.video_dir, skip_download=args.skip_download)
    if args.from_val:
      videos = val_videos(
          video_dir, args.prepare_seed, args.val_fraction, args.max_videos)
      print('Validation pool: %d clips' % len(videos), flush=True)
    else:
      videos = list_videos(video_dir)
    if len(videos) < 2:
      raise SystemExit('Need at least 2 videos in %s' % video_dir)
    rng = random.Random(args.seed)
    pair = rng.sample(videos, 2)

  print('Alignment pair:', flush=True)
  for p in pair:
    print('  ', p, flush=True)

  pair_tfrecord_dir = os.path.abspath(args.pair_tfrecord_dir)
  os.makedirs(os.path.dirname(pair_tfrecord_dir), exist_ok=True)
  staging = pair_tfrecord_dir + '_staging'
  symlink_pair(pair, staging)

  run([
      sys.executable,
      '-m',
      'tcc.dataset_preparation.videos_to_tfrecords',
      '--input_dir',
      staging,
      '--name',
      '%s_val' % PAIR_DATASET_NAME,
      '--output_dir',
      pair_tfrecord_dir,
      '--file_pattern',
      '*.mp4',
      '--files_per_shard',
      '2',
      '--width',
      '224',
      '--height',
      '224',
      '--fps',
      '0',
      '--action_label',
      '-1',
  ], cwd=repo_root)
  shutil.rmtree(staging, ignore_errors=True)

  emb_path = os.path.splitext(args.output)[0] + '_embeddings.npy'
  os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

  run([
      sys.executable,
      '-m',
      'tcc.extract_embeddings',
      '--alsologtostderr',
      '--logdir',
      args.logdir,
      '--dataset',
      PAIR_DATASET_NAME,
      '--split',
      'val',
      '--path_to_tfrecords',
      args.path_to_tfrecords,
      '--save_path',
      emb_path,
      '--keep_data',
      '--keep_labels=false',
      '--max_embs',
      '2',
      '--sample_all_stride',
      str(args.sample_stride),
      '--frames_per_batch',
      '64',
  ], cwd=repo_root)

  viz_cmd = [
      sys.executable,
      '-m',
      'tcc.visualize_alignment',
      '--alsologtostderr',
      '--video_path',
      args.output,
      '--embs_path',
      emb_path,
      '--reference_video',
      '0',
      '--candidate_video',
      '1',
      '--interval',
      '50',
  ]
  if args.use_dtw:
    viz_cmd.append('--use_dtw')
  run(viz_cmd, cwd=repo_root)

  meta_path = os.path.splitext(args.output)[0] + '_pair.json'
  with open(meta_path, 'w') as f:
    json.dump({
        'clip_a': pair[0],
        'clip_b': pair[1],
        'embeddings': emb_path,
        'output': args.output,
        'seed': args.seed,
    }, f, indent=2)

  print('\nDone.', flush=True)
  print('  Alignment video:', args.output, flush=True)
  print('  Embeddings:     ', emb_path, flush=True)
  print('  Pair metadata:  ', meta_path, flush=True)


if __name__ == '__main__':
  main()

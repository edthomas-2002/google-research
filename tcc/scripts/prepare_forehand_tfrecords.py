#!/usr/bin/env python3
"""Build train/val TFRecords from forehand clips in /tmp (synced from S3)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import math
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
DEFAULT_VIDEO_DIR = forehand_data.default_video_dir()
DEFAULT_TFRECORD_DIR = os.environ.get(
    'TFRECORD_DIR', '/tmp/tennis_forehand_rear_tfrecords')
SPLITS_JSON = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'data',
    'tennis_forehand_rear_splits.json',
)


def list_videos(video_dir):
  names = sorted(
      f for f in os.listdir(video_dir)
      if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')))
  return [os.path.join(video_dir, n) for n in names]


def symlink_split(paths, staging_dir):
  if os.path.exists(staging_dir):
    shutil.rmtree(staging_dir)
  os.makedirs(staging_dir)
  for src in paths:
    dst = os.path.join(staging_dir, os.path.basename(src))
    os.symlink(os.path.abspath(src), dst)


def run_tfrecords(name, staging_dir, output_dir, files_per_shard, width, height,
                  fps):
  cmd = [
      sys.executable,
      '-m',
      'tcc.dataset_preparation.videos_to_tfrecords',
      '--input_dir',
      staging_dir,
      '--name',
      name,
      '--output_dir',
      output_dir,
      '--file_pattern',
      '*.mp4',
      '--files_per_shard',
      str(files_per_shard),
      '--width',
      str(width),
      '--height',
      str(height),
      '--action_label',
      '-1',
  ]
  if fps > 0:
    cmd.extend(['--fps', str(fps)])
  repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
  print('Running:', ' '.join(cmd), flush=True)
  subprocess.check_call(cmd, cwd=repo_root)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--video_dir',
      default=DEFAULT_VIDEO_DIR,
      help='Directory containing forehand MP4 clips (default: /tmp, from S3).')
  parser.add_argument(
      '--skip_download',
      action='store_true',
      help='Do not run aws s3 sync; fail if videos are missing.')
  parser.add_argument(
      '--output_dir',
      default=DEFAULT_TFRECORD_DIR,
      help='Where TFRecord shards are written (default: '
           '%s).' % DEFAULT_TFRECORD_DIR)
  parser.add_argument(
      '--val_fraction',
      type=float,
      default=0.1,
      help='Fraction of clips for validation.')
  parser.add_argument(
      '--seed',
      type=int,
      default=42,
      help='Shuffle seed for train/val split.')
  parser.add_argument(
      '--files_per_shard',
      type=int,
      default=50,
      help='Videos per TFRecord shard.')
  parser.add_argument(
      '--max_videos',
      type=int,
      default=0,
      help='If >0, only use this many clips (for quick tests).')
  parser.add_argument(
      '--width',
      type=int,
      default=224)
  parser.add_argument(
      '--height',
      type=int,
      default=224)
  parser.add_argument(
      '--fps',
      type=int,
      default=0,
      help='Target FPS when writing TFRecords (0 = native; use 0 if clips vary).')
  args = parser.parse_args()

  video_dir = forehand_data.ensure_forehand_videos(
      args.video_dir, skip_download=args.skip_download)

  videos = list_videos(video_dir)
  if not videos:
    raise SystemExit('No videos found in %s' % video_dir)

  random.seed(args.seed)
  random.shuffle(videos)
  if args.max_videos > 0:
    videos = videos[:args.max_videos]

  n_val = max(1, int(round(len(videos) * args.val_fraction)))
  if len(videos) < 2:
    raise SystemExit('Need at least 2 videos; found %d' % len(videos))
  if n_val >= len(videos):
    n_val = len(videos) - 1

  val_paths = videos[:n_val]
  train_paths = videos[n_val:]

  output_dir = os.path.abspath(args.output_dir)
  os.makedirs(output_dir, exist_ok=True)

  staging_root = output_dir + '_staging'
  train_stage = os.path.join(staging_root, 'train')
  val_stage = os.path.join(staging_root, 'val')

  print('Total videos: %d (train=%d, val=%d)' %
        (len(videos), len(train_paths), len(val_paths)), flush=True)

  symlink_split(train_paths, train_stage)
  symlink_split(val_paths, val_stage)

  run_tfrecords(
      '%s_train' % DATASET_NAME,
      train_stage,
      output_dir,
      args.files_per_shard,
      args.width,
      args.height,
      args.fps)
  run_tfrecords(
      '%s_val' % DATASET_NAME,
      val_stage,
      output_dir,
      args.files_per_shard,
      args.width,
      args.height,
      args.fps)

  shutil.rmtree(staging_root, ignore_errors=True)

  splits = {'train': len(train_paths), 'val': len(val_paths)}
  os.makedirs(os.path.dirname(SPLITS_JSON), exist_ok=True)
  with open(SPLITS_JSON, 'w') as f:
    json.dump(splits, f, indent=2)

  print('Wrote TFRecords to %s' % output_dir, flush=True)
  print('Wrote split counts to %s: %s' % (SPLITS_JSON, splits), flush=True)


if __name__ == '__main__':
  main()

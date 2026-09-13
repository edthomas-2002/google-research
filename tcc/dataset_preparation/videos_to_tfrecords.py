# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Convert list of videos to tfrecords based on SequenceExample."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import random

from absl import app
from absl import flags
from absl import logging

import cv2
import tensorflow.compat.v2 as tf

from tcc.dataset_preparation.dataset_utils import create_tfrecords

flags.DEFINE_string('input_dir', None, 'Path to videos.')
flags.DEFINE_string('name', None, 'Name of the dataset being created. This will'
                    'be used as a prefix.')
flags.DEFINE_string('file_pattern', '*.mp4', 'Pattern used to searh for files'
                    'in the given directory.')
flags.DEFINE_string('label_file', None, 'Provide a corresponding labels file'
                    'that stores per-frame or per-sequence labels. This info'
                    'will get stored.')
flags.DEFINE_string(
    'output_dir', None,
    'Output directory for TFRecords. Default: /tmp/{name}_tfrecords/ '
    '(CONFIG.PATH_TO_TFRECORDS).')
flags.DEFINE_integer('files_per_shard', 1, 'Number of videos to store in a'
                     'shard.')
flags.DEFINE_boolean('rotate', False, 'Rotate videos by 90 degrees before'
                     'creating tfrecords')
flags.DEFINE_boolean('resize', True, 'Resize videos to a given size.')
flags.DEFINE_integer('width', 224, 'Width of frames in the TFRecord.')
flags.DEFINE_integer('height', 224, 'Height of frames in the TFRecord.')
flags.DEFINE_list(
    'frame_labels', '', 'Comma separated list of descriptions '
    'for labels given on a per frame basis. For example: '
    'winding_up,early_cocking,acclerating,follow_through')
flags.DEFINE_integer('action_label', -1, 'Action label of all videos.')
flags.DEFINE_integer('expected_segments', -1, 'Expected number of segments.')
flags.DEFINE_integer('fps', 0, 'Frames per second of video. If 0, fps will be '
                     'read from metadata of video.')
flags.DEFINE_float(
    'val_fraction', 0.0,
    'If >0, shuffle videos and write {name}_train and {name}_val TFRecords.')
flags.DEFINE_integer('seed', 42, 'Shuffle seed used when val_fraction > 0.')
flags.DEFINE_string(
    'splits_json', None,
    'If set and val_fraction > 0, write {"train": N, "val": M} here.')
flags.DEFINE_boolean(
    'delete_videos', False,
    'If True, delete each source video after its TFRecord shard is written.')
flags.DEFINE_boolean(
    'drop_videos_below_fps', False,
    'If True, exclude videos whose source FPS is below --fps.')
FLAGS = flags.FLAGS


def _list_filenames(input_dir, file_pattern):
  file_glob = os.path.join(input_dir, file_pattern)
  return sorted(os.path.basename(x) for x in tf.io.gfile.glob(file_glob))


def _drop_videos_below_fps(input_dir, filenames, target_fps):
  """Returns filenames meeting target FPS and the number excluded."""
  kept = []
  dropped = 0
  for filename in filenames:
    path = os.path.join(input_dir, filename)
    cap = cv2.VideoCapture(path)
    source_fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0
    cap.release()
    rounded_source_fps = int(source_fps + 0.5)
    if source_fps > 0 and rounded_source_fps >= target_fps:
      kept.append(filename)
    else:
      dropped += 1
      logging.warning(
          'Dropping %s (source FPS: %.3f, rounded FPS: %d, target FPS: %d)',
          path, source_fps, rounded_source_fps, target_fps)
  return kept, dropped


def _write(name, output_dir, filenames):
  create_tfrecords(name, output_dir, FLAGS.input_dir, FLAGS.label_file,
                   FLAGS.file_pattern, FLAGS.files_per_shard,
                   FLAGS.action_label, FLAGS.frame_labels,
                   FLAGS.expected_segments, FLAGS.fps, FLAGS.rotate,
                   FLAGS.resize, FLAGS.width, FLAGS.height,
                   filenames=filenames,
                   delete_videos=FLAGS.delete_videos)


def main(_):
  if not FLAGS.name:
    raise app.UsageError('--name is required.')
  if not FLAGS.input_dir:
    raise app.UsageError('--input_dir is required.')

  output_dir = FLAGS.output_dir or ('/tmp/%s_tfrecords/' % FLAGS.name)
  filenames = _list_filenames(FLAGS.input_dir, FLAGS.file_pattern)
  if not filenames:
    raise ValueError('No files matching %s in %s' %
                     (FLAGS.file_pattern, FLAGS.input_dir))

  dropped = 0
  if FLAGS.drop_videos_below_fps:
    if FLAGS.fps <= 0:
      raise app.UsageError('--drop_videos_below_fps requires --fps > 0.')
    filenames, dropped = _drop_videos_below_fps(
        FLAGS.input_dir, filenames, FLAGS.fps)
    if not filenames:
      raise ValueError('All videos were below the target FPS.')
    logging.info('Keeping %d videos and dropping %d below %d FPS.',
                 len(filenames), dropped, FLAGS.fps)

  if FLAGS.val_fraction > 0:
    files = list(filenames)
    random.Random(FLAGS.seed).shuffle(files)
    n_val = max(1, int(round(len(files) * FLAGS.val_fraction)))
    if n_val >= len(files):
      n_val = len(files) - 1
    val_files = files[:n_val]
    train_files = files[n_val:]
    logging.info('Split %d videos: train=%d val=%d', len(files),
                 len(train_files), len(val_files))
    _write('%s_train' % FLAGS.name, output_dir, train_files)
    _write('%s_val' % FLAGS.name, output_dir, val_files)
    if FLAGS.splits_json:
      splits_dir = os.path.dirname(os.path.abspath(FLAGS.splits_json))
      if splits_dir:
        tf.io.gfile.makedirs(splits_dir)
      with tf.io.gfile.GFile(FLAGS.splits_json, 'w') as f:
        json.dump({
            'train': len(train_files),
            'val': len(val_files),
            'dropped': dropped,
        }, f)
      logging.info('Wrote split counts to %s', FLAGS.splits_json)
  else:
    _write(FLAGS.name, output_dir, filenames)


if __name__ == '__main__':
  app.run(main)

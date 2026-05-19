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

"""List of subsets."""

import json
import os

_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

DATASETS = {
    'pouring': {'train': 70, 'val': 14, 'test': 32},
    'baseball_pitch': {'train': 103, 'val': 63},
    'baseball_swing': {'train': 113, 'val': 57},
    'bench_press': {'train': 113, 'val': 57},
    'bowling': {'train': 134, 'val': 84},
    'clean_and_jerk': {'train': 40, 'val': 42},
    'golf_swing': {'train': 87, 'val': 77},
    'jumping_jacks': {'train': 56, 'val': 56},
    'pushups': {'train': 102, 'val': 105},
    'pullups': {'train': 98, 'val': 100},
    'situp': {'train': 50, 'val': 50},
    'squats': {'train': 114, 'val': 115},
    'tennis_forehand': {'train': 79, 'val': 74},
    'tennis_serve': {'train': 115, 'val': 68},
    # Counts for Forehands/Rear View are written by scripts/prepare_forehand_tfrecords.py
    'tennis_forehand_rear': {'train': 0, 'val': 0},
    # Two-clip alignment subset (see scripts/align_random_pair.py).
    'tennis_forehand_rear_pair': {'train': 0, 'val': 2},
}


def _load_dynamic_splits():
  splits_path = os.path.join(_DATA_DIR, 'tennis_forehand_rear_splits.json')
  if os.path.exists(splits_path):
    with open(splits_path, 'r') as f:
      DATASETS['tennis_forehand_rear'] = json.load(f)


_load_dynamic_splits()


DATASET_TO_NUM_CLASSES = {
    'pouring': 5,
    'baseball_pitch': 4,
    'baseball_swing': 3,
    'bench_press': 2,
    'bowling': 3,
    'clean_and_jerk': 6,
    'golf_swing': 3,
    'jumping_jacks': 4,
    'pushups': 2,
    'pullups': 2,
    'situp': 2,
    'squats': 4,
    'tennis_forehand': 3,
    'tennis_serve': 4,
    'tennis_forehand_rear': 3,
    'tennis_forehand_rear_pair': 3,
}

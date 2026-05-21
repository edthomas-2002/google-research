"""Forehand clip paths and S3 sync into /tmp."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess

DEFAULT_FOREHAND_S3 = 's3://tennis-swing-data/Forehands/'
DEFAULT_VIDEO_DIR = '/tmp/Forehands/Rear View'


def default_video_dir():
  return os.environ.get('FOREHAND_VIDEO_DIR', DEFAULT_VIDEO_DIR)


def default_forehand_s3_uri():
  return os.environ.get('FOREHAND_S3_URI', DEFAULT_FOREHAND_S3)


def _has_videos(video_dir):
  if not os.path.isdir(video_dir):
    return False
  for name in os.listdir(video_dir):
    if name.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
      return True
  return False


def ensure_forehand_videos(video_dir=None, s3_uri=None, skip_download=False):
  """Sync forehand clips from S3 to /tmp if the local Rear View dir is empty."""
  video_dir = os.path.abspath(video_dir or default_video_dir())
  s3_uri = (s3_uri or default_forehand_s3_uri()).rstrip('/') + '/'

  if _has_videos(video_dir):
    return video_dir

  if skip_download:
    raise SystemExit(
        'No videos in %s. Run aws s3 sync or unset FOREHAND_SKIP_DOWNLOAD.' %
        video_dir)

  if os.path.basename(video_dir) == 'Rear View':
    sync_dest = os.path.dirname(video_dir)
  else:
    sync_dest = video_dir
  os.makedirs(sync_dest, exist_ok=True)

  print('Syncing forehand clips from %s to %s ...' % (s3_uri, sync_dest),
        flush=True)
  subprocess.check_call(['aws', 's3', 'sync', s3_uri, sync_dest + '/'])

  if not _has_videos(video_dir):
    raise SystemExit(
        'S3 sync finished but no videos found in %s (check bucket layout).' %
        video_dir)
  print('Forehand videos ready: %s' % video_dir, flush=True)
  return video_dir

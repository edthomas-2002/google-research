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

r"""Training code based on TF Eager."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os

from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v2 as tf

from tcc.algorithms import get_algo
from tcc.config import CONFIG
from tcc.datasets import create_dataset
from tcc.utils import get_lr_fn
from tcc.utils import get_lr_opt_global_step
from tcc.utils import restore_ckpt
from tcc.utils import setup_train_dir
from tcc.utils import Stopwatch
from tcc.utils import to_dict


flags.DEFINE_string('logdir', '/tmp/alignment_logs', 'Path to logs.')
flags.DEFINE_string(
    'persistent_checkpoint_dir', None,
    'If set, also keep one last and one best checkpoint in this directory.')
flags.DEFINE_string('wandb_entity', None, 'Optional W&B entity.')
flags.DEFINE_string('wandb_project', None, 'Optional W&B project.')
flags.DEFINE_string('wandb_run_name', 'tcc', 'Base name for the W&B run.')
flags.DEFINE_boolean('defun', True, 'Defun functions in algo for faster '
                     'training.')
flags.DEFINE_boolean('debug', False, 'Plots detailed summaries on Tensorboard.')
flags.DEFINE_boolean(
    'force_train', False, 'Continue with training even when '
    'train_logs exist. Useful if one has to resume training. '
    'By default switched off to prevent overwriting existing '
    'experiments.')
flags.DEFINE_boolean('visualize', False, 'Visualize images, gradients etc. '
                     'Switched off by for default to speed training up and '
                     'takes less memory.')

FLAGS = flags.FLAGS
layers = tf.keras.layers


def train():
  """Trains model and evaluates on relevant downstream tasks."""
  CONFIG.LOGDIR = FLAGS.logdir
  logdir = CONFIG.LOGDIR
  setup_train_dir(logdir)

  wandb_run = None
  if FLAGS.wandb_project:
    import wandb  # pylint: disable=g-import-not-at-top
    start_timestamp = datetime.datetime.now(
        datetime.timezone.utc).strftime('%Y%m%d-%H%M%SZ')
    run_name = '%s-%s' % (FLAGS.wandb_run_name, start_timestamp)
    wandb_config = to_dict(CONFIG)
    wandb_config['PERSISTENT_CHECKPOINT_DIR'] = (
        FLAGS.persistent_checkpoint_dir)
    wandb_run = wandb.init(
        entity=FLAGS.wandb_entity,
        project=FLAGS.wandb_project,
        name=run_name,
        config=wandb_config)
    wandb_run.define_metric('global_step')
    wandb_run.define_metric('*', step_metric='global_step')

  # Common code for multigpu and single gpu. Set devices here if you don't
  # want to use all the GPUs on the machine. Default is to use all GPUs.
  strategy = tf.distribute.MirroredStrategy()
  with strategy.scope():
    algo = get_algo(CONFIG.TRAINING_ALGO)

    # Setup summary writer.
    summary_writer = tf.summary.create_file_writer(
        os.path.join(logdir, 'train_logs'), flush_millis=10000)

    learning_rate, optimizer, global_step = get_lr_opt_global_step()
    best_val_loss = tf.Variable(
        float('inf'), trainable=False, dtype=tf.float32, name='best_val_loss')
    best_val_step = tf.Variable(
        -1, trainable=False, dtype=tf.int64, name='best_val_step')
    ckpt_manager, _, checkpoint = restore_ckpt(
        logdir=logdir,
        optimizer=optimizer,
        best_val_loss=best_val_loss,
        best_val_step=best_val_step,
        **algo.model)

    last_manager = None
    best_manager = None
    if FLAGS.persistent_checkpoint_dir:
      last_manager = tf.train.CheckpointManager(
          checkpoint,
          directory=os.path.join(FLAGS.persistent_checkpoint_dir, 'last'),
          max_to_keep=1)
      best_manager = tf.train.CheckpointManager(
          checkpoint,
          directory=os.path.join(FLAGS.persistent_checkpoint_dir, 'best'),
          max_to_keep=1)
      persistent_checkpoint = (
          last_manager.latest_checkpoint or
          tf.train.latest_checkpoint(FLAGS.persistent_checkpoint_dir))
      if not ckpt_manager.latest_checkpoint and persistent_checkpoint:
        checkpoint.restore(persistent_checkpoint)
        logging.info('Restored persistent checkpoint: %s',
                     persistent_checkpoint)

    global_step_value = global_step.numpy()

    # Remember in Eager mode learning rate variable needs to be updated
    # manually. Calling lr_fn each iteration to get current learning rate.
    lr_fn = get_lr_fn(CONFIG.OPTIMIZER)

    # Setup Dataset Iterators from train and val datasets.
    batch_size_per_replica = CONFIG.TRAIN.BATCH_SIZE
    total_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
    train_ds = create_dataset('train', mode='train',
                              batch_size=total_batch_size,
                              return_iterator=False)
    train_iterator = strategy.make_dataset_iterator(train_ds)
    val_batch_size = (
        CONFIG.EVAL.BATCH_SIZE * strategy.num_replicas_in_sync)
    val_ds = create_dataset('val', mode='eval',
                            batch_size=val_batch_size,
                            return_iterator=False)
    val_iterator = strategy.make_dataset_iterator(val_ds)

    def train_step(data):
      steps = data['chosen_steps']
      seq_lens = data['seq_lens']
      loss = algo.train_one_iter(data, steps, seq_lens, global_step, optimizer)
      return loss

    def val_step(data):
      steps = data['chosen_steps']
      seq_lens = data['seq_lens']
      embs = algo.call(data, steps, seq_lens, training=False)
      return algo.compute_loss(
          embs,
          steps,
          seq_lens,
          global_step,
          training=False,
          frame_labels=data['frame_labels'],
          seq_labels=data['seq_labels'])

    # This reduction only affects reporting, not the gradients.
    # pylint: disable=g-long-lambda
    dist_train = lambda it: strategy.reduce(
        tf.distribute.ReduceOp.SUM, strategy.experimental_run(train_step, it),
        axis=None)
    dist_val = lambda it: strategy.reduce(
        tf.distribute.ReduceOp.SUM, strategy.experimental_run(val_step, it),
        axis=None) / strategy.num_replicas_in_sync
    # pylint: enable=g-long-lambda
    if FLAGS.defun:
      dist_train = tf.function(dist_train)
      dist_val = tf.function(dist_val)

    stopwatch = Stopwatch()

    try:
      while global_step_value < CONFIG.TRAIN.MAX_ITERS:
        with summary_writer.as_default():
          with tf.summary.record_if(
              global_step_value % CONFIG.LOGGING.REPORT_INTERVAL == 0):

            loss = dist_train(train_iterator)
            global_step_value = global_step.numpy()
            wandb_metrics = {}

            # Update learning rate based in lr_fn.
            learning_rate.assign(lr_fn(learning_rate, global_step))

            tf.summary.scalar('loss', loss, step=global_step)
            tf.summary.scalar('learning_rate', learning_rate, step=global_step)

            # Use the original local checkpoint flow, then persist last/best.
            if global_step_value % CONFIG.CHECKPOINT.SAVE_INTERVAL == 0:
              val_losses = [
                  dist_val(val_iterator)
                  for _ in range(CONFIG.EVAL.VAL_ITERS)
              ]
              val_loss = tf.reduce_mean(tf.stack(val_losses))
              tf.summary.scalar('val_loss', val_loss, step=global_step)
              val_loss_value = float(val_loss.numpy())
              updated_best = (
                  val_loss_value < float(best_val_loss.numpy()))
              if updated_best:
                best_val_loss.assign(val_loss_value)
                best_val_step.assign(global_step_value)
              ckpt_manager.save()
              if last_manager:
                last_manager.save(checkpoint_number=int(global_step.numpy()))
              if updated_best and best_manager:
                best_manager.save(checkpoint_number=int(global_step.numpy()))
                logging.info(
                    'Best checkpoint updated at iter %d (val loss=%.3f).',
                    global_step_value, best_val_loss.numpy())
              logging.info('Validation loss at iter %d: %.3f.',
                           global_step_value, val_loss_value)
              logging.info('Checkpoint saved at iter %d.', global_step_value)
              wandb_metrics.update({
                  'validation/loss': val_loss_value,
                  'validation/best_loss': float(best_val_loss.numpy()),
                  'checkpoint/is_best': int(updated_best),
                  'checkpoint/best_step': int(best_val_step.numpy()),
              })

            time_per_iter = stopwatch.elapsed()

            tf.summary.scalar(
                'timing/time_per_iter', time_per_iter, step=global_step)

            logging.info('Iter[{}/{}], {:.1f}s/iter, Loss: {:.3f}'.format(
                global_step_value, CONFIG.TRAIN.MAX_ITERS, time_per_iter,
                loss.numpy()))
            if global_step_value % CONFIG.LOGGING.REPORT_INTERVAL == 0:
              wandb_metrics.update({
                  'train/loss': float(loss.numpy()),
                  'optimizer/learning_rate': float(learning_rate.numpy()),
                  'performance/seconds_per_iteration': time_per_iter,
                  'performance/examples_per_second': (
                      total_batch_size / time_per_iter),
              })
            if wandb_run and wandb_metrics:
              wandb_run.log(
                  dict(wandb_metrics, global_step=global_step_value))
            # Reset stopwatch after iter is complete.
            stopwatch.reset()

    except KeyboardInterrupt:
      logging.info('Caught keyboard interrupt. Saving model before quitting.')

    finally:
      ckpt_manager.save()
      if last_manager:
        last_manager.save(checkpoint_number=int(global_step.numpy()))
      logging.info('Checkpoint saved at iter %d', global_step_value)
      if wandb_run:
        wandb_run.finish()


def main(_):
  tf.enable_v2_behavior()
  tf.keras.backend.set_learning_phase(1)

  train()

if __name__ == '__main__':
  app.run(main)

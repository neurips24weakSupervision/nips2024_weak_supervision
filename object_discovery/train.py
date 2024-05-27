# coding=utf-8
# Copyright 2023 The Google Research Authors.
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

"""Training loop for object discovery with Slot Attention."""
import datetime
import time

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

import github.data_train as data_utils
import github.model_cnn_unsupervised as model_utils_unsupervised_cnn
import github.model_resnet_unsupervised as model_utils_unsupervised_resnet
import github.model_cnn_label as model_utils_label_cnn
import github.model_resnet_label as model_utils_label_resnet
import github.model_cnn_masks as model_utils_mask_cnn
import github.model_resnet_masks as model_utils_mask_resnet
import github.utils as utils

FLAGS = flags.FLAGS
flags.DEFINE_string("model_dir", "example_dir",
                    "Where to save the checkpoints.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate.")
flags.DEFINE_integer("num_train_steps", 500000, "Number of training steps.")
flags.DEFINE_integer("warmup_steps", 50000,
                     "Number of warmup steps for the learning rate.")
flags.DEFINE_float("decay_rate", 0.5, "Rate for the learning rate decay.")
flags.DEFINE_string("training_strategy", "unsupervised", "Supervision Type: either unsupervised, label, or mask.")
flags.DEFINE_string("model_complexity", "cnn", "Size of the model: either cnn or resnet")
flags.DEFINE_float("weighting_factor",0.1 , "Learning rate.")
flags.DEFINE_integer("decay_steps", 100000,
                     "Number of steps for the learning rate decay.")

def train_step_prediction(batch, model, optimizer,step,max_steps,weighting, isMask):
  """Perform a single training step."""

  # Get the prediction of the models and compute the loss.
  with tf.GradientTape() as tape:
    preds = model(batch["image"], training=True)
    recon_combined, recons, masks, slots, predictions = preds
    loss_value1 = utils.l2_loss(batch["image"], recon_combined)
    if isMask:
      loss_value2 = abs((weighting - weighting/max_steps * step))*utils.hungarian_huber_loss(predictions, batch["target_mask"])
    else:
      loss_value2 = abs((weighting - weighting/max_steps * step))*utils.hungarian_huber_loss(predictions, batch["target"])
    loss_value = loss_value1 + loss_value2
    del recons, masks, slots  # Unused.

  gradients = tape.gradient(loss_value, model.trainable_weights)
  optimizer.apply_gradients(zip(gradients, model.trainable_weights))
  return loss_value, loss_value1, loss_value2

def train_step_reconstruction(batch, model, optimizer, step):
  """Perform a single training step."""

  # Get the prediction of the models and compute the loss.
  with tf.GradientTape() as tape:
    preds = model(batch["image"], training=True)
    recon_combined, recons, masks, slots = preds
    loss_value = utils.l2_loss(batch["image"], recon_combined)
    del recons, masks, slots  # Unused.

  gradients = tape.gradient(loss_value, model.trainable_weights)
  optimizer.apply_gradients(zip(gradients, model.trainable_weights))
  return loss_value


def main(argv):
  del argv
  batch_size = 32
  num_slots = 11
  num_iterations = 3
  weighting_factor = FLAGS.weighting_factor
  base_learning_rate = FLAGS.learning_rate
  num_train_steps = FLAGS.num_train_steps
  warmup_steps = FLAGS.warmup_steps
  decay_rate = FLAGS.decay_rate
  decay_steps = FLAGS.decay_steps
  supervision_type = FLAGS.training_strategy
  complexity_type = FLAGS.model_complexity
  tf.random.set_seed(FLAGS.seed)
  resolution = (128, 128)
  # Build dataset iterators, optimizers and model.
  data_iterator = data_utils.build_clevrtex_iterator(
      batch_size, split="train", resolution=resolution, shuffle=True,
      max_n_objects=6, get_properties=False, apply_crop=True)

  optimizer = tf.keras.optimizers.legacy.Adam(base_learning_rate, epsilon=1e-08)

 # Prepare checkpoint manager.
  global_step = tf.Variable(
      0, trainable=False, name="global_step", dtype=tf.int64)
  
  if supervision_type == "unsupervised":
    if complexity_type == "cnn":
      model = model_utils_unsupervised_cnn.build_model(resolution, batch_size, num_slots,
                                        num_iterations)
    else:
      model = model_utils_unsupervised_resnet.build_model(resolution, batch_size, num_slots,
                                        num_iterations)
  elif supervision_type == "label":
    if complexity_type == "cnn":
      model = model_utils_label_cnn.build_model(resolution, batch_size, num_slots,
                                        num_iterations)
    else:
      model = model_utils_label_resnet.build_model(resolution, batch_size, num_slots,
                                        num_iterations)
  elif supervision_type == "mask":
    if complexity_type == "cnn":
      model = model_utils_mask_cnn.build_model(resolution, batch_size, num_slots,
                                        num_iterations)
    else:
      model = model_utils_mask_resnet.build_model(resolution, batch_size, num_slots,
                                        num_iterations)
  else:
    logging.info("Wrong Parameter Specification, Abort")
    return

  ckpt = tf.train.Checkpoint(
      network=model, optimizer=optimizer, global_step=global_step)
  ckpt_manager = tf.train.CheckpointManager(
      checkpoint=ckpt, directory=FLAGS.model_dir, max_to_keep=200)
  ckpt.restore(ckpt_manager.latest_checkpoint)

  if ckpt_manager.latest_checkpoint:
    logging.info("Restored from %s", ckpt_manager.latest_checkpoint)
  else:
    logging.info("Initializing from scratch")


  start = time.time()
  for _ in range(num_train_steps):
    batch = next(data_iterator)

    # Learning rate warm-up.
    if global_step < warmup_steps:
      learning_rate = base_learning_rate * tf.cast(
          global_step, tf.float32) / tf.cast(warmup_steps, tf.float32)
    else:
      learning_rate = base_learning_rate
    learning_rate = learning_rate * (decay_rate ** (
        tf.cast(global_step, tf.float32) / tf.cast(decay_steps, tf.float32)))
    optimizer.lr = learning_rate.numpy()
    if supervision_type == "unsupervised":
      loss_value = train_step_reconstruction(batch, model, optimizer,global_step.numpy())
      if not global_step % 100:
        logging.info("Step: %s, Loss: %.6f,Time: %s",
                     global_step.numpy(), loss_value,
                     datetime.timedelta(seconds=time.time() - start))
    else:
      loss_value, recon_loss, pred_loss = train_step_prediction(batch, model, optimizer, global_step.numpy(), num_train_steps, weighting_factor, supervision_type=="mask")
      if not global_step % 100:
        logging.info("Step: %s, Loss: %.6f, Loss Recon: %.6f,Loss Pred: %.6f,Time: %s",
                     global_step.numpy(), loss_value,recon_loss,pred_loss,
                     datetime.timedelta(seconds=time.time() - start))




    global_step.assign_add(1)
    if not global_step  % 1000:
      saved_ckpt = ckpt_manager.save()
      logging.info("Saved checkpoint: %s", saved_ckpt)

if __name__ == "__main__":
  app.run(main)

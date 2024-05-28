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

import github.data_test as data_utils
import github.model_cnn_unsupervised as model_utils_unsupervised_cnn
import github.model_resnet_unsupervised as model_utils_unsupervised_resnet
import github.model_cnn_label as model_utils_label_cnn
import github.model_resnet_label as model_utils_label_resnet
import github.model_cnn_masks as model_utils_mask_cnn
import github.model_resnet_masks as model_utils_mask_resnet
import github.utils as utils
import numpy as np
FLAGS = flags.FLAGS
flags.DEFINE_string("model_dir", "example_dir",
                    "Where to save the checkpoints.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("training_strategy", "unsupervised", "Supervision Type: either unsupervised, label, or mask.")
flags.DEFINE_string("model_complexity", "cnn", "Size of the model: either cnn or resnet")
flags.DEFINE_float("weighting_factor",0.1 , "Learning rate.")



def main(argv):
  del argv
  batch_size = 32
  num_slots = 11
  num_iterations = 3
  supervision_type = FLAGS.training_strategy
  complexity_type = FLAGS.model_complexity
  tf.random.set_seed(FLAGS.seed)
  resolution = (128, 128)
  # Build dataset iterators, optimizers and model.
  data_iterator = data_utils.build_clevrtex_iterator(
      batch_size, resolution=resolution, shuffle=True,
      get_properties=False, apply_crop=True)


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
      network=model, global_step=global_step)
  ckpt_manager = tf.train.CheckpointManager(
      checkpoint=ckpt, directory=FLAGS.model_dir, max_to_keep=200)
  ckpt.restore(ckpt_manager.latest_checkpoint)

  if ckpt_manager.latest_checkpoint:
    logging.info("Restored from %s", ckpt_manager.latest_checkpoint)
  else:
    logging.info("Initializing from scratch")

  all_predictions = []
  all_recons = []
  all_masks_predicted = []
  all_images = []
  all_properties = []
  all_slots = []
  all_masks = []
  for i in range(156):  
    batch = next(data_iterator)
    if supervision_type == "unsupervised":
      recons_combined, recons, masks, slots = model.predict(batch["image"])
    else:
      recons_combined, recons, masks, slots, _ = model.predict(batch["image"])
    all_images.append(np.array(batch["image"]))
    all_masks.append(np.array(batch["mask"]))
    all_recons.append(recons)
    all_predictions.append(np.array(recons_combined))
    all_masks_predicted.append(np.array(masks))
    all_properties.append(np.array(batch["target"]))
    all_slots.append(np.array(slots))

  np.save(FLAGS.model_dir + "/all_images.npy", np.array(all_images))
  np.save(FLAGS.model_dir + "/all_masks.npy", np.array(all_masks))
  np.save(FLAGS.model_dir + "/all_recons.npy", np.array(all_recons))
  np.save(FLAGS.model_dir + "/all_predictions.npy", np.array(all_predictions))
  np.save(FLAGS.model_dir + "/all_masks_predicted.npy", np.array(all_masks_predicted))
  np.save(FLAGS.model_dir + "/all_properties.npy", np.array(all_properties))
  np.save(FLAGS.model_dir + "/all_slots.npy", np.array(all_slots))
if __name__ == "__main__":
  app.run(main)

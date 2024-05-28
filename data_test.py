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

"""Data utils."""
import collections
import tensorflow as tf
import tensorflow_datasets as tfds
import logging
def split_masks(tensor):
    num_classes = 11
    labels = tf.range(num_classes, dtype=tf.float32)
    labels = tf.reshape(labels, [1, 1, 1, num_classes])
    tensor_expanded = tf.expand_dims(tensor, -1)
    masks = tf.equal(tensor_expanded, labels)
    masks = tf.cast(masks, tf.float32)
    masks = tf.squeeze(masks, axis=2)
    masks = tf.transpose(masks, [2, 0, 1])
    return masks


def preprocess_clevrtex(features, resolution, apply_crop=False,
                     get_properties=True):
  """Preprocess CLEVRTEX."""
  image = tf.cast(features["image"], dtype=tf.float32)
  image = ((image / 255.0) - 0.5) * 2.0  # Rescale to [-1, 1].
  masks = tf.cast(features["segmentations"], dtype=tf.float32)
  apply_crop = True
  get_properties = True
  if apply_crop:
    crop = ((119 - 96, 119 + 96), (159 - 96, 159 + 96))  # Get center crop.
    image = image[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1], :]
    masks = masks[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1], :]

  image = tf.image.resize(
      image, resolution, method=tf.image.ResizeMethod.BILINEAR)
  masks = tf.image.resize(
      masks, resolution, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  max_n_objects = 11


  masks2 = split_masks(masks)
  masks2 = tf.reshape(masks2, shape=(-1, 128*128))



  material = tf.one_hot(features["instances"]["material"], 90)
  shape_obj = tf.one_hot(features["instances"]["shape"], 4)
  size = tf.one_hot(features["instances"]["size"], 3)
  coords = (features["instances"]["positions"] + 3.) / 6.
  properties_dict = collections.OrderedDict({
      "3d_coords": coords,
      "size": size,
      "material": material,
      "shape": shape_obj,
  })

  properties_tensor = tf.concat(list(properties_dict.values()), axis=1)

  properties_tensor = tf.concat(
      [properties_tensor,
       tf.ones([tf.shape(properties_tensor)[0], 1])], axis=1)

  properties_pad = tf.pad(
      properties_tensor,
      [[0, max_n_objects - tf.shape(properties_tensor)[0],], [0, 0]],
      "CONSTANT")
  masks_dict = collections.OrderedDict({
      "masks": masks2,
  })
  masks_tensor = tf.concat(list(masks_dict.values()), axis=1)
  masks_pad = tf.pad(
      masks_tensor,
      [[0, max_n_objects - tf.shape(masks_tensor)[0],], [0, 0]],
      "CONSTANT")

  features = {
      "image": image,
      "target": properties_pad,
      "target_mask": masks_pad,
      "mask":masks
  }

  return features


def build_clevrtex(resolution=(128, 128), shuffle=False,
                num_eval_examples=512, get_properties=True, apply_crop=False):
  """Build CLEVRTEX dataset."""
  ds,info = tfds.load("clevr_tex:2.0.0", with_info=True, split="testing", shuffle_files=False)
  logging.info(info)

  def _preprocess_fn(x, resolution):
    return preprocess_clevrtex(
        x, resolution, apply_crop=apply_crop, get_properties=get_properties)
  ds = ds.map(lambda x: _preprocess_fn(x, resolution))
  return ds


def build_clevrtex_iterator(batch_size, **kwargs):
  ds = build_clevrtex(**kwargs)
  ds = ds.repeat(-1)
  ds = ds.batch(batch_size, drop_remainder=True)
  return iter(ds)


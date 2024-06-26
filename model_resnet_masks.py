import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import layers, models
from tensorflow.keras import models


"""Calculate position and scaling parameters for position and scaling invariant slot attention"""
def calcSP(attn,resolution,batch_size):
    """
    Args:
    attn: Attention values of slot attention
    resolution: resolution of the last feature map of the encoder
    batch_size: the batch size (in our case 32)
    """

    #build a grid and expand grid and attn to match sizes
    grid = build_grid(resolution)
    grid =  tf.expand_dims(grid, axis=1)
    grid =  tf.broadcast_to(grid, (batch_size,11,resolution[0],resolution[1],2))
    attn = tf.reshape(attn,(batch_size,11,resolution[0],resolution[1]))
    attn = tf.expand_dims(attn, axis= - 1)
    
    #calculate the weighted sum for the position invariant parameter
    gridRel = tf.multiply(grid, attn)
    weighted_sum_p = tf.reduce_sum(gridRel, axis=[2, 3])

    #calculate the scaling invariant parameters
    rel_s_p = weighted_sum_p[:, :, :2]
    rel_s_p = tf.expand_dims(rel_s_p, axis=-2)
    rel_s_p = tf.expand_dims(rel_s_p, axis=-2)
    rel_s_p =  tf.broadcast_to(rel_s_p, (batch_size,11,resolution[0],resolution[1],2))
    grid = grid - rel_s_p 
    grid = tf.math.square(grid)
    attn = attn + 0.00000000001
    gridRel = tf.multiply(grid, attn) 
    weighted_sum_s = tf.reduce_sum(gridRel, axis=[2, 3])
    weighted_sum_s = tf.math.sqrt(weighted_sum_s)
    
    return weighted_sum_p[:, :, :2], weighted_sum_s[:, :, :2]


class SlotAttention(layers.Layer):
  """Slot Attention module."""

  def __init__(self, num_iterations, num_slots, slot_size, mlp_hidden_size,resolution=(16,16),
               epsilon=1e-8):
    """Builds the Slot Attention module.

    Args:
      num_iterations: Number of iterations.
      num_slots: Number of slots.
      slot_size: Dimensionality of slot feature vectors.
      mlp_hidden_size: Hidden layer size of MLP.
      batch_size: the batch size (in our case 32)
      """
    super().__init__()

    #initialize MLPs and learnable slots
    self.num_iterations = num_iterations
    self.num_slots = num_slots
    self.slot_size = slot_size
    self.mlp_hidden_size = mlp_hidden_size
    self.epsilon = epsilon
    self.norm_slots = layers.LayerNormalization()
    self.norm_mlp = layers.LayerNormalization()
    self.layer_norm = layers.LayerNormalization()
    self.slots_mu = self.add_weight(
        initializer="normal",
        shape=[1, 11, self.slot_size],
        dtype=tf.float32,
        name="slots_mu")
    self.gru = layers.GRUCell(self.slot_size)
    self.mlp = tf.keras.Sequential([
      layers.Dense(128, activation="relu"),
      layers.Dense(64)
    ], name="mlp_1")

    self.slot_size = slot_size
    self.resolution = resolution
    self.project_k = tf.keras.Sequential([
      layers.Dense(64)
    ], name="mlp_keys")
    self.project_v = tf.keras.Sequential([
      layers.Dense(64)
    ], name="mlp_values")
    self.project_q = tf.keras.Sequential([
      layers.Dense(64)
    ], name="mlp_slots")
    self.mlp_inputs = tf.keras.Sequential([
        layers.Dense(128, activation="relu"),
        layers.Dense(64)
    ], name="feedforward")

    #relative position MLP
    self.encoder_rel_pos = SoftPositionEmbed_rel(self.resolution,32)

    #absolute position MLP (only applied once at the beginning of slot attention)
    self.encoder_start = SoftPositionEmbed_start(self.resolution,32)

  def call(self, inputs):
    num_inputs = inputs.shape[1]

    #broadcast position invariant parameters to slot size
    s_p = tf.random.uniform(shape=(32, 11, 2), minval=-1, maxval=1)
    s_s = tf.random.normal(shape=(32, 11, 2), mean=0.1, stddev=0.01)

    #initialize slots and broadcast to batch size
    slots = self.slots_mu 
    slots = tf.broadcast_to(slots,(32,11,64))

    #position invariant encoding
    inputs = self.encoder_start(inputs)
    inputs = spatial_flatten(inputs)

    #project keys and values
    inputs_k = self.project_k(inputs)
    inputs_v = self.project_v(inputs)


    for ind in range(self.num_iterations +1):

       #apply relative position encoding
      inputs_k_rel = self.encoder_rel_pos(inputs_k, s_p, s_s)
      inputs_v_rel = self.encoder_rel_pos(inputs_v, s_p, s_s)

      #calculate keys and values
      k = self.mlp_inputs(inputs_k_rel)
      v = self.mlp_inputs(inputs_v_rel)

      #calculate attention
      slots_prev = slots
      slots = self.norm_slots(slots)
      q = self.project_q(slots)
      q *= self.slot_size ** -0.5  
      q =  tf.expand_dims(q, axis=2)
      attn_logits = tf.reduce_sum(k * q, axis=-1)
      attn_logits = tf.transpose(attn_logits, perm=[0, 2, 1])
      attn = tf.nn.softmax(attn_logits, axis=-1)
      attn += self.epsilon
      attn /= tf.reduce_sum(attn, axis=-2, keepdims=True)
      attn2 = tf.expand_dims(attn, -1)
      updates = tf.reduce_sum(tf.transpose(attn2,perm=[0,2,1,3]) * v, axis=2)

      #update position variables
      s_p,s_s = calcSP(spatial_unflatten(tf.transpose(attn,perm=[0,2,1]),num_inputs),(16,16),32)
      s_s = tf.clip_by_value(s_s, clip_value_min=0.001, clip_value_max=5)
      if ind < self.num_iterations:
        slots, _ = self.gru(updates, [slots_prev])
        slots += self.mlp(self.norm_mlp(slots))

    return slots, s_p,s_s,attn


def spatial_broadcast(slots, resolution,rel_s_p,rel_s_s,batch_size):
  """Broadcast slot features to a 2D grid and collapse slot dimension. Broadcast position and scale invariance to grid"""
  slots = tf.reshape(slots, [-1, slots.shape[-1]])[:, None, None, :]
  grid = build_grid(resolution)
  grid =  tf.broadcast_to(grid, (batch_size,11,resolution[0],resolution[1] ,2))    
  rel_s_p = tf.expand_dims(rel_s_p, axis=-2)
  rel_s_p = tf.expand_dims(rel_s_p, axis=-2)  
  rel_s_p =  tf.broadcast_to(rel_s_p, (batch_size,11,resolution[0],resolution[1],2))
  rel_s_s = tf.expand_dims(rel_s_s, axis=-2)
  rel_s_s = tf.expand_dims(rel_s_s, axis=-2)  
  rel_s_s =  tf.broadcast_to(rel_s_s, (batch_size,11,resolution[0],resolution[1],2))
  rel_grid = (grid - rel_s_p) / rel_s_s 
  rel_grid = tf.reshape(rel_grid, (batch_size*11,resolution[0],resolution[1],2))  
  grid = tf.tile(slots, [1, resolution[0], resolution[1], 1])
  return grid, rel_grid


"""Helper function to spatially flatten 2D input to 1D series"""
def spatial_flatten(x):
  return tf.reshape(x, [x.shape[0], x.shape[1], x.shape[2] * x.shape[3], x.shape[-1]])

"""Helper function to spatially re-flatten 1D series to 2D features"""
def spatial_unflatten(x,num_inputs):
  return tf.reshape(x, [x.shape[0],  num_inputs, num_inputs, 11])

def unstack_and_split(x, batch_size, num_channels=3):
  """Unstack batch dimension and split into channels and alpha mask."""
  unstacked = tf.reshape(x, [batch_size, -1] + x.shape.as_list()[1:])
  channels, masks = tf.split(unstacked, [num_channels, 1], axis=-1)
  return channels, masks


class SlotAttentionAutoEncoder(layers.Layer):
  """Slot Attention-based auto-encoder for object discovery."""

  def __init__(self, resolution, num_slots, num_iterations):
    """Builds the Slot Attention-based auto-encoder.

    Args:
      resolution: Tuple of integers specifying width and height of input image.
      num_slots: Number of slots in Slot Attention.
      num_iterations: Number of iterations in Slot Attention.
    """
    super().__init__()
    self.resolution = resolution
    self.num_slots = num_slots
    self.num_iterations = num_iterations
    self.encoder_res = build_resnet34_encoder()

    self.layer_norm = layers.LayerNormalization()

    self.decoder_initial_size = (16, 16)
    self.decoder_cnn = tf.keras.Sequential([
        layers.Conv2DTranspose(
            64, 5, strides=(2, 2), padding="SAME", activation="relu"),
        layers.Conv2DTranspose(
            64, 5, strides=(2, 2), padding="SAME", activation="relu"),
        layers.Conv2DTranspose(
            64, 5, strides=(2, 2), padding="SAME", activation="relu"),
        layers.Conv2DTranspose(
            64, 5, strides=(1, 1), padding="SAME", activation="relu"),
        layers.Conv2DTranspose(
            64, 5, strides=(1, 1), padding="SAME", activation="relu"),
        layers.Conv2DTranspose(
            4, 1, strides=(1, 1), padding="SAME", activation=None)
    ], name="decoder_cnn")



    self.dense_pos_decode = tf.keras.Sequential([
        layers.Dense(128, activation="relu"),
        layers.Dense(64)
    ], name="dense_pos_decode")

    self.slot_attention = SlotAttention(
        num_iterations=self.num_iterations,
        num_slots=self.num_slots,
        slot_size=64,
        mlp_hidden_size=128)

    self.mlp_inputs_decode = tf.keras.Sequential([
        layers.Dense(128, activation="relu"),
        layers.Dense(64)
    ], name="decode")

  def call(self, image):
    #encode image
    x = self.encoder_cnn(image) 

    #apply slot attention
    slots,s_p,s_s,attn = self.slot_attention(x)

     #spatially broadcast slots and relative grid
    x, rel = spatial_broadcast(slots, self.decoder_initial_size,s_p,s_s,32)

    #add relative grid encoding to slots
    x = x + self.dense_pos_decode(rel)
    x = self.mlp_inputs_decode(x)

    #decode cnn
    x = self.decoder_cnn(x)

    #split 3D recons and 1D masks (RGB and alpha-mask)
    recons, masks = unstack_and_split(x, batch_size=image.shape[0])

    #argmax over slots
    masks = tf.nn.softmax(masks, axis=1)

    #recombine RGB and alpha mask
    recon_combined = tf.reduce_sum(recons * masks, axis=1)

    #flatten masks from 2D to 1D
    masks2 = tf.reshape(masks,[32, 11, 128*128]) 
    return recon_combined, recons, masks, slots, masks2, s_p, s_s

def build_grid(resolution):
  """build 2D grid from -1 to 1"""
  ranges = [np.linspace(-1., 1., num=res) for res in resolution]
  grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
  grid = np.stack(grid, axis=-1)
  grid = np.reshape(grid, [resolution[0], resolution[1], -1])
  grid = np.expand_dims(grid, axis=0)
  grid = grid.astype(np.float32)
  return grid

class SoftPositionEmbed_start(layers.Layer):
  """Adds soft positional embedding with learnable projection."""

  def __init__(self, resolution,batch_size):
    """Builds the soft position embedding layer used for absolute position encoding.

    Args:
      resolution: Tuple of integers specifying width and height of grid.
      batch_size: batch size of training/inference (in our case 32)
    """
    super().__init__()
    self.resolution = resolution
    self.dense= tf.keras.Sequential([
        layers.Dense(128, activation="relu"),
        layers.Dense(512)
    ], name="dense")
    self.grid = build_grid(resolution)
    self.batch_size = batch_size
  #build absolute grid and add to input
  def call(self, inputs):
    grid =  tf.expand_dims(self.grid, axis=1)
    grid =  tf.broadcast_to(grid, (self.batch_size,11,self.resolution[0],self.resolution[1] ,2))    
    inputs = tf.expand_dims(inputs,axis=1)
    inputs = tf.broadcast_to(inputs,(self.batch_size,11,self.resolution[0],self.resolution[1],inputs[0].shape[-1]))
    return inputs + self.dense(grid)

class SoftPositionEmbed_rel(layers.Layer):
  """Adds soft positional embedding with learnable projection."""

  def __init__(self, resolution,batch_size):
    """Builds the soft position embedding layer used for relative position encoding.

    Args:
      resolution: Tuple of integers specifying width and height of grid.
      batch_size: batch size of training/inference (in our case 32)
    """
    super().__init__()
    self.resolution = resolution
    self.dense= tf.keras.Sequential([
        layers.Dense(64)
    ], name="dense")
    self.grid = build_grid(resolution)
    self.batch_size = batch_size

  #add relative grid to inputs
  def call(self, inputs, rel_s_p,rel_s_s):
    grid =  tf.expand_dims(self.grid, axis=1)
    grid =  tf.broadcast_to(grid, (self.batch_size,11,self.resolution[0],self.resolution[1] ,2))    
    rel_s_p = tf.expand_dims(rel_s_p, axis=-2)
    rel_s_p = tf.expand_dims(rel_s_p, axis=-2)
    rel_s_p =  tf.broadcast_to(rel_s_p, (self.batch_size,11,self.resolution[0],self.resolution[1],2))

    rel_s_s = tf.expand_dims(rel_s_s, axis=-2)
    rel_s_s = tf.expand_dims(rel_s_s, axis=-2)
    rel_s_s =  tf.broadcast_to(rel_s_s, (self.batch_size,11,self.resolution[0],self.resolution[1],2))
    rel_grid = (grid - rel_s_p) / rel_s_s
    rel_grid = spatial_flatten(rel_grid)
    return inputs + self.dense(rel_grid)



def build_model(resolution, batch_size, num_slots, num_iterations,
                num_channels=3):
  model_def = SlotAttentionAutoEncoder
  image = tf.keras.Input(list(resolution) + [num_channels], batch_size)
  outputs = model_def(resolution, num_slots, num_iterations)(image)
  model = tf.keras.Model(inputs=image, outputs=outputs)
  return model



#definitions for the resnet encoder
def conv_block(input_layer, filters, strides=(1, 1)):
    x = layers.Conv2D(filters, kernel_size=3, strides=strides, padding="same")(input_layer)
    x = layers.LayerNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding="same")(x)
    x = layers.LayerNormalization()(x)
    shortcut = layers.Conv2D(filters, kernel_size=1, strides=strides, padding="same")(input_layer)
    x = layers.LayerNormalization()(x)
    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x

def identity_block(input_layer, filters):
    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding="same")(input_layer)
    x = layers.LayerNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding="same")(x)
    x = layers.LayerNormalization()(x)

    x = layers.add([x, input_layer])
    x = layers.ReLU()(x)
    return x

def build_resnet34_encoder(input_shape=(128, 128, 3)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, kernel_size=3, strides=1, padding="same")(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.ReLU()(x)
    layer_configs = [3, 4, 6, 3]
    filters = 64
    for i, num_blocks in enumerate(layer_configs):
        for j in range(num_blocks):
            if j == 0 and i != 0:
                x = conv_block(x, filters, strides=2)
            else:
                x = identity_block(x, filters)
        
        filters *= 2 
    
    model = models.Model(inputs=inputs, outputs=x)
    return model


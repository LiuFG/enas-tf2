import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages

from src.common_ops import create_weight
from src.common_ops import create_bias


def drop_path(x, keep_prob):
  """Drops out a whole example hiddenstate with the specified probability."""

  batch_size = tf.shape(x)[0]
  noise_shape = [batch_size, 1, 1, 1]
  random_tensor = keep_prob
  random_tensor += tf.random_uniform(noise_shape, dtype=tf.float32)
  binary_tensor = tf.floor(random_tensor)
  x = tf.div(x, keep_prob) * binary_tensor   # 对应位置相除

  return x


def conv(x, filter_size, out_filters, stride, name="conv", padding="SAME",
         data_format="NHWC", seed=None):
  """
  Args:
    stride: [h_stride, w_stride].
  """

  if data_format == "NHWC":
    actual_data_format = "channels_last"
  elif data_format == "NCHW":
    actual_data_format = "channels_first"
  else:
    raise NotImplementedError("Unknown data_format {}".format(data_format))
  x = tf.layers.conv2d(
      x, out_filters, [filter_size, filter_size], stride, padding,
      data_format=actual_data_format,
      kernel_initializer=tf.contrib.keras.initializers.he_normal(seed=seed))

  return x


def fully_connected(x, out_size, name="fc", seed=None):
  in_size = x.get_shape()[-1].value
  with tf.variable_scope(name):
    w = create_weight("w", [in_size, out_size], seed=seed)
  x = tf.matmul(x, w)
  return x


def max_pool(x, k_size, stride, padding="SAME", data_format="NHWC",
             keep_size=False):
  """
  Args:
    k_size: two numbers [h_k_size, w_k_size].
    stride: two numbers [h_stride, w_stride].
  """

  if data_format == "NHWC":
    actual_data_format = "channels_last"
  elif data_format == "NCHW":
    actual_data_format = "channels_first"
  else:
    raise NotImplementedError("Unknown data_format {}".format(data_format))
  out = tf.layers.max_pooling2d(x, k_size, stride, padding,
                                data_format=actual_data_format)

  if keep_size:
    if data_format == "NHWC":
      h_pad = (x.get_shape()[1].value - out.get_shape()[1].value) // 2
      w_pad = (x.get_shape()[2].value - out.get_shape()[2].value) // 2
      out = tf.pad(out, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]])
    elif data_format == "NCHW":
      h_pad = (x.get_shape()[2].value - out.get_shape()[2].value) // 2
      w_pad = (x.get_shape()[3].value - out.get_shape()[3].value) // 2
      out = tf.pad(out, [[0, 0], [0, 0], [h_pad, h_pad], [w_pad, w_pad]])
    else:
      raise NotImplementedError("Unknown data_format {}".format(data_format))
  return out


def global_avg_pool(x, data_format="NHWC"):
  if data_format == "NHWC":
    x = tf.reduce_mean(x, [1, 2])
  elif data_format == "NCHW":
    x = tf.reduce_mean(x, [2, 3])
  else:
    raise NotImplementedError("Unknown data_format {}".format(data_format))
  return x


def relu(x, leaky=0.0):
  return tf.where(tf.greater(x, 0), x, x * leaky)

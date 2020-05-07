import numpy as np
import tensorflow as tf

# @tf.function
def lstm(x, prev_c, prev_h, w):
  ifog = tf.matmul(tf.concat([x, prev_h], axis=1), w)
  i, f, o, g = tf.split(ifog, 4, axis=1)
  i = tf.sigmoid(i)  # 输入门
  f = tf.sigmoid(f)  # 遗忘门
  o = tf.sigmoid(o)  # 输出门
  g = tf.tanh(g)     # 标准RNN
  next_c = i * g + f * prev_c   # C为细胞状态
  next_h = o * tf.tanh(next_c)  # h为之前的输出
  return next_c, next_h


def stack_lstm(x, prev_c, prev_h, w):
  next_c, next_h = [], []
  for layer_id, (_c, _h, _w) in enumerate(zip(prev_c, prev_h, w)):
    inputs = x if layer_id == 0 else next_h[-1]
    curr_c, curr_h = lstm(inputs, _c, _h, _w)
    next_c.append(curr_c)
    next_h.append(curr_h)
  return next_c, next_h

def create_weight(name, shape, initializer=None, trainable=True, seed=None):
  if initializer is None:
    initializer = tf.keras.initializers.he_normal(seed=seed)
  return tf.Variable(name=name, shape=shape, initial_value=initializer(shape), trainable=trainable)

def create_bias(name, shape, initializer=None):
  if initializer is None:
    initializer = tf.constant_initializer(0.0, dtype=tf.float32)
  return tf.get_variable(name, shape, initializer=initializer)


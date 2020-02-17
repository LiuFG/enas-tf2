#-*- coding : utf-8 -*-
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import tensorflow as tf
import time
from src.cifar10.models import Model
from src.cifar10.image_ops import conv
from src.cifar10.image_ops import fully_connected
from src.cifar10.image_ops import relu
from src.cifar10.image_ops import max_pool
from src.cifar10.image_ops import global_avg_pool

from src.utils import count_model_params
from src.utils import get_train_ops
from src.common_ops import create_weight


class GeneralChild(Model):
  def __init__(self,
               data_path=None,
               cutout_size=None,
               whole_channels=False,
               fixed_arc=None,
               out_filters_scale=1,
               num_layers=2,
               num_branches=6,
               out_filters=24,
               keep_prob=1.0,
               train_batch_size=32,
               eval_batch_size=32,
               clip_mode=None,
               grad_bound=None,
               l2_reg=1e-4,
               lr_init=0.1,
               lr_dec_start=0,
               lr_dec_every=10000,
               lr_dec_rate=0.1,
               lr_cosine=False,
               lr_max=None,
               lr_min=None,
               lr_T_0=None,
               lr_T_mul=None,
               optim_algo=None,
               sync_replicas=False,
               num_aggregate=None,
               num_replicas=None,
               data_format="NHWC",
               name="child",
               *args,
               **kwargs
              ):
    super(self.__class__, self).__init__(
      data_path=data_path,
      cutout_size=cutout_size,
      train_batch_size=train_batch_size,
      eval_batch_size=eval_batch_size,
      clip_mode=clip_mode,
      grad_bound=grad_bound,
      l2_reg=l2_reg,
      lr_init=lr_init,
      lr_dec_start=lr_dec_start,
      lr_dec_every=lr_dec_every,
      lr_dec_rate=lr_dec_rate,
      keep_prob=keep_prob,
      optim_algo=optim_algo,
      sync_replicas=sync_replicas,
      num_aggregate=num_aggregate,
      num_replicas=num_replicas,
      data_format=data_format,
      name=name,
      fixed_arc=fixed_arc)

    self.whole_channels = whole_channels
    self.lr_cosine = lr_cosine
    self.lr_max = lr_max
    self.lr_min = lr_min
    self.lr_dec_min = None
    self.lr_T_0 = lr_T_0
    self.lr_T_mul = lr_T_mul
    self.out_filters_ori = out_filters * out_filters_scale
    self.num_layers = num_layers

    self.num_branches = num_branches
    self.fixed_arc = fixed_arc
    self.out_filters_scale = out_filters_scale

    pool_distance = self.num_layers // 3
    self.pool_layers = [pool_distance - 1, 2 * pool_distance - 1]
    self.zero_init = tf.initializers.zeros()
    self.one_init = tf.initializers.ones()
    self.weight = {}
    self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
      lr_init,
      decay_steps=self.lr_dec_every,
      decay_rate=self.lr_dec_rate,
      staircase=True)

    if self.optim_algo == "momentum":
      self.opt = tf.keras.optimizers.RMSprop(learning_rate=self.lr_schedule, momentum=0.9)
    elif self.optim_algo  == "sgd":
      self.opt = tf.keras.optimizers.SGD(learning_rate=self.lr_schedule, momentum=0.9)
    elif self.optim_algo  == "adam":
      self.opt = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)  # beta1=0.0, epsilon=1e-3
    else:
      raise ValueError("Unknown optim_algo {}".format(self.optim_algo ))

    if self.whole_channels:
      self.start_idx = 0
    else:
      self.start_idx = self.num_branches


  def _get_C(self, x):
    """
    Args:
      x: tensor of shape [N, H, W, C] or [N, C, H, W]
    """
    if self.data_format == "NHWC":
      return x.get_shape()[3]
    elif self.data_format == "NCHW":
      return x.get_shape()[1]
    else:
      raise ValueError("Unknown data_format '{0}'".format(self.data_format))

  def _get_HW(self, x):
    """
    Args:
      x: tensor of shape [N, H, W, C] or [N, C, H, W]
    """
    return x.get_shape()[2].value

  def _get_strides(self, stride):
    """
    Args:
      x: tensor of shape [N, H, W, C] or [N, C, H, W]
    """
    if self.data_format == "NHWC":
      return [1, stride, stride, 1]
    elif self.data_format == "NCHW":
      return [1, 1, stride, stride]
    else:
      raise ValueError("Unknown data_format '{0}'".format(self.data_format))

  def _factorized_reduction(self, x, out_filters, stride, is_training, scope=None):
    """Reduces the shape of x without information loss due to striding."""
    assert out_filters % 2 == 0, (
        "Need even number of filters when using this factorized reduction.")
    if stride == 1:
      scopet = scope + "/path_conv"
      inp_c = self._get_C(x)
      if scopet+'_w' not in self.weight.keys() and is_training:
        self.weight[scopet+'/w'] = create_weight(scopet+'/w', [1, 1, inp_c, out_filters])
      w = self.weight[scopet+'/w']
      x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME",
                       data_format=self.data_format)
      x = self.batch_norm(x, is_training, data_format=self.data_format, scope=scopet)
      return x

    stride_spec = self._get_strides(stride)
    # Skip path 1
    path1 = tf.nn.avg_pool(
        x, [1, 1, 1, 1], stride_spec, "VALID", data_format=self.data_format)
    scopet = scope + "/path1_conv"
    inp_c = self._get_C(path1)
    if scopet+'/w' not in self.weight.keys() and is_training:
      self.weight[scopet+'/w'] = create_weight(scopet+"/w", [1, 1, inp_c, out_filters // 2])
    w = self.weight[scopet+'/w']
    path1 = tf.nn.conv2d(path1, w, [1, 1, 1, 1], "SAME",
                         data_format=self.data_format)
  
    # Skip path 2
    # First pad with 0"s on the right and bottom, then shift the filter to
    # include those 0"s that were added.
    if self.data_format == "NHWC":
      pad_arr = [[0, 0], [0, 1], [0, 1], [0, 0]]
      path2 = tf.pad(x, pad_arr)[:, 1:, 1:, :]
      concat_axis = 3
    else:
      pad_arr = [[0, 0], [0, 0], [0, 1], [0, 1]]
      path2 = tf.pad(x, pad_arr)[:, :, 1:, 1:]
      concat_axis = 1
  
    path2 = tf.nn.avg_pool(
        path2, [1, 1, 1, 1], stride_spec, "VALID", data_format=self.data_format)

    scopet = scope + "/path2_conv"
    inp_c = self._get_C(path2)
    if scopet+'/w' not in self.weight.keys() and is_training:
      self.weight[scopet+'/w'] = create_weight(scopet+"/w", [1, 1, inp_c, out_filters // 2])
    w = self.weight[scopet+'/w']
    path2 = tf.nn.conv2d(path2, w, [1, 1, 1, 1], "SAME",
                         data_format=self.data_format)
  
    # Concat and apply BN
    final_path = tf.concat(values=[path1, path2], axis=concat_axis)
    final_path = self.batch_norm(final_path, is_training,
                            data_format=self.data_format, scope=scope+'/final_path')

    return final_path

  def _get_C(self, x):
    """
    Args:
      x: tensor of shape [N, H, W, C] or [N, C, H, W]
    """
    if self.data_format == "NHWC":
      return x.get_shape()[3]
    elif self.data_format == "NCHW":
      return x.get_shape()[1]
    else:
      raise ValueError("Unknown data_format '{0}'".format(self.data_format))

  def batch_norm(self, x, is_training, scope, name="bn", decay=0.9, epsilon=1e-5,
                 data_format="NHWC"):
    if data_format == "NHWC":
      shape = [x.get_shape()[3]]
    elif data_format == "NCHW":
      shape = [x.get_shape()[1]]
    else:
      raise NotImplementedError("Unknown data_format {}".format(data_format))

    if scope + "/bn" + "/offset" not in self.weight.keys() and is_training:
      self.weight[scope + "/bn" + "/offset"] = \
        create_weight(scope + "/bn" + "/offset", shape, initializer=self.zero_init)
      self.weight[scope + "/bn" + "/scale"] = \
        create_weight(scope + "/bn" + "/scale", shape, initializer=self.one_init)
      self.weight[scope + "/bn" + "/moving_mean"] = tf.zeros(shape)
      self.weight[scope + "/bn" + "/moving_variance"] = tf.ones(shape)

    offset = self.weight[scope + "/bn" + "/offset"]
    scale = self.weight[scope + "/bn" + "/scale"]
    moving_mean = self.weight[scope + "/bn" + "/moving_mean"]
    moving_variance = self.weight[scope + "/bn" + "/moving_variance"]
    if True:
      x, mean, variance = tf.compat.v1.nn.fused_batch_norm(
        x, scale, offset, epsilon=epsilon, data_format=data_format,
        is_training=True)
      moving_mean -= (1 - decay) * (moving_mean - mean)
      moving_variance -= (1 - decay) * (moving_variance - mean)
      self.weight[scope + "/bn" + "/moving_mean"] = moving_mean
      self.weight[scope + "/bn" + "/moving_variance"] = moving_variance
    else:
      x, _, _ = tf.compat.v1.nn.fused_batch_norm(x, scale, offset, mean=moving_mean,
                                                 variance=moving_variance,
                                                 epsilon=epsilon, data_format=data_format,
                                                 is_training=False)
    return x

  def _model(self, images, is_training):
    layers = []
    self.start_idx = 0
    self.out_filters = self.out_filters_ori
    # stem_conv
    if self.curr_step == 0 and is_training:
      self.stem_conv = create_weight("stem_conv/w", [3, 3, 3, self.out_filters])
    x = tf.nn.conv2d(images, self.stem_conv, [1, 1, 1, 1], "SAME", data_format=self.data_format)
    x = self.batch_norm(x, is_training, 'stem_conv', data_format=self.data_format)
    layers.append(x)

    for layer_id in range(0, self.num_layers):
      scope = "layer_{0}".format(layer_id)
      if self.fixed_arc is None:
        x = self._enas_layer(layer_id, layers, self.start_idx, self.out_filters, is_training, scope=scope)
      else:
        x = self._fixed_layer(layer_id, layers, self.start_idx, self.out_filters, is_training)
      layers.append(x)
      if layer_id in self.pool_layers:
        if self.fixed_arc is not None:
          self.out_filters *= 2
        scopetmp = scope + "/pool_at_{0}".format(layer_id)
        pooled_layers = []
        for i, layer in enumerate(layers):
          scopett = scopetmp + "_from_{0}".format(i)
          x = self._factorized_reduction(
              layer, self.out_filters, 2, is_training, scope=scopett)
          pooled_layers.append(x)
        layers = pooled_layers

    x = global_avg_pool(x, data_format=self.data_format)
    if is_training:
      x = tf.nn.dropout(x, 1 - self.keep_prob)

    scopetmp = 'fc'
    if self.data_format == "NHWC":
      inp_c = x.get_shape()[3]
    elif self.data_format == "NCHW":
      inp_c = x.get_shape()[1]
    else:
      raise ValueError("Unknown data_format {0}".format(self.data_format))
    if scopetmp+'/w' not in self.weight.keys() and is_training:
      self.weight[scopetmp+'/w'] = create_weight(scopetmp+'/w', [inp_c, 10])
    w = self.weight[scopetmp+'/w']
    x = tf.matmul(x, w)
    return x

  def _enas_layer(self, layer_id, prev_layers, df, out_filters, is_training, scope=None):
    """
    Args:
      layer_id: current layer
      prev_layers: cache of previous layers. for skip connections
      start_idx: where to start looking at. technically, we can infer this
        from layer_id, but why bother...
      is_training: for batch_norm
    """
    inputs = prev_layers[-1]
    if self.whole_channels:
      if self.data_format == "NHWC":
        inp_h = inputs.get_shape()[1]
        inp_w = inputs.get_shape()[2]
        inp_c = inputs.get_shape()[3]
      elif self.data_format == "NCHW":
        inp_c = inputs.get_shape()[1]
        inp_h = inputs.get_shape()[2]
        inp_w = inputs.get_shape()[3]

      count = self.sample_arc[self.start_idx]
      if self.curr_step==0 and is_training:
        self.create_conv_param(inputs.get_shape(), scope=scope)

      scopetmp = scope + "/branch_{}".format(count)
      if count == 0 or count == 2:
        out = self._conv_branch(inputs, count, is_training, out_filters, out_filters,
                                start_idx=0, scope=scopetmp)
      elif count == 1 or count == 3:
        out = self._conv_branch(inputs, count, is_training, out_filters, out_filters,
                                start_idx=0, scope=scopetmp, separable=True)
      elif count == 4:
        out = self._pool_branch(inputs, is_training, out_filters, "avg",
                              start_idx=0, scope=scopetmp)
      elif count == 5:
        out = self._pool_branch(inputs, is_training, out_filters, "max",
                              start_idx=0,  scope=scopetmp)
      else:
        assert False, 'count error'

      if self.data_format == "NHWC":
        # out.set_shape([None, inp_h, inp_w, out_filters])
        out = tf.reshape(out, (-1, inp_h, inp_w, out_filters))
      elif self.data_format == "NCHW":
        # out.set_shape([None, out_filters, inp_h, inp_w])
        out = tf.reshape(out, (-1, out_filters, inp_h, inp_w))
    else:
      count = self.sample_arc[start_idx:start_idx + 2 * self.num_branches]
      branches = []
      with tf.variable_scope("branch_0"):
        branches.append(self._conv_branch(inputs, 3, is_training, count[1],
                                          out_filters, start_idx=count[0]))
      with tf.variable_scope("branch_1"):
        branches.append(self._conv_branch(inputs, 3, is_training, count[3],
                                          out_filters, start_idx=count[2],
                                          separable=True))
      with tf.variable_scope("branch_2"):
        branches.append(self._conv_branch(inputs, 5, is_training, count[5],
                                          out_filters, start_idx=count[4]))
      with tf.variable_scope("branch_3"):
        branches.append(self._conv_branch(inputs, 5, is_training, count[7],
                                          out_filters, start_idx=count[6],
                                          separable=True))
      if self.num_branches >= 5:
        with tf.variable_scope("branch_4"):
          branches.append(self._pool_branch(inputs, is_training, count[9],
                                            "avg", start_idx=count[8]))
      if self.num_branches >= 6:
        with tf.variable_scope("branch_5"):
          branches.append(self._pool_branch(inputs, is_training, count[11],
                                            "max", start_idx=count[10]))

      with tf.variable_scope("final_conv"):
        w = create_weight("w", [self.num_branches * out_filters, out_filters])
        w_mask = tf.constant([False] * (self.num_branches * out_filters), tf.bool)
        new_range = tf.range(0, self.num_branches * out_filters, dtype=tf.int32)
        for i in range(0,self.num_branches):
          start = out_filters * i + count[2 * i]
          new_mask = tf.logical_and(
            start <= new_range, new_range < start + count[2 * i + 1])
          w_mask = tf.logical_or(w_mask, new_mask)
        w = tf.boolean_mask(w, w_mask)
        w = tf.reshape(w, [1, 1, -1, out_filters])

        inp = prev_layers[-1]
        if self.data_format == "NHWC":
          branches = tf.concat(branches, axis=3)
        elif self.data_format == "NCHW":
          branches = tf.concat(branches, axis=1)
          N = tf.shape(inp)[0]
          H = inp.get_shape()[2].value
          W = inp.get_shape()[3].value
          branches = tf.reshape(branches, [N, -1, H, W])
        out = tf.nn.conv2d(
          branches, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
        out = batch_norm(out, is_training, data_format=self.data_format)
        out = tf.nn.relu(out)

    if layer_id > 0:
      if self.whole_channels:
        skip_start = self.start_idx + 1
      else:
        skip_start = self.start_idx + 2 * self.num_branches
      skip = self.sample_arc[skip_start: skip_start + layer_id]

      scopetmp = scope + '/skip'
      res_layers = []
      for i in range(layer_id):
        if skip[i] == 1:
          res_layers.append(prev_layers[i])  #TODO +1
      res_layers.append(out)
      out = tf.add_n(res_layers)   # 所有相加
      out = self.batch_norm(out, is_training, data_format=self.data_format, scope=scopetmp)

    if self.whole_channels:
        self.start_idx += 1 + layer_id
    else:
        self.start_idx += 2 * self.num_branches + layer_id
    return out

  def _fixed_layer(self, layer_id, prev_layers, start_idx, out_filters, is_training):
    """
    Args:
      layer_id: current layer
      prev_layers: cache of previous layers. for skip connections
      start_idx: where to start looking at. technically, we can infer this
        from layer_id, but why bother...
      is_training: for batch_norm
    """

    inputs = prev_layers[-1]
    if self.whole_channels:
      if self.data_format == "NHWC":
        inp_c = inputs.get_shape()[3].value
      elif self.data_format == "NCHW":
        inp_c = inputs.get_shape()[1].value

      count = self.sample_arc[start_idx]
      if count in [0, 1, 2, 3]:
        size = [3, 3, 5, 5]
        filter_size = size[count]
        with tf.variable_scope("conv_1x1"):
          w = create_weight("w", [1, 1, inp_c, out_filters])
          out = tf.nn.relu(inputs)
          out = tf.nn.conv2d(out, w, [1, 1, 1, 1], "SAME",
                             data_format=self.data_format)
          out = batch_norm(out, is_training, data_format=self.data_format)

        with tf.variable_scope("conv_{0}x{0}".format(filter_size)):
          w = create_weight("w", [filter_size, filter_size, out_filters, out_filters])
          out = tf.nn.relu(out)
          out = tf.nn.conv2d(out, w, [1, 1, 1, 1], "SAME",
                             data_format=self.data_format)
          out = self.batch_norm(out, is_training, data_format=self.data_format)
      elif count == 4:
        pass
      elif count == 5:
        pass
      else:
        raise ValueError("Unknown operation number '{0}'".format(count))
    else:
      count = (self.sample_arc[start_idx:start_idx + 2*self.num_branches] *
               self.out_filters_scale)
      branches = []
      total_out_channels = 0
      with tf.variable_scope("branch_0"):
        total_out_channels += count[1]
        branches.append(self._conv_branch(inputs, 3, is_training, count[1]))
      with tf.variable_scope("branch_1"):
        total_out_channels += count[3]
        branches.append(
          self._conv_branch(inputs, 3, is_training, count[3], separable=True))
      with tf.variable_scope("branch_2"):
        total_out_channels += count[5]
        branches.append(self._conv_branch(inputs, 5, is_training, count[5]))
      with tf.variable_scope("branch_3"):
        total_out_channels += count[7]
        branches.append(
          self._conv_branch(inputs, 5, is_training, count[7], separable=True))
      if self.num_branches >= 5:
        with tf.variable_scope("branch_4"):
          total_out_channels += count[9]
          branches.append(
            self._pool_branch(inputs, is_training, count[9], "avg"))
      if self.num_branches >= 6:
        with tf.variable_scope("branch_5"):
          total_out_channels += count[11]
          branches.append(
            self._pool_branch(inputs, is_training, count[11], "max"))

      with tf.variable_scope("final_conv"):
        w = create_weight("w", [1, 1, total_out_channels, out_filters])
        if self.data_format == "NHWC":
          branches = tf.concat(branches, axis=3)
        elif self.data_format == "NCHW":
          branches = tf.concat(branches, axis=1)
        out = tf.nn.relu(branches)
        out = tf.nn.conv2d(out, w, [1, 1, 1, 1], "SAME",
                           data_format=self.data_format)
        out = batch_norm(out, is_training, data_format=self.data_format)

    if layer_id > 0:
      if self.whole_channels:
        skip_start = start_idx + 1
      else:
        skip_start = start_idx + 2 * self.num_branches
      skip = self.sample_arc[skip_start: skip_start + layer_id]
      total_skip_channels = np.sum(skip) + 1

      res_layers = []
      for i in range(layer_id):
        if skip[i] == 1:
          res_layers.append(prev_layers[i])
      prev = res_layers + [out]

      if self.data_format == "NHWC":
        prev = tf.concat(prev, axis=3)
      elif self.data_format == "NCHW":
        prev = tf.concat(prev, axis=1)

      out = prev
      with tf.variable_scope("skip"):
        w = create_weight(
          "w", [1, 1, total_skip_channels * out_filters, out_filters])
        out = tf.nn.relu(out)
        out = tf.nn.conv2d(
          out, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
        out = batch_norm(out, is_training, data_format=self.data_format)

    return out

  def create_bn_param(self, scopetmp):
    self.weight[scopetmp + "/bn" + "/offset"] = \
      create_weight(scopetmp + "/bn" + "/offset", [self.out_filters], initializer=self.zero_init)
    self.weight[scopetmp + "/bn" + "/scale"] = \
      create_weight(scopetmp + "/bn" + "/scale", [self.out_filters], initializer=self.one_init)
    self.weight[scopetmp + "/bn" + "/moving_mean"] = tf.zeros([self.out_filters])
    self.weight[scopetmp + "/bn" + "/moving_variance"] = tf.ones([self.out_filters])

  def create_conv_param(self, shape, scope=None):
    if self.data_format == "NHWC":
      inp_c = shape[3]
    elif self.data_format == "NCHW":
      inp_c = shape[1]
    else:
      assert False, 'data_format error'

    size = [3, 3, 5, 5]
    for id, filter_size in enumerate(size):
      scopetmp = scope + '/branch_{}'.format(id) + "/inp_conv_1"
      self.weight[scopetmp + "/w"] = create_weight(scopetmp + "/w", [1, 1, inp_c, self.out_filters])
      self.create_bn_param(scopetmp)
      if id == 0 or id == 2:
        scope0 = scope + '/branch_{}'.format(id) + "/out_conv_{}".format(filter_size)
        self.weight[scope0 + "/w"] = create_weight(scope0 + "/w", [filter_size, filter_size, self.out_filters, self.out_filters])
        self.create_bn_param(scope0)
      else:
        scopetmp = scope + '/branch_{}'.format(id) + "/separable_conv"
        self.weight[scopetmp + "/w_depth"] = \
          create_weight(scopetmp + "/w_depth", [filter_size, filter_size, self.out_filters, 1])
        self.weight[scopetmp + "/w_point"] = \
          create_weight(scopetmp + "/w_point", [self.out_filters, self.out_filters * 1])
        self.create_bn_param(scopetmp)

    # ======= pool_branch ========
    scopetmp = scope + '/branch_4' + "/conv_1"
    self.weight[scopetmp + '/w'] = create_weight(scopetmp+"w", [1, 1, inp_c, self.out_filters])
    self.create_bn_param(scopetmp)
    scopetmp = scope + '/branch_5' + "/conv_1"
    self.weight[scopetmp + '/w'] = create_weight(scopetmp+"w", [1, 1, inp_c, self.out_filters])
    self.create_bn_param(scopetmp)

  def _conv_branch(self, inputs, id, is_training, count, out_filters,
                   ch_mul=1, start_idx=None, separable=False, scope=None):
    """
    Args:
      start_idx: where to start taking the output channels. if None, assuming
        fixed_arc mode
      count: how many output_channels to take.
    """

    if start_idx is None:
      assert self.fixed_arc is not None, "you screwed up!"

    size = [3, 3, 5, 5]
    filter_size = size[id]
    scopetmp = scope+"/inp_conv_1"
    w = self.weight[scopetmp+"/w"]
    x = tf.nn.conv2d(inputs, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
    x = self.batch_norm(x, is_training, data_format=self.data_format, scope=scopetmp)
    x = tf.nn.relu(x)

    if start_idx is None:
      if separable:
        if self.curr_step == 0 and is_training:
          w_depth = create_weight(
            scope +"w_depth", [self.filter_size, self.filter_size, out_filters, ch_mul])
          w_point = create_weight(scope + "w_point", [1, 1, out_filters * ch_mul, count])
        x = tf.nn.separable_conv2d(x, w_depth, w_point, strides=[1, 1, 1, 1],
                                   padding="SAME", data_format=self.data_format)
        x = self.batch_norm(x, is_training, data_format=self.data_format)
      else:
        if self.curr_step == 0 and is_training:
          w = create_weight("w", [filter_size, filter_size, inp_c, count])
        x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
        x = self.batch_norm(x, is_training, data_format=self.data_format)
    else:
      if separable:
        scope0 = scope + "/separable_conv"
        w_depth = self.weight[scope0 + "/w_depth"]
        w_point = self.weight[scope0 + "/w_point"]
        w_point = tf.reshape(w_point, [1, 1, out_filters * ch_mul, count])

        x = tf.nn.separable_conv2d(x, w_depth, w_point, strides=[1, 1, 1, 1],
                                   padding="SAME", data_format=self.data_format)
        mask = tf.range(0, out_filters, dtype=tf.int32)
        mask = tf.logical_and(start_idx <= mask, mask < start_idx + count)
        x = self.batch_norm_with_mask(
          x, is_training, mask, out_filters, data_format=self.data_format, scope=scope0)
      else:
        scope0 = scope + "/out_conv_{}".format(filter_size)
        w = self.weight[scope0 + "/w"]
        # w = tf.transpose(w, [3, 0, 1, 2])
        # w = w[start_idx:start_idx+count, :, :, :]
        # w = tf.transpose(w, [1, 2, 3, 0])
        x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
        mask = tf.range(0, out_filters, dtype=tf.int32)
        mask = tf.logical_and(start_idx <= mask, mask < start_idx + count)
        x = self.batch_norm_with_mask(
          x, is_training, mask, out_filters, data_format=self.data_format, scope=scope0)
    x = tf.nn.relu(x)
    return x

  def batch_norm_with_mask(self, x, is_training, mask, num_channels, name="bn",
                           decay=0.9, epsilon=1e-3, data_format="NHWC", scope=None):
    # shape = [num_channels]
    # indices = tf.where(mask)
    # indices = tf.cast(indices, dtype=tf.int32)
    # indices = tf.reshape(indices, [-1])

    offset = self.weight[scope + "/bn" + "/offset"]
    scale = self.weight[scope + "/bn" + "/scale"]
    # offset = tf.boolean_mask(offset, mask)
    # scale = tf.boolean_mask(scale, mask)
    moving_mean = self.weight[scope + "/bn" + "/moving_mean"]
    moving_variance = self.weight[scope + "/bn" + "/moving_variance"]

    if True:
      x, mean, variance = tf.compat.v1.nn.fused_batch_norm(
        x, scale, offset, epsilon=epsilon, data_format=data_format,
        is_training=True)
      moving_mean -= (1 - decay) * (tf.boolean_mask(moving_mean, mask) - mean)
      moving_variance -= (1 - decay) * (tf.boolean_mask(moving_variance, mask) - mean)
      self.weight[scope + "/bn" + "/moving_mean"] = moving_mean
      self.weight[scope + "/bn" + "/moving_variance"] = moving_variance
    else:
      masked_moving_mean = tf.boolean_mask(moving_mean, mask)
      masked_moving_variance = tf.boolean_mask(moving_variance, mask)
      x, _, _ = tf.compat.v1.nn.fused_batch_norm(x, scale, offset,
                                       mean=masked_moving_mean,
                                       variance=masked_moving_variance,
                                       epsilon=epsilon, data_format=data_format,
                                       is_training=False)
    return x

  def _pool_branch(self, inputs, is_training, count, avg_or_max, start_idx=None, scope=None):
    """
    Args:
      start_idx: where to start taking the output channels. if None, assuming
        fixed_arc mode
      count: how many output_channels to take.
    """
    if start_idx is None:
      assert self.fixed_arc is not None, "you screwed up!"

    if self.data_format == "NHWC":
      inp_c = inputs.get_shape()[3]
    elif self.data_format == "NCHW":
      inp_c = inputs.get_shape()[1]

    scopetmp = scope + "/conv_1"
    w = self.weight[scopetmp+'/w']
    x = tf.nn.conv2d(inputs, w, [1, 1, 1, 1], "SAME", data_format=self.data_format)
    x = self.batch_norm(x, is_training, data_format=self.data_format, scope=scopetmp)
    x = tf.nn.relu(x)

    scopetmp = scope + "/pool"
    if self.data_format == "NHWC":
      actual_data_format = "channels_last"
    elif self.data_format == "NCHW":
      actual_data_format = "channels_first"
    else:
      assert False, 'data_format error'

    if avg_or_max == "avg":
      x = tf.compat.v1.layers.average_pooling2d(
        x, [3, 3], [1, 1], "SAME", data_format=actual_data_format)
    elif avg_or_max == "max":
      x = tf.compat.v1.layers.max_pooling2d(
        x, [3, 3], [1, 1], "SAME", data_format=actual_data_format)
    else:
      raise ValueError("Unknown pool {}".format(avg_or_max))

    if start_idx is not None:
      if self.data_format == "NHWC":
        x = x[:, :, :, start_idx : start_idx+count]
      elif self.data_format == "NCHW":
        x = x[:, start_idx : start_idx+count, :, :]

    return x

  # override
  def _build_train(self, img, label, step):
    self.curr_step = step
    with tf.GradientTape() as tape:
      tape.watch(self.variables)
      logits = self._model(img, is_training=True)
      log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
      self.loss = tf.reduce_mean(log_probs)
      self.backward(tape)

    train_preds = tf.argmax(logits, axis=1)
    train_preds = tf.cast(train_preds, dtype=tf.int32)
    self.train_acc = tf.equal(train_preds, label)
    self.train_acc = tf.cast(self.train_acc, dtype=tf.int32)
    self.train_acc = tf.reduce_sum(self.train_acc)

  def backward(self, tape, step=0):
    if self.l2_reg > 0:
      l2_losses = []
      for var in self.trainable_variables:
        l2_losses.append(tf.reduce_sum(var ** 2))
      l2_loss = tf.add_n(l2_losses)
      self.loss += self.l2_reg * l2_loss

    grads = tape.gradient(self.loss, self.trainable_variables)

    self.grad_norm = tf.linalg.global_norm(grads)
    # if self.clip_mode is not None:
    #   assert self.grad_bound is not None, "Need grad_bound to clip gradients."
    #   if self.clip_mode == "global":
    #     grads, _ = tf.clip_by_global_norm(grads, self.grad_bound)
    #   elif self.clip_mode == "norm":
    #     clipped = []
    #     for g in grads:
    #       if g is None:
    #           continue
    #       if isinstance(g, tf.IndexedSlices):
    #         c_g = tf.clip_by_norm(g.values, self.grad_bound)
    #         c_g = tf.IndexedSlices(g.indices, c_g)
    #       else:
    #         c_g = tf.clip_by_norm(g, self.grad_bound)
    #       clipped.append(c_g)
    #     grads = clipped
    #   else:
    #     raise NotImplementedError("Unknown clip_mode {}".format(self.clip_mode))

    self.learning_rate = self.lr_schedule(self.curr_step)
    self.opt.apply_gradients(zip(grads, self.trainable_variables))


  # override
  def _build_valid(self, sample_arc):
    self.sample_arc = sample_arc
    valid_acc_list = []
    for count, (img, label) in enumerate(self.valid_dataloader):  # 原代碼好像只valid一個batch
      logits = self._model(img, False)
      valid_preds = tf.argmax(logits, axis=1)
      valid_preds = tf.cast(valid_preds, dtype=tf.int32)
      valid_acc = tf.equal(valid_preds, label)
      valid_acc = tf.cast(valid_acc, dtype=tf.int32)
      valid_acc = tf.reduce_sum(valid_acc)
      valid_acc_list.append(valid_acc/128)
      if count > self.num_valid_batches-2:
        break
    return tf.reduce_mean(valid_acc_list)

  # override
  def _build_test(self):
    logits = self._model(self.x_test, False)
    self.test_preds = tf.argmax(logits, axis=1)
    self.test_preds = tf.to_int32(self.test_preds)
    self.test_acc = tf.equal(self.test_preds, self.y_test)
    self.test_acc = tf.to_int32(self.test_acc)
    self.test_acc = tf.reduce_sum(self.test_acc)

  # override
  def build_valid_rl(self, sample_arc, shuffle=False):
    self.sample_arc = sample_arc
    for count, (img, label) in enumerate(self.valid_shuffle_dataloader):    # 原代碼好像只valid一個batch
      logits = self._model(img, False)
      break
    valid_shuffle_preds = tf.argmax(logits, axis=1)
    valid_shuffle_preds = tf.cast(valid_shuffle_preds, dtype=tf.int32)
    self.valid_shuffle_acc = tf.equal(valid_shuffle_preds, label)
    self.valid_shuffle_acc = tf.cast(self.valid_shuffle_acc, dtype=tf.int32)
    self.valid_shuffle_acc = tf.reduce_sum(self.valid_shuffle_acc)

  def connect_controller_arc(self, sample_arc):
    if self.fixed_arc is None:
      self.sample_arc = sample_arc
    else:
      fixed_arc = np.array([int(x) for x in self.fixed_arc.split(" ") if x])
      self.sample_arc = fixed_arc



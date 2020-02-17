#-*- coding : utf-8 -*-
# coding: utf-8
import sys
import os
import time

import numpy as np
import tensorflow as tf

from src.controller import Controller
from src.utils import get_train_ops
from src.common_ops import stack_lstm

from tensorflow.python.training import moving_averages

class GeneralController(tf.Module):
  def __init__(self,
               search_for="both",
               search_whole_channels=False,
               num_layers=4,
               num_branches=6,
               out_filters=48,
               lstm_size=32,
               lstm_num_layers=2,
               lstm_keep_prob=1.0,
               tanh_constant=None,
               temperature=None,
               lr_init=1e-3,
               lr_dec_start=0,
               lr_dec_every=100,
               lr_dec_rate=0.9,
               l2_reg=0,
               entropy_weight=None,
               clip_mode=None,
               grad_bound=None,
               use_critic=False,
               bl_dec=0.999,
               optim_algo="adam",
               sync_replicas=False,
               num_aggregate=None,
               num_replicas=None,
               skip_target=0.8,
               skip_weight=0.5,
               name="controller",
               *args,
               **kwargs):
    super(GeneralController, self).__init__(name='controller')
    print( "-" * 80)
    print( "Building ConvController")

    self.search_for = search_for
    self.search_whole_channels = search_whole_channels
    self.num_layers = num_layers
    self.num_branches = num_branches
    self.out_filters = out_filters

    self.lstm_size = lstm_size
    self.lstm_num_layers = lstm_num_layers 
    self.lstm_keep_prob = lstm_keep_prob
    self.tanh_constant = tanh_constant
    self.temperature = temperature
    self.lr_init = lr_init
    self.lr_dec_start = lr_dec_start
    self.lr_dec_every = lr_dec_every
    self.lr_dec_rate = lr_dec_rate
    self.l2_reg = l2_reg
    self.entropy_weight = entropy_weight
    self.clip_mode = clip_mode
    self.grad_bound = grad_bound
    self.use_critic = use_critic
    self.bl_dec = bl_dec

    self.skip_target = skip_target
    self.skip_weight = skip_weight

    self.optim_algo = optim_algo
    self.sync_replicas = sync_replicas
    self.num_aggregate = num_aggregate
    self.num_replicas = num_replicas
    self.name = name
    self.baseline = tf.Variable(0.0, dtype=tf.float32, trainable=False)
    self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
          lr_init,
          decay_steps=self.lr_dec_every,
          decay_rate=self.lr_dec_rate,
          staircase=True)

    if self.optim_algo == "momentum":
      self.opt = tf.compat.v1.train.MomentumOptimizer(
        self.lr_schedule, 0.9, use_locking=True, use_nesterov=True)
    elif optim_algo == "sgd":
      self.opt = tf.keras.optimizers.SGD(learning_rate=self.lr_schedule)
    elif optim_algo == "adam":
      # self.child_opt = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)  # beta1=0.0, epsilon=1e-3
      self.opt = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)  # beta1=0.0, epsilon=1e-3
    else:
      raise ValueError("Unknown optim_algo {}".format(optim_algo))
    self.create_para_do = True

  def _create_params(self):
    initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
    self.w_lstm = []
    for layer_id in range(self.lstm_num_layers):
      # tf.random.truncated_normal([2 * self.lstm_size, self.select_num * self.lstm_size])
      w = tf.Variable(initializer([2 * self.lstm_size, 4 * self.lstm_size]),
                      name="{}_layer_{}_w".format('lstm', layer_id))
      self.w_lstm.append(w)

    self.g_emb = tf.Variable(initializer([1, self.lstm_size]), name="g_emb")
    if self.search_whole_channels:
      self.w_emb = tf.Variable(initializer([self.num_branches, self.lstm_size]),
                                 name="emb_w")
      self.w_soft = tf.Variable(initializer([self.lstm_size, self.num_branches]),
                                name="softmax_w")
    else:
      self.w_emb = {"start": [], "count": []}
      for branch_id in range(0, self.num_branches):
        with tf.variable_scope("branch_{}".format(branch_id)):
          self.w_emb["start"].append(tf.Variable(initializer([self.out_filters, self.lstm_size]),
                                   name="w_strart_{}".format(branch_id)))
          self.w_emb["count"].append(tf.Variable(initializer([self.out_filters - 1, self.lstm_size]),
                                                 name="w_count_{}".format(branch_id)))

      self.w_soft = {"start": [], "count": []}
      with tf.variable_scope("softmax"):
        for branch_id in range(0,self.num_branches):
          with tf.variable_scope("branch_{}".format(branch_id)):
            self.w_soft["start"].append(tf.get_variable(
              "w_start", [self.lstm_size, self.out_filters]))
            self.w_soft["count"].append(tf.get_variable(
              "w_count", [self.lstm_size, self.out_filters - 1]))

    self.w_attn_1 = tf.Variable(initializer([self.lstm_size, self.lstm_size]),
                                name="attention_w1")
    self.w_attn_2 = tf.Variable(initializer([self.lstm_size, self.lstm_size]),
                                name="attention_w2")
    self.v_attn = tf.Variable(initializer([self.lstm_size, 1]), name="attention_v")

  def _build_sampler(self):
    """Build the sampler ops and the log_prob ops."""
    # print("-" * 80)
    # print("Build controller sampler")
    if self.create_para_do:
      self._create_params()
      self.create_para_do = False
    anchors = []
    anchors_w_1 = []

    arc_seq = []
    entropys = []
    log_probs = []
    skip_count = []
    skip_penaltys = []

    prev_c = [tf.zeros([1, self.lstm_size], tf.float32) for _ in
              range(0, self.lstm_num_layers)]
    prev_h = [tf.zeros([1, self.lstm_size], tf.float32) for _ in
              range(0, self.lstm_num_layers)]
    inputs = self.g_emb
    skip_targets = tf.constant([1.0 - self.skip_target, self.skip_target],
                               dtype=tf.float32)

    for layer_id in range(0, self.num_layers):
      if self.search_whole_channels:
        timetmp0 = time.time()
        next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
        ddd = time.time() - timetmp0
        prev_c, prev_h = next_c, next_h
        logit = tf.matmul(next_h[-1], self.w_soft)  # w_soft决定输出5个
        if self.temperature is not None:
          logit /= self.temperature
        if self.tanh_constant is not None:
          logit = self.tanh_constant * tf.tanh(logit)
        if self.search_for == "macro" or self.search_for == "branch":
          branch_id = tf.random.categorical(logit, 1, dtype=tf.int32)
          branch_id = tf.reshape(branch_id, [1])
        elif self.search_for == "connection":
          branch_id = tf.constant([0], dtype=tf.int32)
        else:
          raise ValueError("Unknown search_for {}".format(self.search_for))
        arc_seq.append(branch_id)
        log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logit, labels=branch_id)  # 计算选中branch_id的概率
        log_probs.append(log_prob)
        entropy = tf.stop_gradient(log_prob * tf.exp(-log_prob))
        entropys.append(entropy)   # 可作为reward的一部分
        inputs = tf.nn.embedding_lookup(self.w_emb, branch_id)  # embedding
      else:
        for branch_id in range(0, self.num_branches):
          next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
          prev_c, prev_h = next_c, next_h
          logit = tf.matmul(next_h[-1], self.w_soft["start"][branch_id])
          if self.temperature is not None:
            logit /= self.temperature
          if self.tanh_constant is not None:
            logit = self.tanh_constant * tf.tanh(logit)
          start = tf.multinomial(logit, 1)
          start = tf.to_int32(start)
          start = tf.reshape(start, [1])
          arc_seq.append(start)
          log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logit, labels=start)
          log_probs.append(log_prob)
          entropy = tf.stop_gradient(log_prob * tf.exp(-log_prob))
          entropys.append(entropy)
          inputs = tf.nn.embedding_lookup(self.w_emb["start"][branch_id], start)

          next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
          prev_c, prev_h = next_c, next_h
          logit = tf.matmul(next_h[-1], self.w_soft["count"][branch_id])
          if self.temperature is not None:
            logit /= self.temperature
          if self.tanh_constant is not None:
            logit = self.tanh_constant * tf.tanh(logit)
          mask = tf.range(0, limit=self.out_filters-1, delta=1, dtype=tf.int32)
          mask = tf.reshape(mask, [1, self.out_filters - 1])
          mask = tf.less_equal(mask, self.out_filters-1 - start)
          logit = tf.where(mask, x=logit, y=tf.fill(tf.shape(logit), -np.inf))
          count = tf.multinomial(logit, 1)
          count = tf.to_int32(count)
          count = tf.reshape(count, [1])
          arc_seq.append(count + 1)
          log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logit, labels=count)
          log_probs.append(log_prob)
          entropy = tf.stop_gradient(log_prob * tf.exp(-log_prob))
          entropys.append(entropy)
          inputs = tf.nn.embedding_lookup(self.w_emb["count"][branch_id], count)

      next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)  # TODO w_lstm没有更新
      prev_c, prev_h = next_c, next_h

      if layer_id > 0:
        # anchors，w_attn_2，v_attn只有在这里用   # anchors包含了所有输出
        query = tf.concat(anchors_w_1, axis=0)
        query = tf.tanh(query + tf.matmul(next_h[-1], self.w_attn_2))
        query = tf.matmul(query, self.v_attn)
        logit = tf.concat([-query, query], axis=1)
        if self.temperature is not None:
          logit /= self.temperature
        if self.tanh_constant is not None:
          logit = self.tanh_constant * tf.tanh(logit)

        skip = tf.random.categorical(logit, 1, dtype=tf.int32)
        skip = tf.reshape(skip, [layer_id])  # TODO 直接展成layer_id个0/1？
        arc_seq.append(skip)

        skip_prob = tf.sigmoid(logit)
        kl = skip_prob * tf.math.log(skip_prob / skip_targets)
        kl = tf.reduce_sum(kl)
        skip_penaltys.append(kl)

        log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logit, labels=skip)
        log_probs.append(tf.reduce_sum(log_prob, keepdims=True))

        entropy = tf.stop_gradient(
          tf.reduce_sum(log_prob * tf.exp(-log_prob), keepdims=True))
        entropys.append(entropy)

        skip = tf.cast(skip, dtype=tf.float32)
        skip = tf.reshape(skip, [1, layer_id])   # 前几个是否skip
        skip_count.append(tf.reduce_sum(skip))
        inputs = tf.matmul(skip, tf.concat(anchors, axis=0))
        inputs /= (1.0 + tf.reduce_sum(skip))
      else:
        inputs = self.g_emb    # TODO layer_id=1

      anchors.append(next_h[-1])   # anchors包含了所有输出
      anchors_w_1.append(tf.matmul(next_h[-1], self.w_attn_1))

    arc_seq = tf.concat(arc_seq, axis=0)
    self.sample_arc = tf.reshape(arc_seq, [-1])

    entropys = tf.stack(entropys)
    self.sample_entropy = tf.reduce_sum(entropys)

    log_probs = tf.stack(log_probs)
    self.sample_log_prob = tf.reduce_sum(log_probs)

    skip_count = tf.stack(skip_count)
    self.skip_count = tf.reduce_sum(skip_count)

    skip_penaltys = tf.stack(skip_penaltys)
    self.skip_penaltys = tf.reduce_mean(skip_penaltys)
    return self.sample_arc

  def build_trainer(self, child_model, ct_step):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      tape.watch(self.trainable_variables)
      self._build_sampler()
      child_model.build_valid_rl(self.sample_arc)   # 还是普通的valid只不过shuffle了数据，得到valid_shuffle_acc
      self.valid_acc = (tf.cast(child_model.valid_shuffle_acc,dtype=tf.float32 ) /
                        tf.cast(child_model.eval_batch_size, dtype=tf.float32))
      self.reward = self.valid_acc

      normalize = tf.cast(self.num_layers * (self.num_layers - 1) / 2, dtype=tf.float32)
      self.skip_rate = tf.cast(self.skip_count, dtype=tf.float32) / normalize

      if self.entropy_weight is not None:
        self.reward += self.entropy_weight * self.sample_entropy

      self.sample_log_prob = tf.reduce_sum(self.sample_log_prob)
      self.loss = self.sample_log_prob * (self.reward - self.baseline)
      self.baseline.assign_sub((1 - self.bl_dec) * (self.baseline - self.reward))

      if self.skip_weight is not None:
        self.loss += self.skip_weight * self.skip_penaltys

      self.backward(tape)

  def backward(self, tape, step=0):
    """
    Args:
      clip_mode: "global", "norm", or None.
      moving_average: store the moving average of parameters
    """
    if self.l2_reg > 0:
      l2_losses = []
      for var in self.trainable_variables:
        l2_losses.append(tf.reduce_sum(var ** 2))
      l2_loss = tf.add_n(l2_losses)
      self.loss += self.l2_reg * l2_loss

    # compute gradients and update variables
    gradients = tape.gradient(self.loss, self.trainable_variables)
    self.opt.apply_gradients(zip(gradients, self.trainable_variables))

  def eval_controller(self, child_model):
    self._build_sampler()  # 更新下sample_arc
    self.eval_acc = child_model._build_valid(self.sample_arc)
    return self.sample_arc

#-*- coding : utf-8 -*-
# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

# import cPickle as pickle
import shutil
import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from src.utils import Logger
from src.utils import print_user_flags

from src.cifar10.general_controller import GeneralController
from src.cifar10.general_child import GeneralChild

from src.cifar10.micro_controller import MicroController
from src.cifar10.micro_child import MicroChild
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # TODO
tf.get_logger().setLevel('ERROR')
parser = argparse.ArgumentParser(description='enas')
parser.add_argument('--reset_output_dir', default=True,
                    help='Delete output_dir if exists')  # action='store_true'
parser.add_argument('--data_path', default="/home/liufg/data/cifar/cifar10", help='')
parser.add_argument('--output_dir', default="./output/enas_tf2", help='')  # TODO
parser.add_argument('--data_format', default="NCHW", help=" 'NHWC' or 'NCWH'")
parser.add_argument('--search_for', default="macro", help=" Must be [macro|micro]")
parser.add_argument('--train_batch_size', type=int, default=128, metavar='batch_size', help='')  # TODO
parser.add_argument('--eval_batch_size', type=int, default=128, metavar='batch_size', help='')  # TODO
parser.add_argument('--num_epochs', type=int, default=300, metavar='num_epochs', help='')
parser.add_argument('--eval_every_epochs', type=int, default=1, help='How many epochs to eval')  # TODO
parser.add_argument('--controller_training', default=True, help=" 'ce' or 'focal'")

# ==================== child =========================
parser.add_argument('--child_fixed_arc', type=str, default=None, help="")
parser.add_argument('--child_num_layers', type=int, default=12, help='')
parser.add_argument('--child_filter_size', type=int, default=5, help='')
parser.add_argument('--child_out_filters', type=int, default=36, help='')
parser.add_argument('--child_out_filters_scale', type=int, default=1, help='')
parser.add_argument('--child_num_branches', type=int, default=6, help='')
parser.add_argument('--child_block_size', type=int, default=3, help='')
parser.add_argument('--child_num_cell_layers', type=int, default=5, help='')
parser.add_argument('--child_cutout_size', type=int, default=None, help='CutOut size')
parser.add_argument('--child_keep_prob', type=float, default=0.90, help='')
parser.add_argument('--child_optim', default="adam", help='momentum/sgd/adam')
parser.add_argument('--child_train_log_every', type=int, default=50, help='How many steps to log')
# ---------- decay lr -----------
parser.add_argument('--child_lr_dec_every', type=int, default=100, help='')
parser.add_argument('--child_lr_dec_rate', type=float, default=0.1, help='')
parser.add_argument('--child_lr', type=float, default=0.002, help='')  # TODO
# ---------- cosine lr ----------
parser.add_argument('--child_lr_cosine', default=False, help='Use cosine lr schedule')
parser.add_argument('--child_lr_T_0', type=int, default=10, help='for lr schedule')
parser.add_argument('--child_lr_T_mul', type=int, default=2, help='for lr schedule')
parser.add_argument('--child_lr_max', type=float, default=None, help='for lr schedule')
parser.add_argument('--child_lr_min', type=float, default=1e-4, help='for lr schedule')
# ------------ clip --------------
parser.add_argument('--child_clip_mode', default="global", help='global, norm, or None')
parser.add_argument('--child_grad_bound', type=float, default=5.0, help='Gradient clipping')
# --------- wait for test ---------  # TODO
parser.add_argument('--child_skip_pattern', type=str, default=None, help='Must be [dense, None]')
parser.add_argument('--child_sync_replicas', default=False, help='To sync or not to sync')
parser.add_argument('--child_use_aux_heads', default=True, help='To sync or not to sync')
parser.add_argument('--child_num_aggregate', type=int, default=None, help='multi device')
parser.add_argument('--child_num_replicas', type=int, default=1, help='multi device')
parser.add_argument('--child_drop_path_keep_prob', type=float, default=0.6,
                    help='minimum drop_path_keep_prob, only used in fix_layer')
parser.add_argument('--child_l2_reg', type=float, default=0.00025,
                    help='default:1e-4; apply for loss. l1-lasso, l2-ridge')
# ==================== controller ======================
parser.add_argument('--controller_train_steps', type=int, default=50, help='multi device')
parser.add_argument('--controller_log_every', type=int, default=10, help='How many steps to log')
parser.add_argument('--controller_forwards_limit', type=int, default=2, help='multi device')
parser.add_argument('--controller_train_every', type=int, default=1,  # TODO enas为1
                    help='train the controller after this number of epochs')
parser.add_argument('--controller_search_whole_channels', default=True, help='')
parser.add_argument('--controller_use_critic', default=False, help='')
# ------------- lr ---------------
parser.add_argument('--controller_lr', type=float, default=1e-3, help='')
parser.add_argument('--controller_lr_dec_rate', type=float, default=1.0, help='no use')
# ----------- get loss ------------------
parser.add_argument('--controller_entropy_weight', type=float, default=0.0001, help='loss += weight * entropy')
parser.add_argument('--controller_skip_target', type=float, default=0.4, help='for skip_penaltys')
parser.add_argument('--controller_skip_weight', type=float, default=0.8,
                    help='loss += skip_weight * skip_penaltys')
# -------- wait for test ----------   # TODO
parser.add_argument('--controller_l2_reg', type=float, default=0.0, help='')
parser.add_argument('--controller_bl_dec', type=float, default=0.99,
                    help='related to controller loss， wait to define')
# is attention?
parser.add_argument('--controller_tanh_constant', type=float, default=1.10,
                    help='tanh_constant * tf.tanh(logits)')
parser.add_argument('--controller_op_tanh_reduce', type=float, default=2.5, help='reduce tanh_constant')
parser.add_argument('--controller_temperature', type=float, default=None, help='logits /= temperature')
# ------------- multi device -----------------
parser.add_argument('--controller_num_aggregate', type=int, default=1, help='multi device')
parser.add_argument('--controller_num_replicas', type=int, default=1, help='multi device')
parser.add_argument('--controller_sync_replicas', default=False, help='To sync or not to sync')
FLAGS = parser.parse_args()


def get_ops():
  assert FLAGS.search_for is not None, "Please specify --search_for"
  if FLAGS.search_for == "micro":
    ControllerClass = MicroController
    ChildClass = MicroChild
  elif FLAGS.search_for == "macro":
    ControllerClass = GeneralController
    ChildClass = GeneralChild
  else:
    assert False, 'search_for error'

  child_model = ChildClass(
    data_path=FLAGS.data_path,
    use_aux_heads=FLAGS.child_use_aux_heads,
    cutout_size=FLAGS.child_cutout_size,
    whole_channels=FLAGS.controller_search_whole_channels,
    num_layers=FLAGS.child_num_layers,
    num_cells=FLAGS.child_num_cell_layers,
    num_branches=FLAGS.child_num_branches,
    fixed_arc=FLAGS.child_fixed_arc,
    out_filters_scale=FLAGS.child_out_filters_scale,
    out_filters=FLAGS.child_out_filters,
    keep_prob=FLAGS.child_keep_prob,
    drop_path_keep_prob=FLAGS.child_drop_path_keep_prob,
    num_epochs=FLAGS.num_epochs,
    l2_reg=FLAGS.child_l2_reg,
    data_format=FLAGS.data_format,
    train_batch_size=FLAGS.train_batch_size,
    eval_batch_size=FLAGS.eval_batch_size,
    clip_mode="norm",
    grad_bound=FLAGS.child_grad_bound,
    lr_init=FLAGS.child_lr,
    lr_dec_every=FLAGS.child_lr_dec_every,
    lr_dec_rate=FLAGS.child_lr_dec_rate,
    lr_cosine=FLAGS.child_lr_cosine,
    lr_max=FLAGS.child_lr_max,
    lr_min=FLAGS.child_lr_min,
    lr_T_0=FLAGS.child_lr_T_0,
    lr_T_mul=FLAGS.child_lr_T_mul,
    optim_algo=FLAGS.child_optim,
    sync_replicas=FLAGS.child_sync_replicas,
    num_aggregate=FLAGS.child_num_aggregate,
    num_replicas=FLAGS.child_num_replicas,
  )

  if FLAGS.child_fixed_arc is None:
    controller_model = ControllerClass(
      search_for=FLAGS.search_for,
      search_whole_channels=FLAGS.controller_search_whole_channels,
      skip_target=FLAGS.controller_skip_target,
      skip_weight=FLAGS.controller_skip_weight,
      num_cells=FLAGS.child_num_cell_layers,
      num_layers=FLAGS.child_num_layers,
      num_branches=FLAGS.child_num_branches,
      out_filters=FLAGS.child_out_filters,
      lstm_size=64,
      lstm_num_layers=1,
      lstm_keep_prob=1.0,
      tanh_constant=FLAGS.controller_tanh_constant,
      op_tanh_reduce=FLAGS.controller_op_tanh_reduce,
      temperature=FLAGS.controller_temperature,
      lr_init=FLAGS.controller_lr,
      lr_dec_start=0,
      lr_dec_every=1000000,  # never decrease learning rate
      l2_reg=FLAGS.controller_l2_reg,
      entropy_weight=FLAGS.controller_entropy_weight,
      bl_dec=FLAGS.controller_bl_dec,
      use_critic=FLAGS.controller_use_critic,
      optim_algo="adam",
      sync_replicas=FLAGS.controller_sync_replicas,
      num_aggregate=FLAGS.controller_num_aggregate,
      num_replicas=FLAGS.controller_num_replicas)
  else:
    assert not FLAGS.controller_training, (
      "--child_fixed_arc is given, cannot train controller")
    controller_model = None
  return child_model, controller_model


def train():
  child_model, controller_model = get_ops()
  writer = tf.summary.create_file_writer(FLAGS.output_dir)
  # if FLAGS.child_sync_replicas:
  #   sync_replicas_hook = child_ops["optimizer"].make_session_run_hook(True)
  #   hooks.append(sync_replicas_hook)
  # if FLAGS.controller_training and FLAGS.controller_sync_replicas:
  #   sync_replicas_hook = controller_ops["optimizer"].make_session_run_hook(True)   # 分布式
  #   hooks.append(sync_replicas_hook)

  print("-" * 80)
  print("Starting train loop")
  start_time = time.time()
  for count, (img, label) in enumerate(child_model.train_dataloader):
    # plt.imshow(np.transpose(img[1], [1,2,0]))
    # plt.show()
    step = count   # + epoch * child_model.num_train_batches
    epoch = step // child_model.num_train_batches
    timetmp = time.time()
    if step % 1 == 0:
      sample_arc = controller_model._build_sampler()
      # print(sample_arc)
      endtime = time.time() - timetmp
      child_model.connect_controller_arc(sample_arc)

    timetmp = time.time()
    child_model._build_train(img, label, step)
    endtime3 = time.time() - timetmp

    curr_time = time.time()
    if step % FLAGS.child_train_log_every == 0:
      with writer.as_default():
        tf.summary.scalar('child_train_loss', data=child_model.loss, step=step)
        tf.summary.scalar('child_train_acc', data=child_model.train_acc/128, step=step)
      log_string = ""
      log_string += "epoch={:<6d}".format(epoch)
      log_string += "ch_step={:<6d}".format(step)
      log_string += " loss={:<8.6f}".format(child_model.loss)
      log_string += " lr={:<8.4f}".format(child_model.learning_rate)
      log_string += " |g|={:<8.4f}".format(child_model.grad_norm)
      log_string += " tr_acc={:<3d}/{:>3d}".format(child_model.train_acc, FLAGS.train_batch_size)
      log_string += " mins={:<10.2f}".format(float(curr_time - start_time) / 60)
      print(log_string)
    # ================== controller_train & eval acc ==================================
    step_eval_every = child_model.num_train_batches * FLAGS.eval_every_epochs
    if (step + 1) % step_eval_every == 0:
      if (FLAGS.controller_training and (epoch+1) % FLAGS.controller_train_every == 0):
        print("Epoch {}: Training controller".format(epoch))
        # Training controller
        for ct_step in range(FLAGS.controller_train_steps * FLAGS.controller_num_aggregate):
          controller_model.build_trainer(child_model, ct_step)

          if ct_step % FLAGS.controller_log_every == 0:
            curr_time = time.time()
            log_string = ""
            log_string += "ctrl_step={:<6d}".format(ct_step)
            log_string += " loss={:<7.3f}".format(controller_model.loss)
            log_string += " ent={:<5.2f}".format(controller_model.sample_entropy)
            # log_string += " lr={:<6.4f}".format(controller_model.learning_rate)
            # log_string += " |g|={:<8.4f}".format(gn)
            log_string += " acc={:<6.4f}".format(controller_model.valid_acc)
            log_string += " bl={:<5.2f}".format(controller_model.baseline.numpy())
            log_string += " mins={:<.2f}".format(
                float(curr_time - start_time) / 60)
            print(log_string)

        # 验证acc
        print("Here are 10 architectures")
        controller_eval_acc = []
        for _ in range(10):
          arc = controller_model.eval_controller(child_model)
          if FLAGS.search_for == "micro":
            normal_arc, reduce_arc = arc
            print(np.reshape(normal_arc, [-1]))
            print(np.reshape(reduce_arc, [-1]))
          else:
            start = 0
            for layer_id in range(FLAGS.child_num_layers):
              if FLAGS.controller_search_whole_channels:
                end = start + 1 + layer_id
              else:
                end = start + 2 * FLAGS.child_num_branches + layer_id
              print(np.reshape(arc[start: end], [-1]))
              start = end
          print("val_acc={:<6.4f}".format(controller_model.eval_acc))
          controller_eval_acc.append(controller_model.eval_acc)
        controller_eval_acc = tf.reduce_mean(controller_eval_acc)
        with writer.as_default():
          tf.summary.scalar('controller_avg_eval_acc', data=controller_eval_acc, step=step)
    if epoch > FLAGS.num_epochs:
      break

      # print("Epoch {}: Eval".format(epoch))
      # if FLAGS.child_fixed_arc is None:
      #   child_model._build_valid()
      # child_model._build_test()


if __name__ == "__main__":
  print("-" * 80)
  if not os.path.isdir(FLAGS.output_dir):
    print("Path {} does not exist. Creating.".format(FLAGS.output_dir))
    os.makedirs(FLAGS.output_dir)
  elif FLAGS.reset_output_dir:
    print("Path {} exists. Remove and remake.".format(FLAGS.output_dir))
    shutil.rmtree(FLAGS.output_dir)
    os.makedirs(FLAGS.output_dir)

  print("-" * 80)
  log_file = os.path.join(FLAGS.output_dir, "stdout")
  print("Logging to {}".format(log_file))
  sys.stdout = Logger(log_file)

  print_user_flags(FLAGS)
  train()



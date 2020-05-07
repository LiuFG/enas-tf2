#!/bin/bash

export PYTHONPATH="$(pwd)"

python main_macro.py \
  --data_format="NCHW" \
  --search_for="macro" \
  --reset_output_dir=True \
  --data_path="/home/liufg/data/cifar/cifar10" \
  --output_dir="outputs" \
  --child_train_log_every=50 \
  --eval_every_epochs=1 \
  --child_use_aux_heads=Fasle \
  --child_num_layers=12 \
  --child_out_filters=36 \
  --child_l2_reg=0.00025 \
  --child_num_branches=6 \
  --child_num_cell_layers=5 \
  --child_keep_prob=0.90 \
  --child_drop_path_keep_prob=0.60 \
  --child_lr_cosine=Fasle \
  --child_lr_max=0.05 \
  --child_lr_min=0.0005 \
  --child_lr_T_0=10 \
  --child_lr_T_mul=2 \
  --controller_training=True \
  --controller_search_whole_channels=True \
  --controller_entropy_weight=0.0001 \
  --controller_train_every=1 \
  --controller_sync_replicas=1 \
  --controller_num_aggregate=20 \
  --controller_train_steps=50 \
  --controller_lr=0.001 \
  --controller_tanh_constant=1.5 \
  --controller_op_tanh_reduce=2.5 \
  --controller_skip_target=0.4 \
  --controller_skip_weight=0.8 \
  "$@"


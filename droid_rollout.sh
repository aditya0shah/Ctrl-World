#!/bin/bash

# DROID Trajectory Client script
# Connects to DROID server, generates trajectories, evaluates with RFM, and executes best one

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set GPU and memory settings (single-GPU default; omit or set multiple for multi-GPU)
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

# Run trajectory client
# Multi-GPU: add e.g. --gpu_ids "0,1,2,3,4" so each worker uses one GPU
python "${SCRIPT_DIR}/scripts/droid_trajectory_client.py" \
  --server_url tcp://jessezhang2.a.pinggy.link:25363 \
  --rfm_url http://localhost:8004 \
  --task "put plate on table" \
  --num_trajectories 5 \
  --trajectory_length 1 \
  --generation_backend mp \
  --mp_workers 5 \
  --mp_start_method spawn \
  --max_steps 12 \
  --ckpt_path "${SCRIPT_DIR}/checkpoints/Ctrl-World/checkpoint-10000.pt" \
  --pi_ckpt gs://openpi-assets/checkpoints/pi05_droid \
  --task_type pickplace \
  --fps 1.0 \
  --save_dir "${SCRIPT_DIR}/saved_trajectories"
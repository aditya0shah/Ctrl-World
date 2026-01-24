#!/bin/bash

# DROID Trajectory Client script
# Connects to DROID server, generates trajectories, evaluates with RFM, and executes best one

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set GPU and memory settings
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

# Run trajectory client
python "${SCRIPT_DIR}/scripts/droid_trajectory_client.py" \
  --server_url tcp://jessezhang2.a.pinggy.link:25363 \
  --rfm_url http://localhost:8004 \
  --task "put plate on table" \
  --num_trajectories 5 \
  --trajectory_length 1 \
  --max_steps 12 \
  --ckpt_path "${SCRIPT_DIR}/checkpoints/Ctrl-World/checkpoint-10000.pt" \
  --pi_ckpt gs://openpi-assets/checkpoints/pi05_droid \
  --task_type pickplace \
  --fps 1.0 \
  --save_dir "${SCRIPT_DIR}/saved_trajectories"
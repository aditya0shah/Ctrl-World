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
  --rfm_url http://localhost:8000 \
  --task "pick up the cup" \
  --num_trajectories 10 \
  --max_batch_size 8 \
  --trajectory_length 5 \
  --max_steps 4 \
  --ckpt_path "${SCRIPT_DIR}/checkpoints/Ctrl-World/checkpoint-10000.pt" \
  --pi_ckpt gs://openpi-assets/checkpoints/pi05_droid \
  --svd_model_path "${SCRIPT_DIR}/checkpoints/stable-video-diffusion-img2vid" \
  --clip_model_path "${SCRIPT_DIR}/checkpoints/clip-vit-base-patch32" \
  --task_type pickplace \
  --fps 1.0 \
  --save_dir "${SCRIPT_DIR}/saved_trajectories"
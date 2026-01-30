#!/bin/bash

# Rollout script for image-based initialization
# This script runs a closed-loop rollout starting from initial images and joint state
# The pickplace task uses image-based initialization (no ground truth videos needed)
# 
# Required data structure:
#   dataset_example/my_droid/scene_0/
#     - ext_1.png, ext_2.png, wrist.png (3 camera views)
#     - metadata.json (with joint_position, cartesian_position, gripper_position)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set GPU and memory settings
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

# Record start time
START_TIME=$(date +%s)
echo "Starting rollout at $(date '+%Y-%m-%d %H:%M:%S')"

# Run rollout with image-based initialization
python "${SCRIPT_DIR}/scripts/rollout_interact_pi.py" \
  --task_type pickplace \
  --dataset_root_path "${SCRIPT_DIR}/dataset_example" \
  --dataset_meta_info_path "${SCRIPT_DIR}/dataset_meta_info" \
  --dataset_names droid_subset \
  --svd_model_path "${SCRIPT_DIR}/checkpoints/stable-video-diffusion-img2vid" \
  --clip_model_path "${SCRIPT_DIR}/checkpoints/clip-vit-base-patch32" \
  --ckpt_path "${SCRIPT_DIR}/checkpoints/Ctrl-World/checkpoint-10000.pt" \
  --pi_ckpt gs://openpi-assets/checkpoints/pi05_droid \
  --batch_size 8 \
  --num_trajectories 16  # Uncomment to run 16 total trajectories (2 sequential batches of 8)

# Calculate and display elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "========================================="
echo "Rollout completed at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total execution time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "========================================="

# Note: 
# - pi_ckpt uses GCS path which will be automatically downloaded to ~/.cache/openpi on first use
# - Output videos will be saved to: synthetic_traj/Rollouts_interact_pi/video/
# - Trajectory info will be saved to: synthetic_traj/Rollouts_interact_pi/info/
# - batch_size: Number of parallel trajectories per batch
# - num_trajectories: Total number of trajectories to run. If > batch_size, runs sequential batches
#!/bin/bash

# Rollout script for image-based initialization
# This script runs closed-loop rollouts starting from initial images and joint state.
# No batching: each trajectory runs separately (batch_size=1).
# Runs (NUM_GPUS * TRAJECTORIES_PER_GPU) trajectories in parallel, with that many
# trajectories per GPU (multiple trajectories share each GPU).
#
# Required data structure:
#   dataset_example/my_droid/scene_0/
#     - ext_1.png, ext_2.png, wrist.png (3 camera views)
#     - metadata.json (with joint_position, cartesian_position, gripper_position)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Config: NUM_GPUS total GPUs, TRAJECTORIES_PER_GPU on each ---
NUM_GPUS="${NUM_GPUS:-1}"
TRAJECTORIES_PER_GPU="${TRAJECTORIES_PER_GPU:-1}"
NUM_TRAJECTORIES=$((NUM_GPUS * TRAJECTORIES_PER_GPU))

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

run_one() {
  local idx=$1
  local gpu=$((idx % NUM_GPUS))
  CUDA_VISIBLE_DEVICES=$gpu python "${SCRIPT_DIR}/scripts/rollout_interact_pi.py" \
    --task_type pickplace \
    --dataset_root_path "${SCRIPT_DIR}/dataset_example" \
    --dataset_meta_info_path "${SCRIPT_DIR}/dataset_meta_info" \
    --dataset_names droid_subset \
    --svd_model_path "${SCRIPT_DIR}/checkpoints/stable-video-diffusion-img2vid" \
    --clip_model_path "${SCRIPT_DIR}/checkpoints/clip-vit-base-patch32" \
    --ckpt_path "${SCRIPT_DIR}/checkpoints/Ctrl-World/checkpoint-10000.pt" \
    --pi_ckpt gs://openpi-assets/checkpoints/pi05_droid \
    --batch_size 1 \
    --trajectory_index "$idx" \
    --num_trajectories "$NUM_TRAJECTORIES"
}

echo "Running $NUM_TRAJECTORIES trajectories ($TRAJECTORIES_PER_GPU per GPU on $NUM_GPUS GPUs, no batching)."

pids=()
for i in $(seq 0 $((NUM_TRAJECTORIES - 1))); do
  run_one "$i" &
  pids+=($!)
done

for pid in "${pids[@]}"; do
  wait "$pid" || true
done

echo "All trajectories finished."
# Output videos: synthetic_traj/Rollouts_interact_pi/video/
# Trajectory info: synthetic_traj/Rollouts_interact_pi/info/

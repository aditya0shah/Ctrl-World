#!/bin/bash

# Script to generate full-length videos for specific trajectories using rollout_replay_traj
# Runs multiple trajectories in parallel across multiple GPUs
#
# Usage: 
#   NUM_GPUS=2 bash generate_full_videos.sh
#   Or simply: bash generate_full_videos.sh (defaults to 2 GPUs)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Config ---
NUM_GPUS="${NUM_GPUS:-2}"  # Default to 2 GPUs if not set

# Trajectory configurations: [split, traj_id, start_idx]
# The script will auto-calculate interact_num to generate full-length videos
TRAJECTORIES=(
  "train 1 0"
  "train 3 0"
  "val 899 0"
)

NUM_TRAJECTORIES=${#TRAJECTORIES[@]}

echo "=========================================="
echo "Full-Length Video Generation"
echo "=========================================="
echo "Number of trajectories: $NUM_TRAJECTORIES"
echo "Number of GPUs: $NUM_GPUS"
echo "Trajectories per GPU: $((NUM_TRAJECTORIES / NUM_GPUS)) to $(((NUM_TRAJECTORIES + NUM_GPUS - 1) / NUM_GPUS))"
echo ""

# Function to run one trajectory
run_one() {
  local idx=$1
  local split=$2
  local traj_id=$3
  local start_idx=$4
  local gpu=$((idx % NUM_GPUS))
  
  local log_file="${SCRIPT_DIR}/logs/generate_video_${split}_${traj_id}.log"
  
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $gpu: Starting trajectory $split/$traj_id (start_idx=$start_idx)"
  
  # Activate conda environment and run
  CUDA_VISIBLE_DEVICES=$gpu bash -c "
    source ~/.bashrc
    conda activate ctrl-world 2>/dev/null || source activate ctrl-world
    python ${SCRIPT_DIR}/scripts/rollout_replay_traj.py \
      --task_type replay \
      --val_id $traj_id \
      --start_idx $start_idx \
      --split $split
  " > "$log_file" 2>&1
  
  local exit_code=$?
  if [ $exit_code -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $gpu: ✓ Finished trajectory $split/$traj_id"
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $gpu: ✗ Failed trajectory $split/$traj_id (exit code: $exit_code)"
  fi
  
  return $exit_code
}

# Create logs directory
mkdir -p "${SCRIPT_DIR}/logs"
mkdir -p "${SCRIPT_DIR}/synthetic_traj/Rollouts_replay/video"

# Launch all trajectories in parallel
pids=()
for idx in "${!TRAJECTORIES[@]}"; do
  read -r split traj_id start_idx <<< "${TRAJECTORIES[$idx]}"
  run_one "$idx" "$split" "$traj_id" "$start_idx" &
  pids+=($!)
done

# Wait for all to complete
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || ((failed++))
done

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Completed: $((NUM_TRAJECTORIES - failed))/$NUM_TRAJECTORIES"
if [ $failed -gt 0 ]; then
  echo "Failed: $failed"
  echo "Check logs in: logs/"
fi
echo ""
echo "Output videos saved to: synthetic_traj/Rollouts_replay/video/"
echo "Logs saved to: logs/"
echo "=========================================="

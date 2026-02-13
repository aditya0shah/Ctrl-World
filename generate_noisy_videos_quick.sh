#!/bin/bash

# Quick script to generate videos at 5 action noise levels (clean to severe)
# Supports running multiple jobs per GPU concurrently
#
# Usage: 
#   NUM_GPUS=2 JOBS_PER_GPU=3 bash generate_noisy_videos_quick.sh
#   
# Example with 2 GPUs and 3 jobs per GPU = 6 concurrent jobs
# Total: 3 trajectories x 5 noise levels = 15 jobs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Config ---
NUM_GPUS="${NUM_GPUS:-2}"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"  # How many jobs to run on each GPU concurrently
MAX_CONCURRENT=$((NUM_GPUS * JOBS_PER_GPU))

# Trajectory configurations
TRAJECTORIES=(
  "train 1 0"
  "train 3 0"
  "val 899 0"
)

# 5 key action noise levels: clean -> light -> medium -> heavy -> severe
# Format: "name action_noise latent_noise inference_steps"
NOISE_CONFIGS=(
  "clean 0.0 0.0 50"
  "light 0.02 0.0 50"
  "medium 0.05 0.0 50"
  "heavy 0.1 0.0 50"
  "severe 0.2 0.0 50"
)

NUM_TRAJECTORIES=${#TRAJECTORIES[@]}
NUM_NOISE_CONFIGS=${#NOISE_CONFIGS[@]}
TOTAL_JOBS=$((NUM_TRAJECTORIES * NUM_NOISE_CONFIGS))

echo "=========================================="
echo "Action Noise Video Generation (5 levels)"
echo "=========================================="
echo "Trajectories: $NUM_TRAJECTORIES"
echo "Noise levels: $NUM_NOISE_CONFIGS"
echo "Total jobs: $TOTAL_JOBS"
echo "GPUs: $NUM_GPUS"
echo "Jobs per GPU: $JOBS_PER_GPU"
echo "Max concurrent: $MAX_CONCURRENT"
echo ""

# Track GPU slot usage
declare -a GPU_SLOTS
for ((i=0; i<NUM_GPUS; i++)); do
  for ((j=0; j<JOBS_PER_GPU; j++)); do
    GPU_SLOTS+=($i)
  done
done
SLOT_IDX=0

run_one() {
  local gpu=$1
  local split=$2
  local traj_id=$3
  local start_idx=$4
  local noise_name=$5
  local action_noise=$6
  local latent_noise=$7
  local inference_steps=$8
  
  local log_file="${SCRIPT_DIR}/logs/noisy/${noise_name}_${split}_${traj_id}.log"
  mkdir -p "$(dirname "$log_file")"
  
  echo "[$(date '+%H:%M:%S')] GPU $gpu: START ${noise_name} ${split}/${traj_id}"
  
  CUDA_VISIBLE_DEVICES=$gpu bash -c "
    source ~/.bashrc
    conda activate ctrl-world 2>/dev/null || source activate ctrl-world
    python ${SCRIPT_DIR}/scripts/rollout_replay_traj.py \
      --task_type replay \
      --val_id $traj_id \
      --start_idx $start_idx \
      --split $split \
      --action_noise $action_noise \
      --latent_noise $latent_noise \
      --num_inference_steps $inference_steps \
      --noise_config $noise_name
  " > "$log_file" 2>&1
  
  local exit_code=$?
  if [ $exit_code -eq 0 ]; then
    echo "[$(date '+%H:%M:%S')] GPU $gpu: ✓ DONE ${noise_name} ${split}/${traj_id}"
  else
    echo "[$(date '+%H:%M:%S')] GPU $gpu: ✗ FAIL ${noise_name} ${split}/${traj_id}"
  fi
  return $exit_code
}

mkdir -p "${SCRIPT_DIR}/logs/noisy"

# Build job list
declare -a JOBS
for noise_config in "${NOISE_CONFIGS[@]}"; do
  read -r noise_name action_noise latent_noise inference_steps <<< "$noise_config"
  for traj in "${TRAJECTORIES[@]}"; do
    read -r split traj_id start_idx <<< "$traj"
    JOBS+=("$split $traj_id $start_idx $noise_name $action_noise $latent_noise $inference_steps")
  done
done

# Run jobs with controlled concurrency
pids=()
failed=0
job_count=0

for job in "${JOBS[@]}"; do
  read -r split traj_id start_idx noise_name action_noise latent_noise inference_steps <<< "$job"
  
  # Assign GPU in round-robin across all slots
  gpu=${GPU_SLOTS[$((job_count % MAX_CONCURRENT))]}
  
  run_one "$gpu" "$split" "$traj_id" "$start_idx" "$noise_name" "$action_noise" "$latent_noise" "$inference_steps" &
  pids+=($!)
  ((job_count++))
  
  # If we've filled all slots, wait for all to complete before next batch
  if [ $((job_count % MAX_CONCURRENT)) -eq 0 ] && [ $job_count -lt $TOTAL_JOBS ]; then
    echo ""
    echo "[$(date '+%H:%M:%S')] Waiting for batch of $MAX_CONCURRENT jobs to complete..."
    for pid in "${pids[@]}"; do
      wait "$pid" || ((failed++))
    done
    pids=()
    echo "[$(date '+%H:%M:%S')] Batch complete. Starting next batch..."
    echo ""
  fi
done

# Wait for remaining jobs
for pid in "${pids[@]}"; do
  wait "$pid" || ((failed++))
done

echo ""
echo "=========================================="
echo "Done: $((TOTAL_JOBS - failed))/$TOTAL_JOBS succeeded"
if [ $failed -gt 0 ]; then
  echo "Failed: $failed - check logs/noisy/"
fi
echo "Videos: synthetic_traj/Rollouts_replay/video/<noise_level>/"
echo "=========================================="

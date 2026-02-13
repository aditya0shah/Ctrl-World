#!/bin/bash

# Script to generate noisy action videos with multiple samples per (task, noise)
# Each sample gets a unique repeat_id so eval treats them as separate trajectories
#
# Usage: 
#   NUM_GPUS=2 JOBS_PER_GPU=2 bash generate_noisy_videos.sh
#   
# Example: 2 GPUs x 2 jobs/GPU = 4 concurrent jobs
# Total: 3 tasks × 10 noise levels × 5 repeats = 150 jobs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Config ---
NUM_GPUS="${NUM_GPUS:-2}"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
MAX_CONCURRENT=$((NUM_GPUS * JOBS_PER_GPU))

# Trajectory configurations: focus on 3 key tasks
TRAJECTORIES=(
  "train 1 0"
  "train 3 0"
  "val 899 0"
)

# Number of repeat samples per (task, noise) combo for better statistics
NUM_REPEATS=5

# Action noise configurations (includes clean=0.0 as baseline)
# Format: "name action_noise latent_noise inference_steps"
# inference_steps=-1 means use video_length (entire video) per trajectory
NOISE_CONFIGS=(
  "clean 0.0 0.0 -1"
  "action_0.01 0.01 0.0 -1"
  "action_0.02 0.02 0.0 -1"
  "action_0.03 0.03 0.0 -1"
  "action_0.05 0.05 0.0 -1"
  "action_0.07 0.07 0.0 -1"
  "action_0.1 0.1 0.0 -1"
  "action_0.15 0.15 0.0 -1"
  "action_0.2 0.2 0.0 -1"
  "action_0.25 0.25 0.0 -1"
  "action_0.3 0.3 0.0 -1"
)

NUM_TRAJECTORIES=${#TRAJECTORIES[@]}
NUM_NOISE_CONFIGS=${#NOISE_CONFIGS[@]}
TOTAL_JOBS=$((NUM_TRAJECTORIES * NUM_NOISE_CONFIGS * NUM_REPEATS))

# Pre-flight: verify requested GPUs exist
AVAILABLE_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$AVAILABLE_GPUS" -lt "$NUM_GPUS" ]; then
  echo "WARNING: Requested NUM_GPUS=$NUM_GPUS but only $AVAILABLE_GPUS GPU(s) visible."
  echo "  Check: nvidia-smi -L (run inside your salloc session)"
  echo "  Using NUM_GPUS=$AVAILABLE_GPUS instead."
  NUM_GPUS=$AVAILABLE_GPUS
  MAX_CONCURRENT=$((NUM_GPUS * JOBS_PER_GPU))
fi

echo "=========================================="
echo "Action Noise Video Generation (Multiple Samples)"
echo "=========================================="
echo "Tasks: $NUM_TRAJECTORIES"
echo "Noise levels: $NUM_NOISE_CONFIGS"
echo "Repeats per (task, noise): $NUM_REPEATS"
echo "Total jobs: $TOTAL_JOBS"
echo "GPUs: $NUM_GPUS (visible: $AVAILABLE_GPUS)"
echo "Jobs per GPU: $JOBS_PER_GPU"
echo "Max concurrent: $MAX_CONCURRENT"
echo ""
echo "Noise configurations:"
for config in "${NOISE_CONFIGS[@]}"; do
  read -r name action_noise latent_noise steps <<< "$config"
  echo "  $name: action=$action_noise, latent=$latent_noise, steps=$steps"
done
echo ""

# Track GPU slot usage (GPUs 0..(NUM_GPUS-1))
declare -a GPU_SLOTS
for ((i=0; i<NUM_GPUS; i++)); do
  for ((j=0; j<JOBS_PER_GPU; j++)); do
    GPU_SLOTS+=($i)
  done
done

run_one() {
  local gpu=$1
  local split=$2
  local traj_id=$3
  local start_idx=$4
  local noise_name=$5
  local action_noise=$6
  local latent_noise=$7
  local inference_steps=$8
  local repeat_id=$9
  
  local log_file="${SCRIPT_DIR}/logs/noisy/${noise_name}_${split}_${traj_id}_rep${repeat_id}.log"
  mkdir -p "$(dirname "$log_file")"
  
  echo "[$(date '+%H:%M:%S')] GPU $gpu: START ${noise_name} ${split}/${traj_id} rep${repeat_id}"
  
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
      --noise_config $noise_name \
      --repeat_id $repeat_id
  " > "$log_file" 2>&1
  
  local exit_code=$?
  if [ $exit_code -eq 0 ]; then
    echo "[$(date '+%H:%M:%S')] GPU $gpu: ✓ DONE ${noise_name} ${split}/${traj_id} rep${repeat_id}"
  else
    echo "[$(date '+%H:%M:%S')] GPU $gpu: ✗ FAIL ${noise_name} ${split}/${traj_id} rep${repeat_id}"
  fi
  return $exit_code
}

mkdir -p "${SCRIPT_DIR}/logs/noisy"
mkdir -p "${SCRIPT_DIR}/synthetic_traj/Rollouts_replay/video"

# Build job list: for each (traj, noise, repeat) combo
declare -a JOBS
for noise_config in "${NOISE_CONFIGS[@]}"; do
  read -r noise_name action_noise latent_noise inference_steps <<< "$noise_config"
  for traj in "${TRAJECTORIES[@]}"; do
    read -r split traj_id start_idx <<< "$traj"
    for ((rep=0; rep<NUM_REPEATS; rep++)); do
      JOBS+=("$split $traj_id $start_idx $noise_name $action_noise $latent_noise $inference_steps $rep")
    done
  done
done

echo "Starting $TOTAL_JOBS jobs (batches of $MAX_CONCURRENT)..."
echo ""

# Run jobs with controlled concurrency
pids=()
failed=0
job_count=0
batch_num=1

for job in "${JOBS[@]}"; do
  read -r split traj_id start_idx noise_name action_noise latent_noise inference_steps repeat_id <<< "$job"
  
  # Assign GPU in round-robin across all slots
  gpu=${GPU_SLOTS[$((job_count % MAX_CONCURRENT))]}
  
  run_one "$gpu" "$split" "$traj_id" "$start_idx" "$noise_name" "$action_noise" "$latent_noise" "$inference_steps" "$repeat_id" &
  pids+=($!)
  ((job_count++))
  
  # If we've filled all slots, wait for all to complete before next batch
  if [ $((job_count % MAX_CONCURRENT)) -eq 0 ] && [ $job_count -lt $TOTAL_JOBS ]; then
    echo ""
    echo "[$(date '+%H:%M:%S')] Waiting for batch $batch_num ($MAX_CONCURRENT jobs)..."
    for pid in "${pids[@]}"; do
      wait "$pid" || ((failed++))
    done
    pids=()
    ((batch_num++))
    echo "[$(date '+%H:%M:%S')] Batch complete. Starting batch $batch_num..."
    echo ""
  fi
done

# Wait for remaining jobs
echo ""
echo "[$(date '+%H:%M:%S')] Waiting for final batch..."
for pid in "${pids[@]}"; do
  wait "$pid" || ((failed++))
done

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Completed: $((TOTAL_JOBS - failed))/$TOTAL_JOBS"
if [ $failed -gt 0 ]; then
  echo "Failed: $failed"
  echo "Check logs in: logs/noisy/"
fi
echo ""
echo "Output videos organized by noise level:"
echo "  synthetic_traj/Rollouts_replay/video/<noise_config>/"
echo ""
echo "Logs: logs/noisy/"
echo "=========================================="

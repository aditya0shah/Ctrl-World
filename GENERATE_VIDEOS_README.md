# Full-Length Video Generation Guide

This guide explains how to generate full-length trajectory videos using `rollout_replay_traj` with multi-GPU support.

## Overview

The `generate_full_videos.sh` script automates the generation of full-length videos for multiple trajectories in parallel across multiple GPUs. It automatically calculates the required number of interaction steps to cover the entire trajectory length.

## Quick Start

### Basic Usage (2 GPUs)
```bash
bash generate_full_videos.sh
```

### Custom Number of GPUs
```bash
NUM_GPUS=4 bash generate_full_videos.sh
```

### Single GPU
```bash
NUM_GPUS=1 bash generate_full_videos.sh
```

## Current Configuration

The script is configured to generate videos for:
- **train/1**: "Put the blue block in the green bowl" (89 frames)
- **train/3**: "Pick up the longer upright white container from the table and put it in the orange plastic bag" (80 frames)
- **val/899**: "Move the banana to the right" (121 frames)

## How It Works

### GPU Distribution
- Trajectories are distributed round-robin across available GPUs
- Each GPU processes one trajectory at a time
- All trajectories run in parallel

Example with 2 GPUs:
```
GPU 0: train/1, val/899
GPU 1: train/3
```

### Automatic Frame Calculation
The script automatically calculates the required number of interaction steps (`interact_num`) based on:
- Total video length from annotation
- Starting frame index
- Prediction step size (default: 5 frames)

Formula: `interact_num = ceil((video_length - start_idx - pred_step) / (pred_step - 1)) + 1`

## Output

### Video Files
Videos are saved to:
```
synthetic_traj/Rollouts_replay/video/
```

Filename format:
```
time_YYYYMMDD_HHMMSS_traj_{id}_{start_idx}_{pred_step}_{instruction}.mp4
```

### Log Files
Detailed logs for each trajectory:
```
logs/generate_video_{split}_{traj_id}.log
```

## Adding More Trajectories

Edit the `TRAJECTORIES` array in `generate_full_videos.sh`:

```bash
TRAJECTORIES=(
  "train 1 0"      # split, trajectory_id, start_frame
  "train 3 0"
  "val 899 0"
  "train 5 0"      # Add more trajectories here
  "val 199 8"      # Can specify custom start frames
)
```

## Manual Single Trajectory Generation

To generate a single trajectory manually:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rollout_replay_traj.py \
  --svd_model_path checkpoints/stable-video-diffusion-img2vid \
  --clip_model_path checkpoints/clip-vit-base-patch32 \
  --ckpt_path checkpoints/Ctrl-World/checkpoint-10000.pt \
  --dataset_root_path dataset_example \
  --dataset_meta_info_path dataset_meta_info \
  --dataset_names droid_subset \
  --task_type replay \
  --val_id 1 \
  --start_idx 0 \
  --split train \
  --interact_num 23
```

### Parameters
- `--val_id`: Trajectory ID to replay
- `--start_idx`: Starting frame index (default: 0)
- `--split`: Dataset split - `train` or `val` (default: `val`)
- `--interact_num`: Number of interactions (auto-calculated if not specified)

## Monitoring Progress

### During Execution
Watch the console output for real-time status updates:
```
[2026-01-30 10:30:15] GPU 0: Starting trajectory train/1 (start_idx=0)
[2026-01-30 10:30:20] GPU 1: Starting trajectory train/3 (start_idx=0)
...
[2026-01-30 10:45:30] GPU 0: ✓ Finished trajectory train/1
```

### Check Logs
Monitor individual trajectory logs:
```bash
tail -f logs/generate_video_train_1.log
```

### List Generated Videos
```bash
ls -lh synthetic_traj/Rollouts_replay/video/
```

## Troubleshooting

### Out of Memory Errors
If you encounter OOM errors:
1. Reduce the number of GPUs to ensure each GPU has enough memory
2. Check GPU memory usage: `nvidia-smi`
3. Consider processing trajectories sequentially: `NUM_GPUS=1`

### Missing Checkpoints
Ensure the following directories exist:
```
checkpoints/stable-video-diffusion-img2vid/
checkpoints/clip-vit-base-patch32/
checkpoints/Ctrl-World/checkpoint-10000.pt
```

### Failed Trajectories
Check the logs for specific error messages:
```bash
grep -i error logs/generate_video_*.log
```

## Performance Tips

### Optimal GPU Count
- **1-2 GPUs**: Process trajectories sequentially or with minimal parallelism
- **3+ GPUs**: One GPU per trajectory for maximum speed

### SLURM Example
If running on a cluster with SLURM:
```bash
salloc -A yourlab -p gpu-partition -N 1 -c 8 --mem=100G --gres=gpu:2 --time=4:00:00
NUM_GPUS=2 bash generate_full_videos.sh
```

## Advanced Configuration

### Custom Prediction Steps
Edit `config.py` to modify:
- `pred_step`: Number of frames predicted per interaction (default: 5)
- `num_history`: Number of history frames (default: 6)
- `num_frames`: Number of future frames (default: 5)

### Custom Video Parameters
Modify in `config.py`:
- `fps`: Frames per second (default: 7)
- `width`: Video width (default: 320)
- `height`: Video height (default: 192)
- `guidance_scale`: Diffusion guidance scale (default: 2)

## Example Output

After successful execution:
```
==========================================
Summary
==========================================
Completed: 3/3
Failed: 0

Output videos saved to: synthetic_traj/Rollouts_replay/video/
Logs saved to: logs/
==========================================
```

Videos will show:
- Top row: Ground truth video from 3 camera views
- Bottom row: Model-predicted video from 3 camera views

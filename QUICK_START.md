# Quick Start: Generate Full-Length Videos

This guide gets you started quickly with generating full-length trajectory videos.

## TL;DR

```bash
# 1. Check your setup
bash test_setup.sh

# 2. Preview trajectory info
python check_trajectory_info.py

# 3. Generate videos (using 2 GPUs by default)
bash generate_full_videos.sh

# Or specify number of GPUs
NUM_GPUS=4 bash generate_full_videos.sh
```

## What Gets Generated

The default configuration generates full-length videos for:

| Split | ID  | Frames | Interactions | Task                                                                      | Success |
|-------|-----|--------|--------------|---------------------------------------------------------------------------|---------|
| train | 1   | 89     | ~22          | Put the blue block in the green bowl                                      | ✓       |
| train | 3   | 80     | ~19          | Pick up the longer upright white container from the table and put it in the orange plastic bag | ✓       |
| val   | 899 | 121    | ~30          | Move the banana to the right                                              | ✗       |

**Total:** 290 frames, 71 interactions

## Step-by-Step

### Step 1: Verify Setup

Run the setup verification script:

```bash
bash test_setup.sh
```

This checks:
- Required directories and files
- Checkpoint availability
- GPU availability and memory
- Python environment and PyTorch

If all checks pass, you're ready to proceed!

### Step 2: Preview Trajectory Information

See what will be generated:

```bash
python check_trajectory_info.py
```

This shows:
- Video lengths
- Required interactions
- Task instructions
- Estimated processing time for different GPU counts

### Step 3: Generate Videos

Start the generation process:

```bash
# Default: 2 GPUs
bash generate_full_videos.sh

# Or customize GPU count
NUM_GPUS=1 bash generate_full_videos.sh  # Single GPU
NUM_GPUS=4 bash generate_full_videos.sh  # 4 GPUs
```

### Step 4: Monitor Progress

#### Watch Console Output
The script prints real-time status updates:
```
[2026-01-30 10:30:15] GPU 0: Starting trajectory train/1 (start_idx=0)
[2026-01-30 10:30:20] GPU 1: Starting trajectory train/3 (start_idx=0)
...
```

#### Check Individual Logs
Monitor a specific trajectory:
```bash
tail -f logs/generate_video_train_1.log
```

#### Check GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Step 5: View Results

After completion, find your videos:

```bash
ls -lh synthetic_traj/Rollouts_replay/video/
```

Example output:
```
time_20260130_103015_traj_1_0_5_Put_the_blue_block_in_the_green_bowl.mp4
time_20260130_103020_traj_3_0_5_Pick_up_the_longer_upright_white_container.mp4
time_20260130_103025_traj_899_0_5_Move_the_banana_to_the_right.mp4
```

## Video Format

Each video shows 6 concatenated views:
```
┌─────────────────────────────────────────┐
│  Ground Truth Camera 1-2-3 (top row)    │
├─────────────────────────────────────────┤
│  Model Prediction Camera 1-2-3 (bottom) │
└─────────────────────────────────────────┘
```

## Adding More Trajectories

Edit `generate_full_videos.sh` and modify the `TRAJECTORIES` array:

```bash
TRAJECTORIES=(
  "train 1 0"
  "train 3 0"
  "val 899 0"
  # Add more here
  "train 5 0"      # Add train trajectory 5
  "val 199 8"      # Add val trajectory 199, starting at frame 8
)
```

Then re-run:
```bash
bash generate_full_videos.sh
```

## Troubleshooting

### "Out of Memory" Error
**Solution:** Reduce GPU count or run sequentially
```bash
NUM_GPUS=1 bash generate_full_videos.sh
```

### "Checkpoint not found"
**Solution:** Check if checkpoints are downloaded
```bash
ls -lh checkpoints/Ctrl-World/checkpoint-10000.pt
```

### Failed Trajectory
**Solution:** Check the specific log file
```bash
cat logs/generate_video_train_1.log | grep -i error
```

### Slow Generation
**Solution:** Check GPU utilization
```bash
nvidia-smi
```
Ensure GPUs are being used. If idle, check logs for errors.

## Performance Estimates

Based on approximate timing (~2-3 sec per interaction):

| GPUs | Time Estimate |
|------|---------------|
| 1    | ~3.5 minutes  |
| 2    | ~2.0 minutes  |
| 3    | ~1.5 minutes  |
| 4+   | ~1.5 minutes  |

*Actual time depends on GPU model and system load.*

## Advanced Usage

### Generate Single Trajectory Manually

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
  --split train
```

### Custom Start Frame

To start from a specific frame (e.g., frame 10):

```bash
TRAJECTORIES=(
  "train 1 10"   # Start at frame 10
)
```

### Override Interaction Count

By default, `interact_num` is auto-calculated for full-length videos. To override:

```bash
python scripts/rollout_replay_traj.py \
  --val_id 1 \
  --split train \
  --interact_num 15  # Force 15 interactions instead of auto-calculated 22
```

## File Locations

```
dql/
├── generate_full_videos.sh          # Main script
├── test_setup.sh                    # Setup verification
├── check_trajectory_info.py         # Trajectory info preview
├── scripts/
│   └── rollout_replay_traj.py       # Core replay script (modified)
├── logs/                            # Individual trajectory logs
│   ├── generate_video_train_1.log
│   ├── generate_video_train_3.log
│   └── generate_video_val_899.log
└── synthetic_traj/
    └── Rollouts_replay/
        └── video/                   # Output videos
            ├── time_..._traj_1_...mp4
            ├── time_..._traj_3_...mp4
            └── time_..._traj_899_...mp4
```

## Documentation

- `QUICK_START.md` (this file) - Quick start guide
- `GENERATE_VIDEOS_README.md` - Comprehensive documentation
- `CHANGES_SUMMARY.md` - Technical changes made to the codebase

## Getting Help

1. **Check logs:** Look in `logs/` directory for error messages
2. **Run tests:** Use `bash test_setup.sh` to diagnose setup issues
3. **Preview trajectories:** Use `python check_trajectory_info.py` to verify data

## Next Steps

After generating videos:
1. Watch the videos to visually compare ground truth vs predictions
2. Analyze model performance on different task types
3. Use videos for presentations, debugging, or further analysis
4. Generate more trajectories by modifying the trajectory list

Happy video generation!

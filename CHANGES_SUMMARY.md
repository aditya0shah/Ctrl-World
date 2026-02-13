# Changes Summary - Full-Length Video Generation

## Overview
Added support for generating full-length trajectory videos with multi-GPU parallelization for `rollout_replay_traj.py`.

## Files Modified

### 1. `scripts/rollout_replay_traj.py`
**Changes:**
- Added command-line arguments:
  - `--val_id`: Specify trajectory ID to replay
  - `--start_idx`: Specify starting frame index
  - `--split`: Choose between 'train' or 'val' dataset split
  - `--interact_num`: Manually set number of interactions (optional)

- Modified `get_traj_info()` method:
  - Added `split` parameter to support both train and val datasets
  - Dynamically constructs annotation path based on split

- Added automatic `interact_num` calculation:
  - Reads video length from annotation file
  - Calculates required interactions to cover full trajectory
  - Formula: `interact_num = ceil((video_length - start_idx - pred_step) / (pred_step - 1)) + 1`

**Backward Compatibility:**
- All changes are backward compatible
- Default behavior unchanged when arguments not provided
- Existing scripts will continue to work

### 2. `generate_full_videos.sh` (NEW)
**Purpose:** Shell script to generate full-length videos for multiple trajectories in parallel

**Features:**
- Multi-GPU support with automatic load distribution
- Round-robin GPU assignment for trajectories
- Parallel execution of all trajectories
- Comprehensive logging with timestamps
- Error tracking and summary report
- Configurable via environment variable: `NUM_GPUS`

**Default Configuration:**
```bash
TRAJECTORIES=(
  "train 1 0"      # train split, trajectory 1, start at frame 0
  "train 3 0"      # train split, trajectory 3, start at frame 0
  "val 899 0"      # val split, trajectory 899, start at frame 0
)
```

### 3. `GENERATE_VIDEOS_README.md` (NEW)
Comprehensive documentation covering:
- Quick start guide
- How the script works
- GPU distribution strategy
- Automatic frame calculation
- Output file locations
- Adding custom trajectories
- Manual single trajectory generation
- Monitoring and troubleshooting
- Performance tips
- SLURM integration example

### 4. `CHANGES_SUMMARY.md` (NEW)
This file documenting all changes.

## Usage Examples

### Quick Start (2 GPUs)
```bash
bash generate_full_videos.sh
```

### Custom GPU Count
```bash
NUM_GPUS=4 bash generate_full_videos.sh
```

### Single Trajectory (Manual)
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

## Key Improvements

### 1. Multi-GPU Parallelization
- Process multiple trajectories simultaneously
- Automatic GPU assignment (round-robin)
- Configurable GPU count

### 2. Automatic Full-Length Video Generation
- No manual calculation of `interact_num` required
- Automatically determines trajectory length from annotations
- Generates complete videos from start to end

### 3. Train/Val Split Support
- Can replay trajectories from both train and val splits
- Simply specify `--split train` or `--split val`

### 4. Enhanced Logging
- Individual log files per trajectory
- Timestamped progress updates
- Success/failure tracking
- Summary report at completion

### 5. Flexible Configuration
- Easy to add new trajectories to the batch
- Custom start frames supported
- Override auto-calculation with manual `--interact_num`

## Output Structure

```
dql/
├── synthetic_traj/
│   └── Rollouts_replay/
│       └── video/
│           ├── time_20260130_103015_traj_1_0_5_Put_the_blue_block_in_the_green_bowl.mp4
│           ├── time_20260130_103020_traj_3_0_5_Pick_up_the_longer_upright_white_container.mp4
│           └── time_20260130_103025_traj_899_0_5_Move_the_banana_to_the_right.mp4
└── logs/
    ├── generate_video_train_1.log
    ├── generate_video_train_3.log
    └── generate_video_val_899.log
```

## Technical Details

### Video Generation Parameters
- **pred_step**: 5 frames per interaction
- **FPS**: 7 (configurable in config.py)
- **Resolution**: 192x320 (configurable)
- **Views**: 3 camera angles concatenated

### Trajectory Lengths
- train/1: 89 frames → ~23 interactions
- train/3: 80 frames → ~21 interactions  
- val/899: 121 frames → ~30 interactions

### GPU Memory Requirements
- Estimated ~10-15GB per trajectory
- Depends on video length and model size
- Monitor with `nvidia-smi`

## Testing

To verify the changes work correctly:

1. **Test single trajectory:**
   ```bash
   CUDA_VISIBLE_DEVICES=0 python scripts/rollout_replay_traj.py \
     --task_type replay --val_id 1 --start_idx 0 --split train
   ```

2. **Test multi-GPU batch:**
   ```bash
   NUM_GPUS=2 bash generate_full_videos.sh
   ```

3. **Check outputs:**
   ```bash
   ls -lh synthetic_traj/Rollouts_replay/video/
   ls -lh logs/
   ```

## Future Enhancements (Optional)

Possible improvements for future development:
- Dynamic GPU memory monitoring to prevent OOM
- Resume capability for interrupted runs
- Video quality metrics calculation
- Batch processing of entire train/val splits
- Integration with distributed training frameworks
- Real-time progress visualization

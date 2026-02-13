#!/usr/bin/env python3
"""
Quick utility to check trajectory information before generating videos.
Shows video length, required interactions, and estimated processing time.
"""

import json
import os
import sys
from pathlib import Path

def check_trajectory(dataset_dir, split, traj_id, start_idx=0, pred_step=5):
    """Check trajectory info and calculate required interact_num."""
    annotation_path = Path(dataset_dir) / "annotation" / split / f"{traj_id}.json"
    
    if not annotation_path.exists():
        print(f"  ✗ Annotation not found: {annotation_path}")
        return None
    
    with open(annotation_path) as f:
        anno = json.load(f)
    
    video_length = anno.get("video_length", len(anno.get("states", [])))
    instruction = anno.get("texts", [""])[0]
    success = anno.get("success", "unknown")
    
    # Calculate required interact_num
    remaining_frames = video_length - start_idx
    interact_num = max(1, int((remaining_frames - pred_step) / (pred_step - 1)) + 1)
    
    # Estimate processing time (rough estimate: ~2-3 seconds per interaction)
    estimated_time_min = (interact_num * 2) / 60
    estimated_time_max = (interact_num * 3) / 60
    
    return {
        "traj_id": traj_id,
        "split": split,
        "video_length": video_length,
        "start_idx": start_idx,
        "remaining_frames": remaining_frames,
        "interact_num": interact_num,
        "instruction": instruction,
        "success": success,
        "estimated_time_min": estimated_time_min,
        "estimated_time_max": estimated_time_max,
    }

def main():
    # Configuration
    script_dir = Path(__file__).parent
    dataset_dir = script_dir / "dataset_example" / "droid_subset"
    
    # Trajectories to check (same as generate_full_videos.sh)
    trajectories = [
        ("train", "1", 0),
        ("train", "3", 0),
        ("val", "899", 0),
    ]
    
    pred_step = 5  # from config.py
    
    print("=" * 80)
    print("Trajectory Information Check")
    print("=" * 80)
    print()
    
    results = []
    for split, traj_id, start_idx in trajectories:
        print(f"Checking {split}/{traj_id}...")
        info = check_trajectory(dataset_dir, split, traj_id, start_idx, pred_step)
        
        if info:
            results.append(info)
            print(f"  ✓ Video length: {info['video_length']} frames")
            print(f"  ✓ Start index: {info['start_idx']}")
            print(f"  ✓ Remaining: {info['remaining_frames']} frames")
            print(f"  ✓ Required interactions: {info['interact_num']}")
            print(f"  ✓ Instruction: {info['instruction'][:60]}...")
            print(f"  ✓ Success: {info['success']}")
            print(f"  ✓ Est. time: {info['estimated_time_min']:.1f}-{info['estimated_time_max']:.1f} min")
        print()
    
    if results:
        print("=" * 80)
        print("Summary")
        print("=" * 80)
        print()
        
        total_interactions = sum(r["interact_num"] for r in results)
        total_frames = sum(r["remaining_frames"] for r in results)
        total_time_min = sum(r["estimated_time_min"] for r in results)
        total_time_max = sum(r["estimated_time_max"] for r in results)
        
        print(f"Total trajectories: {len(results)}")
        print(f"Total frames: {total_frames}")
        print(f"Total interactions: {total_interactions}")
        print()
        print(f"Estimated total time (sequential): {total_time_min:.1f}-{total_time_max:.1f} min")
        
        # Calculate parallel time based on GPU count
        for num_gpus in [1, 2, 3, 4]:
            if num_gpus > len(results):
                continue
            # Distribute trajectories across GPUs
            traj_per_gpu = (len(results) + num_gpus - 1) // num_gpus
            max_time_per_gpu = 0
            for i in range(num_gpus):
                start = i * traj_per_gpu
                end = min(start + traj_per_gpu, len(results))
                if start < end:
                    gpu_time = sum(r["estimated_time_max"] for r in results[start:end])
                    max_time_per_gpu = max(max_time_per_gpu, gpu_time)
            print(f"  With {num_gpus} GPU(s): ~{max_time_per_gpu:.1f} min")
        
        print()
        print("=" * 80)
        print("Ready to generate videos!")
        print()
        print("Run: bash generate_full_videos.sh")
        print("Or with custom GPUs: NUM_GPUS=2 bash generate_full_videos.sh")
        print("=" * 80)
    else:
        print("No valid trajectories found.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

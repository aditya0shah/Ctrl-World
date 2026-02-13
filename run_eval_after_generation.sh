#!/bin/bash

# Run this after generation completes to produce reward plots
# Or: generation script will call this automatically when done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Running Reward Evaluation"
echo "=========================================="

python "${SCRIPT_DIR}/scripts/eval_noisy_videos_reward.py" \
  --video_dir synthetic_traj/Rollouts_replay/video \
  --output_dir ./noisy_eval_out \
  --rfm_url http://localhost:8001 \
  --roboreward_url http://localhost:8002 \
  --fps 1.0 \
  --rolling_window 5 \
  --view 0

echo ""
echo "=========================================="
echo "Plots saved to: noisy_eval_out/plots/"
echo "  - per_video_*.png (one per trajectory)"
echo "  - summary_final_reward_by_config.png"
echo "=========================================="

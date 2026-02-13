#!/bin/bash

# Quick test script to verify setup before running full video generation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Setup Verification Test"
echo "=========================================="
echo ""

# Check if required directories exist
echo "1. Checking required directories..."
required_dirs=(
  "checkpoints/stable-video-diffusion-img2vid"
  "checkpoints/clip-vit-base-patch32"
  "checkpoints/Ctrl-World"
  "dataset_example/droid_subset/annotation/train"
  "dataset_example/droid_subset/annotation/val"
  "dataset_example/droid_subset/videos/train"
  "dataset_example/droid_subset/videos/val"
  "dataset_meta_info/droid"
)

missing_dirs=0
for dir in "${required_dirs[@]}"; do
  if [ -d "${SCRIPT_DIR}/${dir}" ]; then
    echo "  ✓ ${dir}"
  else
    echo "  ✗ ${dir} (MISSING)"
    ((missing_dirs++))
  fi
done

# Check if checkpoint file exists
echo ""
echo "2. Checking checkpoint file..."
if [ -f "${SCRIPT_DIR}/checkpoints/Ctrl-World/checkpoint-10000.pt" ]; then
  echo "  ✓ checkpoint-10000.pt exists"
  ckpt_size=$(du -h "${SCRIPT_DIR}/checkpoints/Ctrl-World/checkpoint-10000.pt" | cut -f1)
  echo "    Size: ${ckpt_size}"
else
  echo "  ✗ checkpoint-10000.pt (MISSING)"
  ((missing_dirs++))
fi

# Check if annotation files exist
echo ""
echo "3. Checking annotation files..."
annotation_files=(
  "dataset_example/droid_subset/annotation/train/1.json"
  "dataset_example/droid_subset/annotation/train/3.json"
  "dataset_example/droid_subset/annotation/val/899.json"
)

for file in "${annotation_files[@]}"; do
  if [ -f "${SCRIPT_DIR}/${file}" ]; then
    echo "  ✓ ${file}"
  else
    echo "  ✗ ${file} (MISSING)"
    ((missing_dirs++))
  fi
done

# Check GPU availability
echo ""
echo "4. Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
  gpu_count=$(nvidia-smi --list-gpus | wc -l)
  echo "  ✓ nvidia-smi available"
  echo "  ✓ Number of GPUs detected: ${gpu_count}"
  echo ""
  nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | while read line; do
    echo "    ${line}"
  done
else
  echo "  ✗ nvidia-smi not found (GPU support may not be available)"
  ((missing_dirs++))
fi

# Check Python and required packages
echo ""
echo "5. Checking Python environment..."
if command -v python &> /dev/null; then
  python_version=$(python --version 2>&1)
  echo "  ✓ Python: ${python_version}"
  
  # Check if torch is available
  if python -c "import torch" 2>/dev/null; then
    torch_version=$(python -c "import torch; print(torch.__version__)")
    cuda_available=$(python -c "import torch; print(torch.cuda.is_available())")
    echo "  ✓ PyTorch: ${torch_version}"
    echo "  ✓ CUDA available in PyTorch: ${cuda_available}"
  else
    echo "  ✗ PyTorch not found"
    ((missing_dirs++))
  fi
else
  echo "  ✗ Python not found"
  ((missing_dirs++))
fi

# Create output directories if they don't exist
echo ""
echo "6. Creating output directories..."
mkdir -p "${SCRIPT_DIR}/logs"
mkdir -p "${SCRIPT_DIR}/synthetic_traj/Rollouts_replay/video"
echo "  ✓ logs/"
echo "  ✓ synthetic_traj/Rollouts_replay/video/"

# Summary
echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
if [ $missing_dirs -eq 0 ]; then
  echo "✓ All checks passed!"
  echo ""
  echo "You can now run:"
  echo "  bash generate_full_videos.sh"
  echo ""
  echo "Or with custom GPU count:"
  echo "  NUM_GPUS=2 bash generate_full_videos.sh"
else
  echo "✗ ${missing_dirs} check(s) failed"
  echo ""
  echo "Please fix the issues above before running generate_full_videos.sh"
  exit 1
fi
echo "=========================================="

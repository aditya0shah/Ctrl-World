#!/usr/bin/env python3
"""
Download CLIP, SVD, and Ctrl-World checkpoints to checkpoints/ for use with run_rollout.sh.
Target paths (must match run_rollout.sh):
  - checkpoints/clip-vit-base-patch32
  - checkpoints/stable-video-diffusion-img2vid
  - checkpoints/Ctrl-World/checkpoint-10000.pt
"""
from __future__ import annotations

import os
import sys

def main() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub required. Install: pip install huggingface_hub")
        sys.exit(1)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoints = os.path.join(root, "checkpoints")
    os.makedirs(checkpoints, exist_ok=True)

    # CLIP
    clip_dir = os.path.join(checkpoints, "clip-vit-base-patch32")
    if os.path.exists(os.path.join(clip_dir, "config.json")):
        print("CLIP already at", clip_dir)
    else:
        print("Downloading openai/clip-vit-base-patch32 ->", clip_dir)
        snapshot_download(repo_id="openai/clip-vit-base-patch32", local_dir=clip_dir)

    # SVD (Stable Video Diffusion)
    svd_dir = os.path.join(checkpoints, "stable-video-diffusion-img2vid")
    if os.path.exists(os.path.join(svd_dir, "model_index.json")):
        print("SVD already at", svd_dir)
    else:
        print("Downloading stabilityai/stable-video-diffusion-img2vid ->", svd_dir)
        snapshot_download(
            repo_id="stabilityai/stable-video-diffusion-img2vid",
            local_dir=svd_dir,
        )

    # Ctrl-World (checkpoint-10000.pt)
    ctrl_dir = os.path.join(checkpoints, "Ctrl-World")
    ckpt_path = os.path.join(ctrl_dir, "checkpoint-10000.pt")
    if os.path.exists(ckpt_path):
        print("Ctrl-World checkpoint already at", ckpt_path)
    else:
        print("Downloading yjguo/Ctrl-World ->", ctrl_dir)
        snapshot_download(repo_id="yjguo/Ctrl-World", local_dir=ctrl_dir)

    print("Done. Checkpoints ready for run_rollout.sh.")


if __name__ == "__main__":
    main()

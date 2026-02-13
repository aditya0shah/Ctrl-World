"""
SVO → MP4 Conversion Script (runs on robot workstation with ZED SDK)

Converts SVO recordings from franka-data into MP4 videos for Ctrl-World training.
Requires: pyzed (ZED SDK Python bindings), imageio, imageio-ffmpeg

Usage:
    python scripts/convert_svo_to_mp4.py \
        --input_dir franka-data/success/2026-02-06 \
        --output_dir franka_feb06_videos

Output structure:
    franka_feb06_videos/
    ├── 0/{0,1,2}.mp4
    ├── 1/{0,1,2}.mp4
    └── ...
"""

import argparse
import os
import numpy as np

# Camera serial → output video index mapping
CAMERA_MAP = {
    "13263313": 0,
    "23804457": 1,
    "32640896": 2,
}

TARGET_SIZE = (192, 320)  # (H, W) — Ctrl-World expected resolution
SKIP_FACTOR = 3           # ~15Hz → 5Hz
OUTPUT_FPS = 5


def convert_svo_to_mp4(svo_path, output_path, target_size=TARGET_SIZE, skip_factor=SKIP_FACTOR):
    """Extract left-camera frames from SVO, downsample, resize, save as MP4."""
    import pyzed.sl as sl
    import imageio

    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_path)
    init_params.svo_real_time_mode = False

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"  Failed to open {svo_path}: {status}")
        return False

    image = sl.Mat()
    frames = []
    frame_idx = 0

    while True:
        err = zed.grab()
        if err != sl.ERROR_CODE.SUCCESS:
            break
        if frame_idx % skip_factor == 0:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame_np = image.get_data()[:, :, :3]  # drop alpha channel if present
            frames.append(frame_np)
        frame_idx += 1

    zed.close()

    if len(frames) == 0:
        print(f"  No frames extracted from {svo_path}")
        return False

    # Resize frames to target size
    from PIL import Image
    resized_frames = []
    for frame in frames:
        img = Image.fromarray(frame)
        img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
        resized_frames.append(np.array(img))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimwrite(output_path, resized_frames, fps=OUTPUT_FPS, codec="libx264")
    print(f"  Saved {len(resized_frames)} frames → {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert SVO recordings to MP4 for Ctrl-World")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to franka-data/success/2026-02-06")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for MP4 videos")
    args = parser.parse_args()

    # Get sorted list of trajectory folders
    traj_dirs = sorted([
        d for d in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, d))
    ])

    print(f"Found {len(traj_dirs)} trajectory folders")

    for traj_idx, traj_name in enumerate(traj_dirs):
        svo_dir = os.path.join(args.input_dir, traj_name, "recordings", "SVO")
        if not os.path.isdir(svo_dir):
            print(f"Skipping {traj_name}: no SVO directory")
            continue

        print(f"[{traj_idx}/{len(traj_dirs)}] Processing {traj_name}")

        for camera_serial, video_id in CAMERA_MAP.items():
            svo_path = os.path.join(svo_dir, f"{camera_serial}.svo")
            if not os.path.exists(svo_path):
                print(f"  Warning: {svo_path} not found, skipping")
                continue

            output_path = os.path.join(args.output_dir, str(traj_idx), f"{video_id}.mp4")
            convert_svo_to_mp4(svo_path, output_path)

    print("Done!")


if __name__ == "__main__":
    main()

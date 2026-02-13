"""
Convert Franka demonstration data to Ctrl-World training format.

Takes HDF5 trajectory files + pre-converted MP4 videos and produces:
  - annotation JSONs (train/val split)
  - resized MP4 videos (192x320, 5Hz)
  - VAE latent tensors (.pt)
  - normalization statistics (stat.json)
  - meta info sample indices (train_sample.json, val_sample.json)

Usage (with accelerate for multi-GPU VAE encoding):
    accelerate launch scripts/convert_franka_to_training_format.py \
        --h5_dir franka-data/success/2026-02-06 \
        --video_dir franka_feb06_videos \
        --output_dir dataset_example/franka_feb06 \
        --meta_dir dataset_meta_info/franka_feb06 \
        --svd_path checkpoints/stable-video-diffusion-img2vid

Single-GPU:
    python scripts/convert_franka_to_training_format.py \
        --h5_dir franka-data/success/2026-02-06 \
        --video_dir franka_feb06_videos \
        --output_dir dataset_example/franka_feb06 \
        --meta_dir dataset_meta_info/franka_feb06 \
        --svd_path checkpoints/stable-video-diffusion-img2vid
"""

import argparse
import json
import os
import random

import h5py
import imageio
import numpy as np
import torch
from diffusers.models import AutoencoderKLTemporalDecoder
from torch.utils.data import Dataset

NUM_CAMERAS = 3
TARGET_SIZE = (192, 320)  # (H, W)
RGB_SKIP = 3
VAE_BATCH_SIZE = 64
NUM_TRAIN = 27  # first 27 → train, last 3 → val

# Sequence sampling params (matching create_meta_info.py)
SEQUENCE_LENGTH = 8
SEQUENCE_INTERVAL = 1
START_INTERVAL = 1


def get_sorted_traj_dirs(h5_dir):
    """Get sorted list of trajectory directory names."""
    dirs = sorted([
        d for d in os.listdir(h5_dir)
        if os.path.isdir(os.path.join(h5_dir, d))
    ])
    return dirs


def read_h5_trajectory(h5_path):
    """Read trajectory data from HDF5 file, return dict of lists."""
    with h5py.File(h5_path, "r") as f:
        task = f.attrs["current_task"]
        success = bool(f.attrs["success"])

        obs_car = f["observation/robot_state/cartesian_position"][:].tolist()
        obs_gripper = f["observation/robot_state/gripper_position"][:].tolist()
        obs_joint = f["observation/robot_state/joint_positions"][:].tolist()
        act_car = f["action/cartesian_position"][:].tolist()
        act_joint = f["action/joint_position"][:].tolist()
        act_gripper = f["action/gripper_position"][:].tolist()
        act_joint_vel = f["action/joint_velocity"][:].tolist()

    return {
        "task": task,
        "success": success,
        "observation.state.cartesian_position": obs_car,
        "observation.state.joint_position": obs_joint,
        "observation.state.gripper_position": obs_gripper,
        "action.cartesian_position": act_car,
        "action.joint_position": act_joint,
        "action.gripper_position": act_gripper,
        "action.joint_velocity": act_joint_vel,
    }


class FrankaLatentDataset(Dataset):
    """Dataset that processes one trajectory per __getitem__ call.
    Used with accelerate DataLoader for multi-GPU VAE encoding."""

    def __init__(self, h5_dir, video_dir, output_dir, svd_path, device):
        self.h5_dir = h5_dir
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.device = device

        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
            svd_path, subfolder="vae"
        ).to(device)

        self.traj_dirs = get_sorted_traj_dirs(h5_dir)
        print(f"Found {len(self.traj_dirs)} trajectories")

    def __len__(self):
        return len(self.traj_dirs)

    def __getitem__(self, idx):
        traj_name = self.traj_dirs[idx]
        data_type = "train" if idx < NUM_TRAIN else "val"
        h5_path = os.path.join(self.h5_dir, traj_name, "trajectory.h5")

        # Read HDF5 data
        traj_info = read_h5_trajectory(h5_path)

        # Process videos: load MP4 → resize → save resized MP4 + VAE latents
        video_dir = os.path.join(self.video_dir, str(idx))
        first_video_frames = None

        for video_id in range(NUM_CAMERAS):
            video_path = os.path.join(video_dir, f"{video_id}.mp4")
            if not os.path.exists(video_path):
                print(f"  Warning: {video_path} not found, skipping trajectory {idx}")
                return 0

            # Load and process video (matching extract_latent.py:91-115)
            video = np.asarray(imageio.mimread(video_path))  # (T, H, W, C)
            frames = torch.tensor(video).permute(0, 3, 1, 2).float() / 255.0 * 2 - 1
            # Videos are already at 5Hz from SVO conversion, no need to skip again
            # But resize to exact target size in case of mismatch
            x = torch.nn.functional.interpolate(
                frames, size=TARGET_SIZE, mode="bilinear", align_corners=False
            )

            if first_video_frames is None:
                first_video_frames = x

            # Save resized MP4
            resize_video = ((x / 2.0 + 0.5).clamp(0, 1) * 255)
            resize_video = resize_video.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            out_video_dir = os.path.join(self.output_dir, "videos", data_type, str(idx))
            os.makedirs(out_video_dir, exist_ok=True)
            imageio.mimwrite(
                os.path.join(out_video_dir, f"{video_id}.mp4"),
                resize_video, fps=5, codec="libx264",
            )

            # Encode VAE latents
            x_device = x.to(self.device)
            with torch.no_grad():
                latents = []
                for i in range(0, len(x_device), VAE_BATCH_SIZE):
                    batch = x_device[i : i + VAE_BATCH_SIZE]
                    latent = (
                        self.vae.encode(batch)
                        .latent_dist.sample()
                        .mul_(self.vae.config.scaling_factor)
                        .cpu()
                    )
                    latents.append(latent)
                latent_tensor = torch.cat(latents, dim=0)

            out_latent_dir = os.path.join(self.output_dir, "latent_videos", data_type, str(idx))
            os.makedirs(out_latent_dir, exist_ok=True)
            torch.save(latent_tensor, os.path.join(out_latent_dir, f"{video_id}.pt"))

        # Build annotation JSON (matching extract_latent.py:117-151)
        num_frames = first_video_frames.shape[0] if first_video_frames is not None else 0

        # Downsample states to match video frame rate
        cartesian_pose = np.array(traj_info["observation.state.cartesian_position"])
        cartesian_gripper = np.array(traj_info["observation.state.gripper_position"])[:, None]
        cartesian_states = np.concatenate(
            (cartesian_pose, cartesian_gripper), axis=-1
        )[::RGB_SKIP].tolist()

        info = {
            "texts": [traj_info["task"]],
            "episode_id": idx,
            "success": int(traj_info["success"]),
            "video_length": num_frames,
            "state_length": len(cartesian_states),
            "raw_length": len(traj_info["observation.state.cartesian_position"]),
            "videos": [
                {"video_path": f"videos/{data_type}/{idx}/0.mp4"},
                {"video_path": f"videos/{data_type}/{idx}/1.mp4"},
                {"video_path": f"videos/{data_type}/{idx}/2.mp4"},
            ],
            "latent_videos": [
                {"latent_video_path": f"latent_videos/{data_type}/{idx}/0.pt"},
                {"latent_video_path": f"latent_videos/{data_type}/{idx}/1.pt"},
                {"latent_video_path": f"latent_videos/{data_type}/{idx}/2.pt"},
            ],
            "states": cartesian_states,
            "observation.state.cartesian_position": traj_info["observation.state.cartesian_position"],
            "observation.state.joint_position": traj_info["observation.state.joint_position"],
            "observation.state.gripper_position": traj_info["observation.state.gripper_position"],
            "action.cartesian_position": traj_info["action.cartesian_position"],
            "action.joint_position": traj_info["action.joint_position"],
            "action.gripper_position": traj_info["action.gripper_position"],
            "action.joint_velocity": traj_info["action.joint_velocity"],
        }

        ann_dir = os.path.join(self.output_dir, "annotation", data_type)
        os.makedirs(ann_dir, exist_ok=True)
        with open(os.path.join(ann_dir, f"{idx}.json"), "w") as f:
            json.dump(info, f, indent=2)

        print(f"  [{idx}] {data_type}: {traj_name} → {num_frames} frames, {len(cartesian_states)} states")
        return 0


def compute_statistics(h5_dir, meta_dir):
    """Compute 1st/99th percentile of 7D state across all trajectories."""
    traj_dirs = get_sorted_traj_dirs(h5_dir)

    all_states = []
    for traj_name in traj_dirs:
        h5_path = os.path.join(h5_dir, traj_name, "trajectory.h5")
        with h5py.File(h5_path, "r") as f:
            cart = f["observation/robot_state/cartesian_position"][:]  # (N, 6)
            grip = f["observation/robot_state/gripper_position"][:][:, None]  # (N, 1)
            states = np.concatenate((cart, grip), axis=-1)  # (N, 7)
            all_states.append(states)

    all_states = np.concatenate(all_states, axis=0)
    state_01 = np.percentile(all_states, 1, axis=0).tolist()
    state_99 = np.percentile(all_states, 99, axis=0).tolist()

    stat = {"state_01": state_01, "state_99": state_99}
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, "stat.json"), "w") as f:
        json.dump(stat, f)

    print(f"Statistics saved to {meta_dir}/stat.json")
    print(f"  state_01: {state_01}")
    print(f"  state_99: {state_99}")


def create_meta_samples(output_dir, meta_dir):
    """Generate train_sample.json and val_sample.json following create_meta_info.py pattern."""
    os.makedirs(meta_dir, exist_ok=True)

    for data_type in ["train", "val"]:
        ann_dir = os.path.join(output_dir, "annotation", data_type)
        if not os.path.isdir(ann_dir):
            print(f"  Warning: {ann_dir} not found, skipping {data_type} samples")
            continue

        ann_files = [f for f in os.listdir(ann_dir) if f.endswith(".json")]
        samples = []

        for ann_file in ann_files:
            ann_path = os.path.join(ann_dir, ann_file)
            with open(ann_path, "r") as f:
                ann = json.load(f)

            n_frames = ann["video_length"]
            traj_len = int(SEQUENCE_LENGTH * SEQUENCE_INTERVAL)
            end_idx = n_frames - int(traj_len * 0.5)
            if end_idx < 1:
                end_idx = 1

            for start_frame in range(0, end_idx, START_INTERVAL):
                sample = {
                    "episode_id": ann["episode_id"],
                    "frame_ids": [start_frame],
                }
                samples.append(sample)

        random.shuffle(samples)

        out_path = os.path.join(meta_dir, f"{data_type}_sample.json")
        with open(out_path, "w") as f:
            json.dump(samples, f, indent=4)

        print(f"  {data_type}: {len(samples)} samples from {len(ann_files)} trajectories → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert Franka data to Ctrl-World training format")
    parser.add_argument("--h5_dir", type=str, required=True,
                        help="Path to franka-data/success/2026-02-06")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Path to pre-converted MP4 videos (from convert_svo_to_mp4.py)")
    parser.add_argument("--output_dir", type=str, default="dataset_example/franka_feb06",
                        help="Output dataset directory")
    parser.add_argument("--meta_dir", type=str, default="dataset_meta_info/franka_feb06",
                        help="Output meta info directory")
    parser.add_argument("--svd_path", type=str, default="checkpoints/stable-video-diffusion-img2vid",
                        help="Path to SVD checkpoint (contains vae subfolder)")
    parser.add_argument("--skip_latents", action="store_true",
                        help="Skip VAE latent extraction (for debugging)")
    args = parser.parse_args()

    # Step 1: Compute normalization statistics (CPU only, no GPU needed)
    print("=" * 60)
    print("Step 1: Computing normalization statistics...")
    print("=" * 60)
    compute_statistics(args.h5_dir, args.meta_dir)

    if args.skip_latents:
        # CPU-only mode: just create annotations from HDF5 + copy videos
        print("=" * 60)
        print("Step 2: Creating annotations (skip_latents mode)...")
        print("=" * 60)
        traj_dirs = get_sorted_traj_dirs(args.h5_dir)
        for idx, traj_name in enumerate(traj_dirs):
            data_type = "train" if idx < NUM_TRAIN else "val"
            h5_path = os.path.join(args.h5_dir, traj_name, "trajectory.h5")
            traj_info = read_h5_trajectory(h5_path)

            # Check video exists and get frame count
            video_path = os.path.join(args.video_dir, str(idx), "0.mp4")
            if os.path.exists(video_path):
                video = np.asarray(imageio.mimread(video_path))
                num_frames = len(video)
            else:
                print(f"  Warning: {video_path} not found")
                num_frames = len(traj_info["observation.state.cartesian_position"]) // RGB_SKIP

            cartesian_pose = np.array(traj_info["observation.state.cartesian_position"])
            cartesian_gripper = np.array(traj_info["observation.state.gripper_position"])[:, None]
            cartesian_states = np.concatenate(
                (cartesian_pose, cartesian_gripper), axis=-1
            )[::RGB_SKIP].tolist()

            info = {
                "texts": [traj_info["task"]],
                "episode_id": idx,
                "success": int(traj_info["success"]),
                "video_length": num_frames,
                "state_length": len(cartesian_states),
                "raw_length": len(traj_info["observation.state.cartesian_position"]),
                "videos": [
                    {"video_path": f"videos/{data_type}/{idx}/{v}.mp4"}
                    for v in range(NUM_CAMERAS)
                ],
                "latent_videos": [
                    {"latent_video_path": f"latent_videos/{data_type}/{idx}/{v}.pt"}
                    for v in range(NUM_CAMERAS)
                ],
                "states": cartesian_states,
                "observation.state.cartesian_position": traj_info["observation.state.cartesian_position"],
                "observation.state.joint_position": traj_info["observation.state.joint_position"],
                "observation.state.gripper_position": traj_info["observation.state.gripper_position"],
                "action.cartesian_position": traj_info["action.cartesian_position"],
                "action.joint_position": traj_info["action.joint_position"],
                "action.gripper_position": traj_info["action.gripper_position"],
                "action.joint_velocity": traj_info["action.joint_velocity"],
            }

            ann_dir = os.path.join(args.output_dir, "annotation", data_type)
            os.makedirs(ann_dir, exist_ok=True)
            with open(os.path.join(ann_dir, f"{idx}.json"), "w") as f:
                json.dump(info, f, indent=2)

            print(f"  [{idx}] {data_type}: {traj_name}")
    else:
        # Full mode: use accelerate + VAE
        print("=" * 60)
        print("Step 2: Processing videos + extracting VAE latents...")
        print("=" * 60)
        from accelerate import Accelerator

        accelerator = Accelerator()
        dataset = FrankaLatentDataset(
            h5_dir=args.h5_dir,
            video_dir=args.video_dir,
            output_dir=args.output_dir,
            svd_path=args.svd_path,
            device=accelerator.device,
        )
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, num_workers=0, pin_memory=True,
        )
        data_loader = accelerator.prepare_data_loader(data_loader)
        for idx, _ in enumerate(data_loader):
            if idx % 5 == 0 and accelerator.is_main_process:
                print(f"  Processed {idx}/{len(dataset)} trajectories")

    # Step 3: Create meta info samples
    print("=" * 60)
    print("Step 3: Creating meta info samples...")
    print("=" * 60)
    create_meta_samples(args.output_dir, args.meta_dir)

    print("=" * 60)
    print("Done! Output:")
    print(f"  Dataset:   {args.output_dir}/")
    print(f"  Meta info: {args.meta_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()

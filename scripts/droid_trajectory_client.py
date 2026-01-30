#!/usr/bin/env python3
"""
DROID Trajectory Client with RFM Selection

Connects to DROID trajectory server, generates multiple trajectories using Ctrl-World agent,
evaluates each with RFM rewards, selects the best one, and sends it to the server for execution.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import multiprocessing as mp
import os
import pickle
import socket
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import wm_args

# Import agent class from rollout_interact_pi (same directory)
# Use importlib to avoid circular imports
import importlib.util
rollout_module_path = Path(__file__).parent / "rollout_interact_pi.py"
spec = importlib.util.spec_from_file_location("rollout_interact_pi", rollout_module_path)
rollout_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rollout_module)
agent = rollout_module.agent

from use_reward_eval_server import (
    compute_rewards_per_frame,
    extract_rewards_from_server_output,
    make_progress_sample,
    post_evaluate_batch_npy,
)

# ============================================================================
# Multiprocess trajectory generation (no batching)
# ============================================================================

# NOTE: These globals are initialized inside worker processes via _mp_init_worker().
_MP_AGENT_OBJ = None
_MP_TASK = None
_MP_TRAJECTORY_LENGTH = None


def _mp_init_worker(
    cfg: Any,
    task: str,
    trajectory_length: int,
    gpu_ids: Optional[List[int]] = None,
    worker_rank: Optional[Any] = None,
) -> None:
    """
    Worker initializer: constructs the agent once per process.

    We keep the agent instance in a process-global so we don't reload weights for every trajectory.
    When gpu_ids and worker_rank are provided, sets CUDA_VISIBLE_DEVICES so this process uses
    gpu_ids[worker_rank] (for multi-GPU parallelism).
    """
    global _MP_AGENT_OBJ, _MP_TASK, _MP_TRAJECTORY_LENGTH
    if gpu_ids is not None and worker_rank is not None:
        gpu_id = gpu_ids[worker_rank]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    _MP_TASK = task
    _MP_TRAJECTORY_LENGTH = trajectory_length
    _MP_AGENT_OBJ = agent(cfg)


def _mp_init_worker_multi_gpu(
    cfg: Any,
    task: str,
    trajectory_length: int,
    gpu_ids: List[int],
    counter: Any,
    lock: Any,
) -> None:
    """Wrapper initializer: assigns this process a GPU via shared counter, then inits agent."""
    with lock:
        worker_rank = counter.value
        counter.value += 1
    _mp_init_worker(
        cfg,
        task,
        trajectory_length,
        gpu_ids=gpu_ids,
        worker_rank=worker_rank,
    )


def _mp_generate_one(scene_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker job: generate exactly 1 trajectory from scene_data.
    """
    global _MP_AGENT_OBJ, _MP_TASK, _MP_TRAJECTORY_LENGTH
    if _MP_AGENT_OBJ is None:
        raise RuntimeError("Worker agent not initialized. This should not happen.")
    trajs = generate_trajectories(
        _MP_AGENT_OBJ,
        scene_data,
        _MP_TASK,
        num_trajectories=1,
        trajectory_length=_MP_TRAJECTORY_LENGTH,
    )
    return trajs[0]


# ============================================================================
# Socket Communication Module
# ============================================================================

def recvall(sock: socket.socket, n: int) -> Optional[bytes]:
    """Helper to receive n bytes or return None if EOF is hit."""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)


def send_msg(sock: socket.socket, msg: Any) -> None:
    """Send a message with length prefix over socket."""
    msg_bytes = pickle.dumps(msg)
    msg_len = struct.pack(">I", len(msg_bytes))
    sock.sendall(msg_len + msg_bytes)


def recv_msg(sock: socket.socket) -> Optional[Any]:
    """Receive a length-prefixed message from socket."""
    # Read message length (4 bytes)
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack(">I", raw_msglen)[0]
    # Read the message data
    return pickle.loads(recvall(sock, msglen))


def connect_to_server(url: str, max_retries: int = 3, retry_delay: float = 1.0) -> socket.socket:
    """Parse tcp://hostname:port and connect to server with retries."""
    if not url.startswith("tcp://"):
        raise ValueError(f"URL must start with 'tcp://', got: {url}")
    url = url[6:]  # Remove 'tcp://'
    if ":" not in url:
        raise ValueError(f"URL must include port, got: {url}")
    hostname, port_str = url.rsplit(":", 1)
    port = int(port_str)
    
    last_error = None
    for attempt in range(max_retries):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(300.0)  # 300 second timeout (5 min default for remote connections)
        
        try:
            sock.connect((hostname, port))
            return sock
        except (socket.error, OSError) as e:
            last_error = e
            sock.close()
            if attempt < max_retries - 1:
                print(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                raise ConnectionError(f"Failed to connect to {hostname}:{port} after {max_retries} attempts: {e}")
    
    raise ConnectionError(f"Failed to connect to {hostname}:{port}: {last_error}")


def reset_scene(sock: socket.socket) -> Dict[str, Any]:
    """Send RESET command and receive SCENE_DATA."""
    send_msg(sock, {"type": "RESET"})
    response = recv_msg(sock)
    if response is None:
        raise ConnectionError("Server closed connection during RESET")
    if response.get("type") != "SCENE_DATA":
        raise ValueError(f"Expected SCENE_DATA, got: {response.get('type')}")
    return response


def get_scene(sock: socket.socket) -> Dict[str, Any]:
    """Send GET_SCENE command and receive SCENE_DATA."""
    send_msg(sock, {"type": "GET_SCENE"})
    response = recv_msg(sock)
    if response is None:
        raise ConnectionError("Server closed connection during GET_SCENE")
    if response.get("type") != "SCENE_DATA":
        raise ValueError(f"Expected SCENE_DATA, got: {response.get('type')}")
    return response


def execute_trajectory(sock: socket.socket, trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """Send EXECUTE_TRAJECTORY command and receive TRAJECTORY_RESULT."""
    msg = {
        "type": "EXECUTE_TRAJECTORY",
        "trajectory": trajectory,
    }
    send_msg(sock, msg)
    response = recv_msg(sock)
    if response is None:
        raise ConnectionError("Server closed connection during EXECUTE_TRAJECTORY")
    if response.get("type") != "TRAJECTORY_RESULT":
        raise ValueError(f"Expected TRAJECTORY_RESULT, got: {response.get('type')}")
    return response


def close_connection(sock: socket.socket) -> None:
    """Send CLOSE command and close socket."""
    try:
        send_msg(sock, {"type": "CLOSE"})
    except Exception:
        pass  # Server may have already closed
    finally:
        sock.close()


# ============================================================================
# Trajectory Generation Module
# ============================================================================

def scene_data_to_images_and_state(scene_data: Dict[str, Any]) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Extract images and robot state from SCENE_DATA."""
    required_keys = ["left_image", "right_image", "wrist_image"]
    for key in required_keys:
        if key not in scene_data:
            raise ValueError(f"Missing required key '{key}' in scene_data")
    
    images = [
        scene_data["left_image"],      # (H, W, 3) uint8
        scene_data["right_image"],     # (H, W, 3) uint8
        scene_data["wrist_image"],     # (H, W, 3) uint8
    ]
    
    # Validate image shapes
    for i, img in enumerate(images):
        if not isinstance(img, np.ndarray):
            raise TypeError(f"Image {i} is not a numpy array, got {type(img)}")
        if len(img.shape) != 3 or img.shape[2] != 3:
            raise ValueError(f"Image {i} has invalid shape {img.shape}, expected (H, W, 3)")
    
    metadata = scene_data.get("metadata", {})
    if "joint_positions" not in metadata:
        raise ValueError("Missing 'joint_positions' in metadata")
    
    return images, metadata


def generate_trajectories(
    agent_obj: agent,
    scene_data: Dict[str, Any],
    task: str,
    num_trajectories: int,
    trajectory_length: int = 1,
) -> List[Dict[str, Any]]:
    """
    Generate N trajectories in parallel from scene data.
    
    Args:
        agent_obj: Initialized agent object
        scene_data: SCENE_DATA from server
        task: Task description string
        num_trajectories: Number of trajectories to generate
        trajectory_length: Number of interaction steps per trajectory (default: 1)
    
    Returns:
        List of trajectory dicts, each containing:
            - joint_vel: numpy array (T, 7)
            - predicted_video: numpy array (T, H, W, 3) for RFM evaluation
            - joint_pos: numpy array (T, 7)
            - state_fk: numpy array (T, 7)
    """
    images, metadata = scene_data_to_images_and_state(scene_data)
    
    # Extract robot state
    joint_position = np.array(metadata.get("joint_positions", []))  # (7,)
    if len(joint_position) != 7:
        raise ValueError(f"Expected 7 joint positions, got {len(joint_position)}")
    
    cartesian_position = metadata.get("cartesian_position")
    gripper_position = metadata.get("gripper_position", 0.0)
    
    # Handle gripper_position as single value or array
    if isinstance(gripper_position, (list, np.ndarray)):
        if len(gripper_position) > 0:
            gripper_position = float(gripper_position[0])
        else:
            gripper_position = 0.0
    else:
        gripper_position = float(gripper_position)
    
    # Initialize agent from images
    # Convert images to temporary file paths or use in-memory approach
    # For now, we'll use a temporary directory approach
    import tempfile
    import cv2
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        image_paths = []
        for i, img in enumerate(images):
            # Convert to RGB if needed
            if img.shape[2] == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.dtype == np.uint8 else img
            else:
                img_rgb = img
            
            # Save to temp file
            img_path = os.path.join(tmpdir, f"view_{i}.png")
            cv2.imwrite(img_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            image_paths.append(img_path)
        
        # Initialize agent
        video_dict, video_latents, initial_eef, initial_joint = agent_obj.init_from_images(
            image_paths,
            joint_position,
            cartesian_position,
            gripper_position,
        )
    
    # Initialize history buffers for each trajectory
    batch_size = num_trajectories
    his_cond_batch = [[] for _ in range(batch_size)]
    his_joint_batch = [[] for _ in range(batch_size)]
    his_eef_batch = [[] for _ in range(batch_size)]
    
    first_latent = torch.cat([v[0] for v in video_latents], dim=1).unsqueeze(0)  # (1, 4, 72, 40)
    initial_joint_single = torch.from_numpy(initial_joint[None, :]).to(agent_obj.device).to(agent_obj.dtype)  # (1, 8)
    initial_eef_single = torch.from_numpy(initial_eef[None, :]).to(agent_obj.device).to(agent_obj.dtype)  # (1, 7)
    
    for traj_idx in range(batch_size):
        for i in range(agent_obj.args.num_history * 4):
            his_cond_batch[traj_idx].append(first_latent.clone())
            his_joint_batch[traj_idx].append(initial_joint_single.clone())
            his_eef_batch[traj_idx].append(initial_eef_single.clone())
    
    video_dict_pred_batch = [[v[0:1] for v in video_dict] for _ in range(batch_size)]
    
    # Generate trajectories
    trajectories = []
    interact_num = trajectory_length
    
    for step in range(interact_num):
        # Collect inputs from all trajectories for batched policy and world model forward pass
        # Prepare batched inputs for policy
        videos_batch = []
        states_batch = []
        joints_batch = []
        
        for traj_idx in range(batch_size):
            # Get current state for this trajectory
            current_joint = his_joint_batch[traj_idx][-1][0]  # (8,)
            current_pose = his_eef_batch[traj_idx][-1][0]  # (7,)
            # Convert tensors to numpy arrays if needed (forward_policy_batch expects numpy arrays)
            if isinstance(current_joint, torch.Tensor):
                current_joint = current_joint.cpu().float().numpy()
            if isinstance(current_pose, torch.Tensor):
                current_pose = current_pose.cpu().float().numpy()
            # Extract last frame from each video view
            # Handle both (1, H, W, 3) and (num_frames, h, w, c) shapes
            current_obs = []
            for v in video_dict_pred_batch[traj_idx]:
                if len(v.shape) == 4:
                    # Shape is (num_frames, h, w, c) or (1, H, W, 3)
                    # Use v[0] if first dim is 1, otherwise v[-1]
                    current_obs.append(v[0] if v.shape[0] == 1 else v[-1])
                else:
                    # Already 3D: (h, w, c)
                    current_obs.append(v)
            
            videos_batch.append(current_obs)
            states_batch.append(current_pose)
            joints_batch.append(current_joint)
        
        # BATCHED policy forward pass - processes all trajectories at once
        policy_outputs, joint_positions, cartesian_poses = agent_obj.forward_policy_batch(
            videos_batch, states_batch, joints_batch, task
        )
        
        # Prepare world model inputs for each trajectory (after policy divergence)
        action_conds = []
        current_latents = []
        his_latents = []
        history_idx = agent_obj.args.history_idx
        
        for traj_idx in range(batch_size):
            # Convert tensors to numpy arrays if needed for numpy operations
            his_pose_elems = []
            for idx in history_idx:
                elem = his_eef_batch[traj_idx][idx][0]  # (7,)
                if isinstance(elem, torch.Tensor):
                    elem = elem.cpu().float().numpy()
                # Ensure element is 2D: (1, 7)
                if elem.ndim == 1:
                    elem = elem[None, :]  # (1, 7)
                his_pose_elems.append(elem)
            his_pose = np.concatenate(his_pose_elems, axis=0)  # (num_history, 7)
            action_cond = np.concatenate([his_pose, cartesian_poses[traj_idx]], axis=0)  # (num_history+num_frames, 7)
            his_latent = torch.cat([his_cond_batch[traj_idx][idx] for idx in history_idx], dim=0).unsqueeze(0)  # (1, num_history, 4, 72, 40)
            current_latent = his_cond_batch[traj_idx][-1]  # (1, 4, 72, 40)
            
            action_conds.append(action_cond)
            current_latents.append(current_latent)
            his_latents.append(his_latent)
        
        # Batched world model forward pass
        action_cond_batch = np.stack(action_conds, axis=0)
        current_latent_batch = torch.cat(current_latents, dim=0)
        his_latent_batch = torch.cat(his_latents, dim=0)
        text_batch = [task if agent_obj.args.text_cond else None] * batch_size
        
        videos_cat_batch, true_videos, video_dict_pred_batch_output, predict_latents = agent_obj.forward_wm(
            action_cond_batch,
            None,
            current_latent_batch,
            his_cond=his_latent_batch,
            text=text_batch,
        )
        
        # Update history buffers and collect trajectories
        for traj_idx in range(batch_size):
            # Extract latents for this trajectory
            if batch_size == 1:
                traj_latents = predict_latents
            else:
                traj_latents = [v[traj_idx:traj_idx+1] for v in predict_latents]
            
            # Update history
            joint_update = joint_positions[traj_idx][agent_obj.args.pred_step-1:agent_obj.args.pred_step]
            his_joint_batch[traj_idx].append(joint_update)
            
            cartesian_update = cartesian_poses[traj_idx][agent_obj.args.pred_step-1:agent_obj.args.pred_step]
            his_eef_batch[traj_idx].append(cartesian_update)
            
            # Combine latents
            if batch_size == 1:
                combined_latent = torch.cat([v[agent_obj.args.pred_step-1:agent_obj.args.pred_step] for v in traj_latents], dim=2)
            else:
                combined_latent = torch.cat([v[0, agent_obj.args.pred_step-1:agent_obj.args.pred_step] for v in traj_latents], dim=2)
            his_cond_batch[traj_idx].append(combined_latent)
            
            # Update video predictions
            if batch_size == 1:
                video_dict_pred_batch[traj_idx] = video_dict_pred_batch_output
            else:
                video_dict_pred_batch[traj_idx] = [v[traj_idx] for v in video_dict_pred_batch_output]
            
            # Store trajectory data (only on first step, or accumulate)
            # Extract predicted video frames for this step
            # video_dict_pred_batch_output is a list of 3 arrays (one per camera view)
            # Each array: (batch_size, num_frames, H, W, 3) or (num_frames, H, W, 3) for batch_size=1
            if batch_size == 1:
                # Single batch: each view is (num_frames, H, W, 3)
                video_view0_step = video_dict_pred_batch_output[0][:agent_obj.args.pred_step-1]  # (pred_step-1, H, W, 3)
            else:
                # Batched: each view is (batch_size, num_frames, H, W, 3)
                video_view0_step = video_dict_pred_batch_output[0][traj_idx, :agent_obj.args.pred_step-1]  # (pred_step-1, H, W, 3)
            
            if step == 0:
                trajectories.append({
                    "joint_vel": policy_outputs[traj_idx]["joint_vel"].copy(),  # (T, 7)
                    "predicted_video": [video_view0_step],  # List of (T, H, W, 3) arrays
                    "joint_pos": policy_outputs[traj_idx]["joint_pos"].copy(),  # (T, 7)
                    "state_fk": policy_outputs[traj_idx]["state_fk"].copy(),  # (T, 7)
                })
            else:
                # Accumulate trajectory data
                trajectories[traj_idx]["joint_vel"] = np.concatenate([
                    trajectories[traj_idx]["joint_vel"],
                    policy_outputs[traj_idx]["joint_vel"]
                ], axis=0)
                trajectories[traj_idx]["joint_pos"] = np.concatenate([
                    trajectories[traj_idx]["joint_pos"],
                    policy_outputs[traj_idx]["joint_pos"]
                ], axis=0)
                trajectories[traj_idx]["state_fk"] = np.concatenate([
                    trajectories[traj_idx]["state_fk"],
                    policy_outputs[traj_idx]["state_fk"]
                ], axis=0)
                # Accumulate video frames
                trajectories[traj_idx]["predicted_video"].append(video_view0_step)
    
    # Concatenate all video frames for RFM evaluation
    # Use view0 (ext_1) for RFM evaluation as per run_rollout_reward_comparison.py
    for traj_idx in range(batch_size):
        # Concatenate all accumulated video frames
        video_frames_list = trajectories[traj_idx]["predicted_video"]
        video_view0_full = np.concatenate(video_frames_list, axis=0)  # (total_frames, H, W, 3)
        trajectories[traj_idx]["predicted_video"] = video_view0_full
    
    return trajectories


def generate_trajectories_multiprocess(
    cfg: Any,
    scene_data: Dict[str, Any],
    task: str,
    num_trajectories: int,
    trajectory_length: int,
    mp_workers: int,
    mp_start_method: str = "spawn",
    gpu_ids: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate N trajectories concurrently using N separate processes (no batching).

    This is intentionally different from the default batched mode:
    - Each worker process has its own agent/model instance
    - Each worker generates exactly 1 trajectory
    - Parallelism scales with mp_workers (typically set = num_trajectories)
    - If gpu_ids is provided, workers are assigned one GPU each (gpu_ids[worker_rank])
    """
    if num_trajectories <= 0:
        return []
    if mp_workers <= 0:
        mp_workers = num_trajectories
    mp_workers = min(mp_workers, num_trajectories)
    if gpu_ids is not None and len(gpu_ids) < mp_workers:
        mp_workers = len(gpu_ids)

    # With CUDA + multiprocessing, spawn is the safest default.
    ctx = mp.get_context(mp_start_method)

    if gpu_ids is not None:
        counter = ctx.Value("i", 0)
        lock = ctx.Lock()
        initializer = _mp_init_worker_multi_gpu
        initargs = (cfg, task, trajectory_length, gpu_ids, counter, lock)
    else:
        initializer = _mp_init_worker
        initargs = (cfg, task, trajectory_length)

    with cf.ProcessPoolExecutor(
        max_workers=mp_workers,
        mp_context=ctx,
        initializer=initializer,
        initargs=initargs,
    ) as ex:
        futures = [ex.submit(_mp_generate_one, scene_data) for _ in range(num_trajectories)]
        # Wait for all in submission order so trajectories are deterministic and we only eval after all are done.
        trajectories = [fut.result() for fut in futures]

    return trajectories


# ============================================================================
# RFM Evaluation Module
# ============================================================================

def evaluate_trajectory_with_rfm(
    video_frames: np.ndarray,
    task: str,
    rfm_url: str,
    fps: float = 1.0,
    timeout_s: float = 120.0,
    max_retries: int = 2,
) -> float:
    """
    Evaluate trajectory video with RFM and return final reward.
    
    Args:
        video_frames: Video frames array (T, H, W, 3) uint8
        task: Task description string
        rfm_url: RFM evaluation server URL
        fps: FPS for evaluation
        timeout_s: HTTP timeout
        max_retries: Maximum number of retry attempts
    
    Returns:
        Final reward value (last frame's reward)
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            rewards, success_probs = compute_rewards_per_frame(
                eval_server_url=rfm_url,
                video_frames=video_frames,
                task=task,
                max_frames=16,  # Default from use_reward_eval_server.py
                batch_size=32,
                timeout_s=timeout_s,
            )
            
            if rewards.size == 0:
                return 0.0
            
            # Return final reward (last frame)
            return float(rewards[-1])
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                print(f"  RFM evaluation attempt {attempt + 1}/{max_retries} failed: {e}. Retrying...")
                time.sleep(0.5)
            else:
                print(f"  Warning: RFM evaluation failed after {max_retries} attempts: {e}")
                return 0.0  # Return 0 on failure
    
    return 0.0


# ============================================================================
# Trajectory Selection
# ============================================================================

def select_best_trajectory(
    trajectories: List[Dict[str, Any]],
    rfm_rewards: List[float],
) -> Tuple[int, Dict[str, Any]]:
    """
    Select trajectory with highest RFM reward.
    
    Args:
        trajectories: List of trajectory dicts
        rfm_rewards: List of final RFM rewards
    
    Returns:
        (best_idx, best_trajectory)
    """
    if not trajectories or not rfm_rewards:
        raise ValueError("Empty trajectories or rewards list")
    
    best_idx = max(range(len(rfm_rewards)), key=lambda i: rfm_rewards[i])
    return best_idx, trajectories[best_idx]


# ============================================================================
# Format Conversion
# ============================================================================

def trajectory_to_server_format(
    trajectory: Dict[str, Any],
    task: str,
) -> Dict[str, Any]:
    """
    Convert trajectory to server EXECUTE_TRAJECTORY format.
    
    Args:
        trajectory: Trajectory dict with joint_vel, joint_pos, state_fk
        task: Task description string
    
    Returns:
        Server-formatted trajectory dict
    """
    joint_vel = trajectory["joint_vel"]  # (T, 7)
    joint_pos = trajectory.get("joint_pos")
    state_fk = trajectory.get("state_fk")
    
    # Convert numpy arrays to lists
    joint_vel_list = joint_vel.tolist()  # List of 7-element lists
    
    server_traj = {
        "joint_vel": joint_vel_list,
        "instructions": task,
        "start_idx": 0,
        "end_idx": len(joint_vel_list) - 1,
    }
    
    if joint_pos is not None:
        server_traj["joint_pos"] = joint_pos.tolist()
    
    if state_fk is not None:
        server_traj["state_fk"] = state_fk.tolist()
    
    return server_traj


# ============================================================================
# Main Loop
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DROID Trajectory Client with RFM Selection",
    )
    parser.add_argument(
        "--server_url",
        type=str,
        default="tcp://localhost:6000",
        help="TCP server URL (e.g., tcp://localhost:6000)",
    )
    parser.add_argument(
        "--rfm_url",
        type=str,
        default="http://localhost:8004",
        help="RFM evaluation server URL",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task description string",
    )
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=5,
        help="Number of trajectories to generate per step",
    )
    parser.add_argument(
        "--trajectory_length",
        type=int,
        default=1,
        help="Length of each trajectory in interaction steps",
    )
    parser.add_argument(
        "--generation_backend",
        type=str,
        default="batched",
        choices=["batched", "mp"],
        help="How to generate candidate trajectories: "
        "'batched' runs one GPU-batched forward pass; "
        "'mp' runs separate processes concurrently (no batching).",
    )
    parser.add_argument(
        "--mp_workers",
        type=int,
        default=0,
        help="Number of worker processes when --generation_backend mp. "
        "0 means use --num_trajectories.",
    )
    parser.add_argument(
        "--mp_start_method",
        type=str,
        default="spawn",
        choices=["spawn", "forkserver", "fork"],
        help="Multiprocessing start method for --generation_backend mp. "
        "Use 'spawn' for best CUDA safety.",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Comma-separated GPU IDs for multi-GPU multiprocessing (e.g. '0,1,2,3,4'). "
        "When set with --generation_backend mp, each worker uses one GPU. "
        "If unset, workers share the default GPU(s) from the environment.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=12,
        help="Maximum number of steps to execute",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to Ctrl-World checkpoint",
    )
    parser.add_argument(
        "--pi_ckpt",
        type=str,
        default=None,
        help="Path to PI policy checkpoint",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="pickplace",
        help="Task type for config",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="FPS for RFM evaluation",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Optional directory to save selected trajectories",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort on first error",
    )
    
    args = parser.parse_args()

    # Parse --gpu_ids for multi-GPU multiprocessing
    gpu_ids: Optional[List[int]] = None
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
    
    # Initialize config
    cfg = wm_args(task_type=args.task_type)
    if args.ckpt_path:
        cfg.ckpt_path = args.ckpt_path
    if args.pi_ckpt:
        cfg.pi_ckpt = args.pi_ckpt
    
    # Initialize agent
    print("Initializing agent...")
    agent_obj = None
    if args.generation_backend == "batched":
        agent_obj = agent(cfg)
        print("Agent initialized successfully")
    else:
        # In multiprocess mode, each worker initializes its own agent.
        if gpu_ids is not None:
            print(f"Multiprocess generation mode: {len(gpu_ids)} GPU(s) {gpu_ids}")
        else:
            print("Multiprocess generation mode: workers will initialize agents")
    
    # Connect to server
    print(f"Connecting to server: {args.server_url}")
    sock = None
    try:
        sock = connect_to_server(args.server_url)
        print("Connected to server")
        
        # Reset scene to get initial state
        print("Resetting scene...")
        scene_data = reset_scene(sock)
        print("Scene reset complete")
        
        # Main loop
        step = 0
        all_rfm_rewards = []
        
        while step < args.max_steps:
            print(f"\n{'='*60}")
            print(f"Step {step + 1}/{args.max_steps}")
            print(f"{'='*60}")
            
            # Generate trajectories
            print(f"Generating {args.num_trajectories} trajectories...")
            try:
                if args.generation_backend == "batched":
                    trajectories = generate_trajectories(
                        agent_obj,
                        scene_data,
                        args.task,
                        args.num_trajectories,
                        args.trajectory_length,
                    )
                else:
                    trajectories = generate_trajectories_multiprocess(
                        cfg=cfg,
                        scene_data=scene_data,
                        task=args.task,
                        num_trajectories=args.num_trajectories,
                        trajectory_length=args.trajectory_length,
                        mp_workers=args.mp_workers,
                        mp_start_method=args.mp_start_method,
                        gpu_ids=gpu_ids,
                    )
                print(f"Generated {len(trajectories)} trajectories")
            except Exception as e:
                print(f"Error generating trajectories: {e}")
                if args.strict:
                    raise
                break
            
            # Evaluate with RFM
            print(f"Evaluating trajectories with RFM...")
            rfm_rewards = []
            for i, traj in enumerate(trajectories):
                print(f"  Evaluating trajectory {i+1}/{len(trajectories)}...")
                try:
                    reward = evaluate_trajectory_with_rfm(
                        traj["predicted_video"],
                        args.task,
                        args.rfm_url,
                        args.fps,
                    )
                    rfm_rewards.append(reward)
                    print(f"    RFM reward: {reward:.4f}")
                except Exception as e:
                    print(f"    RFM evaluation failed: {e}")
                    rfm_rewards.append(0.0)
                    if args.strict:
                        raise
            
            # Select best trajectory
            if not rfm_rewards or all(r == 0.0 for r in rfm_rewards):
                print("Warning: All RFM rewards are 0, selecting first trajectory")
                best_idx = 0
            else:
                best_idx, best_trajectory = select_best_trajectory(trajectories, rfm_rewards)
                print(f"Selected trajectory {best_idx} with RFM reward: {rfm_rewards[best_idx]:.4f}")
            
            best_trajectory = trajectories[best_idx]
            all_rfm_rewards.append(rfm_rewards[best_idx])
            
            # Convert to server format
            server_trajectory = trajectory_to_server_format(best_trajectory, args.task)
            
            # Save trajectory if requested
            if args.save_dir:
                import os
                import json
                save_path = Path(args.save_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                traj_file = save_path / f"trajectory_step_{step:02d}.json"
                with open(traj_file, "w") as f:
                    json.dump({
                        "step": step,
                        "rfm_reward": rfm_rewards[best_idx],
                        "trajectory": {
                            "joint_vel": best_trajectory["joint_vel"].tolist(),
                            "joint_pos": best_trajectory["joint_pos"].tolist() if "joint_pos" in best_trajectory else None,
                            "state_fk": best_trajectory["state_fk"].tolist() if "state_fk" in best_trajectory else None,
                        },
                    }, f, indent=2)
                print(f"Saved trajectory to {traj_file}")
            
            # Execute trajectory
            print("Executing trajectory on server...")
            try:
                result = execute_trajectory(sock, server_trajectory)
                print(f"Execution result: success={result.get('success')}, num_steps={result.get('num_steps')}")
                
                if not result.get("success", False):
                    error = result.get("error", "Unknown error")
                    print(f"Execution failed: {error}")
                    if args.strict:
                        print("Strict mode: aborting due to execution failure")
                        break
                    # Continue to next step even on failure
                    print("Continuing despite execution failure...")
                
                # Check if done
                if result.get("done", False) or result.get("truncated", False):
                    print("Task completed or truncated")
                    break
                
                # Get next scene
                if "next_scene" in result:
                    scene_data = result["next_scene"]
                    print("Received next scene")
                else:
                    # Fallback: request scene
                    try:
                        scene_data = get_scene(sock)
                        print("Requested next scene")
                    except Exception as e:
                        print(f"Error getting next scene: {e}")
                        if args.strict:
                            raise
                        break
                    
            except (ConnectionError, socket.error, OSError) as e:
                print(f"Socket error executing trajectory: {e}")
                if args.strict:
                    raise
                print("Attempting to reconnect...")
                try:
                    sock.close()
                except Exception:
                    pass
                try:
                    sock = connect_to_server(args.server_url)
                    print("Reconnected to server")
                    # Try to get current scene
                    scene_data = get_scene(sock)
                except Exception as reconnect_error:
                    print(f"Failed to reconnect: {reconnect_error}")
                    break
            except Exception as e:
                print(f"Error executing trajectory: {e}")
                if args.strict:
                    raise
                break
            
            step += 1
        
        # Summary
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        print(f"Total steps: {step}")
        print(f"Average RFM reward: {np.mean(all_rfm_rewards) if all_rfm_rewards else 0:.4f}")
        print(f"Max RFM reward: {max(all_rfm_rewards) if all_rfm_rewards else 0:.4f}")
        print(f"Min RFM reward: {min(all_rfm_rewards) if all_rfm_rewards else 0:.4f}")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        if args.strict:
            raise
    finally:
        if sock:
            close_connection(sock)
            print("Connection closed")


if __name__ == "__main__":
    main()

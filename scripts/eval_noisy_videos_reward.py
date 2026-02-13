#!/usr/bin/env python3
"""
Evaluate noisy/clean rollout videos with RFM (port 8000) and RoboReward (port 8005):
crop to single view, compute per-frame rewards, rolling average, and plot per-video
(clean at center, action-noise variants as bounds) and summary plots.

Supports:
  - clean / light / medium / heavy / severe (from generate_noisy_videos_quick.sh)
  - action_0.01, action_0.02, action_0.03, etc. (from generate_noisy_videos.sh)

Usage:
  python scripts/eval_noisy_videos_reward.py --video_dir synthetic_traj/Rollouts_replay/video \\
    --output_dir ./noisy_eval_out --rfm_url http://localhost:8001 --roboreward_url http://localhost:8002
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def check_reward_server(url: str, timeout: float = 3.0) -> tuple[bool, str]:
    """
    Check if a reward eval server is reachable (any HTTP response = up).
    Returns (reachable, message).
    """
    base = url.rstrip("/")
    try:
        r = requests.get(base, timeout=timeout)
        return True, f"OK (HTTP {r.status_code})"
    except requests.exceptions.ConnectTimeout:
        return False, "timeout"
    except requests.exceptions.ConnectionError as e:
        return False, f"connection failed: {e}"
    except Exception as e:
        return False, str(e)


def trajectory_key_from_stem(stem: str) -> str:
    """
    From replay stem time_{uuid}_traj_{split}_{val_id}_{start}_{text_id}[_repN],
    return traj_{split}_{val_id}_{start}_{text_id}[_repN] so we can group across noise configs.
    If repeat_id is present (_repN), it's included in the key so each repeat is a unique trajectory.
    """
    parts = stem.split("_")
    if "traj" not in parts:
        return stem
    try:
        j = parts.index("traj")
        return "_".join(parts[j:])
    except ValueError:
        return stem


def get_task_from_stem(stem: str, info_dir: Path | None = None) -> str:
    """
    Resolve task string for reward eval. For replay stems:
    time_*_traj_{split}_{val_id}_{start}_{text_id}[_repN] -> task = text_id with spaces (ignore repN).
    """
    parts = stem.split("_")
    if "traj" not in parts or len(parts) < 5:
        return stem.replace("_", " ")
    j = parts.index("traj")
    # traj, split, val_id, start_idx, then rest is text_id (possibly ending with repN)
    if j + 4 >= len(parts):
        return stem.replace("_", " ")
    text_id_parts = parts[j + 4 :]
    # Remove _repN suffix if present (last part starting with "rep" and followed by digits)
    if text_id_parts and text_id_parts[-1].startswith("rep") and text_id_parts[-1][3:].isdigit():
        text_id_parts = text_id_parts[:-1]
    text_id = "_".join(text_id_parts)
    return text_id.replace("_", " ")


def run_split_video(
    video_path: Path,
    output_dir: Path,
    project_root: Path,
    layout: str = "auto",
    row: str = "bottom",
) -> list[Path]:
    """Split one 3-view video into view0, view1, view2; return paths to single-view mp4s."""
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "split_rollout_views_for_rfm.py"),
        str(video_path),
        "--output-dir",
        str(output_dir),
        "--layout",
        layout,
        "--row",
        row,
    ]
    subprocess.run(cmd, cwd=str(project_root), check=True, capture_output=True)
    stem = video_path.stem
    return [
        output_dir / f"{stem}_view0.mp4",
        output_dir / f"{stem}_view1.mp4",
        output_dir / f"{stem}_view2.mp4",
    ]


def run_reward_eval(
    video_path: Path,
    task: str,
    eval_url: str,
    out_npy: Path,
    fps: float,
    project_root: Path,
) -> None:
    """Run use_reward_eval_server; writes out_npy."""
    cmd = [
        sys.executable,
        str(project_root / "use_reward_eval_server.py"),
        "--video",
        str(video_path),
        "--task",
        task,
        "--eval-server-url",
        eval_url,
        "--out",
        str(out_npy),
        "--fps",
        str(fps),
    ]
    subprocess.run(cmd, cwd=str(project_root), check=True, capture_output=True)


def _is_connection_error(stderr_text: str) -> bool:
    """True if stderr indicates a transient connection/server error (worth retrying)."""
    t = (stderr_text or "").lower()
    return (
        "remote end closed connection" in t
        or "connection aborted" in t
        or "connectionerror" in t
        or "connection refused" in t
        or "remotedisconnected" in t
    )


def _is_corrupt_video_error(stderr_text: str) -> bool:
    """True if stderr indicates the video file is corrupt (re-split and retry)."""
    t = (stderr_text or "").lower()
    return (
        "moov atom not found" in t
        or "error reading" in t
        or "invalid data found when processing input" in t
        or "error extracting frames" in t
    )


def run_reward_eval_with_retries(
    view_path: Path,
    task: str,
    eval_url: str,
    out_npy: Path,
    fps: float,
    project_root: Path,
    run_split_video_fn: Callable[[], None],
    log_fn: Callable[[str], None] | None = None,
    max_retries: int = 3,
    retry_sleep_s: float = 5.0,
) -> bool:
    """
    Run reward eval with retries. On connection errors, retry up to max_retries.
    On corrupt-video errors, re-run split once and retry eval. Returns True if out_npy exists.
    """
    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    last_stderr = ""

    for attempt in range(max_retries):
        try:
            run_reward_eval(view_path, task, eval_url, out_npy, fps, project_root)
            return out_npy.exists()
        except subprocess.CalledProcessError as e:
            last_stderr = (e.stderr or b"").decode("utf-8", errors="replace").strip()

            # Corrupt video: re-split once then retry
            if _is_corrupt_video_error(last_stderr) and view_path.exists():
                _log(f"  Corrupt split video detected; re-splitting and retrying...")
                try:
                    view_path.unlink()
                    run_split_video_fn()
                except Exception as reex:
                    _log(f"  Re-split failed: {reex}")
                    if attempt == max_retries - 1:
                        _log(f"  Eval failed: {last_stderr[:500]}")
                    continue
                if not view_path.exists():
                    _log(f"  Re-split did not produce {view_path.name}")
                    continue
                # Retry eval (no extra sleep)
                try:
                    run_reward_eval(view_path, task, eval_url, out_npy, fps, project_root)
                    return out_npy.exists()
                except subprocess.CalledProcessError as e2:
                    last_stderr = (e2.stderr or b"").decode("utf-8", errors="replace").strip()
                    _log(f"  Eval failed after re-split: {last_stderr[:500]}")
                continue

            # Connection error: retry with backoff
            elif _is_connection_error(last_stderr) and attempt < max_retries - 1:
                _log(f"  Connection error (attempt {attempt + 1}/{max_retries}), retrying in {retry_sleep_s}s...")
                time.sleep(retry_sleep_s)
            else:
                _log(f"  Eval failed: {last_stderr[:500]}")
    return out_npy.exists()


def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean; window must be odd or we use (window-1)//2 padding."""
    if arr.size == 0:
        return arr
    window = max(1, min(window, len(arr)))
    half = window // 2
    out = np.full_like(arr, np.nan, dtype=np.float64)
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        out[i] = np.nanmean(arr[lo:hi])
    return out


def discover_videos_by_trajectory(video_dir: Path) -> dict[str, dict[str, Path]]:
    """
    video_dir has subdirs = noise configs (clean, heavy, light, action_0.01, ...).
    Each subdir has .mp4 files with stem time_*_traj_...
    Return: { trajectory_key: { noise_config: path } }
    """
    video_dir = Path(video_dir)
    if not video_dir.exists():
        return {}

    traj_to_config_paths: dict[str, dict[str, Path]] = {}
    for sub in sorted(video_dir.iterdir()):
        if not sub.is_dir():
            continue
        noise_config = sub.name
        for mp4 in sorted(sub.glob("*.mp4")):
            key = trajectory_key_from_stem(mp4.stem)
            if key not in traj_to_config_paths:
                traj_to_config_paths[key] = {}
            traj_to_config_paths[key][noise_config] = mp4

    return traj_to_config_paths


def ensure_rewards(
    trajectory_key: str,
    noise_config: str,
    video_path: Path,
    single_views_dir: Path,
    cache_dir: Path,
    task: str,
    rfm_url: str,
    roboreward_url: str,
    fps: float,
    project_root: Path,
    view_index: int = 0,
    log_fn: Callable[[str], None] | None = None,
    rfm_only: bool = False,
) -> tuple[Path | None, Path | None]:
    """
    Split video to single views (if needed), run RFM and optionally RoboReward eval, save .npy in cache.
    Returns (path_to_rfm_npy, path_to_roboreward_npy) or (None, None) on failure.
    """
    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    stem = video_path.stem
    view_path = single_views_dir / f"{stem}_view{view_index}.mp4"

    if not view_path.exists():
        try:
            run_split_video(video_path, single_views_dir, project_root)
        except Exception as e:
            _log(f"  {noise_config}: split failed: {e}")
            return None, None
        if not view_path.exists():
            _log(f"  {noise_config}: split ran but {view_path.name} missing")
            return None, None

    run_dir = cache_dir / trajectory_key / noise_config
    run_dir.mkdir(parents=True, exist_ok=True)
    rfm_npy = run_dir / f"view{view_index}_rfm_rewards.npy"
    robo_npy = run_dir / f"view{view_index}_roboreward_rewards.npy"

    def _resplit() -> None:
        run_split_video(video_path, single_views_dir, project_root)

    def _log_prefixed(msg: str) -> None:
        _log(f"  {noise_config}: {msg}")

    if not rfm_npy.exists():
        run_reward_eval_with_retries(
            view_path,
            task,
            rfm_url,
            rfm_npy,
            fps,
            project_root,
            _resplit,
            log_fn=_log_prefixed,
        )
        if not rfm_npy.exists():
            _log(f"  {noise_config}: RFM eval failed after retries")
    if not rfm_only and not robo_npy.exists():
        run_reward_eval_with_retries(
            view_path,
            task,
            roboreward_url,
            robo_npy,
            fps,
            project_root,
            _resplit,
            log_fn=_log_prefixed,
        )
        if not robo_npy.exists():
            _log(f"  {noise_config}: RoboReward eval failed after retries")

    return (rfm_npy if rfm_npy.exists() else None, None if rfm_only else (robo_npy if robo_npy.exists() else None))


def normalize_to_npoints(rewards: np.ndarray, n_pts: int = 100) -> np.ndarray:
    """Interpolate reward curve to n_pts on [0, 1] for alignment across videos."""
    if rewards.size == 0:
        return np.full(n_pts, np.nan)
    x_orig = np.linspace(0, 1, len(rewards))
    x_norm = np.linspace(0, 1, n_pts)
    return np.interp(x_norm, x_orig, rewards.astype(np.float64))


def plot_per_video(
    trajectory_key: str,
    task_label: str,
    results: dict[str, dict[str, np.ndarray]],
    center_config: str,
    other_configs: list[str],
    plots_dir: Path,
    rolling_window: int,
    n_pts: int = 100,
    rfm_only: bool = False,
) -> None:
    """
    results[noise_config] = { "rfm": array, "roboreward": array } (raw per-frame).
    center_config = "clean" or reference; other_configs = list of variant names.
    Plot: 1) RFM only (clean center, bounds from others), 2) RoboReward only (if not rfm_only), 3) both (if not rfm_only).
    """
    # Align all to n_pts and apply rolling mean
    def smooth_and_norm(arr: np.ndarray) -> np.ndarray:
        r = np.asarray(arr, dtype=np.float64)
        if r.size == 0:
            return np.full(n_pts, np.nan)
        smoothed = rolling_mean(r, rolling_window)
        return normalize_to_npoints(smoothed, n_pts)

    x = np.linspace(0, 1, n_pts)
    safe_key = re.sub(r'[^\w\-]', '_', trajectory_key)[:80]

    # --- 1) RFM only ---
    fig, ax = plt.subplots(figsize=(8, 4))
    if center_config in results and "rfm" in results[center_config]:
        center_rfm = smooth_and_norm(results[center_config]["rfm"])
        ax.plot(x, center_rfm, color="black", linewidth=2.5, label=center_config, zorder=10)
    other_rfm = []
    for c in other_configs:
        if c in results and "rfm" in results[c]:
            other_rfm.append(smooth_and_norm(results[c]["rfm"]))
    if other_rfm:
        other_arr = np.array(other_rfm)
        ax.fill_between(
            x,
            np.nanmin(other_arr, axis=0),
            np.nanmax(other_arr, axis=0),
            alpha=0.35,
            color="steelblue",
            label="other variants (min–max)",
        )
        for i, c in enumerate(other_configs):
            if c in results and "rfm" in results[c]:
                ax.plot(x, smooth_and_norm(results[c]["rfm"]), alpha=0.7, label=c)
    ax.set_xlabel("Normalized time")
    ax.set_ylabel("Reward (RFM)")
    ax.set_title(f"RFM (8000) — {task_label}")
    ax.legend(loc="best", fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(plots_dir / f"per_video_{safe_key}_rfm.png", dpi=150)
    plt.close(fig)

    if rfm_only:
        # Only RFM plot and final-by-config (RFM only) below
        pass
    else:
        # --- 2) RoboReward only ---
        fig, ax = plt.subplots(figsize=(8, 4))
        if center_config in results and "roboreward" in results[center_config]:
            center_robo = smooth_and_norm(results[center_config]["roboreward"])
            ax.plot(x, center_robo, color="black", linewidth=2.5, label=center_config, zorder=10)
        other_robo = []
        for c in other_configs:
            if c in results and "roboreward" in results[c]:
                other_robo.append(smooth_and_norm(results[c]["roboreward"]))
        if other_robo:
            other_arr = np.array(other_robo)
            ax.fill_between(
                x,
                np.nanmin(other_arr, axis=0),
                np.nanmax(other_arr, axis=0),
                alpha=0.35,
                color="darkorange",
                label="other variants (min–max)",
            )
            for c in other_configs:
                if c in results and "roboreward" in results[c]:
                    ax.plot(x, smooth_and_norm(results[c]["roboreward"]), alpha=0.7, label=c)
        ax.set_xlabel("Normalized time")
        ax.set_ylabel("Reward (RoboReward)")
        ax.set_title(f"RoboReward (8005) — {task_label}")
        ax.legend(loc="best", fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(plots_dir / f"per_video_{safe_key}_roboreward.png", dpi=150)
        plt.close(fig)

        # --- 3) Both models ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        if center_config in results and "rfm" in results[center_config]:
            ax1.plot(x, smooth_and_norm(results[center_config]["rfm"]), color="black", linewidth=2, label=f"{center_config} (RFM)")
        for c in other_configs:
            if c in results and "rfm" in results[c]:
                ax1.plot(x, smooth_and_norm(results[c]["rfm"]), alpha=0.7, label=c)
        ax1.set_ylabel("Reward (RFM)")
        ax1.set_title("RFM (8000)")
        ax1.legend(loc="best", fontsize=7)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        if center_config in results and "roboreward" in results[center_config]:
            ax2.plot(x, smooth_and_norm(results[center_config]["roboreward"]), color="black", linewidth=2, label=f"{center_config} (Robo)")
        for c in other_configs:
            if c in results and "roboreward" in results[c]:
                ax2.plot(x, smooth_and_norm(results[c]["roboreward"]), alpha=0.7, label=c)
        ax2.set_xlabel("Normalized time")
        ax2.set_ylabel("Reward (RoboReward)")
        ax2.set_title("RoboReward (8005)")
        ax2.legend(loc="best", fontsize=7)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        fig.suptitle(f"Both models — {task_label}", y=1.02)
        plt.tight_layout()
        fig.savefig(plots_dir / f"per_video_{safe_key}_both.png", dpi=150)
        plt.close(fig)

    # --- 4) Final reward by config (bar chart) ---
    configs = [center_config] + [c for c in other_configs if c in results]
    rfm_fin = []
    robo_fin = []
    for c in configs:
        if c in results:
            r = results[c].get("rfm")
            rfm_fin.append(float(r[-1]) if r is not None and len(r) else np.nan)
            r = results[c].get("roboreward")
            robo_fin.append(float(r[-1]) if r is not None and len(r) else np.nan)
        else:
            rfm_fin.append(np.nan)
            robo_fin.append(np.nan)
    x_pos = np.arange(len(configs))
    if rfm_only:
        fig, ax1 = plt.subplots(figsize=(5, 4))
        ax1.bar(x_pos, rfm_fin, color="steelblue", alpha=0.8)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(configs, rotation=45, ha="right")
        ax1.set_ylabel("Final reward (RFM)")
        ax1.set_title("RFM final reward by config")
        ax1.set_ylim(0, 1)
        fig.suptitle(f"Final reward by config — {task_label}", y=1.02)
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.bar(x_pos, rfm_fin, color="steelblue", alpha=0.8)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(configs, rotation=45, ha="right")
        ax1.set_ylabel("Final reward (RFM)")
        ax1.set_title("RFM final reward by config")
        ax1.set_ylim(0, 1)
        ax2.bar(x_pos, robo_fin, color="darkorange", alpha=0.8)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(configs, rotation=45, ha="right")
        ax2.set_ylabel("Final reward (RoboReward)")
        ax2.set_title("RoboReward final reward by config")
        ax2.set_ylim(0, 1)
        fig.suptitle(f"Final reward by config — {task_label}", y=1.02)
    plt.tight_layout()
    fig.savefig(plots_dir / f"per_video_{safe_key}_final_by_config.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Eval noisy/clean rollout videos with RFM and RoboReward; plot per-video and summaries.",
    )
    ap.add_argument(
        "--video_dir",
        type=str,
        default="synthetic_traj/Rollouts_replay/video",
        help="Root dir with subdirs = noise configs (clean, heavy, action_0.01, ...)",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default="./noisy_eval_out",
        help="Output dir: cache rewards here, write plots to output_dir/plots",
    )
    ap.add_argument("--rfm_url", type=str, default="http://localhost:8000", help="RFM eval server (RFM-RL)")
    ap.add_argument("--roboreward_url", type=str, default="http://localhost:8005", help="RoboReward eval server")
    ap.add_argument("--fps", type=float, default=1.0, help="FPS for use_reward_eval_server")
    ap.add_argument(
        "--rolling_window",
        type=int,
        default=5,
        help="Window size for rolling average over timesteps",
    )
    ap.add_argument(
        "--center",
        type=str,
        default="clean",
        help="Config to use as center/reference (e.g. clean). If missing, first available is used.",
    )
    ap.add_argument(
        "--max_trajectories",
        type=int,
        default=None,
        help="Limit number of trajectories to process (default: all)",
    )
    ap.add_argument(
        "--view",
        type=int,
        default=0,
        help="View index for reward eval (0=ext1, 1=ext2, 2=wrist)",
    )
    ap.add_argument(
        "--check_servers_only",
        action="store_true",
        help="Only check if RFM and RoboReward servers are reachable, then exit.",
    )
    ap.add_argument(
        "--rfm_only",
        action="store_true",
        help="Only run RFM eval and plots; skip RoboReward (e.g. when RoboReward server is down).",
    )
    args = ap.parse_args()

    def log(msg: str) -> None:
        print(f"[eval_noisy_videos_reward] {msg}", flush=True)

    # Check reward servers
    rfm_ok, rfm_msg = check_reward_server(args.rfm_url)
    log(f"RFM ({args.rfm_url}): {'reachable' if rfm_ok else 'NOT reachable'} — {rfm_msg}")
    if not args.rfm_only:
        robo_ok, robo_msg = check_reward_server(args.roboreward_url)
        log(f"RoboReward ({args.roboreward_url}): {'reachable' if robo_ok else 'NOT reachable'} — {robo_msg}")
    if args.check_servers_only:
        if args.rfm_only:
            sys.exit(0 if rfm_ok else 1)
        sys.exit(0 if (rfm_ok and robo_ok) else 1)

    project_root = Path(__file__).resolve().parent.parent
    video_dir = Path(args.video_dir)
    if not video_dir.is_absolute():
        video_dir = (project_root / video_dir).resolve()
    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "reward_cache"
    single_views_dir = output_dir / "single_views"
    plots_dir = output_dir / "plots"
    cache_dir.mkdir(parents=True, exist_ok=True)
    single_views_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    traj_to_configs = discover_videos_by_trajectory(video_dir)
    if not traj_to_configs:
        log(f"No videos found under {video_dir}. Expect subdirs (clean, heavy, action_0.01, ...) with .mp4 files.")
        sys.exit(1)

    # Decide center config: prefer args.center if present in any traj, else first alphabetically
    all_configs = set()
    for configs in traj_to_configs.values():
        all_configs.update(configs.keys())
    all_configs = sorted(all_configs)
    center_config = args.center if args.center in all_configs else (all_configs[0] if all_configs else "clean")
    other_configs = [c for c in all_configs if c != center_config]
    log(f"Center config: {center_config}, other configs: {other_configs}")

    trajectories = sorted(traj_to_configs.keys())
    if args.max_trajectories is not None:
        trajectories = trajectories[: args.max_trajectories]
    log(f"Processing {len(trajectories)} trajectories")

    for ti, traj_key in enumerate(trajectories):
        config_paths = traj_to_configs[traj_key]
        # Get task from first available stem
        first_stem = next((Path(p).stem for p in config_paths.values()), traj_key)
        task = get_task_from_stem(first_stem)
        task_label = task[:50] + "…" if len(task) > 50 else task

        log(f"Trajectory {ti+1}/{len(trajectories)}: {traj_key}")
        results: dict[str, dict[str, np.ndarray]] = {}

        for noise_config, video_path in config_paths.items():
            rfm_npy, robo_npy = ensure_rewards(
                traj_key,
                noise_config,
                video_path,
                single_views_dir,
                cache_dir,
                task,
                args.rfm_url,
                args.roboreward_url,
                args.fps,
                project_root,
                view_index=args.view,
                log_fn=log,
                rfm_only=args.rfm_only,
            )
            if rfm_npy is not None:
                results[noise_config] = results.get(noise_config, {})
                results[noise_config]["rfm"] = np.load(rfm_npy)
            if not args.rfm_only and robo_npy is not None:
                results[noise_config] = results.get(noise_config, {})
                results[noise_config]["roboreward"] = np.load(robo_npy)

        if not results:
            log(f"  No rewards for {traj_key}, skipping plots")
            continue

        plot_per_video(
            traj_key,
            task_label,
            results,
            center_config,
            other_configs,
            plots_dir,
            args.rolling_window,
            rfm_only=args.rfm_only,
        )

    # Aggregate summary: mean final reward by config across trajectories
    all_rfm_by_config: dict[str, list[float]] = {}
    all_robo_by_config: dict[str, list[float]] = {}
    for traj_key in trajectories:
        config_paths = traj_to_configs[traj_key]
        for noise_config in config_paths:
            run_dir = cache_dir / traj_key / noise_config
            rfm_npy = run_dir / f"view{args.view}_rfm_rewards.npy"
            robo_npy = run_dir / f"view{args.view}_roboreward_rewards.npy"
            if rfm_npy.exists():
                r = np.load(rfm_npy)
                if r.size > 0:
                    all_rfm_by_config.setdefault(noise_config, []).append(float(r[-1]))
            if not args.rfm_only and robo_npy.exists():
                r = np.load(robo_npy)
                if r.size > 0:
                    all_robo_by_config.setdefault(noise_config, []).append(float(r[-1]))

    if all_rfm_by_config or all_robo_by_config:
        configs_sorted = sorted(set(all_rfm_by_config.keys()) | set(all_robo_by_config.keys()))
        x_pos = np.arange(len(configs_sorted))
        if args.rfm_only:
            fig, ax1 = plt.subplots(figsize=(6, 4))
            rfm_means = [np.mean(all_rfm_by_config.get(c, [np.nan])) for c in configs_sorted]
            rfm_stds = [np.std(all_rfm_by_config.get(c, [np.nan])) if all_rfm_by_config.get(c) else 0.0 for c in configs_sorted]
            ax1.bar(x_pos, rfm_means, yerr=rfm_stds, color="steelblue", alpha=0.8, capsize=4)
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(configs_sorted, rotation=45, ha="right")
            ax1.set_ylabel("Mean final reward (RFM)")
            ax1.set_title("Aggregate: RFM final reward by config")
            ax1.set_ylim(0, 1)
            fig.suptitle("Summary across all trajectories (RFM only)", y=1.02)
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            rfm_means = [np.mean(all_rfm_by_config.get(c, [np.nan])) for c in configs_sorted]
            rfm_stds = [np.std(all_rfm_by_config.get(c, [np.nan])) if all_rfm_by_config.get(c) else 0.0 for c in configs_sorted]
            ax1.bar(x_pos, rfm_means, yerr=rfm_stds, color="steelblue", alpha=0.8, capsize=4)
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(configs_sorted, rotation=45, ha="right")
            ax1.set_ylabel("Mean final reward (RFM)")
            ax1.set_title("Aggregate: RFM final reward by config")
            ax1.set_ylim(0, 1)
            robo_means = [np.mean(all_robo_by_config.get(c, [np.nan])) for c in configs_sorted]
            robo_stds = [np.std(all_robo_by_config.get(c, [np.nan])) if all_robo_by_config.get(c) else 0.0 for c in configs_sorted]
            ax2.bar(x_pos, robo_means, yerr=robo_stds, color="darkorange", alpha=0.8, capsize=4)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(configs_sorted, rotation=45, ha="right")
            ax2.set_ylabel("Mean final reward (RoboReward)")
            ax2.set_title("Aggregate: RoboReward final reward by config")
            ax2.set_ylim(0, 1)
            fig.suptitle("Summary across all trajectories", y=1.02)
        plt.tight_layout()
        fig.savefig(plots_dir / "summary_final_reward_by_config.png", dpi=150)
        plt.close(fig)
        log("Wrote plots/summary_final_reward_by_config.png")

    log(f"Done. Output: {output_dir}")
    log(f"  reward_cache/<traj_key>/<noise_config>/view*_rfm_rewards.npy, view*_roboreward_rewards.npy")
    log(f"  plots/per_video_*_rfm.png, *_roboreward.png, *_both.png, *_final_by_config.png")
    log(f"  plots/summary_final_reward_by_config.png (if any rewards)")


if __name__ == "__main__":
    main()

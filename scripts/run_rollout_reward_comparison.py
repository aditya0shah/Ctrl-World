#!/usr/bin/env python3
"""
Run X rollout trajectories, split into single-view videos, evaluate with RFM-RL and
RoboReward, and produce per-run and aggregate RFM vs RoboReward comparison plots.

Usage:
  python scripts/run_rollout_reward_comparison.py --num_trajectories 5 --task_type pickplace \\
    --output_dir ./benchmark_$(date +%Y%m%d_%H%M%S) [optional passthrough for rollout]

Requires roboreward url on --roboreward_url (default http://localhost:8003) and RFM-RL url on
--rfm-rl-url (default http://localhost:8004).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_task_from_info_or_stem(stem: str, info_dir: Path) -> str:
    """
    Resolve task string for use_reward_eval_server from stem and optional info JSON.

    - Regex on stem: ^(.+)_\d+_(.+)$ → (prefix, text_id).
    - Glob {info_dir}/{prefix}_*_{text_id}.json; use first match and read
      ``instructions`` as task.
    - If none: task = text_id.replace("_", " ").
    """
    mob = re.match(r"^(.+)_\d+_(.+)$", stem)
    if not mob:
        return stem.replace("_", " ")
    prefix, text_id = mob.group(1), mob.group(2)
    pattern = str(info_dir / f"{prefix}_*_{text_id}.json")
    matches = glob.glob(pattern)
    if matches:
        with open(matches[0]) as f:
            data = json.load(f)
        return data.get("instructions", text_id.replace("_", " "))
    return text_id.replace("_", " ")


def trajectory_slug_from_stem(stem: str) -> str:
    """
    From stem {task_type}_time_{uuid}_traj_{val_id}_{start}_{skip}_{text_id},
    return time_{uuid}_traj_{val_id}.
    """
    parts = stem.split("_")
    if "time" not in parts or "traj" not in parts:
        return stem
    try:
        i = parts.index("time")
        j = parts.index("traj")
        if j + 1 < len(parts):
            return "_".join(parts[i : j + 2])
    except ValueError:
        pass
    return stem


def run_rollout(
    task_type: str,
    save_dir: Path,
    project_root: Path,
    passthrough: dict,
    env: dict | None,
    *,
    show_logs: bool = True,
) -> list[Path]:
    """Run rollout_interact_pi once; parse stdout for 'Saving video to <path>' and return those paths.
    If show_logs=True, stream stdout/stderr and return [] (rely on fallback to find new videos)."""
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "rollout_interact_pi.py"),
        "--task_type", task_type,
        "--save_dir", str(save_dir),
    ]
    for k, v in passthrough.items():
        if v is not None:
            cmd.extend([f"--{k}", str(v)])
    if show_logs:
        out = subprocess.run(cmd, cwd=str(project_root), env=env)
        if out.returncode != 0:
            raise RuntimeError(f"rollout_interact_pi exited {out.returncode}")
        return []
    out = subprocess.run(
        cmd,
        cwd=str(project_root),
        env=env,
        capture_output=True,
        text=True,
    )
    paths = []
    for line in (out.stdout or "").splitlines():
        m = re.search(r"Saving video to (\S+)", line)
        if m:
            p = Path(m.group(1))
            if p.exists():
                paths.append(p)
    if out.returncode != 0:
        sys.stderr.write(out.stderr or "")
        raise RuntimeError(f"rollout_interact_pi exited {out.returncode}")
    return paths


def run_split(video_dir: Path, single_views: Path, project_root: Path) -> None:
    """Run split_rollout_views_for_rfm on video_dir, output to single_views."""
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "split_rollout_views_for_rfm.py"),
        str(video_dir),
        "-o", str(single_views),
    ]
    subprocess.run(cmd, cwd=str(project_root), check=True)


def run_reward_eval(
    video_path: Path,
    task: str,
    eval_url: str,
    out_npy: Path,
    fps: float,
    project_root: Path,
) -> None:
    """Run use_reward_eval_server; writes out_npy, {stem}_success_probs.npy, {stem}_progress_success.png."""
    cmd = [
        sys.executable,
        str(project_root / "use_reward_eval_server.py"),
        "--video", str(video_path),
        "--task", task,
        "--eval-server-url", eval_url,
        "--out", str(out_npy),
        "--fps", str(fps),
    ]
    subprocess.run(cmd, cwd=str(project_root), check=True)


def _load_final_mean(npy_path: Path) -> tuple[float, float]:
    """Return (final_reward, mean_reward); (nan, nan) if missing."""
    if not npy_path.exists():
        return (float("nan"), float("nan"))
    r = np.load(npy_path)
    if r.size == 0:
        return (float("nan"), float("nan"))
    return (float(r[-1]), float(np.mean(r)))


def _rel(output_dir: Path, p: str | Path) -> str:
    """Path relative to output_dir, or original if not under it."""
    try:
        return str(Path(p).resolve().relative_to(Path(output_dir).resolve()))
    except ValueError:
        return str(p)


def build_report(
    manifest: list[dict],
    output_dir: Path,
    rfm_url: str,
    roboreward_url: str,
) -> str:
    """Build REPORT.md with per-run metrics and 'trajectories to look at'."""
    lines: list[str] = []
    lines.append("# Reward comparison report\n")
    lines.append("## Summary\n")
    lines.append(f"- **Runs:** {len(manifest)}")
    lines.append(f"- **RFM-RL:** `{rfm_url}`")
    lines.append(f"- **RoboReward:** `{roboreward_url}`")
    lines.append("")

    # Per-run: load rewards and compute derived
    rows: list[dict] = []
    for m in manifest:
        run_dir = Path(m["run_folder"])
        v0_rfm_fin, v0_rfm_avg = _load_final_mean(run_dir / "view0_rfm_rewards.npy")
        v0_robo_fin, v0_robo_avg = _load_final_mean(run_dir / "view0_roboreward_rewards.npy")
        v1_rfm_fin, v1_rfm_avg = _load_final_mean(run_dir / "view1_rfm_rewards.npy")
        v1_robo_fin, v1_robo_avg = _load_final_mean(run_dir / "view1_roboreward_rewards.npy")

        rfm_robo_delta = max(
            abs(v0_rfm_fin - v0_robo_fin) if np.isfinite(v0_rfm_fin) and np.isfinite(v0_robo_fin) else 0.0,
            abs(v1_rfm_fin - v1_robo_fin) if np.isfinite(v1_rfm_fin) and np.isfinite(v1_robo_fin) else 0.0,
        )
        view_delta = max(
            abs(v0_rfm_fin - v1_rfm_fin) if np.isfinite(v0_rfm_fin) and np.isfinite(v1_rfm_fin) else 0.0,
            abs(v0_robo_fin - v1_robo_fin) if np.isfinite(v0_robo_fin) and np.isfinite(v1_robo_fin) else 0.0,
        )
        mean_final = np.nanmean([v0_rfm_fin, v0_robo_fin, v1_rfm_fin, v1_robo_fin])

        rows.append({
            "run": m["run_index"],
            "slug": m["trajectory_slug"],
            "task": m["task"],
            "video_rel": _rel(output_dir, m["video_path"]),
            "run_folder_rel": _rel(output_dir, m["run_folder"]),
            "v0_rfm_fin": v0_rfm_fin, "v0_rfm_avg": v0_rfm_avg,
            "v0_robo_fin": v0_robo_fin, "v0_robo_avg": v0_robo_avg,
            "v1_rfm_fin": v1_rfm_fin, "v1_rfm_avg": v1_rfm_avg,
            "v1_robo_fin": v1_robo_fin, "v1_robo_avg": v1_robo_avg,
            "rfm_robo_delta": rfm_robo_delta,
            "view_delta": view_delta,
            "mean_final": mean_final,
        })

    def _f(x: float) -> str:
        return f"{x:.3f}" if np.isfinite(x) else "—"

    def _task(s: str) -> str:
        t = s.replace("|", "/")[:26]
        return t + "…" if len(s) > 26 else t

    # Table
    lines.append("## Per-run metrics\n")
    lines.append("| Run | Trajectory | Task | v0 RFM (fin) | v0 Robo (fin) | v1 RFM (fin) | v1 Robo (fin) | RFM–Robo Δ | View Δ | Mean fin |")
    lines.append("|-----|------------|------|--------------|---------------|--------------|---------------|------------|--------|----------|")
    for r in rows:
        lines.append(
            f"| {r['run']} | `{r['slug']}` | {_task(r['task'])} | "
            f"{_f(r['v0_rfm_fin'])} | {_f(r['v0_robo_fin'])} | {_f(r['v1_rfm_fin'])} | {_f(r['v1_robo_fin'])} | "
            f"{_f(r['rfm_robo_delta'])} | {_f(r['view_delta'])} | {_f(r['mean_final'])} |"
        )
    lines.append("")

    # Trajectories to look at
    lines.append("## Trajectories to look at\n")
    valid = [r for r in rows if np.isfinite(r["mean_final"])]
    if not valid:
        lines.append("(No runs with complete rewards.)\n")
        return "\n".join(lines)

    # Best by mean final
    by_mean = sorted(valid, key=lambda r: r["mean_final"], reverse=True)
    lines.append("### Highest final reward (likely successes)\n")
    for r in by_mean[:3]:
        lines.append(f"- **run_{r['run']:02d}** `{r['slug']}` — mean final = {r['mean_final']:.3f}")
        lines.append(f"  - Video: `{r['video_rel']}`")
        lines.append(f"  - Run folder: `{r['run_folder_rel']}`")
    lines.append("")

    # Worst by mean final
    lines.append("### Lowest final reward (likely failures)\n")
    for r in by_mean[-3:][::-1]:
        lines.append(f"- **run_{r['run']:02d}** `{r['slug']}` — mean final = {r['mean_final']:.3f}")
        lines.append(f"  - Video: `{r['video_rel']}`")
        lines.append(f"  - Run folder: `{r['run_folder_rel']}`")
    lines.append("")

    # Biggest RFM vs Robo disagreement
    by_rr = sorted(valid, key=lambda r: r["rfm_robo_delta"], reverse=True)
    lines.append("### Biggest RFM vs RoboReward disagreement\n")
    lines.append("(Worth inspecting: models reach different conclusions.)\n")
    n_rr = 0
    for r in by_rr[:3]:
        if r["rfm_robo_delta"] < 0.05:
            continue
        n_rr += 1
        lines.append(f"- **run_{r['run']:02d}** `{r['slug']}` — Δ = {r['rfm_robo_delta']:.3f} (v0: RFM {_f(r['v0_rfm_fin'])} vs Robo {_f(r['v0_robo_fin'])}, v1: RFM {_f(r['v1_rfm_fin'])} vs Robo {_f(r['v1_robo_fin'])})")
        lines.append(f"  - Video: `{r['video_rel']}`")
        lines.append(f"  - Run folder: `{r['run_folder_rel']}`")
    if n_rr == 0:
        lines.append("(None with Δ ≥ 0.05.)\n")
    lines.append("")

    # Biggest view0 vs view1 disagreement
    by_view = sorted(valid, key=lambda r: r["view_delta"], reverse=True)
    lines.append("### Biggest view0 vs view1 disagreement\n")
    lines.append("(Reward depends on camera view; worth checking.)\n")
    n_view = 0
    for r in by_view[:3]:
        if r["view_delta"] < 0.05:
            continue
        n_view += 1
        lines.append(f"- **run_{r['run']:02d}** `{r['slug']}` — view Δ = {r['view_delta']:.3f}")
        lines.append(f"  - Video: `{r['video_rel']}`")
        lines.append(f"  - Run folder: `{r['run_folder_rel']}`")
    if n_view == 0:
        lines.append("(None with view Δ ≥ 0.05.)\n")

    return "\n".join(lines)


def _extract_look_at_section(report_md: str) -> str:
    """Extract '## Trajectories to look at' for console printing."""
    marker = "## Trajectories to look at"
    idx = report_md.find(marker)
    if idx < 0:
        return ""
    chunk = report_md[idx:].strip()
    header = "[run_rollout_reward_comparison] REPORT — trajectories to look at\n"
    return header + "─" * 60 + "\n" + chunk + "\n" + "─" * 60


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run rollout reward comparison: X rollouts, split, RFM+RoboReward eval, plots.",
    )
    ap.add_argument("--num_trajectories", type=int, required=True, help="Number of rollout runs (X)")
    ap.add_argument("--task_type", type=str, default="pickplace", help="Task type for rollout")
    ap.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Root for rollouts, runs, plots (default: ./benchmark_results_<timestamp>)",
    )
    ap.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="If set, overrides {output_dir}/rollouts as save_dir for rollout",
    )
    ap.add_argument("--rfm_url", type=str, default="http://localhost:8004", help="RFM-RL eval server")
    ap.add_argument("--roboreward_url", type=str, default="http://localhost:8003", help="RoboReward eval server")
    ap.add_argument("--fps", type=float, default=1.0, help="FPS for use_reward_eval_server")
    # Passthrough for rollout (only passed when non-None)
    ap.add_argument("--svd_model_path", type=str, default=None)
    ap.add_argument("--clip_model_path", type=str, default=None)
    ap.add_argument("--ckpt_path", type=str, default=None)
    ap.add_argument("--pi_ckpt", type=str, default=None)
    ap.add_argument("--dataset_root_path", type=str, default=None)
    ap.add_argument("--dataset_meta_info_path", type=str, default=None)
    ap.add_argument("--dataset_names", type=str, default=None)
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Abort on first rollout or reward-eval failure",
    )
    ap.add_argument(
        "--show_rollout_logs",
        action="store_true",
        help="Stream rollout_interact_pi stdout/stderr instead of capturing (noisier but shows progress)",
    )
    ap.add_argument(
        "--num_parallel_rollouts",
        type=int,
        default=1,
        help="Run this many rollouts in parallel (default 1). Each uses its own model load; if OOM, lower XLA_PYTHON_CLIENT_MEM_FRACTION or reduce this.",
    )
    args = ap.parse_args()

    if args.output_dir is None:
        from datetime import datetime
        args.output_dir = f"./benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir)
    save_dir = Path(args.save_dir) if args.save_dir else (output_dir / "rollouts")
    project_root = Path(__file__).resolve().parent.parent

    # Resolve task_name for paths (Rollouts_interact_pi for interact_pi)
    try:
        from config import wm_args
        _cfg = wm_args(task_type=args.task_type)
        task_name = getattr(_cfg, "task_name", "Rollouts_interact_pi")
    except Exception:
        task_name = "Rollouts_interact_pi"

    video_dir = save_dir / task_name / "video"
    info_dir = save_dir / task_name / "info"
    single_views = video_dir / "single_views"
    runs_dir = output_dir / "runs"
    plots_dir = output_dir / "plots"

    runs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")

    passthrough = {
        "svd_model_path": args.svd_model_path,
        "clip_model_path": args.clip_model_path,
        "ckpt_path": args.ckpt_path,
        "pi_ckpt": args.pi_ckpt,
        "dataset_root_path": args.dataset_root_path,
        "dataset_meta_info_path": args.dataset_meta_info_path,
        "dataset_names": args.dataset_names,
    }

    # 1) Record existing videos so we can detect new ones if parsing fails
    pre = set(video_dir.glob("*.mp4")) if video_dir.exists() else set()

    def log(msg: str) -> None:
        print(f"[run_rollout_reward_comparison] {msg}", flush=True)

    n_par = max(1, int(args.num_parallel_rollouts))
    log(f"output_dir={output_dir} save_dir={save_dir} num_trajectories={args.num_trajectories} num_parallel_rollouts={n_par}")

    # 2) Run rollouts
    new_videos: list[Path] = []
    if n_par <= 1:
        for i in range(args.num_trajectories):
            log(f"Rollout run {i+1}/{args.num_trajectories} starting...")
            try:
                paths = run_rollout(
                    args.task_type, save_dir, project_root, passthrough, env,
                    show_logs=args.show_rollout_logs,
                )
                new_videos.extend(paths)
                log(f"Rollout run {i+1}/{args.num_trajectories} done, got {len(paths)} video(s).")
            except Exception as e:
                log(f"Rollout run {i+1}/{args.num_trajectories} failed: {e}")
                if args.strict:
                    raise
    else:
        log(f"Running {args.num_trajectories} rollouts with {n_par} in parallel...")
        with ThreadPoolExecutor(max_workers=n_par) as ex:
            futures = [
                ex.submit(
                    run_rollout,
                    args.task_type,
                    save_dir,
                    project_root,
                    passthrough,
                    env,
                    show_logs=args.show_rollout_logs,
                )
                for _ in range(args.num_trajectories)
            ]
            for f in as_completed(futures):
                try:
                    paths = f.result()
                    new_videos.extend(paths)
                except Exception as e:
                    log(f"Rollout failed: {e}")
                    if args.strict:
                        raise
        log("All rollouts finished.")
    if not new_videos:
        log("No 'Saving video to' lines parsed; using fallback (newest .mp4 in video_dir).")
        post = set(video_dir.glob("*.mp4")) if video_dir.exists() else set()
        added = sorted(post - pre, key=lambda p: p.stat().st_mtime)
        new_videos = added[-args.num_trajectories:] if len(added) >= args.num_trajectories else added
    if not new_videos:
        log("No videos to process. Exiting.")
        sys.exit(1)
    log(f"Total videos to process: {len(new_videos)}")

    # 3) Split
    log("Splitting videos into single-view (view0, view1, view2)...")
    run_split(video_dir, single_views, project_root)
    log("Split done.")

    # 4) Reward eval and per-run plots
    log("Reward eval (view0/view1 × RFM / RoboReward) and per-run plots...")
    manifest = []
    for i, video_path in enumerate(new_videos):
        log(f"Processing run {i+1}/{len(new_videos)}: {video_path.name}")
        stem = video_path.stem
        slug = trajectory_slug_from_stem(stem)
        run_dir = runs_dir / f"run_{i:02d}_{slug}"
        run_dir.mkdir(parents=True, exist_ok=True)

        task = get_task_from_info_or_stem(stem, info_dir)
        view0_path = single_views / f"{stem}_view0.mp4"
        view1_path = single_views / f"{stem}_view1.mp4"

        reward_files = {}
        for v, vname in [(0, "view0"), (1, "view1")]:
            vpath = single_views / f"{stem}_view{v}.mp4"
            if not vpath.exists():
                continue
            reward_files[vname] = {}
            for m, url, mkey in [
                ("rfm", args.rfm_url, "rfm"),
                ("roboreward", args.roboreward_url, "roboreward"),
            ]:
                out_npy = run_dir / f"view{v}_{m}_rewards.npy"
                log(f"Eval run {i} {vname} {mkey}...")
                try:
                    run_reward_eval(vpath, task, url, out_npy, args.fps, project_root)
                    reward_files[vname][mkey] = str(out_npy)
                except Exception as e:
                    log(f"Reward eval run {i} {vname} {mkey} failed: {e}")
                    if args.strict:
                        raise

        # meta.json
        meta = {
            "run_index": i,
            "trajectory_slug": slug,
            "video_path": str(video_path),
            "view0_path": str(view0_path) if view0_path.exists() else None,
            "view1_path": str(view1_path) if view1_path.exists() else None,
            "task": task,
        }
        with open(run_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Per-run plots: view0 and view1, RFM vs RoboReward
        plot_files = {}
        for v, vname in [(0, "view0"), (1, "view1")]:
            rfm_npy = run_dir / f"view{v}_rfm_rewards.npy"
            robo_npy = run_dir / f"view{v}_roboreward_rewards.npy"
            if rfm_npy.exists() and robo_npy.exists():
                rfm = np.load(rfm_npy)
                robo = np.load(robo_npy)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(rfm, label="RFM-RL")
                ax.plot(robo, label="RoboReward")
                ax.set_xlabel("Frame")
                ax.set_ylabel("Reward")
                ax.legend()
                title = f"RFM-RL vs RoboReward — run_{i:02d} {slug} — {vname} (ext1)" if v == 0 else f"RFM-RL vs RoboReward — run_{i:02d} {slug} — {vname} (ext2)"
                ax.set_title(title)
                plt.tight_layout()
                plot_path = run_dir / f"plot_{vname}_RFM_vs_RoboReward.png"
                fig.savefig(plot_path, dpi=150)
                plt.close(fig)
                plot_files[vname] = str(plot_path)

        manifest.append({
            "run_index": i,
            "trajectory_slug": slug,
            "run_folder": str(run_dir),
            "task": task,
            "video_path": str(video_path),
            "view0_path": str(view0_path) if view0_path.exists() else None,
            "view1_path": str(view1_path) if view1_path.exists() else None,
            "reward_files": reward_files,
            "plot_files": plot_files,
        })

    # 5) Aggregate plots: interpolate to 100 pts on [0,1], mean ± std per (model, view)
    log("Building aggregate plots...")
    n_runs = len(new_videos)
    n_pts = 100
    x_norm = np.linspace(0, 1, n_pts)

    for v, vname in [(0, "view0"), (1, "view1")]:
        rfm_stack, robo_stack = [], []
        for i in range(n_runs):
            run_dir = Path(manifest[i]["run_folder"])
            for arr, p in [(rfm_stack, run_dir / f"view{v}_rfm_rewards.npy"), (robo_stack, run_dir / f"view{v}_roboreward_rewards.npy")]:
                if p.exists():
                    r = np.load(p)
                    x_orig = np.linspace(0, 1, len(r))
                    arr.append(np.interp(x_norm, x_orig, r))
        fig, ax = plt.subplots(figsize=(8, 4))
        if rfm_stack:
            a = np.array(rfm_stack)
            ax.fill_between(x_norm, np.mean(a, axis=0) - np.std(a, axis=0), np.mean(a, axis=0) + np.std(a, axis=0), alpha=0.3)
            ax.plot(x_norm, np.mean(a, axis=0), label="RFM-RL")
        if robo_stack:
            a = np.array(robo_stack)
            ax.fill_between(x_norm, np.mean(a, axis=0) - np.std(a, axis=0), np.mean(a, axis=0) + np.std(a, axis=0), alpha=0.3)
            ax.plot(x_norm, np.mean(a, axis=0), label="RoboReward")
        ax.set_xlabel("Normalized time")
        ax.set_ylabel("Reward")
        ax.set_title(f"Aggregate {vname} — mean ± std over {n_runs} runs")
        ax.legend()
        ax.set_ylim(0, 1)
        plt.tight_layout()
        fig.savefig(plots_dir / f"aggregate_{vname}.png", dpi=150)
        plt.close(fig)

    # 6) manifest.json and _LABELS.txt
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    with open(output_dir / "_LABELS.txt", "w") as f:
        f.write("run_XX: run index (00, 01, ...)\n")
        f.write("time_YYYYMMDD_HHMMSS: timestamp from trajectory video stem\n")
        f.write("traj_ZZZ: validation id (e.g. scene_1, 0004)\n")
        f.write("trajectory_slug = time_YYYYMMDD_HHMMSS_traj_ZZZ — use to find a trajectory from a graph.\n")

    # 7) Detailed report: metrics and "trajectories to look at"
    report_md = build_report(manifest, output_dir, args.rfm_url, args.roboreward_url)
    report_path = output_dir / "REPORT.md"
    with open(report_path, "w") as f:
        f.write(report_md)
    log(f"Report written: {report_path}")

    # Print "trajectories to look at" to console
    look_at = _extract_look_at_section(report_md)
    if look_at:
        print(flush=True)
        print(look_at, flush=True)

    log(f"Done. Output: {output_dir}")
    log("  manifest.json, _LABELS.txt, REPORT.md")
    log("  runs/run_XX_<slug>/ — meta.json, view*_rfm/roboreward_rewards.npy, plot_*_RFM_vs_RoboReward.png")
    log("  plots/aggregate_view0.png, plots/aggregate_view1.png")


if __name__ == "__main__":
    main()

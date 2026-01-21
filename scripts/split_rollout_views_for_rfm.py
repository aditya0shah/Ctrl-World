#!/usr/bin/env python3
"""
Split 3-camera side-by-side rollout videos into 3 separate single-view videos
for RFM evaluation.

Supports two layouts from rollout_interact_pi.py:
  1. 3 side-by-side: 3 views in one row, equal width. (T, H, 3*W, C)
  2. 3x2 grid (GT + predicted): 3 columns, 2 rows (top=GT, bottom=predicted).
     (T, 2*H, 3*W, C). By default uses the bottom row (world model output).
     Use --row top to use the top row (ground truth from dataset).

Output: {stem}_view0.mp4, {stem}_view1.mp4, {stem}_view2.mp4
  (view0=left, view1=center, view2=right; for DROID-style, often ext1, ext2, wrist)

Example:
  python scripts/split_rollout_views_for_rfm.py --input synthetic_traj/Rollouts_interact_pi/video
  python scripts/split_rollout_views_for_rfm.py --input path/to/video.mp4 --output-dir ./single_views
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import mediapy

try:
    from decord import VideoReader, cpu

    def _read_frames(path: Path) -> tuple[np.ndarray, float]:
        vr = VideoReader(str(path), ctx=cpu(0), num_threads=1)
        frames = vr.get_batch(np.arange(len(vr))).asnumpy()
        try:
            fps = float(vr.get_avg_fps())
        except Exception:
            fps = 4.0
        del vr
        return frames, fps

except ImportError:
    VideoReader = None

    def _read_frames(path: Path) -> tuple[np.ndarray, float]:
        import imageio

        r = imageio.get_reader(str(path), "ffmpeg")
        frames = np.stack([r.get_data(i) for i in range(len(r))])
        fps = r.get_meta_data().get("fps", 4.0)
        r.close()
        return frames, fps


def split_video(
    video_path: str | Path,
    output_dir: Path | None = None,
    output_stem: str | None = None,
    fps: float | None = None,
    layout: str = "auto",
    row: str = "bottom",
) -> list[Path]:
    """
    Split a 3-view rollout video into 3 single-view videos.

    Args:
        video_path: Path to the side-by-side .mp4
        output_dir: Where to write the 3 videos. Default: same as input.
        output_stem: Stem for output filenames. Default: input stem.
        fps: FPS for output. Default: read from input, fallback 4.
        layout: "auto" | "sidebyside" | "grid". auto: detect from H vs W.
        row: "bottom" | "top". For grid: which row to use (bottom=pred, top=GT). Ignored for sidebyside.

    Returns:
        List of 3 output paths.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    frames, meta_fps = _read_frames(video_path)
    out_fps = fps if fps is not None else meta_fps

    T, H, W, C = frames.shape
    if W % 3 != 0:
        raise ValueError(
            f"Video width {W} is not divisible by 3. Expected 3 equal-width columns. path={video_path}"
        )

    view_w = W // 3
    out_dir = Path(output_dir) if output_dir else video_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem if output_stem is not None else video_path.stem

    # Detect or use layout
    if layout == "auto":
        # Grid: height is 2x the width of one view (square tiles in 3x2)
        if H == 2 * view_w:
            layout = "grid"
        else:
            layout = "sidebyside"
    elif layout not in ("sidebyside", "grid"):
        raise ValueError(f"layout must be 'auto', 'sidebyside', or 'grid', got {layout!r}")

    if layout == "sidebyside":
        # 3 views horizontally
        view_h = H
        base = frames  # (T, H, W, C)
    else:
        # 3x2 grid: pick one row
        view_h = H // 2
        if H != 2 * view_h:
            raise ValueError(
                f"Grid layout expects H=2*view_h (H={H}, view_w={view_w}). "
                "Use --layout sidebyside if this is 3-in-a-row only."
            )
        if row == "bottom":
            base = frames[:, view_h:, :, :]  # (T, view_h, W, C)
        elif row == "top":
            base = frames[:, :view_h, :, :]  # (T, view_h, W, C)
        else:
            raise ValueError(f"row must be 'bottom' or 'top', got {row!r}")

    out_paths = []
    for i in range(3):
        view = base[:, :, i * view_w : (i + 1) * view_w, :].copy()  # (T, view_h, view_w, C)
        out_name = f"{stem}_view{i}.mp4"
        out_path = out_dir / out_name
        mediapy.write_video(str(out_path), view, fps=out_fps)
        out_paths.append(out_path)

    return out_paths


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Split 3-view side-by-side rollout videos into 3 single-view videos for RFM."
    )
    ap.add_argument(
        "input",
        type=str,
        help="Input .mp4 path or directory of .mp4 files",
    )
    ap.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory. Default: same as input file, or <input>/single_views when input is a dir",
    )
    ap.add_argument(
        "--layout",
        type=str,
        choices=("auto", "sidebyside", "grid"),
        default="auto",
        help="Layout: auto (detect), sidebyside (3 in a row), grid (3x2, use --row to pick top/bottom)",
    )
    ap.add_argument(
        "--row",
        type=str,
        choices=("bottom", "top"),
        default="bottom",
        help="For grid: bottom=world model output (default), top=ground truth",
    )
    ap.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output FPS. Default: from input or 4",
    )
    ap.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Append to output stem before _viewN, e.g. --suffix _pred",
    )
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"Error: {inp} does not exist", file=sys.stderr)
        sys.exit(1)

    if inp.is_file():
        if inp.suffix.lower() != ".mp4":
            print(f"Error: expected .mp4, got {inp.suffix}", file=sys.stderr)
            sys.exit(1)
        files = [inp]
        out_dir = Path(args.output_dir) if args.output_dir else inp.parent
    else:
        files = sorted(inp.glob("*.mp4"))
        if not files:
            print(f"No .mp4 under {inp}", file=sys.stderr)
            sys.exit(1)
        out_dir = Path(args.output_dir) if args.output_dir else (inp / "single_views")

    stem_suffix = args.suffix
    failed = 0
    for f in files:
        stem = f.stem + stem_suffix if stem_suffix else None
        try:
            paths = split_video(
                f,
                output_dir=out_dir,
                output_stem=stem,
                fps=args.fps,
                layout=args.layout,
                row=args.row,
            )
            print(f"{f.name} -> {paths[0].name}, {paths[1].name}, {paths[2].name}")
        except Exception as e:
            print(f"Error {f}: {e}", file=sys.stderr)
            failed += 1

    if failed:
        print(f"Failed: {failed}/{len(files)}", file=sys.stderr)
        sys.exit(1)
    print(f"Wrote {len(files) * 3} videos to {out_dir}")


if __name__ == "__main__":
    main()

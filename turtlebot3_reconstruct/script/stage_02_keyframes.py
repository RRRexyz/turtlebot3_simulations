from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recon.config import Config, load_config
from recon.io_utils import ProjectPaths, save_json, setup_logger
from recon.keyframe_selector import KeyframeSelectionStats, select_keyframes, summarize_keyframes
from recon.sync_utils import load_synced_frames
from recon.types import FrameRecord, KeyframeRecord


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the keyframe selection stage."""

    parser = argparse.ArgumentParser(description="Select keyframes from synchronized RGB-D frames.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "default.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--robots",
        nargs="*",
        default=None,
        help="Optional subset of robots to process. Defaults to all robots in config.",
    )
    return parser.parse_args()


def _keyframe_json_payload(robot_name: str, keyframes: list[KeyframeRecord], stats: KeyframeSelectionStats) -> dict[str, object]:
    """Build the JSON payload saved to disk."""

    return {
        "robot_name": robot_name,
        "raw_frame_count": stats.raw_frame_count,
        "keyframe_count": stats.keyframe_count,
        "compression_ratio": stats.compression_ratio,
        "keyframes": [
            {
                "keyframe_index": keyframe.keyframe_index,
                "frame_index": keyframe.frame_index,
                "timestamp": keyframe.timestamp,
                "rgb_path": str(keyframe.frame.rgb_path),
                "depth_path": str(keyframe.frame.depth_path),
                "pose_matrix": keyframe.frame.pose.matrix,
            }
            for keyframe in keyframes
        ],
    }


def _plot_keyframe_stats(
    frames: list[FrameRecord],
    keyframes: list[KeyframeRecord],
    output_path: Path,
    robot_name: str,
) -> None:
    """Save a compact visualization of trajectory coverage and frame compression."""

    if not frames:
        return

    frame_positions = np.array([frame.pose.matrix[:3, 3] for frame in frames], dtype=np.float64)
    keyframe_positions = np.array([keyframe.frame.pose.matrix[:3, 3] for keyframe in keyframes], dtype=np.float64)
    frame_times = np.array([frame.timestamp for frame in frames], dtype=np.float64)
    keyframe_indices = np.array([keyframe.frame_index for keyframe in keyframes], dtype=np.int32)
    keyframe_times = np.array([keyframe.timestamp for keyframe in keyframes], dtype=np.float64)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(frame_positions[:, 0], frame_positions[:, 1], color="#9aa0a6", linewidth=1.5, label="All frames")
    axes[0].scatter(keyframe_positions[:, 0], keyframe_positions[:, 1], color="#d93025", s=25, label="Keyframes")
    axes[0].set_title("Trajectory XY")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(frame_times, np.arange(len(frames)), color="#1a73e8", linewidth=1.5, label="All frames")
    axes[1].scatter(keyframe_times, keyframe_indices, color="#f29900", s=30, label="Keyframes")
    axes[1].set_title("Frame Index Over Time")
    axes[1].set_xlabel("timestamp (s)")
    axes[1].set_ylabel("frame index")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    figure.suptitle(robot_name)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output_path), dpi=160)
    plt.close(figure)


def process_robot(config: Config, robot_name: str, paths: ProjectPaths) -> KeyframeSelectionStats:
    """Run keyframe selection for one robot and save outputs."""

    stage_dir = paths.stage_dir("stage_02_keyframes", robot_name)
    logger = setup_logger("stage_02_keyframes", paths.log_file("stage_02_keyframes", robot_name))
    frames = load_synced_frames(config, robot_name)
    keyframes = select_keyframes(frames, config)
    stats = summarize_keyframes(frames, keyframes, config)

    save_json(stage_dir / "keyframes.json", _keyframe_json_payload(robot_name, keyframes, stats))
    save_json(stage_dir / "keyframe_stats.json", asdict(stats))
    _plot_keyframe_stats(frames, keyframes, stage_dir / "keyframe_stats.png", robot_name)

    logger.info(
        "Processed %s | frames=%d keyframes=%d compression=%.3f",
        robot_name,
        stats.raw_frame_count,
        stats.keyframe_count,
        stats.compression_ratio,
    )
    return stats


def run_stage(config: Config, robot_names: Iterable[str] | None = None) -> list[tuple[str, KeyframeSelectionStats]]:
    """Run keyframe selection for all or selected robots."""

    paths = ProjectPaths(config=config)
    selected_robots = list(robot_names) if robot_names is not None else list(config.robots)
    return [(robot_name, process_robot(config, robot_name, paths)) for robot_name in selected_robots]


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    config = load_config(args.config)
    results = run_stage(config, args.robots)

    for robot_name, stats in results:
        print(
            f"{robot_name}: frames={stats.raw_frame_count}, "
            f"keyframes={stats.keyframe_count}, "
            f"compression={stats.compression_ratio:.3f}"
        )


if __name__ == "__main__":
    main()

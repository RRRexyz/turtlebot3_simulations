from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recon.config import Config, load_config
from recon.fragment_builder import (
    FragmentBuildStats,
    build_fragment_records,
    load_keyframes_from_json,
    summarize_fragments,
)
from recon.io_utils import ProjectPaths, save_json, save_pcd, setup_logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for fragment construction."""

    parser = argparse.ArgumentParser(description="Build local fragments from selected keyframes.")
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
    parser.add_argument(
        "--keyframes-per-fragment",
        type=int,
        default=None,
        help="Optional override for fragment size. Use 0 for one fragment per robot.",
    )
    return parser.parse_args()


def _fragment_json_payload(fragments: list, pointcloud_paths: list[Path]) -> list[dict[str, object]]:
    """Serialize fragment metadata for downstream stages."""

    payload: list[dict[str, object]] = []
    for fragment, pointcloud_path in zip(fragments, pointcloud_paths):
        payload.append(
            {
                "fragment_index": fragment.fragment_index,
                "robot_name": fragment.robot_name,
                "pointcloud_path": str(pointcloud_path),
                "global_pose": fragment.global_pose,
                "keyframes": [
                    {
                        "keyframe_index": keyframe.keyframe_index,
                        "frame_index": keyframe.frame_index,
                        "timestamp": keyframe.timestamp,
                        "rgb_path": str(keyframe.frame.rgb_path),
                        "depth_path": str(keyframe.frame.depth_path),
                        "pose_matrix": keyframe.frame.pose.matrix,
                    }
                    for keyframe in fragment.keyframes
                ],
            }
        )
    return payload


def process_robot(config: Config, robot_name: str, paths: ProjectPaths, keyframes_per_fragment: int | None) -> FragmentBuildStats:
    """Build and save fragments for one robot."""

    stage_dir = paths.stage_dir("stage_03_fragments", robot_name)
    logger = setup_logger("stage_03_fragments", paths.log_file("stage_03_fragments", robot_name))
    keyframe_json_path = paths.stage_file("stage_02_keyframes", "keyframes.json", robot_name)
    keyframes = load_keyframes_from_json(keyframe_json_path)
    fragments, pointclouds = build_fragment_records(
        keyframes=keyframes,
        config=config,
        keyframes_per_fragment=keyframes_per_fragment,
    )

    pointcloud_paths: list[Path] = []
    for fragment, pointcloud in zip(fragments, pointclouds):
        pointcloud_path = stage_dir / f"fragment_{fragment.fragment_index:03d}.ply"
        save_pcd(pointcloud_path, pointcloud)
        fragment.pointcloud_path = pointcloud_path
        pointcloud_paths.append(pointcloud_path)

    stats = summarize_fragments(robot_name, keyframes, pointclouds)
    save_json(stage_dir / "fragments.json", _fragment_json_payload(fragments, pointcloud_paths))
    save_json(stage_dir / "fragment_stats.json", asdict(stats))

    logger.info(
        "Processed %s | keyframes=%d fragments=%d points=%d",
        robot_name,
        stats.keyframe_count,
        stats.fragment_count,
        stats.total_point_count,
    )
    return stats


def run_stage(
    config: Config,
    robot_names: Iterable[str] | None = None,
    keyframes_per_fragment: int | None = None,
) -> list[tuple[str, FragmentBuildStats]]:
    """Run fragment construction for all or selected robots."""

    paths = ProjectPaths(config)
    selected_robots = list(robot_names) if robot_names is not None else list(config.robots)
    results: list[tuple[str, FragmentBuildStats]] = []
    for robot_name in selected_robots:
        stats = process_robot(config, robot_name, paths, keyframes_per_fragment)
        results.append((robot_name, stats))
    return results


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    config = load_config(args.config)
    results = run_stage(config, args.robots, args.keyframes_per_fragment)

    for robot_name, stats in results:
        print(
            f"{robot_name}: keyframes={stats.keyframe_count}, "
            f"fragments={stats.fragment_count}, "
            f"points={stats.total_point_count}"
        )


if __name__ == "__main__":
    main()

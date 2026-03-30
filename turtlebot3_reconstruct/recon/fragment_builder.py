from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recon.config import Config, load_config
from recon.io_utils import ProjectPaths, load_json
from recon.rgbd_utils import build_pointcloud_result
from recon.types import FragmentRecord, FrameRecord, KeyframeRecord, PoseRecord


@dataclass
class FragmentBuildStats:
    """Summary information for one robot's fragment construction."""

    robot_name: str
    keyframe_count: int
    fragment_count: int
    total_point_count: int


def load_keyframes_from_json(path: str | Path) -> list[KeyframeRecord]:
    """Load keyframes produced by stage_02_keyframes.py."""

    payload = load_json(path)
    robot_name = str(payload["robot_name"])
    keyframes: list[KeyframeRecord] = []
    for item in payload["keyframes"]:
        frame_index = int(item["frame_index"])
        timestamp = float(item["timestamp"])
        pose_matrix = np.asarray(item["pose_matrix"], dtype=np.float64)
        pose = PoseRecord(
            robot_name=robot_name,
            frame_index=frame_index,
            timestamp=timestamp,
            matrix=pose_matrix,
        )
        frame = FrameRecord(
            robot_name=robot_name,
            frame_index=frame_index,
            timestamp=timestamp,
            rgb_path=Path(item["rgb_path"]),
            depth_path=Path(item["depth_path"]),
            pose=pose,
        )
        keyframes.append(
            KeyframeRecord(
                keyframe_index=int(item["keyframe_index"]),
                frame=frame,
                downsampled_pose=pose_matrix.copy(),
            )
        )
    return keyframes


def split_keyframes_into_groups(
    keyframes: Sequence[KeyframeRecord],
    keyframes_per_fragment: int,
) -> list[list[KeyframeRecord]]:
    """Split keyframes into consecutive fragment groups."""

    if not keyframes:
        return []
    if keyframes_per_fragment <= 0:
        return [list(keyframes)]

    groups: list[list[KeyframeRecord]] = []
    for start_index in range(0, len(keyframes), keyframes_per_fragment):
        groups.append(list(keyframes[start_index : start_index + keyframes_per_fragment]))
    return groups


def merge_keyframe_pointclouds(
    keyframes: Sequence[KeyframeRecord],
    config: Config,
) -> o3d.geometry.PointCloud:
    """Fuse multiple keyframes into one fragment point cloud in world coordinates."""

    merged_pointcloud = o3d.geometry.PointCloud()
    for keyframe in keyframes:
        result = build_pointcloud_result(
            keyframe,
            config,
            coordinate_frame="world",
        )
        merged_pointcloud += result.pointcloud

    if len(merged_pointcloud.points) == 0:
        return merged_pointcloud

    merged_pointcloud = merged_pointcloud.voxel_down_sample(config.pointcloud_voxel_size)
    merged_pointcloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=max(config.pointcloud_voxel_size * 2.0, 0.05),
            max_nn=30,
        )
    )
    return merged_pointcloud


def build_fragment_records(
    keyframes: Sequence[KeyframeRecord],
    config: Config,
    keyframes_per_fragment: int | None = None,
) -> tuple[list[FragmentRecord], list[o3d.geometry.PointCloud]]:
    """Build fragment records and fused point clouds from consecutive keyframes."""

    resolved_group_size = config.fragment.keyframes_per_fragment if keyframes_per_fragment is None else keyframes_per_fragment
    keyframe_groups = split_keyframes_into_groups(keyframes, resolved_group_size)

    fragments: list[FragmentRecord] = []
    pointclouds: list[o3d.geometry.PointCloud] = []
    for fragment_index, keyframe_group in enumerate(keyframe_groups):
        if not keyframe_group:
            continue

        fragment_pointcloud = merge_keyframe_pointclouds(keyframe_group, config)
        first_pose = keyframe_group[0].frame.pose.matrix.copy()
        fragments.append(
            FragmentRecord(
                robot_name=keyframe_group[0].robot_name,
                fragment_index=fragment_index,
                keyframes=list(keyframe_group),
                global_pose=first_pose,
            )
        )
        pointclouds.append(fragment_pointcloud)

    return fragments, pointclouds


def summarize_fragments(robot_name: str, keyframes: Sequence[KeyframeRecord], pointclouds: Sequence[o3d.geometry.PointCloud]) -> FragmentBuildStats:
    """Summarize fragment construction for logs and JSON output."""

    return FragmentBuildStats(
        robot_name=robot_name,
        keyframe_count=len(keyframes),
        fragment_count=len(pointclouds),
        total_point_count=sum(len(pointcloud.points) for pointcloud in pointclouds),
    )


def main() -> None:
    """Simple example for manual verification."""

    config_path = PROJECT_ROOT / "config" / "default.yaml"
    config = load_config(config_path)
    paths = ProjectPaths(config)
    keyframe_json = paths.stage_file("stage_02_keyframes", "keyframes.json", config.robots[1])
    keyframes = load_keyframes_from_json(keyframe_json)
    fragments, pointclouds = build_fragment_records(keyframes, config)
    stats = summarize_fragments(config.robots[1], keyframes, pointclouds)

    print("Robot:", stats.robot_name)
    print("Keyframes:", stats.keyframe_count)
    print("Fragments:", stats.fragment_count)
    print("Total points:", stats.total_point_count)


if __name__ == "__main__":
    main()

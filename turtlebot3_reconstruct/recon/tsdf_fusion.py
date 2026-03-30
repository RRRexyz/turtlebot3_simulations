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
from recon.pair_selector import load_all_keyframes
from recon.rgbd_utils import create_camera_intrinsic, create_rgbd_image
from recon.types import KeyframeRecord, PoseRecord


@dataclass
class TSDFIntegrationStats:
    """Summary of the global TSDF integration step."""

    integrated_keyframe_count: int
    mesh_vertex_count: int
    mesh_triangle_count: int
    point_count: int


def create_tsdf_volume(config: Config) -> o3d.pipelines.integration.ScalableTSDFVolume:
    """Construct the configured scalable TSDF volume."""

    color_type = getattr(o3d.pipelines.integration.TSDFVolumeColorType, config.tsdf.color_type)
    return o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=config.tsdf.voxel_length,
        sdf_trunc=config.tsdf.sdf_trunc,
        color_type=color_type,
    )


def load_optimized_pose_map(path: str | Path) -> dict[str, np.ndarray]:
    """Load optimized keyframe poses exported by stage_05_optimize."""

    payload = load_json(path)
    optimized_pose_map: dict[str, np.ndarray] = {}
    for item in payload:
        optimized_pose_map[str(item["key"])] = np.asarray(item["optimized_pose"], dtype=np.float64)
    return optimized_pose_map


def apply_optimized_poses_to_keyframes(
    keyframes: Sequence[KeyframeRecord],
    optimized_pose_map: dict[str, np.ndarray],
) -> list[KeyframeRecord]:
    """Return keyframes with poses replaced by optimized graph poses when available."""

    updated_keyframes: list[KeyframeRecord] = []
    for keyframe in keyframes:
        optimized_pose = optimized_pose_map.get(keyframe.key, keyframe.frame.pose.matrix)
        updated_pose = PoseRecord(
            robot_name=keyframe.robot_name,
            frame_index=keyframe.frame_index,
            timestamp=keyframe.frame.pose.timestamp,
            matrix=np.asarray(optimized_pose, dtype=np.float64),
        )
        updated_frame = keyframe.frame.__class__(
            robot_name=keyframe.frame.robot_name,
            frame_index=keyframe.frame.frame_index,
            timestamp=keyframe.frame.timestamp,
            rgb_path=keyframe.frame.rgb_path,
            depth_path=keyframe.frame.depth_path,
            pose=updated_pose,
        )
        updated_keyframes.append(
            keyframe.__class__(
                keyframe_index=keyframe.keyframe_index,
                frame=updated_frame,
                downsampled_pose=np.asarray(optimized_pose, dtype=np.float64),
                pointcloud_path=keyframe.pointcloud_path,
            )
        )
    return updated_keyframes


def integrate_keyframes_into_volume(
    volume: o3d.pipelines.integration.ScalableTSDFVolume,
    keyframes: Sequence[KeyframeRecord],
    config: Config,
) -> int:
    """Integrate optimized keyframes into the TSDF volume."""

    integrated_count = 0
    for keyframe in keyframes:
        rgbd_image = create_rgbd_image(keyframe, config)
        intrinsic = create_camera_intrinsic(config, keyframe.robot_name)
        extrinsic = np.linalg.inv(keyframe.frame.pose.matrix)
        volume.integrate(rgbd_image, intrinsic, extrinsic)
        integrated_count += 1
    return integrated_count


def extract_fused_mesh(volume: o3d.pipelines.integration.ScalableTSDFVolume) -> o3d.geometry.TriangleMesh:
    """Extract and post-process the fused mesh."""

    mesh = volume.extract_triangle_mesh()
    if len(mesh.vertices) > 0:
        mesh.compute_vertex_normals()
    return mesh


def extract_fused_pointcloud(volume: o3d.pipelines.integration.ScalableTSDFVolume) -> o3d.geometry.PointCloud:
    """Extract the fused point cloud."""

    return volume.extract_point_cloud()


def integrate_global_map(
    config: Config,
    keyframes: Sequence[KeyframeRecord],
    optimized_pose_map: dict[str, np.ndarray],
) -> tuple[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud, TSDFIntegrationStats]:
    """Fuse all keyframes into a global TSDF map and extract final outputs."""

    updated_keyframes = apply_optimized_poses_to_keyframes(keyframes, optimized_pose_map)
    volume = create_tsdf_volume(config)
    integrated_count = integrate_keyframes_into_volume(volume, updated_keyframes, config)
    mesh = extract_fused_mesh(volume)
    pointcloud = extract_fused_pointcloud(volume)
    stats = TSDFIntegrationStats(
        integrated_keyframe_count=integrated_count,
        mesh_vertex_count=len(mesh.vertices),
        mesh_triangle_count=len(mesh.triangles),
        point_count=len(pointcloud.points),
    )
    return mesh, pointcloud, stats


def main() -> None:
    """Simple example for manual verification on optimized poses."""

    config_path = PROJECT_ROOT / "config" / "default.yaml"
    config = load_config(config_path)
    paths = ProjectPaths(config)
    optimized_pose_map = load_optimized_pose_map(paths.stage_file("stage_05_optimize", "optimized_poses.json"))
    keyframes = load_all_keyframes(config)
    mesh, pointcloud, stats = integrate_global_map(config, keyframes, optimized_pose_map)

    print("Integrated keyframes:", stats.integrated_keyframe_count)
    print("Mesh vertices:", stats.mesh_vertex_count)
    print("Point count:", stats.point_count)


if __name__ == "__main__":
    main()

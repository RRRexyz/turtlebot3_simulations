from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Literal

import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recon.config import Config, load_config
from recon.io_utils import ProjectPaths, load_json, load_rgbd_image
from recon.sync_utils import load_synced_frames
from recon.types import FrameRecord, KeyframeRecord


CoordinateFrame = Literal["camera", "world"]


@dataclass
class PointCloudBuildResult:
    """Point cloud generation result with explicit coordinate frame."""

    pointcloud: o3d.geometry.PointCloud
    rgbd_image: o3d.geometry.RGBDImage
    intrinsic: o3d.camera.PinholeCameraIntrinsic
    coordinate_frame: CoordinateFrame


def _frame_from_record(record: FrameRecord | KeyframeRecord) -> FrameRecord:
    """Normalize frame-like inputs."""

    if isinstance(record, KeyframeRecord):
        return record.frame
    return record


def create_camera_intrinsic(config: Config, robot_name: str | None = None) -> o3d.camera.PinholeCameraIntrinsic:
    """Construct camera intrinsics from exported metadata or config fallback."""

    if robot_name is not None:
        paths = ProjectPaths(config)
        intrinsic_path = paths.exported_camera_intrinsic_json(robot_name)
        if intrinsic_path.exists():
            data = load_json(intrinsic_path)
            return o3d.camera.PinholeCameraIntrinsic(
                width=int(data["width"]),
                height=int(data["height"]),
                fx=float(data["fx"]),
                fy=float(data["fy"]),
                cx=float(data["cx"]),
                cy=float(data["cy"]),
            )

    camera = config.camera
    return o3d.camera.PinholeCameraIntrinsic(
        width=camera.width,
        height=camera.height,
        fx=camera.fx,
        fy=camera.fy,
        cx=camera.cx,
        cy=camera.cy,
    )


def create_rgbd_image(record: FrameRecord | KeyframeRecord, config: Config) -> o3d.geometry.RGBDImage:
    """Load RGB and depth images and create an Open3D RGBDImage."""

    frame = _frame_from_record(record)
    return load_rgbd_image(frame.rgb_path, frame.depth_path, config)


def create_pointcloud(record: FrameRecord | KeyframeRecord, config: Config) -> o3d.geometry.PointCloud:
    """Create a raw point cloud from one synchronized frame."""

    frame = _frame_from_record(record)
    rgbd_image = create_rgbd_image(frame, config)
    intrinsic = create_camera_intrinsic(config, frame.robot_name)
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)


def voxel_downsample(pointcloud: o3d.geometry.PointCloud, voxel_size: float) -> o3d.geometry.PointCloud:
    """Downsample a point cloud using a voxel grid."""

    if voxel_size <= 0.0:
        return pointcloud
    return pointcloud.voxel_down_sample(voxel_size)


def estimate_normals(
    pointcloud: o3d.geometry.PointCloud,
    voxel_size: float,
    max_nn: int = 30,
) -> o3d.geometry.PointCloud:
    """Estimate normals using a search radius derived from voxel size."""

    if len(pointcloud.points) == 0:
        return pointcloud

    search_radius = max(voxel_size * 2.0, 0.05)
    pointcloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=max_nn)
    )
    return pointcloud


def remove_outliers(
    pointcloud: o3d.geometry.PointCloud,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> o3d.geometry.PointCloud:
    """Remove sparse outlier points with a statistical filter."""

    if len(pointcloud.points) < nb_neighbors:
        return pointcloud
    filtered_pointcloud, _ = pointcloud.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    return filtered_pointcloud


def preprocess_pointcloud(
    pointcloud: o3d.geometry.PointCloud,
    config: Config,
    voxel_size: float | None = None,
    estimate_normals_flag: bool = True,
    remove_outliers_flag: bool = True,
) -> o3d.geometry.PointCloud:
    """Apply standard point cloud preprocessing for reconstruction."""

    resolved_voxel_size = config.pointcloud_voxel_size if voxel_size is None else voxel_size
    processed = voxel_downsample(pointcloud, resolved_voxel_size)
    if remove_outliers_flag:
        processed = remove_outliers(processed)
    if estimate_normals_flag:
        processed = estimate_normals(processed, resolved_voxel_size)
    return processed


def build_pointcloud_result(
    record: FrameRecord | KeyframeRecord,
    config: Config,
    coordinate_frame: CoordinateFrame = "camera",
    voxel_size: float | None = None,
    estimate_normals_flag: bool = True,
    remove_outliers_flag: bool = True,
) -> PointCloudBuildResult:
    """Create a preprocessed point cloud in camera or world coordinates."""

    frame = _frame_from_record(record)
    rgbd_image = create_rgbd_image(frame, config)
    intrinsic = create_camera_intrinsic(config, frame.robot_name)
    pointcloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    pointcloud = preprocess_pointcloud(
        pointcloud,
        config,
        voxel_size=voxel_size,
        estimate_normals_flag=estimate_normals_flag,
        remove_outliers_flag=remove_outliers_flag,
    )

    if coordinate_frame == "world":
        pointcloud.transform(frame.pose.matrix)
    elif coordinate_frame != "camera":
        raise ValueError(f"Unsupported coordinate_frame '{coordinate_frame}'.")

    return PointCloudBuildResult(
        pointcloud=pointcloud,
        rgbd_image=rgbd_image,
        intrinsic=intrinsic,
        coordinate_frame=coordinate_frame,
    )


def main() -> None:
    """Simple example for manual verification on exported data."""

    config_path = PROJECT_ROOT / "config" / "default.yaml"
    config = load_config(config_path)
    robot_name = config.robots[1]
    frames = load_synced_frames(config, robot_name)
    if not frames:
        print("No synchronized frames found for", robot_name)
        return

    result = build_pointcloud_result(frames[0], config, coordinate_frame="world")
    print("Robot:", robot_name)
    print("Coordinate frame:", result.coordinate_frame)
    print("Point count:", len(result.pointcloud.points))


if __name__ == "__main__":
    main()
